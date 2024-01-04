import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CURL_CA_BUNDLE'] = ''

import torch
import argparse
import contexttimer
from colorama import Fore, Style
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, AutoModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration
from datasets import load_dataset

from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2, multi_speculative_sampling, beam_speculative_sampling
from sampling import BiLD_sampling
from globals import Decoder
import json
from time import process_time_ns
from tqdm import tqdm
import evaluate as hf_evaluate

import numpy as np
import random


def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--input', type=str, default="Any recommendations for my holidays in Abu Dhabi?")
    parser.add_argument('--approx_model_name', type=str, default="facebook/opt-125m")
    parser.add_argument('--target_model_name', type=str, default="facebook/opt-350m")
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    parser.add_argument('--seed', '-s', type=int, default=None, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
    parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
    parser.add_argument('--max_tokens', '-M', type=int, default=20, help='max token number generated.')
    parser.add_argument('--gamma', '-g', type=int, default=8, help='guess time.')
    parser.add_argument('--width', '-w', type=int, default=4, help='guess width.')
    parser.add_argument('--acc_rate_head_path', type=str, default=None)
    parser.add_argument('--log_file', type=str, default="logs/log.txt")
    parser.add_argument('--dataset', type=str, default='cnndm')
    args = parser.parse_args()
    return args


def color_print(text):
    print(Fore.RED + text + Style.RESET_ALL)
    
def benchmark(fn, print_prefix, use_profiler=True, *args, **kwargs):
    TEST_TIME = 10
    profile_filename = f"./profile_logs/{print_prefix}"
    
    with contexttimer.Timer() as t:
        if use_profiler:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1, skip_first=0),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_filename),
                record_shapes=False,
                profile_memory=False,
                # with_stack=True
            ) as prof:
                for _ in range(TEST_TIME): 
                    output = fn(*args, **kwargs)
                    prof.step()
        else:
            for _ in range(TEST_TIME): 
                output = fn(*args, **kwargs)

    print(f"\n [benchmark] {print_prefix}, tokens/sec: {len(output[0]) / (t.elapsed / TEST_TIME)}, {t.elapsed / TEST_TIME} sec generates {len(output[0])} tokens")

def my_benchmark(print_prefix, model, input_ids, max_length, top_k, top_p, pad_token_id):
    TEST_TIME = 10
    with contexttimer.Timer() as t:
        for _ in range(TEST_TIME):
            set_seed(42)
            output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, do_sample=False, top_k=top_k, top_p=top_p,
                    pad_token_id=pad_token_id)
    print(f"\n [benchmark] {print_prefix}, tokens/sec: {len(output[0]) / t.elapsed / TEST_TIME}, {t.elapsed / TEST_TIME} sec generates {len(output[0])} tokens")


def generate(input_text, approx_model_name, target_model_name, num_tokens=20, gamma = 4,
             random_seed = None, verbose = False, use_benchmark = False, use_profiling = False):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True)
  
    Decoder().set_tokenizer(tokenizer)
    
    print(f"begin loading models: \n {approx_model_name} \n {target_model_name}")
    if 't5' in approx_model_name:
        small_model = AutoModelForSeq2SeqLM.from_pretrained(approx_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True)

    else:
        small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True)

    if 't5' in target_model_name:
        large_model = AutoModelForSeq2SeqLM.from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True)

    else:
        large_model = AutoModelForCausalLM.from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True)

    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch_device)

    top_k = 20
    top_p = 0.9
    repeats = 10

 
    print("BEAM")
    total_time = 0
    ave_time = 0
    for _ in range(repeats):
        torch.manual_seed(123)
        t = process_time_ns()
        output = multi_speculative_sampling(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed)
        total_time += (process_time_ns()-t)/1e9
        ave_time += (process_time_ns()-t)/1e9/len(output[0])

    print('BEAM model time', total_time/repeats, ave_time/repeats)



    print("DIVERSE")
    total_time = 0
    ave_time = 0
    for _ in range(repeats):
        torch.manual_seed(123)
        t = process_time_ns()
        output = multi_speculative_sampling(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed, strategy="diverse")
        total_time += (process_time_ns()-t)/1e9
        ave_time += (process_time_ns()-t)/1e9/len(output[0])

    print('DIVERSE model time', total_time/repeats, ave_time/repeats)




    print("conventional")
    total_time = 0
    ave_time = 0
    for _ in range(repeats):
        torch.manual_seed(123)
        t = process_time_ns()
        output = speculative_sampling(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed)
        total_time += (process_time_ns()-t)/1e9
        ave_time += (process_time_ns()-t)/1e9/len(output[0])

    print('conventional model time', total_time/repeats, ave_time/repeats)

def get_score(output, target_model, input_len):
    with torch.no_grad():
        if target_model.config.is_encoder_decoder == False:
            logits = target_model(output).logits
            logits = logits[:,:-1,:]
            logits = torch.nn.functional.log_softmax(logits, dim=-1)
            logits = torch.gather(logits,
                          dim = -1,
                          index = output[:,1:,None])
            if logits.isnan().any():
                print(logits.size())
                print(logits)
                print(old_logits)
                xxx = input()

            return torch.mean(logits[:,input_len-1:,:])
        else:
            logits = target_model(output[:, :input_len], decoder_input_ids=output[:,input_len:]).logits
            logits = logits[:, :-1, :]
            logits = torch.nn.functional.log_softmax(logits, dim=-1)
            logits = torch.gather(logits,
                                  dim = -1,
                                  index = output[:, input_len+1:, None])
            return torch.mean(logits)


def evaluate(approx_model_name, target_model_name, 
        dataset, 
        acc_rate_head_path = None, num_tokens=20, 
            #gamma = 4, width=1,
             random_seed = None, verbose = False, log_file = "logs/log.txt"):
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_f = open(log_file, 'w')
    
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True)
    tokenizer2 = AutoTokenizer.from_pretrained(target_model_name, trust_remote_code=True)
    print(approx_model_name, file=log_f)
    print(target_model_name, file=log_f)

    vocab1 = tokenizer.get_vocab()
    vocab2 = tokenizer2.get_vocab()
    if vocab1 == vocab2:
        print("Vocabularies are the same. Proceed")
    else:
        print("Vocabularies are different.")
        print("Vocabularies are different.", file=log_f)
        return
    Decoder().set_tokenizer(tokenizer)
    
    print(f"begin loading models: \n {approx_model_name} \n {target_model_name}")
    if 't5' in approx_model_name:
        small_model = AutoModelForSeq2SeqLM.from_pretrained(approx_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True)

    else:
        if 'AWQ' in approx_model_name:
            from awq import AutoAWQForCausalLM
            small_model = AutoAWQForCausalLM.from_quantized(approx_model_name, fuse_layers=True,
                                          trust_remote_code=True, safetensors=True,
                                          device_map="auto")
        else:
            small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True)

    if 't5' in target_model_name:
        large_model = AutoModelForSeq2SeqLM.from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True)

    else:
        large_model = AutoModelForCausalLM.from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True)

    if acc_rate_head_path is not None:
        final_dim = 768
        acc_rate_head = torch.nn.Sequential(
                                torch.nn.Linear(final_dim, 100),
                                torch.nn.ReLU(),
                                torch.nn.Linear(100,1)
                                )
        state_dict = torch.load(acc_rate_head_path)
        acc_rate_head.load_state_dict(state_dict)
        acc_rate_head.to(torch_device)
    else:
        acc_rate_head = None



    top_k = 20
    top_p = 0.9
    repeats = 10
    
    if dataset == 'cnndm':
        dataset = load_dataset('cnn_dailymail', '3.0.0', split='test[:10]')
        prefix = 'Summarize: '
        input_dataset = [tokenizer.encode(prefix + s['article'], return_tensors="pt", max_length=1024) for s in dataset]
        output_dataset = [[s['highlights']] for s in dataset]
    else:
        raise RuntimeError(f"Unrecognized dataset {dataset}")
    # split dataset based on input length
    #length_interval = [100,200,400,800]


    # split dataset based on input length
    #length_interval = [100,200,400,800]
    length_interval = [100000]

    rouge = hf_evaluate.load('rouge') 

    for i in range(len(length_interval)):
        u = length_interval[i]
        if i > 0:
            l = length_interval[i-1]
        else:
            l = 0
        ds = [pt for pt in input_dataset if (pt.size(-1) < u and pt.size(-1) >= l)]
        print(f'input length {l}-{u}, {len(ds)} data in total')
        total_input_tokens = sum([d.size(1) for d in ds])
        print('total_input_tokens', total_input_tokens)
        # small model github implementation
         
        total_time = 0
        total_token = 0
        approx_time = 0
        target_time = 0
        other_time = 0
        target_times = 0
        scores = []
        pred_seq = []

        
        for input_ids in tqdm(ds):
            input_ids = input_ids.to(torch_device)
            t = process_time_ns()
            output = autoregressive_sampling(input_ids, small_model, num_tokens, eos_token_id = tokenizer.eos_token_id, 
                    top_k = top_k, top_p=top_p, pad_token_id = tokenizer.pad_token_id)
            total_time += process_time_ns() - t
            total_token += len(output[0]) - input_ids.size(1)
            score = get_score(output, large_model, input_ids.size(1))
            scores.append(score.item())
            pred_seq.append(tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True))

        print(f'\nsmall model (gpu) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token, prob score = {np.mean(scores)}')
        print(f'\nsmall model (gpu) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token, prob score = {np.mean(scores)}', file=log_f)
        print(pred_seq[0])
        print(output_dataset[0])
        rouge_score = rouge.compute(predictions = pred_seq, references = output_dataset)
        print(f'rouge score = {rouge_score}')
        print(f'rouge score = {rouge_score}', file=log_f)

        # large model github implementation
        total_time = 0
        total_token = 0
        approx_time = 0
        target_time = 0
        other_time = 0
        target_times = 0
        scores = []
        pred_seq = []
        for input_ids in tqdm(ds):
            input_ids = input_ids.to(torch_device)
            t = process_time_ns()
            output = autoregressive_sampling(input_ids, large_model, num_tokens, eos_token_id = tokenizer.eos_token_id, 
                    top_k = top_k, top_p=top_p, pad_token_id = tokenizer.pad_token_id)
            total_time += process_time_ns() - t
            total_token += len(output[0]) - input_ids.size(1)
            score = get_score(output, large_model, input_ids.size(1))
            scores.append(score.item())
            pred_seq.append(tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True))


        print(f'\nlarge model total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token, prob_score = {np.mean(scores)}', file=log_f)
        print(f'\nlarge model total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token, prob_score = {np.mean(scores)}')
        rouge_score = rouge.compute(predictions = pred_seq, references = output_dataset)
        print(f'rouge score = {rouge_score}')
        print(f'rouge score = {rouge_score}', file=log_f)
        print(pred_seq[0])
        print(output_dataset[0])
       
        # large model beam sample 
        total_time = 0
        total_token = 0
        approx_time = 0
        target_time = 0
        other_time = 0
        target_times = 0
        scores = []
        pred_seq = []
        for input_ids in tqdm(ds):
            input_ids = input_ids.to(torch_device)
            t = process_time_ns()
            output = large_model.generate(input_ids, max_new_tokens = num_tokens, num_return_sequences=1, do_sample=True, top_k=top_k, top_p=top_p,
                    num_beams = 4,
                    pad_token_id=tokenizer.eos_token_id)
            #autoregressive_sampling(input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)
            total_time += process_time_ns() - t

            if large_model.config.is_encoder_decoder == True:
                output = torch.cat((input_ids, output), dim=-1)
            total_token += len(output[0]) - input_ids.size(1)
            score = get_score(output, large_model, input_ids.size(1))
            if score.isnan().any():
                print(input_ids.size())
                print(output)
                print(output.size())
                print(score)
                xxx = input()
            scores.append(score.item())
            pred_seq.append(tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True))


        print(f'\nlarge model beam sample width {4} total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token, prob_score = {np.mean(scores)}', file=log_f)
        print(f'\nlarge model beam sample width {4} total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token, prob_score = {np.mean(scores)}')
        rouge_score = rouge.compute(predictions = pred_seq, references = output_dataset)
        print(f'rouge score = {rouge_score}')
        print(f'rouge score = {rouge_score}', file=log_f)
        
        # convetional speculative decoding
        total_time = 0
        total_token = 0
        approx_time = 0
        target_time = 0
        other_time = 0
        target_times = 0
        total_acc_len = 0
        acc_rate = []
        target_times = 0
        approx_times = 0
        scores = []
        pred_seq = []
        for input_ids in tqdm(ds):
            input_ids = input_ids.to(torch_device)
            t = process_time_ns()
            output, details = speculative_sampling(input_ids, small_model, large_model, 
                    eos_token_id = tokenizer.eos_token_id,
                    pad_token_id = tokenizer.pad_token_id,
                    max_len = num_tokens, 
                    top_k = top_k, top_p=top_p, random_seed = random_seed, details=True)
            total_time += process_time_ns() - t
            total_token += len(output[0])- input_ids.size(1)
            approx_time += details['approx_time']
            target_time += details['target_time']
            other_time += details['other_time']
            total_acc_len += np.sum(details['acc_len'])
            acc_rate.append(details['acc_rate'])
            target_times += details['target_call_times']
            approx_times += details['approx_call_times']
            score = get_score(output, large_model, input_ids.size(1))
            scores.append(score.item())
            pred_seq.append(tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True))

        print(f'\n google speculative decoding (with KVCache) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token', file=log_f)
        print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}", file=log_f)
        print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}", file=log_f)
        print(f"prob score = {np.mean(scores)}", file=log_f)

        print(f'\n google speculative decoding (with KVCache) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token')
        print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}")
        print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}")
        print(f"prob score = {np.mean(scores)}")

        rouge_score = rouge.compute(predictions = pred_seq, references = output_dataset)
        print(f'rouge score = {rouge_score}')
        print(f'rouge score = {rouge_score}', file=log_f)
 
        
        # BiLD speculative decoding
        for fallback_thres in [0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,0.9]:
            for rollback_thres in range(1,10):
                total_time = 0
                total_token = 0
                approx_time = 0
                target_time = 0
                other_time = 0
                target_times = 0
                total_acc_len = 0
                acc_rate = []
                target_times = 0
                approx_times = 0
                scores = []
                pred_seq = []

                for input_ids in tqdm(ds):

                    input_ids = input_ids.to(torch_device)
                    t = process_time_ns()
            
                    output, details = BiLD_sampling(input_ids, small_model, large_model, 
                      fallback_thres = fallback_thres, rollback_thres = rollback_thres, 
                      gamma = 10, eos_token_id = tokenizer.eos_token_id, 
                      pad_token_id = tokenizer.pad_token_id, max_len = num_tokens, 
                      top_k = top_k, top_p=top_p, 
                      random_seed = random_seed, details=True)

                    total_time += process_time_ns() - t
                    total_token += len(output[0])- input_ids.size(1)
                    approx_time += details['approx_time']
                    target_time += details['target_time']
                    other_time += details['other_time']
                    total_acc_len += np.sum(details['acc_len'])
                    acc_rate.append(details['acc_rate'])
                    target_times += details['target_call_times']
                    approx_times += details['approx_call_times']
                    score = get_score(output, large_model, input_ids.size(1))
                    scores.append(score.item())
                    pred_seq.append(tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True))


                print(f'\n BiLD decoding {(fallback_thres, rollback_thres)} total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token', file=log_f)
                print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}", file=log_f)
                print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}", file=log_f)
                print(f"prob score = {np.mean(scores)}", file=log_f)       
        
                print(f'\n BiLD decoding {(fallback_thres, rollback_thres)} total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token')
                print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}")
                print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}")
                print(f"prob score = {np.mean(scores)}")  

                rouge_score = rouge.compute(predictions = pred_seq, references = output_dataset)
                print(f'rouge score = {rouge_score}')
                print(f'rouge score = {rouge_score}', file=log_f)
                #break
            #break

        # true beam speculative decoding
        for gamma in [2,4,6,8]:
            for width in [2,4,6,8]:
                if gamma * width > 32:
                    break
                total_time = 0
                total_token = 0
                approx_time = 0
                target_time = 0
                other_time = 0
                target_times = 0
                total_acc_len = 0
                acc_rate = []
                target_times = 0
                approx_times = 0
                scores = []
                pred_seq = []
                for input_ids in tqdm(ds):

                    input_ids = input_ids.to(torch_device)
                    t = process_time_ns()

                    output, details = beam_speculative_sampling(input_ids, small_model, large_model, 
                      eos_token_id = tokenizer.eos_token_id,
                      pad_token_id = tokenizer.pad_token_id, max_len = num_tokens, 
                      gamma = gamma, width=width, top_k = top_k, top_p=top_p, 
                      random_seed = random_seed, details=True)

                    total_time += process_time_ns() - t
                    total_token += len(output[0])- input_ids.size(1)
                    approx_time += details['approx_time']
                    target_time += details['target_time']
                    other_time += details['other_time']
                    total_acc_len += np.sum(details['acc_len'])
                    acc_rate.append(details['acc_rate'])
                    target_times += details['target_call_times']
                    approx_times += details['approx_call_times']
                    score = get_score(output, large_model, input_ids.size(1))
                    scores.append(score.item())
                    pred_seq.append(tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True))

                print(f'\n true beam speculative decoding (gamma {gamma}, width {width}) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token', file=log_f)
                print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}", file=log_f)
                print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}", file=log_f)
                print(f"prob score = {np.mean(scores)}", file=log_f)       
        
                print(f'\n true beam speculative decoding (with KVCache) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token')
                print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}")
                print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}")
                print(f"prob score = {np.mean(scores)}")  
                rouge_score = rouge.compute(predictions = pred_seq, references = output_dataset)
                print(f'rouge score = {rouge_score}')
                print(f'rouge score = {rouge_score}', file=log_f)
                #break
            #break

         
         
        # beam speculative decoding
        for gamma in [2,4,6,8]:
            for width in [2,4,6,8]:
                if gamma * width > 32:
                    break
 
                total_time = 0
                total_token = 0
                approx_time = 0
                target_time = 0
                other_time = 0
                target_times = 0
                total_acc_len = 0
                acc_rate = []
                target_times = 0
                approx_times = 0
                scores = []
                pred_seq = []
                for input_ids in tqdm(ds):

                    input_ids = input_ids.to(torch_device)
                    t = process_time_ns()

                    output, details = multi_speculative_sampling(input_ids, small_model, large_model, 
                      eos_token_id = tokenizer.eos_token_id,
                      pad_token_id = tokenizer.pad_token_id, max_len = num_tokens, 
                      gamma = gamma, width=width, top_k = top_k, top_p=top_p, 
                      random_seed = random_seed, details=True)

                    total_time += process_time_ns() - t
                    total_token += len(output[0])- input_ids.size(1)
                    approx_time += details['approx_time']
                    target_time += details['target_time']
                    other_time += details['other_time']
                    total_acc_len += np.sum(details['acc_len'])
                    acc_rate.append(details['acc_rate'])
                    target_times += details['target_call_times']
                    approx_times += details['approx_call_times']
                    score = get_score(output, large_model, input_ids.size(1))
                    scores.append(score.item())
                    pred_seq.append(tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True))

                print(f'\n beam speculative decoding (gamma {gamma}, width {width}) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token', file=log_f)
                print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}", file=log_f)
                print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}", file=log_f)
                print(f"prob score = {np.mean(scores)}", file=log_f)       
        
                print(f'\n beam speculative decoding (gamma {gamma}, width {width}) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token')
                print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}")
                print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}")
                print(f"prob score = {np.mean(scores)}")  
                rouge_score = rouge.compute(predictions = pred_seq, references = output_dataset)
                print(f'rouge score = {rouge_score}')
                print(f'rouge score = {rouge_score}', file=log_f)
                #break
            #break

       
        # iid beam speculative decoding
        for gamma in [2,4,6,8]:
            for width in [2,4,6,8]:
                if gamma * width > 32:
                    break
 
                total_time = 0
                total_token = 0
                approx_time = 0
                target_time = 0
                other_time = 0
                target_times = 0
                total_acc_len = 0
                acc_rate = []
                target_times = 0
                approx_times = 0
                scores = []
                pred_seq = []
                for input_ids in tqdm(ds):

                    input_ids = input_ids.to(torch_device)
                    t = process_time_ns()


                    output, details = multi_speculative_sampling(input_ids, small_model, large_model,  
                      eos_token_id = tokenizer.eos_token_id,
                      pad_token_id = tokenizer.pad_token_id, max_len = num_tokens, 
                      gamma = gamma, top_k = top_k, top_p=top_p, width = width,
                      random_seed = random_seed, 
                      details=True, strategy='iid')

                    total_time += process_time_ns() - t
                    total_token += len(output[0])- input_ids.size(1)
                    approx_time += details['approx_time']
                    target_time += details['target_time']
                    other_time += details['other_time']
                    total_acc_len += np.sum(details['acc_len'])
                    acc_rate.append(details['acc_rate'])
                    target_times += details['target_call_times']
                    approx_times += details['approx_call_times']
                    score = get_score(output, large_model, input_ids.size(1))
                    scores.append(score.item())
                    pred_seq.append(tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True))


                print(f'\niid speculative decoding (gamma {gamma}, width {width}) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token', file=log_f)
                print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}", file=log_f)
                print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}", file=log_f)
                print(f"prob score = {np.mean(scores)}", file=log_f)

                print(f'\niid speculative decoding (gamma {gamma}, width {width}) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token')
                print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}")
                print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}")
                print(f"prob score = {np.mean(scores)}")
                rouge_score = rouge.compute(predictions = pred_seq, references = output_dataset)
                print(f'rouge score = {rouge_score}')
                print(f'rouge score = {rouge_score}', file=log_f)
                #break
            #break





if __name__ == "__main__":
    args = parse_arguments()
    evaluate(args.approx_model_name, args.target_model_name, 
            dataset = args.dataset,
            num_tokens=args.max_tokens, 
            #gamma=args.gamma, width=args.width,
             acc_rate_head_path = args.acc_rate_head_path,
             log_file = args.log_file,
             random_seed = args.seed, verbose=args.verbose)
    #generate(args.input, args.approx_model_name, args.target_model_name, num_tokens=args.max_tokens, gamma=args.gamma,
    #         random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark)
