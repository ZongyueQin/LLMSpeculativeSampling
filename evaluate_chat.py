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
import time
import numpy as np
import random
import subprocess
import pickle

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
    parser.add_argument('--dataset', type=str, default='chatalpaca')
    parser.add_argument('--max_seconds', type=int, default=7200, help='timeout seconds')
    args = parser.parse_args()
    return args


def color_print(text):
    print(Fore.RED + text + Style.RESET_ALL)
    

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

def get_total_power(outputs, t1, t2, fname):
    with open(fname, 'wb') as f:
        pickle.dump((outputs, t1, t2), f)
    x = [out.strip().split() for out in outputs]
#    for xx in x:
#        if len(xx) < 2:
#            print(outputs)
#            print(x)
#            print(xx)
    x = [[float(xx[0]), float(xx[1])] for xx in x if len(xx) >= 2] # it seems possible that the last output of nvidia-smi is missing
    total_power = 0
    for timestamp, power in x:
        if timestamp > t1 and timestamp < t2:
            total_power += power
    return total_power


def evaluate(approx_model_name, target_model_name, 
        dataset_name, 
        acc_rate_head_path = None, num_tokens=20, 
        max_seconds = 7200,
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
        if 'GPTQ' in approx_model_name:
            small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, 
                                                       device_map="auto",
                                                       trust_remote_code=True)
            small_model.generation_config.pad_token_id = tokenizer.eos_token_id

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
    elif 'GPTQ' in target_model_name:
        large_model = AutoModelForCausalLM.from_pretrained(target_model_name, 
                                                       device_map="auto",
                                                       trust_remote_code=True)
        large_model.generation_config.pad_token_id = tokenizer.eos_token_id

    else:
        large_model = AutoModelForCausalLM.from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True)


    top_k = 20
    top_p = 0.9
    repeats = 10
    
    if dataset_name == 'chatalpaca':

        if 't5' in approx_model_name:
            prefix = ''
            postfix = ''
            #BiLD_params = [(0.2,2), (0.3,1)]
            BiLD_params = []
            multi_params = [(2,1,8), (2,2,8), (2,4,8), (2,6,8), (2,8,8),(2,2,2), (2,2,4), (2,2,8), (2,2,16),(2,2,32)]
        elif 'opt' in approx_model_name:
            prefix = ''
            postfix = ''
#            BiLD_params = [(0.2, 2), (0.3, 2), (0.3, 3)]
#            multi_params = [(4,4), (6,4), (4,6)]
            BiLD_params = []
#            multi_params = [(2,1,8), (2,2,8), (2,4,8), (2,6,8), (2,8,8),(2,2,2), (2,2,4), (2,2,8), (2,2,16),(2,2,32)]
            multi_params = [(6,1,8), (6,2,8), (6,4,8), (6,6,8), (6,8,8)]
            iid_params = []

        elif 'GPTQ' in approx_model_name:
            prefix = ""
            postfix = ''
            #BiLD_params = [(0.2,2)]
#            BiLD_params = [(0.9,1),(0.9,2)]
            BiLD_params = []
            multi_params = [(8,1,2), (8,2,2), (8,1,8), (8,2,8)]#,(4,6,8),(4,8,8),(6,4,8)]
 #           iid_params = [(4,2), (4,6), (4,8)]
            iid_params = []
            #multi_params = [(4,2)]
        else:
            prefix = ''
            postfix = ''

        import json
        raw_data = []
        with open('chatalpaca-10k.json', 'r') as f:
            for line in f:
                raw_data.append(json.loads(line)) 
        input_dataset = []
        output_dataset = []
        for conver in raw_data:
            conv_list = conver['conversations']
            input_text = ""
            for turn in conv_list:
                if turn['from'] == 'human':
                    input_text += turn['value']+'\n'
                else:
                    input_dataset.append(input_text)
                    output_dataset.append(turn['value'])
                    input_text += turn['value']+'\n'

        input_dataset = [tokenizer.encode(prefix + s + postfix, return_tensors="pt") for s in input_dataset]
    else:
        raise RuntimeError(f"Unrecognized dataset {dataset_name}")
    # split dataset based on input length
    #length_interval = [100,200,400,800]


    # split dataset based on input length
    #length_interval = [100,200,400,800]
    length_interval = [100000]

    bleu = hf_evaluate.load('bleu') 
    prefix = "/llmss/LLMSpeculativeSampling/logs/"
    approx_model_name = os.path.basename(approx_model_name)
    target_model_name = os.path.basename(target_model_name)

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

        large_model_cnt = 0
        
        # large model github implementation
        total_time = 0
        total_token = 0
        approx_time = 0
        target_time = 0
        other_time = 0
        target_times = 0
        scores = []
        pred_seq = []
        P = subprocess.Popen("exec python3 -u gpu_power_monitor.py",shell=True, text=True, stdout=subprocess.PIPE)
        t1 = time.time()
        for input_ids in tqdm(ds):
            large_model_cnt += 1

            input_ids = input_ids.to(torch_device)
            t = process_time_ns()
            output = autoregressive_sampling(input_ids, large_model, num_tokens, eos_token_id = tokenizer.eos_token_id, 
                    top_k = top_k, top_p=top_p, pad_token_id = tokenizer.pad_token_id)
            total_time += process_time_ns() - t
            total_token += len(output[0]) - input_ids.size(1)
            score = get_score(output, large_model, input_ids.size(1))
            scores.append(score.item())
            pred_seq.append(tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True))
            if total_time / 1e9 > max_seconds:
                print(f'terminated at {large_model_cnt}', file=log_f)
                print(f'terminated at {large_model_cnt}')
                break
        t2 = time.time()
        P.kill()
        P.wait()
        outputs = P.stdout.readlines()
        fname = os.path.join(prefix, f"{approx_model_name}_{target_model_name}_{dataset_name}_large_model.pkl")
        power_total = get_total_power(outputs, t1, t2, fname)


        print(f'\nlarge model total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token, prob_score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}', file=log_f)
        print(f'\nlarge model total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token, prob_score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}')
        bleu_score = bleu.compute(predictions = pred_seq, references = output_dataset[:large_model_cnt])
        print(f'bleu score = {bleu_score}')
        print(f'bleu score = {bleu_score}', file=log_f)
        print(f'total power consumption: {power_total}')
        print(f'total power consumption: {power_total}', file=log_f)
        print(f'power/token: {power_total/total_token}')
        print(f'power/token: {power_total/total_token}', file=log_f)
        time_limit = total_time/1e9/total_token
        quality_limit = np.mean(scores)



        # small model github implementation
        total_time = 0
        total_token = 0
        approx_time = 0
        target_time = 0
        other_time = 0
        target_times = 0
        scores = []
        pred_seq = []
        cnt = 0
        P = subprocess.Popen("exec python3 -u gpu_power_monitor.py",shell=True, text=True, stdout=subprocess.PIPE)
        t1 = time.time()
        
        for input_ids in tqdm(ds):
            cnt += 1
            input_ids = input_ids.to(torch_device)
            t = process_time_ns()
            output = autoregressive_sampling(input_ids, small_model, num_tokens, eos_token_id = tokenizer.eos_token_id, 
                    top_k = top_k, top_p=top_p, pad_token_id = tokenizer.pad_token_id)
            total_time += process_time_ns() - t
            total_token += len(output[0]) - input_ids.size(1)
            score = get_score(output, large_model, input_ids.size(1))
            scores.append(score.item())
            pred_seq.append(tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True))
            if total_time / 1e9 > max_seconds:
                print(f'terminated at {cnt}', file=log_f)
                print(f'terminated at {cnt}')
                break

        t2 = time.time()
        P.kill()
        P.wait()
        outputs = P.stdout.readlines()
        fname = os.path.join(prefix, f"{approx_model_name}_{target_model_name}_{dataset_name}_small_model.pkl")
        power_total = get_total_power(outputs, t1, t2, fname)


        print(f'\nsmall model (gpu) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token, prob score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}')
        print(f'\nsmall model (gpu) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token, prob score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}', file=log_f)
        bleu_score = bleu.compute(predictions = pred_seq, references = output_dataset[:cnt])
        print(f'bleu score = {bleu_score}')
        print(f'bleu score = {bleu_score}', file=log_f)
        print(f'total power consumption: {power_total}')
        print(f'total power consumption: {power_total}', file=log_f)
        print(f'power/token: {power_total/total_token}')
        print(f'power/token: {power_total/total_token}', file=log_f)
        

        """ 
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
        bleu_score = bleu.compute(predictions = pred_seq, references = output_dataset)
        print(f'bleu score = {bleu_score}')
        print(f'bleu score = {bleu_score}', file=log_f)
        """        
        
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
        cnt = 0
        P = subprocess.Popen("exec python3 -u gpu_power_monitor.py",shell=True, text=True, stdout=subprocess.PIPE)
        t1 = time.time()

        #print(tokenizer.eos_token_id)
        #print(tokenizer.pad_token_id)
        for input_ids in tqdm(ds):
            cnt += 1
            input_ids = input_ids.to(torch_device)
            t = process_time_ns()
            #print(input_ids)
            output, details = speculative_sampling(input_ids, small_model, large_model, 
                    eos_token_id = tokenizer.eos_token_id,
                    pad_token_id = tokenizer.pad_token_id,
                    max_len = num_tokens, 
                    top_k = top_k, top_p=top_p, random_seed = random_seed, details=True)
            #print(output.size())
            #print(output)
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
            #print(pred_seq[-1])
            #xxx = input()
            if total_time / 1e9 > max_seconds:
                print(f'terminated at {cnt}', file=log_f)
                print(f'terminated at {cnt}')
                break

        t2 = time.time()
        P.kill()
        P.wait()
        outputs = P.stdout.readlines()
        fname = os.path.join(prefix, f"{approx_model_name}_{target_model_name}_{dataset_name}_ss.pkl")
        power_total = get_total_power(outputs, t1, t2, fname)

        print(f'\n google speculative decoding (with KVCache) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token', file=log_f)
        print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}", file=log_f)
        print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}", file=log_f)
        print(f"prob score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}", file=log_f)

        print(f'\n google speculative decoding (with KVCache) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token')
        print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}")
        print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}")
        print(f"prob score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}")

        bleu_score = bleu.compute(predictions = pred_seq, references = output_dataset[:cnt])
        print(f'bleu score = {bleu_score}')
        print(f'bleu score = {bleu_score}', file=log_f)
        print(f'total power consumption: {power_total}')
        print(f'total power consumption: {power_total}', file=log_f)
        print(f'power/token: {power_total/total_token}')
        print(f'power/token: {power_total/total_token}', file=log_f)


        
        # BiLD speculative decoding
        BiLD_stop = False
#        for fallback_thres in [0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,0.9]:
#            for rollback_thres in range(10,0,-1):
        if True:
            for fallback_thres, rollback_thres in BiLD_params:
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
                cnt = 0
                P = subprocess.Popen("exec python3 -u gpu_power_monitor.py",shell=True, text=True, stdout=subprocess.PIPE)
                t1 = time.time()

                for input_ids in tqdm(ds):
                    cnt += 1

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

                    if total_time / 1e9 > max_seconds:
                        print(f'terminated at {cnt}', file=log_f)
                        print(f'terminated at {cnt}')
                        break

                t2 = time.time()
                P.kill()
                P.wait()
                outputs = P.stdout.readlines()
                fname = os.path.join(prefix, f"{approx_model_name}_{target_model_name}_{dataset_name}_BiLD_{fallback_thres}_{rollback_thres}.pkl")

                power_total = get_total_power(outputs, t1, t2, fname)

                print(f'\n BiLD decoding {(fallback_thres, rollback_thres)} total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token', file=log_f)
                print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}", file=log_f)
                print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}", file=log_f)
                print(f"prob score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}", file=log_f)       
        
                print(f'\n BiLD decoding {(fallback_thres, rollback_thres)} total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token')
                print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}")
                print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}")
                print(f"prob score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}")  

                bleu_score = bleu.compute(predictions = pred_seq, references = output_dataset[:cnt])
                print(f'bleu score = {bleu_score}')
                print(f'bleu score = {bleu_score}', file=log_f)
                print(f'total power consumption: {power_total}')
                print(f'total power consumption: {power_total}', file=log_f)
                print(f'power/token: {power_total/total_token}')
                print(f'power/token: {power_total/total_token}', file=log_f)
#                if np.mean(scores) > quality_limit and total_time/1e9/total_token < time_limit:
#                    print(f'Early Stop')
#                    print(f'Early Stop', file=log_f)
#                    BiLD_stop = True
#                    break
#            if BiLD_stop == True:
#                break



                #break
           # break
        
        # true beam speculative decoding
#        for gamma in [2,4,6,8]:
#            for width in [1, 2,4,6,8]:
        if True:
            for gamma, width, num_beams in multi_params:
#                num_beams = max(width, 2)
#                if gamma * width > 32:
#                    break
                
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
                cnt = 0
                P = subprocess.Popen("exec python3 -u gpu_power_monitor.py",shell=True, text=True, stdout=subprocess.PIPE)
                t1 = time.time()


                for input_ids in tqdm(ds):
                    cnt += 1

                    input_ids = input_ids.to(torch_device)
                    t = process_time_ns()

                    output, details = beam_speculative_sampling(input_ids, small_model, large_model, 
                      eos_token_id = tokenizer.eos_token_id,
                      pad_token_id = tokenizer.pad_token_id, max_len = num_tokens, 
                      gamma = gamma, width=width, num_beams = num_beams,
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
                    if total_time / 1e9 > max_seconds:
                        print(f'terminated at {cnt}', file=log_f)
                        print(f'terminated at {cnt}')
                        break

                t2 = time.time()
                P.kill()
                P.wait()
                outputs = P.stdout.readlines()
                fname = os.path.join(prefix, f"{approx_model_name}_{target_model_name}_{dataset_name}_true_beam_{gamma}_{width}.pkl")
                power_total = get_total_power(outputs, t1, t2, fname)


                print(f'\n true beam speculative decoding (gamma {gamma}, width {width}, num_beams {num_beams}) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token', file=log_f)
                print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}", file=log_f)
                print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}", file=log_f)
                print(f"prob score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}", file=log_f)       
        
                print(f'\n true beam speculative decoding (gamma {gamma}, width {width}, num_beams {num_beams}) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token')
                print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}")
                print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}")
                print(f"prob score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}")  
                bleu_score = bleu.compute(predictions = pred_seq, references = output_dataset[:cnt])
                print(f'bleu score = {bleu_score}')
                print(f'bleu score = {bleu_score}', file=log_f)
            #    break
            #break
                print(f'total power consumption: {power_total}')
                print(f'total power consumption: {power_total}', file=log_f)
                print(f'power/token: {power_total/total_token}')
                print(f'power/token: {power_total/total_token}', file=log_f)


         
        """  
        # beam speculative decoding
#        for gamma in [2,4,6,8]:
#            for width in [2,4,6,8]:
        if True:
            for gamma, width in multi_params:
                num_beams = num_beams
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
                cnt = 0
                P = subprocess.Popen("exec python3 -u gpu_power_monitor.py",shell=True, text=True, stdout=subprocess.PIPE)
                t1 = time.time()


                for input_ids in tqdm(ds):
                    cnt += 1

                    input_ids = input_ids.to(torch_device)
                    t = process_time_ns()

                    output, details = multi_speculative_sampling(input_ids, small_model, large_model, 
                      eos_token_id = tokenizer.eos_token_id,
                      pad_token_id = tokenizer.pad_token_id, max_len = num_tokens, 
                      gamma = gamma, width=width, num_beams = num_beams,
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
                    if total_time / 1e9 > max_seconds:
                        print(f'terminated at {cnt}', file=log_f)
                        print(f'terminated at {cnt}')
                        break

                t2 = time.time()
                P.kill()
                P.wait()
                outputs = P.stdout.readlines()
                fname = os.path.join(prefix, f"{approx_model_name}_{target_model_name}_{dataset_name}_beam_{gamma}_{width}.pkl")
                power_total = get_total_power(outputs, t1, t2, fname)


                print(f'\n beam speculative decoding (gamma {gamma}, width {width}) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token', file=log_f)
                print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}", file=log_f)
                print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}", file=log_f)
                print(f"prob score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}", file=log_f)       
        
                print(f'\n beam speculative decoding (gamma {gamma}, width {width}) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token')
                print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}")
                print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}")
                print(f"prob score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}")  
                bleu_score = bleu.compute(predictions = pred_seq, references = output_dataset[:cnt])
                print(f'bleu score = {bleu_score}')
                print(f'bleu score = {bleu_score}', file=log_f)
             #   break
            #break
                print(f'total power consumption: {power_total}')
                print(f'total power consumption: {power_total}', file=log_f)
                print(f'power/token: {power_total/total_token}')
                print(f'power/token: {power_total/total_token}', file=log_f)
        """

       
        # iid beam speculative decoding
#        for gamma in [2,4,6,8]:
#            for width in [2,4,6,8]:
        if True:
            for gamma, width in iid_params:

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
                cnt = 0
                P = subprocess.Popen("exec python3 -u gpu_power_monitor.py",shell=True, text=True, stdout=subprocess.PIPE)
                t1 = time.time()


                for input_ids in tqdm(ds):
                    cnt += 1

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

                    if total_time / 1e9 > max_seconds:
                        print(f'terminated at {cnt}', file=log_f)
                        print(f'terminated at {cnt}')
                        break

                t2 = time.time()
                P.kill()
                P.wait()
                outputs = P.stdout.readlines()
                fname = os.path.join(prefix, f"{approx_model_name}_{target_model_name}_{dataset_name}_iid_{gamma}_{width}.pkl")
                power_total = get_total_power(outputs, t1, t2, fname)


                print(f'\niid speculative decoding (gamma {gamma}, width {width}) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token', file=log_f)
                print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}", file=log_f)
                print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}", file=log_f)
                print(f"prob score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}", file=log_f)

                print(f'\niid speculative decoding (gamma {gamma}, width {width}) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token')
                print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}")
                print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}")
                print(f"prob score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}")
                bleu_score = bleu.compute(predictions = pred_seq, references = output_dataset[:cnt])
                print(f'bleu score = {bleu_score}')
                print(f'bleu score = {bleu_score}', file=log_f)
             #   break
            #break

                print(f'total power consumption: {power_total}')
                print(f'total power consumption: {power_total}', file=log_f)
                print(f'power/token: {power_total/total_token}')
                print(f'power/token: {power_total/total_token}', file=log_f)





if __name__ == "__main__":
    args = parse_arguments()
    evaluate(args.approx_model_name, args.target_model_name, 
            dataset_name = args.dataset,
            num_tokens=args.max_tokens, 
            max_seconds = args.max_seconds,
            #gamma=args.gamma, width=args.width,
             acc_rate_head_path = args.acc_rate_head_path,
             log_file = args.log_file,
             random_seed = args.seed, verbose=args.verbose)
    #generate(args.input, args.approx_model_name, args.target_model_name, num_tokens=args.max_tokens, gamma=args.gamma,
    #         random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark)
