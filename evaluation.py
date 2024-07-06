import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3,6'

import pandas as pd
import torch
import argparse
import contexttimer
from colorama import Fore, Style
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, AutoModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration
from datasets import load_dataset

from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2, multi_speculative_sampling, mjsd_speculative_sampling
from sampling import beam_speculative_sampling_v2 as beam_speculative_sampling
from sampling import BiLD_sampling
from sampling import random_width_beam_sampling
from sampling.models.modeling_llama import LlamaForCausalLM
from sampling.models.modeling_opt import OPTForCausalLM
from sampling.utils import exact_match_references, execution_accuracy_references 

from globals import Decoder
import json
from time import process_time_ns
from tqdm import tqdm
import time
import numpy as np
import random
import subprocess
import pickle
import evaluate as hf_evaluate
#from pyJoules.energy_meter import measure_energy
hf_token = os.environ['HFTOKEN']

def find_fields_MYSQL_like(db_name, spider_schema):
  df = spider_schema[spider_schema['Database name'] == db_name]
  df = df.groupby(' Table Name')
  output = ""
  for name, group in df:
    output += "Table " +name+ ', columns = ['
    for index, row in group.iterrows():
      output += row[" Field Name"]+','
    output = output[:-1]
    output += "]\n"
  return output

def creatiing_schema(DATASET_JSON):
    schema_df = pd.read_json(DATASET_JSON)
    schema_df = schema_df.drop(['column_names','table_names'], axis=1)
    schema = []
    f_keys = []
    p_keys = []
    for index, row in schema_df.iterrows():
        tables = row['table_names_original']
        col_names = row['column_names_original']
        col_types = row['column_types']
        foreign_keys = row['foreign_keys']
        primary_keys = row['primary_keys']
        for col, col_type in zip(col_names, col_types):
            index, col_name = col
            if index == -1:
                for table in tables:
                    schema.append([row['db_id'], table, '*', 'text'])
            else:
                schema.append([row['db_id'], tables[index], col_name, col_type])
        for primary_key in primary_keys:
            index, column = col_names[primary_key]
            p_keys.append([row['db_id'], tables[index], column])
        for foreign_key in foreign_keys:
            first, second = foreign_key
            first_index, first_column = col_names[first]
            second_index, second_column = col_names[second]
            f_keys.append([row['db_id'], tables[first_index], tables[second_index], first_column, second_column])
    spider_schema = pd.DataFrame(schema, columns=['Database name', ' Table Name', ' Field Name', ' Type'])
    spider_primary = pd.DataFrame(p_keys, columns=['Database name', 'Table Name', 'Primary Key'])
    spider_foreign = pd.DataFrame(f_keys,
                        columns=['Database name', 'First Table Name', 'Second Table Name', 'First Table Foreign Key',
                                 'Second Table Foreign Key'])
    return spider_schema,spider_primary,spider_foreign


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
    parser.add_argument('--dataset', type=str, default='wmt')
    parser.add_argument('--max_seconds', type=int, default=7200, help='timeout seconds')
    parser.add_argument('--run_flexflow',action='store_true', default=False, help='run flexflow specinfer as baseline')
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
    first_one = True
    for timestamp, power in x:
        if timestamp > t1 and timestamp < t2:
            if first_one:
                first_one = False
            else:
                total_power += power
    return total_power


#@measure_energy
def evaluate(approx_model_name, target_model_name, 
        dataset_name, 
        acc_rate_head_path = None, num_tokens=20, 
        max_seconds = 7200,
            #gamma = 4, width=1,
            run_flexflow = False,
             random_seed = None, verbose = False, log_file = "logs/log.txt"):
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_f = open(log_file, 'w')
    
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True, token=hf_token)

    #tokenizer2 = AutoTokenizer.from_pretrained(target_model_name, trust_remote_code=True, token=hf_token)
    print(approx_model_name, file=log_f)
    print(target_model_name, file=log_f)

    vocab1 = tokenizer.get_vocab()
    #vocab2 = tokenizer2.get_vocab()
    #if vocab1 == vocab2:
    #    print("Vocabularies are the same. Proceed")
    #else:
    #    print("Vocabularies are different.")
    #    print("Vocabularies are different.", file=log_f)
    #    return
    Decoder().set_tokenizer(tokenizer)
    
    print(f"begin loading models: \n {approx_model_name} \n {target_model_name}")
    if 'llama' in approx_model_name and 'GPTQ' not in approx_model_name:
        small_model = LlamaForCausalLM.from_pretrained(approx_model_name, 
                                                       torch_dtype=torch.float32,
                                                       device_map="auto",
                                                       trust_remote_code=True,
                                                       token=hf_token)
    elif 'opt' in approx_model_name:
        small_model = OPTForCausalLM.from_pretrained(approx_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True,
                                                       token=hf_token)


    else:
        small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if 'Llama-3' in target_model_name:
         large_model = LlamaForCausalLM.from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       offload_folder="offload",
                                                       trust_remote_code=True,
                                                       cache_dir='/local2/shared_cache_huggingface/',
#                                                       _commit_hash='cfe96d938c52db7c6d936f99370c0801b24233c4',
                                                       local_files_only=True,
                                                       )
        
    elif 'llama' in target_model_name and '70' in target_model_name:

        large_model = LlamaForCausalLM.from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       offload_folder="offload",
                                                       trust_remote_code=True,
                                                       cache_dir='/local2/ma/cache/huggingface/',
                                                       _commit_hash='cfe96d938c52db7c6d936f99370c0801b24233c4',
                                                       local_files_only=True,
                                                       )
                                                       #token=hf_token)
    elif 'llama' in target_model_name:
        large_model = LlamaForCausalLM.from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float32,
                                                       device_map="auto",
                                                       offload_folder="offload",
                                                       trust_remote_code=True,
                                                       token=hf_token,
                                                       )
                                                       #token=hf_token)


    elif 'opt' in target_model_name:
        large_model = OPTForCausalLM .from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       offload_folder="offload",
                                                       trust_remote_code=True,
                                                       token=hf_token)


    else:
        large_model = AutoModelForCausalLM.from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       offload_folder="offload",
                                                       trust_remote_code=True)
    top_k = 10
    top_p = 0.0
    repeats = 1
    
    if dataset_name == 'cnndm':
        dataset = load_dataset('cnn_dailymail', '3.0.0', split='test')
        prefix = 'Summarize: '
        postfix = ""
        if 'opt' in approx_model_name:
            BiLD_params = [(0.3,2)]
            multi_params = [(4,1,8),(4,2,8)]
            iid_params = [ (4,2), ]
        elif 'llama' in approx_model_name:
            BiLD_params = [(0.9, 3)]
            multi_params = [(4,1,8)]
            iid_params = [(4,4)]
            prefix = "[INST] <<SYS>> Please Summarize <</SYS>>"
            postfix = '[/INST]'
        input_dataset = [tokenizer.encode(prefix + s['article'] + postfix, return_tensors="pt", max_length=512, truncation=True) for s in dataset]
        output_dataset = [[s['highlights']] for s in dataset]
    elif dataset_name == 'squad':
        dataset = load_dataset('squad', split='validation')
        examples = """[INST] <<SYS>> You are a good student. You need to answer the question using the exact words from the context. Below are some examples of how to answer questions based on context<</SYS>>
Example 1
Context: Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.
Question: To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?
Answer: Saint Bernadette Soubirous

Example 2
Context: The university is the major seat of the Congregation of Holy Cross (albeit not its official headquarters, which are in Rome). Its main seminary, Moreau Seminary, is located on the campus across St. Joseph lake from the Main Building. Old College, the oldest building on campus and located near the shore of St. Mary lake, houses undergraduate seminarians. Retired priests and brothers reside in Fatima House (a former retreat center), Holy Cross House, as well as Columba Hall near the Grotto. The university through the Moreau Seminary has ties to theologian Frederick Buechner. While not Catholic, Buechner has praised writers from Notre Dame and Moreau Seminary created a Buechner Prize for Preaching. 
Question: What is the primary seminary of the Congregation of the Holy Cross? 
Answer: Moreau Seminary

Now, answer the following question[/INST]
"""
        input_texts = [examples + 
                       "Context: " + s["context"] + '\n'+
                       "Question: " + s["question"] + ' \n'+
                       "Answer:" for s in dataset]
        input_dataset = [tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True) for text in input_texts]
        output_dataset = [s["answers"]["text"] for s in dataset]
    elif dataset_name == 'spider':
        import json
        dataset = json.load(open("spider/spider/dev.json"))
        spider_schema,spider_primary,spider_foreign = creatiing_schema("./spider/spider/tables.json")

        examples = """[INST] <<SYS>> You are a SQL expert. You need to write the correct SQL based on the user question and database schemas. Below are some examples <</SYS>>
Example 1
Schema:
Table department, columns = [*,Department_ID,Name,Creation,Ranking,Budget_in_Billions,Num_Employees]
Table head, columns = [*,head_ID,name,born_state,age]
Table management, columns = [*,department_ID,head_ID,temporary_acting]
Foreign_keys = [management.head_ID = head.head_ID,management.department_ID = department.Department_ID]
Question: "How many heads of the departments are older than 56 ?"
SQL: SELECT count(*) FROM head WHERE age  >  56; 

Example 2
Schema:
Table department, columns = [*,Department_ID,Name,Creation,Ranking,Budget_in_Billions,Num_Employees]
Table head, columns = [*,head_ID,name,born_state,age]
Table management, columns = [*,department_ID,head_ID,temporary_acting]
Foreign_keys = [management.head_ID = head.head_ID,management.department_ID = department.Department_ID]
Question: "what are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?"
SQL: SELECT DISTINCT T1.creation FROM department AS T1 JOIN management AS T2 ON T1.department_id  =  T2.department_id JOIN head AS T3 ON T2.head_id  =  T3.head_id WHERE T3.born_state  =  'Alabama'; 

"""
        input_texts = [examples + 
                       "Schema:\n" + find_fields_MYSQL_like(s["db_id"], spider_schema) + "\n" + 
                       "Question: " + s["question"] + "\n" + 
                       "SQL:" for s in dataset]
        input_dataset = [tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True) for text in input_texts]
        output_dataset = [s["db_id"] + "[SQL]" + s["query"] for s in dataset] 
    elif dataset_name == 'ChatGPT':
        dataset = load_dataset('MohamedRashad/ChatGPT-prompts',split='train')
        input_dataset = [tokenizer.encode(s, return_tensors="pt", max_length=512, truncation=True) for s in dataset['human_prompt']]
        output_dataset = [s for s in dataset['chatgpt_response']]
        if 'opt' in approx_model_name:
            BiLD_params = [(0.9,2)]
            multi_params = [(4,1,8)]
            iid_params = [(4,4)]
        elif 'llama' in approx_model_name:
            BiLD_params = [(0.9,2)]
            multi_params = [(4,1,8)]
            iid_params = [(4,4)]
        else:
            prefix = ''
            postfix = ''

    elif dataset_name == 'chatalpaca':

        if 'opt' in approx_model_name:
            prefix = ''
            postfix = ''
            BiLD_params = [(0.3, 2)]
            multi_params = [(4,1,8)]
            iid_params = [(4,2)]

        elif 'llama' in approx_model_name:
            prefix = ""
            postfix = ''
            BiLD_params = [(0.9,1)]
            multi_params = [(4,1,8)]
            iid_params = [(4,2)]
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


    length_interval = [100000]

    prefix = "./logs/"
    approx_model_name = os.path.basename(approx_model_name)
    target_model_name = os.path.basename(target_model_name)


    rouge = hf_evaluate.load('rouge')
    if dataset_name == 'squad':
        em = exact_match_references
    elif dataset_name == 'spider':
        em = execution_accuracy_references
    else:
        em = None

    ori_output_dataset = output_dataset
    for i in range(repeats):
        #u = length_interval[i]
        u = 100000
        l = 0
        ds = [pt for pt in input_dataset if (pt.size(-1) < u and pt.size(-1) >= l)]
#        ds = [ds[12] for k in range(100)]
        ds = ds[:100]
        output_dataset = ori_output_dataset[:100]
#        output_dataset = [ori_output_dataset[12] for k in range(100)]

        print(f'input length {l}-{u}, {len(ds)} data in total')
        total_input_tokens = sum([d.size(1) for d in ds])
        print('total_input_tokens', total_input_tokens)


        large_model_cnt = 0
        
        # large model github implementation
#        time.sleep(100)
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
        p_list_list = []
        for input_ids in tqdm(ds):
            large_model_cnt += 1
            #if large_model_cnt % 4 == 0:
            #    time.sleep(0.025)

            input_ids = input_ids.to(torch_device)
            t = process_time_ns()
            output = autoregressive_sampling(input_ids, large_model, num_tokens, eos_token_id = tokenizer.eos_token_id, 
                    top_k = top_k, top_p=top_p, pad_token_id = tokenizer.pad_token_id)
 #           p_list_list.append(p_list)
            total_time += process_time_ns() - t
            total_token += len(output[0]) - input_ids.size(1)
            score = get_score(output, large_model, input_ids.size(1))
            scores.append(score.item())
            if dataset_name == 'squad':
                pred_seq.append(tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True).split('\n')[0])
            elif dataset_name == 'spider':
                pred_seq.append(tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True).split(';')[0])
            else:
                pred_seq.append(tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True))

            if total_time / 1e9 > max_seconds:
                print(f'terminated at {large_model_cnt}', file=log_f)
                print(f'terminated at {large_model_cnt}')
                break
            

            #print(tokenizer.decode(input_ids.cpu().tolist()[0], skip_special_tokens=True))
#            print('output')
#            print(pred_seq[large_model_cnt-1])
#            print('ansewr')
#            print(output_dataset[large_model_cnt-1])
            #xxx = input()
            
        t2 = time.time()
        P.kill()
        P.wait()
        outputs = P.stdout.readlines()
        fname = os.path.join(prefix, f"{approx_model_name}_{target_model_name}_{dataset_name}_large_model.pkl")
        power_total = get_total_power(outputs, t1, t2, fname)
#        torch.save(p_list_list, 'as_p_list.pth')

        print(t1, t2)

        print(f'\nlarge model total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token, prob_score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}', file=log_f)
        print(f'\nlarge model total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token, prob_score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}')
        print(f'total power consumption: {power_total}')
        print(f'total power consumption: {power_total}', file=log_f)
        print(f'power/token: {power_total/total_token}')
        print(f'power/token: {power_total/total_token}', file=log_f)
        time_limit = total_time/1e9/total_token
        quality_limit = np.mean(scores)

        rouge_score = rouge.compute(predictions = pred_seq, references = output_dataset[:large_model_cnt])
        print(f'rouge score = {rouge_score}')
        print(f'rouge score = {rouge_score}', file=log_f)

        em_score = em(predictions = pred_seq, references = output_dataset[:large_model_cnt])
        print(f'em score = {em_score}')
        print(f'em score = {em_score}', file=log_f)
#        print(pred_seq)




                
        """ 
        # convetional speculative decoding
#        time.sleep(100)
        
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
        target_model_time = 0
        target_pre_cache_time = 0
        target_post_prob_time = 0
        

        #print(tokenizer.eos_token_id)
        #print(tokenizer.pad_token_id
        
        for input_ids in tqdm(ds):
            cnt += 1
#            if cnt % 16 == 0:
#                time.sleep(0.025)


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
            target_model_time += details['target_model_time']
            target_pre_cache_time += details['target_pre_cache_time']
            target_post_prob_time += details['target_post_prob_time']
            score = get_score(output, large_model, input_ids.size(1))
            scores.append(score.item())
            if dataset_name == 'squad':
                pred_seq.append(tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True).split('\n')[0])
            elif dataset_name == 'spider':
                pred_seq.append(tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True).split(';')[0])
            else:
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

        print(t1, t2)
        print(f'\n google speculative decoding (with KVCache) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token', file=log_f)
        print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}", file=log_f)
        print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}", file=log_f)
        print(f"prob score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}", file=log_f)

        print(f'\n google speculative decoding (with KVCache) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token')
        print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}")
        print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}")
        print(f"prob score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}")


        print(f'total power consumption: {power_total}')
        print(f'total power consumption: {power_total}', file=log_f)
        print(f'power/token: {power_total/total_token}')
        print(f'power/token: {power_total/total_token}', file=log_f)
        print(f'target_model_time: {target_model_time/1e9}, pre cache time: {target_pre_cache_time/1e9}, post prob time: {target_post_prob_time/1e9}')
        print(f'target_model_time: {target_model_time/1e9}, pre cache time: {target_pre_cache_time/1e9}, post prob time: {target_post_prob_time/1e9}', file=log_f)
        if cnt > large_model_cnt:
            rouge_score = rouge.compute(predictions = pred_seq[:large_model_cnt], references = output_dataset[:large_model_cnt])
        else:
            rouge_score = rouge.compute(predictions = pred_seq[:cnt], references = output_dataset[:cnt])

        print(f'rouge score = {rouge_score}')
        print(f'rouge score = {rouge_score}', file=log_f)
        em_score = em(predictions = pred_seq, references = output_dataset[:large_model_cnt])
        print(f'em score = {em_score}')
        print(f'em score = {em_score}', file=log_f)

        """
        
        
        # BiLD speculative decoding
        BiLD_stop = False
        if False:
            for fallback_thres, rollback_thres in BiLD_params:
#                time.sleep(100)
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
                    #if cnt % 16 == 0:
                    #    time.sleep(0.025)


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
                print(t1, t2)

                print(f'\n BiLD decoding {(fallback_thres, rollback_thres)} total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token', file=log_f)
                print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}", file=log_f)
                print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}", file=log_f)
                print(f"prob score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}", file=log_f)       
        
                print(f'\n BiLD decoding {(fallback_thres, rollback_thres)} total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token')
                print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}")
                print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}")
                print(f"prob score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}")  

                print(f'total power consumption: {power_total}')
                print(f'total power consumption: {power_total}', file=log_f)
                print(f'power/token: {power_total/total_token}')
                print(f'power/token: {power_total/total_token}', file=log_f)
        
        # mjsd speculative decoding
        if False:
            for params in multi_params:
                if len(params) == 2:
                    gamma, width = params[0], params[1]
                    tau = 0.1
                    num_beams = width
                elif len(params) == 3:
                    gamma, width, num_beams = params[0], params[1], params[2]
                    tau = 0.1
                else:
                    assert len(params) == 4
                    gamma, width, num_beams, tau = params[0], params[1], params[2], params[3]

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

                    output, details = mjsd_speculative_sampling(input_ids, small_model, large_model, 
                      eos_token_id = tokenizer.eos_token_id,
                      pad_token_id = tokenizer.pad_token_id, max_len = num_tokens, 
                      gamma = gamma, width=width, num_beams = num_beams, accept_thres = tau,
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

                print(t1, t2)

                print(f'\n mjsd speculative decoding (gamma {gamma}, width {width}, num beams {num_beams}, {tau}) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token', file=log_f)
                print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}", file=log_f)
                print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}", file=log_f)
                print(f"prob score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}", file=log_f)       
        
                print(f'\n mjsd speculative decoding (gamma {gamma}, width {width}, num beams {num_beams}, {tau}) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token')
                print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}")
                print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}")
                print(f"prob score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}")  
            #    break
            #break
                print(f'total power consumption: {power_total}')
                print(f'total power consumption: {power_total}', file=log_f)
                print(f'power/token: {power_total/total_token}')
                print(f'power/token: {power_total/total_token}', file=log_f)
                if cnt > large_model_cnt:
                    rouge_score = rouge.compute(predictions = pred_seq[:large_model_cnt], references = output_dataset[:large_model_cnt])
                else:
                    rouge_score = rouge.compute(predictions = pred_seq[:cnt], references = output_dataset[:cnt])
                print(f'rouge score = {rouge_score}')
                print(f'rouge score = {rouge_score}', file=log_f)


       
        # iid beam speculative decoding
        if False:
            for gamma, width in iid_params:

              #  if gamma * width > 32:
              #      break
 
#                time.sleep(100)
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
                    #if cnt % 16 == 0:
                    #    time.sleep(0.025)


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

                print(t1, t2)

                print(f'\niid speculative decoding (gamma {gamma}, width {width}) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token', file=log_f)
                print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}", file=log_f)
                print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}", file=log_f)
                print(f"prob score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}", file=log_f)

                print(f'\niid speculative decoding (gamma {gamma}, width {width}) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token')
                print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}")
                print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}")
                print(f"prob score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}")
             #   break
            #break

                print(f'total power consumption: {power_total}')
                print(f'total power consumption: {power_total}', file=log_f)
                print(f'power/token: {power_total/total_token}')
                print(f'power/token: {power_total/total_token}', file=log_f)

   
        for width in [3]:
            for gamma in [1,2]:

#        if True:
#            for gamma, width in multi_params:
                num_beams = width
#                num_beams = width
#                time.sleep(100)
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

                total_counts = {1.:0, 12.:0}
                for input_ids in tqdm(ds):
                    cnt += 1
                    #if cnt % 16 == 0:
                    #    time.sleep(0.025)


                    input_ids = input_ids.to(torch_device)
                    t = process_time_ns()

                    output, details = beam_speculative_sampling(input_ids, small_model, large_model, 
                      eos_token_id = tokenizer.eos_token_id,
                      pad_token_id = tokenizer.pad_token_id, max_len = num_tokens, 
                      gamma = 4, width=width, num_beams = num_beams, min_num_beams = gamma,
                      top_k = top_k, top_p=top_p, 
                      random_seed = random_seed, details=True)
#                    val, counts = output[0,input_ids.size(1):].unique(return_counts=True)
#                    print(output[0,input_ids.size(1):])
                #    xxx = input()
#                    print(val, counts)
#                    for v,c in zip(val,counts):
#                        total_counts[float(v)] += c
                    #print(output)
                    #print(input_ids)

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
                    if score.isnan().any():
                        print(input_ids)
                        print(output)
                        print(tokenizer.eos_token_id)
                        raise RuntimeError('score nan')
                    #    xxx = input()
                    scores.append(score.item())
                    if dataset_name == 'squad':
                        pred_seq.append(tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True).split('\n')[0])
                    elif dataset_name == 'spider':
                        pred_seq.append(tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True).split(';')[0])
                    else:
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
 #               print(total_counts)
 #               xxx = input()


                print(f'\n true beam speculative decoding (gamma {gamma}, width {width}) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token', file=log_f)
                print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}", file=log_f)
                print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}", file=log_f)
                print(f"prob score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}", file=log_f)       
        
                print(f'\n true beam speculative decoding (gamma {gamma}, width {width}) total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token')
                print(f"approx time {approx_time/1e9}, target time {target_time/1e9}, other time {other_time/1e9}")
                print(f"average accepted len {total_acc_len/target_times}, target call times {target_times}, acc rate {np.mean(acc_rate)}, approx call times {approx_times}")
                print(f"prob score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}")  
            #    break
            #break
                print(f'total power consumption: {power_total}')
                print(f'total power consumption: {power_total}', file=log_f)
                print(f'power/token: {power_total/total_token}')
                print(f'power/token: {power_total/total_token}', file=log_f)
                if cnt > large_model_cnt:
                    rouge_score = rouge.compute(predictions = pred_seq[:large_model_cnt], references = output_dataset[:large_model_cnt])
                    em_score = em(predictions = pred_seq, references = output_dataset[:large_model_cnt], debug = False)
                else:
                    rouge_score = rouge.compute(predictions = pred_seq[:cnt], references = output_dataset[:cnt])
                    em_score = em(predictions = pred_seq[:cnt], references = output_dataset[:cnt], debug = False)

                print(f'rouge score = {rouge_score}')
                print(f'rouge score = {rouge_score}', file=log_f)
                print(f'em score = {em_score}')
                print(f'em score = {em_score}', file=log_f)



        for max_beams in [1,3,4]:
            for min_beams in [1,2,3]:
                if min_beams > max_beams:
                    break
                if min_beams > 1:
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
                p_list_list = []
                total_counts = {1.:0, 12.:0}
                for input_ids in tqdm(ds):
                    cnt += 1
                    #if cnt % 16 == 0:
                    #    time.sleep(0.025)


                    input_ids = input_ids.to(torch_device)
                    t = process_time_ns()
            
                    output = random_width_beam_sampling(input_ids, large_model, num_tokens,
                        max_num_beams = max_beams, min_num_beams = min_beams,
                        eos_token_id = tokenizer.eos_token_id, 
                        top_k = top_k, top_p=top_p, pad_token_id = tokenizer.pad_token_id)
                #    p_list_list.append(p_list)
#                    val, counts = output[0,input_ids.size(1):].unique(return_counts=True)
#                    for v,c in zip(val,counts):
#                        total_counts[float(v)] += c


                    total_time += process_time_ns() - t
                    total_token += len(output[0])- input_ids.size(1)
                    score = get_score(output, large_model, input_ids.size(1))
                    scores.append(score.item())

                    if dataset_name == 'squad':
                        pred_seq.append(tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True).split('\n')[0])
                    elif dataset_name == 'spider':
                        pred_seq.append(tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True).split(';')[0])
                    else:
                        pred_seq.append(tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True))

                    """
                    print(tokenizer.decode(input_ids.cpu().tolist()[0], skip_special_tokens=True))
                    print('output')
                    print(pred_seq[cnt-1])
                    print('ansewr')
                    print(output_dataset[cnt-1])
                    print(rouge.compute(predictions=pred_seq[-1:], references = output_dataset[cnt-1:cnt]))
                    xxx = input()
                    """
                    if total_time / 1e9 > max_seconds:
                        print(f'terminated at {cnt}', file=log_f)
                        print(f'terminated at {cnt}')
                        break
#                print(total_counts)
#                xxx = input()

                t2 = time.time()
   #             torch.save(p_list_list, 'rb_p_list.pth')
                P.kill()
                P.wait()
                outputs = P.stdout.readlines()
                fname = os.path.join(prefix, f"{approx_model_name}_{target_model_name}_{dataset_name}_rwbd_{max_beams}_{min_beams}.pkl")
                power_total = get_total_power(outputs, t1, t2, fname)
                print(t1, t2)

                print(f'\n rwbd decoding {(max_beams, min_beams)} total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token', file=log_f)
                print(f"prob score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}", file=log_f)       
        
                print(f'\n rwbd decoding {(max_beams, min_beams)} total time {total_time/1e9} s, total tokens {total_token}, average time {total_time/1e9/total_token} s/token')
                print(f"prob score = {np.mean(scores)}, prob score cut = {np.mean(scores[:large_model_cnt])}")  

                print(f'total power consumption: {power_total}')
                print(f'total power consumption: {power_total}', file=log_f)
                print(f'power/token: {power_total/total_token}')
                print(f'power/token: {power_total/total_token}', file=log_f)

                if cnt > large_model_cnt:
                    rouge_score = rouge.compute(predictions = pred_seq[:large_model_cnt], references = output_dataset[:large_model_cnt])
                    em_score = em(predictions = pred_seq, references = output_dataset[:large_model_cnt])
                else:
                    rouge_score = rouge.compute(predictions = pred_seq[:cnt], references = output_dataset[:cnt])
                    em_score = em(predictions = pred_seq[:cnt], references = output_dataset[:cnt])

                print(f'rouge score = {rouge_score}')
                print(f'rouge score = {rouge_score}', file=log_f)

                print(f'em score = {em_score}')
                print(f'em score = {em_score}', file=log_f)
   #             print(pred_seq)



if __name__ == "__main__":
    args = parse_arguments()
    evaluate(args.approx_model_name, args.target_model_name, 
            dataset_name = args.dataset,
            num_tokens=args.max_tokens, 
            max_seconds = args.max_seconds,
            #gamma=args.gamma, width=args.width,
             acc_rate_head_path = args.acc_rate_head_path,
             log_file = args.log_file,
             random_seed = args.seed, verbose=args.verbose,
             run_flexflow = args.run_flexflow)
    
