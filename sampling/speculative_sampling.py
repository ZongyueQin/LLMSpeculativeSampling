import torch
from tqdm import tqdm
import torch
import torch.nn.functional as F

from sampling.kvcache_model import KVCacheModel
from sampling.utils import norm_logits, sample, max_fn
from globals import Decoder
from time import process_time_ns
import numpy as np
import os

@torch.no_grad()
def beam_speculative_sampling(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         eos_token_id, pad_token_id, max_len : int , gamma : int = 4, width : int = 8, num_beams: int = 8,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None,
                         details : bool = False) -> torch.Tensor:
    if pad_token_id is None:
        pad_token_id = eos_token_id

    seq_len = prefix.shape[1]
    ori_eos_cnt = (prefix == eos_token_id).int().sum()
    T = seq_len + max_len
    acc_len = []
    acc_rate = []

    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    assert prefix.shape[0] == 1, "input batch size must be 1"

    approx_time = 0
    target_time = 0
    sample_time = 0
    target_call_times = 0
    approx_call_times = 0

    if approx_model.config.is_encoder_decoder == True:
        encoder_outputs = approx_model.get_encoder()(
                    prefix, return_dict=True
                    )
        for key, val in encoder_outputs.items():
            if key != 'last_hidden_state':
                del encdoer_outputs[key]
        output_prefix = torch.LongTensor([[pad_token_id]]).to(prefix.device)
        T = max_len
    else:
        output_prefix = prefix

    start_t = process_time_ns()


#    with tqdm(total=T, desc="speculative sampling") as pbar:
    if True:
        while output_prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            prefix_len = output_prefix.shape[1]

            # generate x of size width * (prefix_len+gamma)
            tt = process_time_ns()
            #num_beams = max(4, width)
            if approx_model.config.is_encoder_decoder:
                encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state[0:1]

                out = approx_model_cache.beam_sample_with_kv_cache(
                       output_prefix, 
                       gamma=gamma, 
                       num_beams=num_beams, 
                       top_k=top_k, top_p=top_p,
                       num_return_sequences=width,
                       return_dict_in_generate = True,
                       output_scores = True,
                       encoder_outputs = encoder_outputs,
                       ret_seq_scores = True
                       )
            else:
                out = approx_model_cache.beam_sample_with_kv_cache(
                       output_prefix, 
                       gamma=gamma, 
                       num_beams=num_beams, 
                       top_k=top_k, top_p=top_p,
                       num_return_sequences=width,
                       return_dict_in_generate = True,
                       output_scores = True,
                       ret_seq_scores = True
                       )


            x = out['sequences'] # width * (prefix_len+gamma)
            q, seq_q = out['scores'] # tuples of gamma * (width * vocab) ?
 
            inc_len = x.shape[1] - prefix_len
            approx_call_times += 1
            approx_time += process_time_ns() - tt

            tt = process_time_ns()
            if target_model.config.is_encoder_decoder == False:
                _ = target_model_cache.generate(x, 1)
            else:
                _ = target_model_cache.generate(
                        prefix.repeat_interleave(width, dim=0),
                        1, 
                        decoder_input_ids = x)
            target_call_times += 1
            p = target_model_cache._prob_history
            target_time += process_time_ns() - tt

            for w in range(width):
                cur_target_p = 0
                for i in range(gamma):
                    if prefix_len + i >= x.size(1):
                        break
                    j = x[w, prefix_len+i]
                    cur_target_p += torch.log(p[w, prefix_len + i - 1, j])
                    cur_draft_p = seq_q[w, i]
                    acc_rate.append((torch.exp(cur_target_p)/cur_draft_p).item())
                    if acc_rate[-1] > 1:
                        acc_rate[-1] = 1



            tt = process_time_ns()
            is_all_accept = False
            n = prefix_len - 1
            max_n = prefix_len - 1
            max_l = 0
            choice = 0
            for w in range(width):
                cur_n = prefix_len - 1
                cur_l = 0
                cur_all_accept = True

                cur_target_p = 0
                for i in range(inc_len):
                    if prefix_len + i >= x.size(1):
                        break
                    if random_seed:
                        torch.manual_seed(random_seed)
                    r = torch.rand(1, device = p.device)
                    j = x[w, prefix_len + i]

                    cur_target_p += torch.log(p[w, prefix_len + i - 1, j])
                    cur_draft_p = seq_q[w, i]
               
                    if r < torch.min(torch.tensor([1], device=q.device), torch.exp(cur_target_p)/cur_draft_p):
                        cur_l += 1
                    # accept, and update n
                        cur_n += 1
                    else:
                        # reject
                        cur_all_accept = False
                        break
                if cur_l > max_l:
                    assert cur_n > max_n, f"cur_n {cur_n}, max_n {max_n}"
                    max_n = cur_n
                    max_l = cur_l
                    choice = w
                    if cur_all_accept == True:
                        is_all_accept = True
                        break
            acc_len.append(max_l)

            n = max_n
         
            output_prefix = x[choice:choice+1, :n + 1]
        
            approx_model_cache.rollback(n+1, choice)
 
            if is_all_accept:
                t = sample(p[choice:choice+1, -1, :])
                target_model_cache.rollback(n+2, choice)

            else:
                #print(torch.sum(p[choice,n,:]).item(), torch.sum(q[choice,max_l,:]).item())
                #tmp = p[choice,n,:] - q[choice,max_l,:]
                #print(torch.sum((tmp>0).float()).item(), torch.sum((tmp<=0).float()).item())
                #print(torch.sum((p[choice,n,:]==q[choice,max_l,:]).float()).item())

                #t = sample(max_fn(p[choice:choice+1, n, :] - q[choice:choice+1, max_l, :]))
                t = sample(max_fn(p[choice:choice+1, n, :]))

                target_model_cache.rollback(n+1, choice)
           
            output_prefix = torch.cat((output_prefix, t), dim=1)
            #pbar.update(n - pbar.n)
            mask = (output_prefix == eos_token_id)
            if mask.int().sum() > ori_eos_cnt:
                mask = torch.cumsum(mask.float(), dim=1)
                mask = (mask < ori_eos_cnt+1)
                end = mask.int().sum()
                if end < mask.size(1):
                    mask[:, end] = True
                output_prefix = output_prefix[mask][None,:] 
                break




            sample_time += process_time_ns() - tt

    if approx_model.config.is_encoder_decoder:
        output_prefix = torch.cat((prefix, output_prefix), dim=1)

    if verbose:
        print('approx model time', approx_time/1e9)
        print('target model time', target_time/1e9)
        print('other time', sample_time/1e9)
        print('inner overall time', (process_time_ns()-start_t)/1e9)
        print('acc len', np.mean(acc_len), len(acc_len), acc_len)
    if details == True:
        d = {
                'approx_time': approx_time,
                'target_time': target_time,
                'other_time': sample_time,
                'acc_len': acc_len,
                'acc_rate': np.mean(acc_rate),
                'target_call_times': target_call_times,
                'approx_call_times': approx_call_times
            }
        return output_prefix, d
    else:
        return output_prefix


@torch.no_grad()
def multi_speculative_sampling(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         eos_token_id, pad_token_id, max_len : int ,
                         gamma : int = 4, width : int = 8, num_beams = None, strategy : str = "beam", 
                         acc_rate_head = None, acc_rate_thres = 0.4, 
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None,
                         details : bool = False) -> torch.Tensor:
    seq_len = prefix.shape[1]
    ori_eos_cnt = (prefix == eos_token_id).int().sum()
    T = seq_len + max_len
    acc_len = []
    acc_rate = []

    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)

    """ 
    print(prefix.size())
    out = approx_model_cache.beam_sample_with_kv_cache(prefix, gamma=gamma, num_beams=width, 
            top_k=top_k, top_p=top_p,
            num_return_sequences=width,
          )
    print(out['sequences'].size())
    xxx = input('')
    prefix = out['sequences'][0:1]
    approx_model_cache.rollback(prefix.size(1), 0)
    out = approx_model_cache.beam_sample_with_kv_cache(prefix, gamma=gamma, num_beams=width, 
            top_k=top_k, top_p=top_p,
            num_return_sequences=width)
    print(out['sequences'].size())
    xxx = input('')
    return None, None
    """

    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    approx_time = 0
    target_time = 0
    sample_time = 0
    target_call_times = 0
    approx_call_times = 0

    start_t = process_time_ns()

    if approx_model.config.is_encoder_decoder == True:
        encoder_outputs = approx_model.get_encoder()(
                    prefix, return_dict=True
                    )
        for key, val in encoder_outputs.items():
            if key != 'last_hidden_state':
                del encdoer_outputs[key]
        output_prefix = torch.LongTensor([[pad_token_id]]).to(prefix.device)
        T = max_len
    else:
        output_prefix = prefix
    model_kwargs = {}



#    with tqdm(total=T, desc="speculative sampling") as pbar:
    if True:
        while output_prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            prefix_len = output_prefix.shape[1]

            # generate x of size width * (prefix_len+gamma)
            tt = process_time_ns()
            if strategy == "beam":
                #out = approx_model.generate(x, num_beams=width, 
                #        output_scores=True, return_dict_in_generate=True, 
                #        num_return_sequences=width,
                #        top_k = top_k,
                #        top_p = top_p,
                #        do_sample=True,
                #        max_new_tokens = gamma)
                if num_beams is None:
                    num_beams = max(4, width)

                if approx_model.config.is_encoder_decoder:
                    encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state[0:1]

                    out = approx_model_cache.beam_sample_with_kv_cache(
                       output_prefix, 
                       gamma=gamma, 
                       num_beams=num_beams, 
                       top_k=top_k, top_p=top_p,
                       num_return_sequences=width,
                       return_dict_in_generate = True,
                       output_scores = True,
                       encoder_outputs = encoder_outputs
                       )
                else:
                    out = approx_model_cache.beam_sample_with_kv_cache(
                       output_prefix, 
                       gamma=gamma, 
                       num_beams=num_beams, 
                       top_k=top_k, top_p=top_p,
                       num_return_sequences=width,
                       return_dict_in_generate = True,
                       output_scores = True,
                       )

                x = out['sequences'] # width * (prefix_len+gamma)

                q = out['scores'] # tuples of gamma * (width * vocab) ?
                #q = torch.stack(q, dim=1) # width * gamma * vocab
                #for w in range(width):
                #   for i in range(q.shape[1]):
                       #q[w:w+1, i, :] = norm_logits(q[w:w+1, i, :],
                       #                  temperature, top_k, top_p)
                #       q[w:w+1, i, :] = F.softmax(q[w:w+1, i, :], dim=1)
            elif strategy == 'acc_beam':
                assert (acc_rate_head is not None)
                out = approx_model_cache.beam_sample_with_kv_cache(output_prefix, 
                     gamma=gamma, 
                     num_beams=width, 
                     top_k=top_k, top_p=top_p,
                     num_return_sequences=width,
                     return_dict_in_generate = True,
                     output_scores = True,
                     acc_rate_head = acc_rate_head,
                     acc_rate_thres = acc_rate_thres,
                     **model_kwargs
                     )

                x = out['sequences'] # width * (prefix_len+gamma)

                q = out['scores'] # tuples of gamma * (width * vocab) ?

            elif strategy == "diverse":
                raise NotImplementedError
                #out = approx_model.generate(x, num_beams=width, num_beam_groups=4, 
                #        diversity_penalty=0.1,
                #        output_scores=True, return_dict_in_generate=True, 
                #        top_k = top_k,
                #        top_p = top_p,
                #        do_sample=False,
                #        num_return_sequences=width,
                #        max_new_tokens = gamma)

                #x = out['sequences'] # width * (prefix_len+gamma)
                #q = out['scores'] # tuples of gamma * (width * vocab) ?
                #q = torch.stack(q, dim=1) # width * gamma * vocab
                #for w in range(width):
                #   for i in range(q.shape[1]):
                       #q[w:w+1, i, :] = norm_logits(q[w:w+1, i, :],
                       #                  temperature, top_k, top_p)
                #       q[w:w+1, i, :] = F.softmax(q[w:w+1, i, :], dim=1)


            elif strategy == 'iid':
                if approx_model.config.is_encoder_decoder == False:
                    out = approx_model_cache.generate(output_prefix,
                        gamma,
                        multi = width,
                        strategy = 'iid')
                else:
                    out = approx_model_cache.generate(prefix,
                        gamma,
                        decoder_input_ids = output_prefix,
                        multi = width,
                        strategy = 'iid')

                q = approx_model_cache._prob_history[:,prefix_len-1:,:]
                x = out

            else:
                raise RuntimeError("Strategy not implemented "+strategy)

            inc_len = x.shape[1] - prefix_len
            approx_call_times += 1



            approx_time += process_time_ns() - tt

            tt = process_time_ns()
   #         p = target_model(x).logits 
            if target_model.config.is_encoder_decoder == False:
                _ = target_model_cache.generate(x, 1)
            else:
                _ = target_model_cache.generate(
                        prefix.repeat_interleave(width, dim=0),
                        1, 
                        decoder_input_ids = x)


            target_call_times += 1
            p = target_model_cache._prob_history
            #for w in range(width):
            #    for i in range(p.shape[1]):
            #        p[w:w+1, i, :] = norm_logits(p[w:w+1, i, :],
            #                temperature, top_k, top_p)
            target_time += process_time_ns() - tt
            """
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                q = approx_model(x).logits
                next_tok = sample(norm_logits(q[:, -1, :], 
                                  temperature, top_k, top_p))
                x = torch.cat((x, next_tok), dim=1)
            # normalize the logits
            for i in range(q.shape[1]):
                q[:,i,:] = norm_logits(q[:,i,:],
                                temperature, top_k, top_p)
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            p = target_model(x).logits
            for i in range(p.shape[1]):
                p[:,i,:] = norm_logits(p[:,i,:],
                                temperature, top_k, top_p)

            """
            for w in range(width):
                for i in range(gamma):
                    if prefix_len + i >= x.size(1):
                        break
                    j = x[w, prefix_len+i]
                    acc_rate.append((p[w, prefix_len + i - 1, j] / q[w, i, j]).item())
                    if acc_rate[-1] > 1:
                        acc_rate[-1] = 1
                    if q[w,i,j] == 0:
                        acc_rate[-1] = 0
            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            tt = process_time_ns()
            
            is_all_accept = False
            n = prefix_len - 1
            max_n = prefix_len - 1
            max_l = 0
            choice = 0
            for w in range(width):
                cur_n = prefix_len - 1
                cur_l = 0
                cur_all_accept = True
                for i in range(inc_len):
                    if prefix_len + i >= x.size(1):
                        break
                    if random_seed:
                        torch.manual_seed(random_seed)
                    r = torch.rand(1, device = p.device)
                    j = x[w, prefix_len + i]

               
                    if r < torch.min(torch.tensor([1], device=q.device), p[w, prefix_len + i - 1, j] / q[w, i, j]):
                        cur_l += 1
                    # accept, and update n
                        cur_n += 1
                    else:
                        # reject
                        cur_all_accept = False
                        break
                if cur_l > max_l:
                    assert cur_n > max_n, f"cur_n {cur_n}, max_n {max_n}"
                    max_n = cur_n
                    max_l = cur_l
                    choice = w
                    if cur_all_accept == True:
                        is_all_accept = True
                        break
            acc_len.append(max_l)

            n = max_n
         
            output_prefix = x[choice:choice+1, :n + 1]
        
            approx_model_cache.rollback(n+1, choice)
           
            if is_all_accept:
                t = sample(p[choice:choice+1, -1, :])
                target_model_cache.rollback(n+2, choice)

            else:
                #print(torch.sum(p[choice,n,:]).item(), torch.sum(q[choice,max_l,:]).item())
                #tmp = p[choice,n,:] - q[choice,max_l,:]
                #print(torch.sum((tmp>0).float()).item(), torch.sum((tmp<=0).float()).item())
                #print(torch.sum((p[choice,n,:]==q[choice,max_l,:]).float()).item())
                new_p = max_fn(p[choice:choice+1, n, :] - q[choice:choice+1, max_l, :])

                try:
                    t = sample(new_p)
                except Exception as e:
                    # it seems it is possible to sample x where p = 0 and q = 0
                    t = sample(p[choice:choice+1, n, :])
                    #print(new_p.sum())
                    #print((p[choice:choice+1, n, :] - q[choice:choice+1, max_l, :]).max())
                    #print(p[choice:choice+1, n, :].sum())
                    #print(q[choice:choice+1, max_l, :].sum())
                    #j = x[choice, n+1]
                    #print(p[choice, n, j], q[choice, max_l, j])
                    #raise RuntimeError(f'{e}')


                target_model_cache.rollback(n+1, choice)
           
            output_prefix = torch.cat((output_prefix, t), dim=1)
            #pbar.update(n - pbar.n)
            sample_time += process_time_ns() - tt

            mask = (output_prefix == eos_token_id)
            if mask.int().sum() > ori_eos_cnt:
                mask = torch.cumsum(mask.float(), dim=1)
                mask = (mask < ori_eos_cnt+1)
                end = mask.int().sum()
                if end < mask.size(1):
                    mask[:, end] = True
                output_prefix = output_prefix[mask][None,:] 
                break




    
    if approx_model.config.is_encoder_decoder:
        output_prefix = torch.cat((prefix, output_prefix), dim=1)

    if verbose:
        print('approx model time', approx_time/1e9)
        print('target model time', target_time/1e9)
        print('other time', sample_time/1e9)
        print('inner overall time', (process_time_ns()-start_t)/1e9)
        print('acc len', np.mean(acc_len), len(acc_len), acc_len)
    if details == True:
        d = {
                'approx_time': approx_time,
                'target_time': target_time,
                'other_time': sample_time,
                'acc_len': acc_len,
                'acc_rate': np.mean(acc_rate),
                'target_call_times': target_call_times,
                'approx_call_times': approx_call_times
            }
        return output_prefix, d
    else:
        return output_prefix

@torch.no_grad()
def BiLD_sampling(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         gamma, eos_token_id,  
                         pad_token_id, fallback_thres, rollback_thres,
                         max_len : int ,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None,
                         details : bool = False) -> torch.Tensor:
    seq_len = prefix.shape[1]
    ori_eos_cnt = (prefix == eos_token_id).int().sum()
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    assert approx_model.device == target_model.device
    
    device = target_model.device
    
    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    
    approx_time = 0
    target_time = 0
    sample_time = 0
    approx_call_times = 0
    target_call_times = 0

    start_t = process_time_ns()

    acc_rate = []
    acc_len = []

    if pad_token_id is None:
        pad_token_id = eos_token_id
    decoder_input_ids = torch.LongTensor([[pad_token_id]]).to(prefix.device)

    if approx_model.config.is_encoder_decoder == False:
        last_check = seq_len - 1
    else:
        last_check = 0

    while prefix.shape[1] + decoder_input_ids.shape[1] - 1 < T:
        #print('loop')
        # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        tt = process_time_ns()
        
        if approx_model.config.is_encoder_decoder == False:
            x = approx_model_cache.generate(prefix, 1)
            prefix_len = prefix.shape[1]
        else:
            x = approx_model_cache.generate(prefix, 1, decoder_input_ids = decoder_input_ids)
            prefix_len = decoder_input_ids.shape[1]

        q = approx_model_cache._prob_history[:,prefix_len-1:,:]

        approx_call_times += 1
        
        approx_time += process_time_ns() - tt
        tt = process_time_ns()

        if torch.max(q[:,-1,:]) < fallback_thres or x.size(1)-last_check-1 >= gamma:
            # use large model to check
            ttt = process_time_ns()
        
            if target_model.config.is_encoder_decoder == False:
                _ = target_model_cache.generate(x, 1)
            else:
                _ = target_model_cache.generate(prefix, 1, decoder_input_ids = x)
            target_call_times += 1

            target_time += process_time_ns() - ttt
            p = target_model_cache._prob_history
            n = x.size(1) - 1
            l = 0
            for i in range(last_check, x.size(1)-1):
                j = x[:, i+1]
                if -p[:, i, j].log() > rollback_thres:
                    n = i
                    break
                l += 1
            acc_len.append(l)
            if approx_model.config.is_encoder_decoder == False:
                prefix = x[:, :n+1]
            else:
                decoder_input_ids = x[:, :n+1]
                
            approx_model_cache.rollback(n+1)
            t = sample(p[:, n, :])
            target_model_cache.rollback(n+1)
            last_check = n+1

            if approx_model.config.is_encoder_decoder == False:
                prefix = torch.cat((prefix, t), dim=1)
            else:
                decoder_input_ids = torch.cat((decoder_input_ids, t), dim=1)


        else: # continue
            if approx_model.config.is_encoder_decoder:
                decoder_input_ids = x
            else:
                prefix = x

        if approx_model.config.is_encoder_decoder:
            out = decoder_input_ids
        else:
            out = prefix

        mask = (out == eos_token_id)
        if mask.int().sum() > ori_eos_cnt:
            mask = torch.cumsum(mask.float(), dim=1)
            mask = (mask < ori_eos_cnt+1)
            end = mask.int().sum()
            if end < mask.size(1):
                mask[:, end] = True
            out = out[mask][None,:] 
            break



        sample_time += process_time_ns() - tt

    if approx_model.config.is_encoder_decoder == True:
        out = torch.cat((prefix, out), dim=1)

    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
        print(f"Acc rate: {np.mean(acc_rate)}")
        print('approx model time', approx_time/1e9)
        print('target model time', target_time/1e9)
        print('other time', sample_time/1e9)
        print('inner overall time', (process_time_ns()-start_t)/1e9)
        print('acc len', np.mean(acc_len), len(acc_len), acc_len)
    if details == True:
        d = {
                'approx_time': approx_time,
                'target_time': target_time,
                'other_time': sample_time,
                'acc_len': acc_len,
                'acc_rate': np.mean(acc_rate),
                'target_call_times': target_call_times,
                'approx_call_times': approx_call_times

            }
        return out, d
    else:
        return out



@torch.no_grad()
def speculative_sampling(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         eos_token_id, pad_token_id,
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None,
                         details : bool = False) -> torch.Tensor:
    """
    Google version Speculative Sampling.
    https://arxiv.org/pdf/2211.17192.pdf
        
    Adapted with KV Cache Optimization.
        
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    ori_eos_cnt = (prefix == eos_token_id).int().sum()
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    assert approx_model.device == target_model.device
    
    device = target_model.device
    
    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    
    approx_time = 0
    target_time = 0
    sample_time = 0
    approx_call_times = 0
    target_call_times = 0

    start_t = process_time_ns()

    acc_rate = []
    acc_len = []

    if pad_token_id is None:
        pad_token_id = eos_token_id
    decoder_input_ids = torch.LongTensor([[pad_token_id]]).to(prefix.device)

    while prefix.shape[1] + decoder_input_ids.shape[1] - 1 < T:
        #print('loop')
        # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        tt = process_time_ns()
        
        if approx_model.config.is_encoder_decoder == False:
            x = approx_model_cache.generate(prefix, gamma)
            prefix_len = prefix.shape[1]
        else:
            x = approx_model_cache.generate(prefix, gamma, decoder_input_ids = decoder_input_ids)
            prefix_len = decoder_input_ids.shape[1]

        approx_call_times += 1
        
        approx_time += process_time_ns() - tt
        tt = process_time_ns()
        
        if target_model.config.is_encoder_decoder == False:
            _ = target_model_cache.generate(x, 1)
        else:
            _ = target_model_cache.generate(prefix, 1, decoder_input_ids = x)
        target_call_times += 1

        target_time += process_time_ns() - tt
        tt = process_time_ns()
        
        n = prefix_len + gamma - 1

        for i in range(gamma):
            j = x[:, prefix_len + i]
            acc_rate.append(((target_model_cache._prob_history[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j])).item())
            if acc_rate[-1] > 1:
                acc_rate[-1] = 1
        

        l = 0
        for i in range(gamma):
            if random_seed:
                torch.manual_seed(random_seed)
            r = torch.rand(1, device = device)
            j = x[:, prefix_len + i]
            
            if r > (target_model_cache._prob_history[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
                # reject
                n = prefix_len + i - 1
                break
            
            if verbose:
                print(f"approx guess accepted {j[0]}: \033[31m{Decoder().decode(torch.tensor([j]))}\033[0m")

            accepted_count += 1
            l += 1
        acc_len.append(l)
        
        # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        if approx_model.config.is_encoder_decoder == False:
            prefix = x[:, :n + 1]
        else:
            decoder_input_ids = x[:, :n+1]
        
        approx_model_cache.rollback(n+1)
        #print('after roll back')
        #print(approx_model_cache._prob_history.size())
        assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"
        
        if n < prefix_len + gamma - 1:
            # reject someone, sample from the pos n
            t = sample(max_fn(target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, n, :]))
            if verbose:
                print(f"target resamples at position {n}: \033[34m{Decoder().decode(t)}\033[0m")
            resample_count += 1
            target_model_cache.rollback(n+1)
        else:
            # all approx model decoding accepted
            assert n == target_model_cache._prob_history.shape[1] - 1
            t = sample(target_model_cache._prob_history[:, -1, :])
            if verbose:
                print(f"target samples {n}: \033[35m{Decoder().decode(t)}\033[0m")
            target_sample_count += 1
            target_model_cache.rollback(n+2)
        
        
        if approx_model.config.is_encoder_decoder == False:
            prefix = torch.cat((prefix, t), dim=1)
            out = prefix
        else:
            decoder_input_ids = torch.cat((decoder_input_ids, t), dim=1)
            out = decoder_input_ids

        mask = (out == eos_token_id)
        if mask.int().sum() > ori_eos_cnt:
            mask = torch.cumsum(mask.float(), dim=1)
            mask = (mask < ori_eos_cnt+1)
            end = mask.int().sum()
            if end < mask.size(1):
                mask[:, end] = True
            out = out[mask][None,:] 
            break

        sample_time += process_time_ns() - tt

        #print(f'n={n}, l={l}')

    if approx_model.config.is_encoder_decoder == True:
        out = torch.cat((prefix, out), dim=1)

    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
        print(f"Acc rate: {np.mean(acc_rate)}")
        print('approx model time', approx_time/1e9)
        print('target model time', target_time/1e9)
        print('other time', sample_time/1e9)
        print('inner overall time', (process_time_ns()-start_t)/1e9)
        print('acc len', np.mean(acc_len), len(acc_len), acc_len)
    if details == True:
        d = {
                'approx_time': approx_time,
                'target_time': target_time,
                'other_time': sample_time,
                'acc_len': acc_len,
                'acc_rate': np.mean(acc_rate),
                'target_call_times': target_call_times,
                'approx_call_times': approx_call_times

            }
        return out, d
    else:
        return out


@torch.no_grad()
def speculative_sampling_v2(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, random_seed : int = None, details : bool = False) -> torch.Tensor:
    """
    DeepMind version Speculative Sampling.
    Accelerating Large Language Model Decoding with Speculative Sampling
    https://arxiv.org/abs/2302.01318
    No KV Cache Optimization
    
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"
    approx_time = 0
    target_time = 0
    sample_time = 0


    acc_rate = []
    acc_len = []


    #with tqdm(total=T, desc="speculative sampling") as pbar:
    if True:
        while prefix.shape[1] < T:
            tt = process_time_ns()
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            x = prefix
            prefix_len = prefix.shape[1]
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                q = approx_model(x).logits
                next_tok = sample(norm_logits(q[:, -1, :], 
                                  temperature, top_k, top_p))
                x = torch.cat((x, next_tok), dim=1)
            
            # normalize the logits
            for i in range(q.shape[1]):
                q[:,i,:] = norm_logits(q[:,i,:],
                                temperature, top_k, top_p)
            approx_time += process_time_ns() - tt
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            tt = process_time_ns()
            p = target_model(x).logits
            for i in range(p.shape[1]):
                p[:,i,:] = norm_logits(p[:,i,:],
                                temperature, top_k, top_p)
            target_time += process_time_ns() - tt

            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            tt = process_time_ns()
            
            is_all_accept = True
            n = prefix_len - 1
            l = 0
            for i in range(gamma):
                if random_seed:
                    torch.manual_seed(random_seed)
                r = torch.rand(1, device = p.device)
                j = x[:, prefix_len + i]
                acc_rate.append(torch.min(torch.tensor([1], device=q.device), p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j]).item())
                
                if r < torch.min(torch.tensor([1], device=q.device), p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j]):
                    # accept, and update n
                    n += 1
                    l += 1
                else:
                    # reject
                    try:
                        t = sample(max_fn(p[:, n, :] - q[:, n, :]))
                    except Exception as e:
                        print(e)
                        print('reject, r, t: ', r, p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j])
                        print(prefix_len+i-1, n)
                        print(torch.sum(p[:,n,:]), torch.sum(q[:,n,:]))
                        print(torch.sum(((p[:,n,:]-q[:,n,:])>0).float()))
                        raise RuntimeError(f'{e}')

                    is_all_accept = False
                    break
            acc_len.append(l)
         
            prefix = x[:, :n + 1]
            
            if is_all_accept:
                t = sample(p[:, -1, :])
            
            prefix = torch.cat((prefix, t), dim=1)
            sample_time += process_time_ns() - tt
#            pbar.update(n - pbar.n)
    if details == True:
        d = {
                'approx_time': approx_time,
                'target_time': target_time,
                'other_time': sample_time,
                'acc_len': acc_len,
                'acc_rate': np.mean(acc_rate)
            }
        return prefix, d
    else:
        return prefix


