import torch
from torch.nn import functional as F
import pickle
import re
import string
import sqlite3

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = exact_match_score(prediction, ground_truth)
        scores_for_ground_truths.append(score)
#    if max(scores_for_ground_truths) < 1:
#        print(prediction, ground_truths)
    return max(scores_for_ground_truths)

def exact_match_references(predictions, references, debug = False):

    em = 0
    for pred, refer in zip(predictions, references):
        score = metric_max_over_ground_truths(pred, refer)
        em += score
        if debug == True and score < 1.:
            print("pred")
            print(pred)
            print("refer")
            print(refer)
    return {"exact_match": 100.0 * em / len(predictions)}

def match(result1, result2):
    str_result1 = set()
    str_result2 = set()
    for row in result1:
        str_row = [str(c) for c in row]
        str_row.sort()
        str_result1.add(tuple(str_row))
    for row in result2:
        str_row = [str(c) for c in row]
        str_row.sort()
        str_result2.add(tuple(str_row))

    return str_result1 == str_result2

def execution_accuracy(db, pred, sql):
    conn = sqlite3.connect(f"./spider/spider/database/{db}/{db}.sqlite")
    conn.text_factory = str
    cur = conn.cursor()
    gt_result = cur.execute(sql).fetchall()
    try:
        result = cur.execute(pred).fetchall()
    except:
        return 0
    spider_acc = float(match(result, gt_result))
    return spider_acc


def execution_accuracy_references(predictions, references):
    em = 0
    for pred, refer in zip(predictions, references):
        db = refer.split("[SQL]")[0]
        sql = refer.split("[SQL]")[1]
        em += execution_accuracy(db, pred, sql)
    return {"execution accuracy": 100.0 * em / len(predictions)}

def get_seq_att_mask(input_cnt, all_input_idx, all_beam_idx, all_next_token, input_len, pad_token_id, device='cpu'):
    """
    Return sequences of shape input_cnt * max_seq_len
    Return tree attention mask of shape input_cnt * max_seq_len * max_seq_len

    """
    ret_seq = [[] for i in range(input_cnt)]
    ret_att_mask = [[] for i in range(input_cnt)]
    position_ids = [[] for i in range(input_cnt)]

    last_beam_att_mask = [[] for i in range(all_input_idx[0].numel())]
    max_seq_len = 0
    pos = [[i,-1] for i in range(input_cnt)]
    position = input_len
    for input_idx_list, beam_idx_list, next_token_list in zip(all_input_idx, all_beam_idx, all_next_token):
        cur_beam_att_mask = []
        for j in range(input_idx_list.numel()):
            input_idx = input_idx_list[j].item()
            next_token = next_token_list[j].item()
            beam_idx = beam_idx_list[j].item()

            cur_seq_len = len(ret_seq[input_idx])
            pos.append([input_idx, cur_seq_len])
            ret_seq[input_idx].append(next_token)
            position_ids[input_idx].append(position)

            if len(ret_seq[input_idx]) > max_seq_len:
              max_seq_len = len(ret_seq[input_idx])
            prev_seq_len = len(last_beam_att_mask[beam_idx])

            mask = last_beam_att_mask[beam_idx] + [False for k in range(cur_seq_len-prev_seq_len)]+[True]
            ret_att_mask[input_idx].append(mask)
            cur_beam_att_mask.append(mask)
        last_beam_att_mask = cur_beam_att_mask
        position += 1

    """ padding """
    for i in range(input_cnt):
      position_ids[i] += [0 for j in range(max_seq_len-len(ret_seq[i]))]
      ret_seq[i] += [pad_token_id for j in range(max_seq_len-len(ret_seq[i]))]

      for j in range(len(ret_att_mask[i])):
        ret_att_mask[i][j] += [False for k in range(max_seq_len-len(ret_att_mask[i][j]))]
      ret_att_mask[i] += [[False for k in range(max_seq_len)] for j in range(max_seq_len-len(ret_att_mask[i]))]

    ret_seq = torch.LongTensor(ret_seq).to(device)
    ret_att_mask = torch.Tensor(ret_att_mask)
    full_att_mask = torch.ones(input_cnt, max_seq_len, max_seq_len+input_len).bool()
    full_att_mask[:,:,input_len:] = ret_att_mask
    full_att_mask = full_att_mask.to(device)
    pos = torch.LongTensor(pos).to(device)
    position_ids = torch.LongTensor(position_ids).to(device)
    return ret_seq, full_att_mask, pos, position_ids


# copy from https://github.com/LeeSinLiang/microGPT/blob/ed40cf9780dbeb180adfe94c227d4aa97e69250e/gpt.py
def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
#    """ for debugging """
#    return logits

    if top_k is not None and top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p is not None and top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits


def norm_logits(logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    #assert logits.isnan().any() == False
    ori_logits = logits
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = torch.log_softmax(logits, dim=1).exp()
#    sample(probs, logits = logits)
    #if probs.isnan().any() or probs.isinf().any() or (probs<0).any():
    #    probs = torch.softmax(ori_logits, dim=1)
    if probs.isnan().any() or probs.isinf().any() or (probs<0).any():
        print(torch.logical_not(logits.isinf()).any())
        print(logits[probs.isnan()])
        print(logits[probs.isinf()])
        raise RuntimeError('norm logits error')


    return probs


def sample(probs : torch.Tensor, num_samples: int = 1):
    if torch.numel(probs.nonzero()) < num_samples:
        replacement = True
    else:
        replacement = False
    """ for debugging """
    replacement = True
    try:
        idx_next = torch.multinomial(probs, num_samples=num_samples, replacement = replacement)
    except:
        print((probs<0).any(), probs.isnan().any(), probs.isinf().any())
        raise RuntimeError('prob error')
    #if (idx_next.item() == 0) and False:
    #    raise RuntimeError
    return idx_next


def max_fn(x):
    """
        norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    if x.dim() > 1:
        x_max_sum = torch.sum(x_max, dim=1, keepdim=True)
    else:
        x_max_sum = torch.sum(x_max)
    return x_max / x_max_sum
