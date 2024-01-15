import torch

from tqdm import tqdm
from sampling.utils import norm_logits, sample

@torch.no_grad()
def autoregressive_sampling(x : torch.Tensor, model : torch.nn.Module, N : int, eos_token_id : int,  
                            temperature : float = 1, top_k : int = 0, top_p : float = 0, pad_token_id = None):
    n = len(x)
    T = len(x) + N

    past_key_values = None
    if pad_token_id is None:
        pad_token_id = eos_token_id

    decoder_x = torch.LongTensor([[pad_token_id]]).to(x.device)
    while n < T:
        # outputs = model(x)
        if past_key_values:
            if model.config.is_encoder_decoder:
                last_ids = decoder_x[:, -1]
                if last_ids.dim() == 1:
                    last_ids = torch.unsqueeze(last_ids, 0)
                outputs = model(x, decoder_input_ids = last_ids, past_key_values = past_key_values, use_cache = True)
            else:
                last_ids = x[:, -1]
                if last_ids.dim() == 1:
                    last_ids = torch.unsqueeze(last_ids, 0)
                outputs = model(last_ids, past_key_values = past_key_values, use_cache = True)

        else:
            if model.config.is_encoder_decoder:
                outputs = model(x, decoder_input_ids = decoder_x)
            else:
                outputs = model(x)
        last_p = norm_logits(outputs.logits[::, -1, :], temperature, top_k, top_p)
        past_key_values = outputs.past_key_values
        idx_next = sample(last_p)
        if model.config.is_encoder_decoder:
            decoder_x = torch.cat((decoder_x, idx_next), dim=1)
        else:
            x = torch.cat((x, idx_next), dim=1)
        n += 1
        if idx_next == eos_token_id:
            break

    if model.config.is_encoder_decoder:
        x = torch.cat((x, decoder_x), dim=1)
    return x

