import torch
import math
from typing import Dict, Tuple, Any, Union
from .block_metadata import BlockLayout

def dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    if causal:
        raise NotImplementedError("Causal masking is not fully supported yet in reference dense_attention.")
    
    head_dim = q.size(-1)
    scale = 1.0 / math.sqrt(head_dim)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    
    output = torch.matmul(attn_weights, v)
    return output

def semantic_block_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    layout: BlockLayout, 
    causal: bool = False, 
    return_debug: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    kv_tokens = k.size(-2)
    layout.validate(kv_tokens)
    
    selected_indices = layout.selected_token_indices()
    
    if not selected_indices:
        out = torch.zeros_like(q)
        if return_debug:
            return out, {
                "selected_token_count": 0,
                "selected_block_names": [],
                "total_kv_tokens": kv_tokens,
                "selected_kv_tokens": 0
            }
        return out

    idx_tensor = torch.tensor(selected_indices, dtype=torch.long, device=k.device)
    
    selected_k = k.index_select(-2, idx_tensor)
    selected_v = v.index_select(-2, idx_tensor)
    
    output = dense_attention(q, selected_k, selected_v, causal=causal)
    
    if return_debug:
        debug_info = {
            "selected_token_count": layout.selected_token_count(),
            "selected_block_names": [b.name for b in layout.selected_blocks()],
            "total_kv_tokens": kv_tokens,
            "selected_kv_tokens": len(selected_indices)
        }
        return output, debug_info
        
    return output
