from typing import Dict

def attention_flops(batch: int, heads: int, query_tokens: int, kv_tokens: int, head_dim: int) -> int:
    return 4 * batch * heads * query_tokens * kv_tokens * head_dim

def kv_read_bytes(batch: int, heads: int, kv_tokens: int, head_dim: int, dtype_bytes: int = 2) -> int:
    return 2 * batch * heads * kv_tokens * head_dim * dtype_bytes

def semantic_attention_cost(batch: int, heads: int, query_tokens: int, total_kv_tokens: int, selected_kv_tokens: int, head_dim: int, dtype_bytes: int = 2) -> Dict[str, int]:
    return {
        "flops": attention_flops(batch, heads, query_tokens, selected_kv_tokens, head_dim), 
        "kv_bytes": kv_read_bytes(batch, heads, selected_kv_tokens, head_dim, dtype_bytes)
    }

def savings_report(batch: int, heads: int, query_tokens: int, total_kv_tokens: int, selected_kv_tokens: int, head_dim: int, dtype_bytes: int = 2) -> Dict[str, float]:
    dense_flops = attention_flops(batch, heads, query_tokens, total_kv_tokens, head_dim)
    sem_flops = attention_flops(batch, heads, query_tokens, selected_kv_tokens, head_dim)
    dense_kv = kv_read_bytes(batch, heads, total_kv_tokens, head_dim, dtype_bytes)
    sem_kv = kv_read_bytes(batch, heads, selected_kv_tokens, head_dim, dtype_bytes)
    
    flops_saved_pct = (dense_flops - sem_flops) / dense_flops * 100 if dense_flops > 0 else 0
    kv_bytes_saved_pct = (dense_kv - sem_kv) / dense_kv * 100 if dense_kv > 0 else 0
    selected_fraction = selected_kv_tokens / total_kv_tokens if total_kv_tokens > 0 else 0
    
    return {
        "dense_flops": dense_flops,
        "semantic_flops": sem_flops,
        "dense_kv_bytes": dense_kv,
        "semantic_kv_bytes": sem_kv,
        "flops_saved_pct": flops_saved_pct,
        "kv_bytes_saved_pct": kv_bytes_saved_pct,
        "selected_fraction": selected_fraction
    }
