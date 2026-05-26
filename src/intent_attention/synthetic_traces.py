import random
from typing import Dict, List
from .block_metadata import BlockPolicy, SemanticBlock, BlockLayout

def generate_agentic_layout(total_tokens: int, doc_blocks: int = 4, tool_blocks: int = 2, recent_tokens: int = 2048, score_threshold: float = 0.5) -> BlockLayout:
    blocks = []
    curr = 0
    
    sys_tok = min(512, total_tokens)
    if sys_tok > 0:
        blocks.append(SemanticBlock("system_prompt", curr, curr + sys_tok, BlockPolicy.ALWAYS))
        curr += sys_tok
        
    mem_tok = min(256, total_tokens - curr)
    if mem_tok > 0:
        blocks.append(SemanticBlock("memory_summary", curr, curr + mem_tok, BlockPolicy.ATTEND))
        curr += mem_tok
        
    for i in range(doc_blocks):
        dt = min(1024, total_tokens - curr)
        if dt <= 0: break
        score = random.random()
        pol = BlockPolicy.ATTEND if score >= score_threshold else BlockPolicy.SKIP
        blocks.append(SemanticBlock(f"retrieved_doc_{i}", curr, curr + dt, pol, score))
        curr += dt
        
    for i in range(tool_blocks):
        tt = min(512, total_tokens - curr)
        if tt <= 0: break
        score = random.random()
        pol = BlockPolicy.ATTEND if score >= score_threshold else BlockPolicy.SKIP
        blocks.append(SemanticBlock(f"tool_output_{i}", curr, curr + tt, pol, score))
        curr += tt
        
    if curr < total_tokens:
        recent_start = max(curr, total_tokens - recent_tokens)
        if recent_start > curr:
            blocks.append(SemanticBlock("ignored_context", curr, recent_start, BlockPolicy.SKIP))
        blocks.append(SemanticBlock("recent_context", recent_start, total_tokens, BlockPolicy.RECENT))
        
    return BlockLayout(blocks)

def random_layout(total_tokens: int, num_blocks: int = 10) -> BlockLayout:
    blocks = []
    curr = 0
    tok_per = total_tokens // num_blocks
    for i in range(num_blocks):
        end = curr + tok_per if i < num_blocks - 1 else total_tokens
        blocks.append(SemanticBlock(f"block_{i}", curr, end, random.choice(list(BlockPolicy))))
        curr = end
    return BlockLayout(blocks)

def layout_from_policy_dict(policy_dict: Dict[str, BlockPolicy], block_sizes: Dict[str, int]) -> BlockLayout:
    blocks = []
    curr = 0
    for name, pol in policy_dict.items():
        sz = block_sizes.get(name, 0)
        blocks.append(SemanticBlock(name, curr, curr + sz, pol))
        curr += sz
    return BlockLayout(blocks)
