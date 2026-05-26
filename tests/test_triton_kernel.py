import pytest
import torch
from intent_attention.triton_kernel import is_cuda_available, semantic_block_attention_triton
from intent_attention.block_metadata import SemanticBlock, BlockPolicy, BlockLayout

@pytest.mark.skipif(not is_cuda_available(), reason="Requires CUDA to run Triton Kernels")
def test_triton_kernel_execution():
    q = torch.randn(1, 1, 16, 64).cuda()
    k = torch.randn(1, 1, 128, 64).cuda()
    v = torch.randn(1, 1, 128, 64).cuda()
    
    layout = BlockLayout([
        SemanticBlock("a", 0, 128, BlockPolicy.ALWAYS)
    ])
    
    with pytest.raises(NotImplementedError):
        semantic_block_attention_triton(q, k, v, layout)
