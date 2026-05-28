from __future__ import annotations

from typing import Optional

import pytest
import torch

from intent_attention.block_metadata import BlockLayout, BlockPolicy, SemanticBlock
from intent_attention.cost_model import savings_report
from intent_attention.hf_patch import _extract_layer_idx, _is_attention_module, patch_model


def test_extract_layer_idx() -> None:
    assert _extract_layer_idx("model.layers.0.self_attn") == 0
    assert _extract_layer_idx("transformer.h.2.attn") == 2
    assert _extract_layer_idx("embed_tokens") == 0


def test_is_attention_module() -> None:
    class DummyAttn(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(64, 64)
            self.k_proj = torch.nn.Linear(64, 64)
            self.v_proj = torch.nn.Linear(64, 64)
            self.o_proj = torch.nn.Linear(64, 64)

    attn = DummyAttn()
    assert _is_attention_module(attn) is True

    empty = torch.nn.Module()
    assert _is_attention_module(empty) is False


class SimpleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(100, 64)
        self.layers = torch.nn.ModuleList([SimpleDecoderLayer() for _ in range(3)])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        return h


class SimpleDecoderLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = SimpleAttention()
        self.mlp = torch.nn.Linear(64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.self_attn(x))


class SimpleAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(64, 64)
        self.k_proj = torch.nn.Linear(64, 64)
        self.v_proj = torch.nn.Linear(64, 64)
        self.o_proj = torch.nn.Linear(64, 64)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        bsz, q_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, q_len, 1, 64).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, 1, 64).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, 1, 64).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, 64)
        return self.o_proj(out)


def test_patch_model_with_simple_model() -> None:
    model = SimpleModel()
    input_ids = torch.randint(0, 100, (1, 32))

    layout = BlockLayout([
        SemanticBlock("system", 0, 8, BlockPolicy.ALWAYS),
        SemanticBlock("docs", 8, 24, BlockPolicy.ATTEND, score=0.9),
        SemanticBlock("rest", 24, 32, BlockPolicy.SKIP),
    ])

    call_count = 0

    def layout_fn(layer_idx: int) -> Optional[BlockLayout]:
        nonlocal call_count
        call_count += 1
        return layout

    patch_model(model, layout_fn, verbose=False)

    with torch.no_grad():
        out = model(input_ids)

    assert out.shape == (1, 32, 64)
    assert torch.isfinite(out).all()
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert call_count == 3


@pytest.mark.skipif(True, reason="requires internet to download model; run manually")
def test_patch_tiny_gpt2() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-GPT2")
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-GPT2")
    model.eval()

    inputs = tokenizer("Hello, world!", return_tensors="pt")

    def layout_fn(layer_idx: int) -> Optional[BlockLayout]:
        return BlockLayout([
            SemanticBlock("system", 0, 2, BlockPolicy.ALWAYS),
            SemanticBlock("rest", 2, 6, BlockPolicy.ATTEND, score=0.9),
        ])

    patch_model(model, layout_fn, verbose=False)

    with torch.no_grad():
        out = model(**inputs)

    assert torch.isfinite(out.logits).all()
