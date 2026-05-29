"""Tests for kv_quant.py."""
from __future__ import annotations

import torch

from intent_attention.kv_quant import (
    quantise_k_perchannel, dequantise_k,
    quantise_v_pertoken, dequantise_v,
    KVQuantStore, QuantisedPage,
)


def test_quantise_k_roundtrip():
    k = torch.randn(64, 128)
    k_int8, scale, zero = quantise_k_perchannel(k)
    k_deq = dequantise_k(k_int8, scale, zero)
    assert k_deq.shape == k.shape
    error = (k.float() - k_deq.float()).abs().mean().item()
    assert error < 1.0


def test_quantise_v_roundtrip():
    v = torch.randn(64, 128)
    v_int8, scale, zero = quantise_v_pertoken(v)
    v_deq = dequantise_v(v_int8, scale, zero)
    assert v_deq.shape == v.shape
    assert v_int8.dtype == torch.int8


def test_quantised_page_dequantise():
    k = torch.randn(64, 128)
    v = torch.randn(64, 128)
    k_int8, ks, kz = quantise_k_perchannel(k)
    v_int8, vs, vz = quantise_v_pertoken(v)
    page = QuantisedPage(k_int8, v_int8, ks, kz, vs, vz)
    k_deq = page.dequantise_k()
    v_deq = page.dequantise_v()
    assert k_deq.shape == k.shape
    assert v_deq.shape == v.shape


def test_kv_quant_store():
    store = KVQuantStore(page_size=64)
    k = torch.randn(64, 128)
    v = torch.randn(64, 128)
    store.append_page(0, k, v)
    k_get, v_get = store.get_block_kv(0)
    assert k_get is not None
    assert v_get is not None
    assert k_get.shape == k.shape
    assert v_get.shape == v.shape


def test_kv_quant_store_residual():
    store = KVQuantStore(residual_r=32)
    store.update_residual(torch.randn(64, 128), torch.randn(64, 128))
    store.update_residual(torch.randn(64, 128), torch.randn(64, 128))
    res_k, res_v = store.get_residual()
    assert res_k is not None
    assert res_k.shape[0] <= 32


def test_kv_quant_store_missing_block():
    store = KVQuantStore()
    k_get, v_get = store.get_block_kv(42)
    assert k_get is None
    assert v_get is None


def test_kv_quant_store_memory_bytes():
    store = KVQuantStore(page_size=64)
    store.append_page(0, torch.randn(64, 128), torch.randn(64, 128))
    mem = store.memory_bytes()
    assert "quantised_bytes" in mem
    assert "residual_fp16_bytes" in mem
    assert mem["quantised_bytes"] > 0


def test_kv_quant_store_snr():
    store = KVQuantStore(page_size=64)
    k = torch.randn(64, 128)
    v = torch.randn(64, 128)
    store.append_page(0, k, v)
    snr = store.snr_db(0, k, v)
    assert "k_snr_db" in snr
    assert "v_snr_db" in snr


def test_quantise_k_default_group_size():
    k = torch.randn(128, 64)
    k_int8, scale, zero = quantise_k_perchannel(k)
    assert k_int8.dtype == torch.int8


def test_quantise_v_pertoken_shape():
    v = torch.randn(32, 64)
    v_int8, scale, zero = quantise_v_pertoken(v)
    assert v_int8.shape == (32, 64)
    assert scale.shape == (32,)
