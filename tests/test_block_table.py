import torch
from intent_attention.block_metadata import SemanticBlock, BlockPolicy, BlockLayout
from intent_attention.block_table import BlockTable


def test_block_table_creation():
    layout = BlockLayout([
        SemanticBlock("a", 0, 64, BlockPolicy.ALWAYS),
        SemanticBlock("b", 64, 128, BlockPolicy.SKIP),
        SemanticBlock("c", 128, 192, BlockPolicy.ATTEND, score=0.8),
    ])
    bt = BlockTable(block_size=64)
    table, tokens = bt.create_block_table(layout, 192)
    assert tokens == 128
    assert len(table) == 2
    assert table[0].item() == 0
    assert table[1].item() == 2


def test_block_table_empty():
    layout = BlockLayout([
        SemanticBlock("a", 0, 64, BlockPolicy.SKIP),
    ])
    bt = BlockTable(block_size=64)
    table, tokens = bt.create_block_table(layout, 64)
    assert tokens == 0
    assert table.numel() == 0


def test_block_table_spanning_multiple_pages():
    layout = BlockLayout([
        SemanticBlock("a", 0, 150, BlockPolicy.ALWAYS),
    ])
    bt = BlockTable(block_size=64)
    table, tokens = bt.create_block_table(layout, 150)
    assert tokens == 150
    assert len(table) == 3
    assert table.tolist() == [0, 1, 2]


def test_block_table_mid_block_start():
    layout = BlockLayout([
        SemanticBlock("a", 60, 130, BlockPolicy.ALWAYS),
    ])
    bt = BlockTable(block_size=64)
    table, tokens = bt.create_block_table(layout, 130)
    assert tokens == 70
    assert table[0].item() == 0
    assert table[1].item() == 1
    assert table[2].item() == 2
    assert len(table) == 3


def test_block_table_two_pages():
    layout = BlockLayout([
        SemanticBlock("a", 32, 96, BlockPolicy.ALWAYS),
    ])
    bt = BlockTable(block_size=64)
    table, tokens = bt.create_block_table(layout, 96)
    assert tokens == 64
    assert len(table) == 2
    assert table.tolist() == [0, 1]


def test_block_table_dtype():
    layout = BlockLayout([
        SemanticBlock("a", 0, 64, BlockPolicy.ALWAYS),
    ])
    bt = BlockTable(block_size=64)
    table, _ = bt.create_block_table(layout, 64)
    assert table.dtype == torch.int32
