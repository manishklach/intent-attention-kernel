import torch
from intent_attention.block_metadata import SemanticBlock, BlockPolicy, BlockLayout
from intent_attention.block_table import BlockTable

def test_block_table_creation():
    layout = BlockLayout([
        SemanticBlock("a", 0, 64, BlockPolicy.ALWAYS),
        SemanticBlock("b", 64, 128, BlockPolicy.SKIP),
        SemanticBlock("c", 128, 192, BlockPolicy.ATTEND)
    ])
    
    bt = BlockTable(block_size=64)
    table, tokens = bt.create_block_table(layout, 192)
    
    assert tokens == 128
    assert len(table) == 2
    assert table[0].item() == 0  # Logical tokens 0-63 (Page 0)
    assert table[1].item() == 2  # Logical tokens 128-191 (Page 2)
