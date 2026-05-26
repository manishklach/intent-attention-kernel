import pytest
from intent_attention.block_metadata import SemanticBlock, BlockPolicy, BlockLayout

def test_valid_layout():
    layout = BlockLayout([SemanticBlock("a", 0, 10, BlockPolicy.ALWAYS)])
    layout.validate(10)
    assert layout.selected_token_count() == 10

def test_overlapping_fails():
    layout = BlockLayout([
        SemanticBlock("a", 0, 10, BlockPolicy.ALWAYS),
        SemanticBlock("b", 5, 15, BlockPolicy.SKIP)
    ])
    with pytest.raises(ValueError):
        layout.validate(15)

def test_unsorted_fails():
    layout = BlockLayout([
        SemanticBlock("b", 10, 20, BlockPolicy.SKIP),
        SemanticBlock("a", 0, 10, BlockPolicy.ALWAYS)
    ])
    with pytest.raises(ValueError):
        layout.validate(20)

def test_empty_name_fails():
    layout = BlockLayout([SemanticBlock("", 0, 10, BlockPolicy.ALWAYS)])
    with pytest.raises(ValueError):
        layout.validate(10)
