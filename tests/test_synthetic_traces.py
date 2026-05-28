from intent_attention.block_metadata import BlockPolicy
from intent_attention.synthetic_traces import (
    generate_agentic_layout,
    layout_from_policy_dict,
    random_layout,
)


class TestGenerateAgenticLayout:
    def test_basic(self):
        layout = generate_agentic_layout(8192)
        layout.validate(8192)
        assert layout.selected_token_count() <= layout.total_token_count()
        names = [b.name for b in layout.blocks]
        assert "system_prompt" in names
        assert "recent_context" in names

    def test_deterministic_with_seed(self):
        a = generate_agentic_layout(8192, seed=42)
        b = generate_agentic_layout(8192, seed=42)
        assert [b.name for b in a.blocks] == [b.name for b in b.blocks]
        assert a.selected_token_count() == b.selected_token_count()

    def test_different_seeds_differ(self):
        a = generate_agentic_layout(8192, seed=1)
        b = generate_agentic_layout(8192, seed=99)
        # They may occasionally match but that's astronomically unlikely
        block_names_match = [b.name for b in a.blocks] == [b.name for b in b.blocks]
        selected_count_match = a.selected_token_count() == b.selected_token_count()
        assert not (
            block_names_match and selected_count_match
        ), "Expected different seeds to produce different layouts"

    def test_small_total_tokens(self):
        layout = generate_agentic_layout(128)
        layout.validate(128)
        names = [b.name for b in layout.blocks]
        assert "system_prompt" in names

    def test_all_blocks_validate(self):
        for size in [512, 2048, 8192, 65536]:
            layout = generate_agentic_layout(size)
            layout.validate(size)

    def test_selected_policies_are_valid(self):
        layout = generate_agentic_layout(8192, doc_blocks=0, tool_blocks=0)
        for block in layout.selected_blocks():
            assert block.policy in (
                BlockPolicy.ALWAYS,
                BlockPolicy.ATTEND,
                BlockPolicy.RECENT,
                BlockPolicy.GLOBAL,
            )


class TestRandomLayout:
    def test_basic(self):
        layout = random_layout(1000, num_blocks=5, seed=0)
        layout.validate(1000)
        assert len(layout.blocks) == 5

    def test_deterministic(self):
        a = random_layout(1000, num_blocks=10, seed=7)
        b = random_layout(1000, num_blocks=10, seed=7)
        assert a.selected_token_count() == b.selected_token_count()

    def test_attend_blocks_have_scores(self):
        layout = random_layout(1000, num_blocks=20, seed=42)
        for b in layout.blocks:
            if b.policy == BlockPolicy.ATTEND:
                assert b.score is not None


class TestLayoutFromPolicyDict:
    def test_basic(self):
        policies = {
            "sys": BlockPolicy.ALWAYS,
            "docs": BlockPolicy.ATTEND,
            "skip": BlockPolicy.SKIP,
        }
        sizes = {"sys": 100, "docs": 200, "skip": 50}
        layout = layout_from_policy_dict(policies, sizes)
        layout.validate(350)
        assert layout.selected_token_count() == 300

    def test_empty_dict(self):
        layout = layout_from_policy_dict({}, {})
        assert len(layout.blocks) == 0
        layout.validate(0)
