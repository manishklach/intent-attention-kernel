from intent_attention.synthetic_traces import generate_agentic_layout

def test_generate_agentic_layout():
    layout = generate_agentic_layout(8192)  # Increase size so recent_context is reached
    layout.validate(8192)
    assert layout.selected_token_count() <= layout.total_token_count()
    names = [b.name for b in layout.blocks]
    assert "system_prompt" in names
    assert "recent_context" in names
