# ai_plugin.py



from slm_core.intent_expander_fields import generate_intent_expansion1
from slm_core.intent_expander_values import generate_intent_expansion2

def generate_intent_expansion_plugin(question: str, field_only=False, debug: bool = True):
    """
    Unified AI plugin for field and value detection.

    Args:
        question: User query text
        field_only: If True, only detects fields (skip value expansion)
        debug: Prints debug info if True

    Returns:
        dict with "fields" (detected fields) and "expanded_text" (value expansion)
    """
    if debug:
        print("\n[AI intent plugin: field detection]\n")
    selected_fields = generate_intent_expansion1(question)

    expanded_values = ""
    if not field_only:
        if debug:
            print("\n[AI intent plugin: value expansion]\n")
        expanded_values = generate_intent_expansion2(question, selected_fields)

    return {
        "fields": selected_fields,
        "expanded_text": expanded_values,
    }
