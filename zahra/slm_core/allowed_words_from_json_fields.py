# allowed_words_from_json_fields.py



from typing import Any, Set


def extract_allowed_words(data: Any) -> Set[str]:
    keys: Set[str] = set()

    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(k, str):
                keys.add(k)
            if isinstance(v, (dict, list)):
                keys |= extract_allowed_words(v)

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                keys |= extract_allowed_words(item)

    return keys



