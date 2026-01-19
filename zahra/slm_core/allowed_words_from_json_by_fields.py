# allowed_words_from_json_by_fields.py



from typing import Any, Set, Iterable


def extract_values_by_fields(data: Any, allowed_fields: Iterable[str]) -> Set[str]:
    """
    Extract string values ONLY from fields listed in allowed_fields.
    """
    values: Set[str] = set()
    allowed_fields = set(allowed_fields)

    def walk(obj: Any):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in allowed_fields:
                    collect(v)
                else:
                    walk(v)

        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    def collect(obj: Any):
        if isinstance(obj, str):
            values.add(obj)
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                collect(item)
        elif isinstance(obj, dict):
            for v in obj.values():
                collect(v)

    walk(data)
    return values
