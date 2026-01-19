



from typing import Any, Set


def extract_values_only(data: Any) -> Set[str]:
    values: Set[str] = set()

    def walk(obj: Any):
        if isinstance(obj, dict):
            for v in obj.values():   # ✅ values ONLY
                walk(v)

        elif isinstance(obj, (list, set, tuple)):
            for item in obj:
                walk(item)

        elif isinstance(obj, str):
            values.add(obj)

        # ❌ ignore bool, int, float, None, everything else

    walk(data)
    return values

