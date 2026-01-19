# main.py



import json
from collections import defaultdict
from difflib import get_close_matches, SequenceMatcher
from slm_core.ai_response import answer_question
from slm_core.ai_plugin import generate_intent_expansion_plugin  


FUZZY_CUTOFF = 1
SKILL_BOOST_CUTOFF = 1


def is_context_empty(ctx: str | None) -> bool:
    return not ctx or not ctx.strip()


# ============================
# Employee Loading
# ============================
import os
import json

def load_employees():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "data", "employees_v2.json")

    print("USING EMPLOYEE DATA:", path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def flatten_employee_values(employee):
    values = set()
    value_to_field = {}

    for key, value in employee.items():
        if isinstance(value, bool):
            s = str(value).lower()
            values.add(s)
            value_to_field[s] = key
            continue
        if isinstance(value, (int, float)):
            continue
        if isinstance(value, list):
            for v in value:
                if isinstance(v, (int, float)):
                    continue
                s = str(v).lower()
                values.add(s)
                value_to_field[s] = key
        else:
            s = str(value).lower()
            values.add(s)
            value_to_field[s] = key

    return values, value_to_field


def detect_requested_fields(tokens, sample_employee):
    fields = []
    keys = list(sample_employee.keys())
    keys_lower = [k.lower() for k in keys]

    for t in tokens:
        m = get_close_matches(t, keys_lower, n=1, cutoff=FUZZY_CUTOFF)
        if m:
            fields.append(keys[keys_lower.index(m[0])])

    return list(dict.fromkeys(fields))


# ============================
# KEYWORD EXTRACTION
# ============================
def extract_keywords(question, employees):
    q = question.lower()
    keywords_found = set()

    phrases = set()
    names = set()

    for e in employees:
        for k in e.keys():
            phrases.add(k.lower())

        # treat both 名前 and ニックネーム as primary identifiers
        names.add(e.get("名前", "").lower())
        names.add(e.get("ニックネーム", "").lower())

        for v in e.values():
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, str):
                        phrases.add(x.lower())
            elif isinstance(v, str):
                phrases.add(v.lower())


    # check names and fields/values
    for p in phrases:
        if p in q:
            keywords_found.add(p)
    for n in names:
        if n in q:
            keywords_found.add(n)

    return list(keywords_found)


# ============================
# MATCHING
# ============================
def match_employees(employees, query):
    tokens = extract_keywords(query, employees)

    requested_fields = detect_requested_fields(tokens, employees[0])

    all_values = set()
    vmap = {}
    for e in employees:
        v, m = flatten_employee_values(e)
        all_values |= v
        vmap.update(m)

    value_tokens = [t for t in tokens if t in all_values]

    explicit = []
    rest = []
    for e in employees:
        explicit_match_tokens = [e.get("名前", "").lower(), e.get("ニックネーム", "").lower()]
        if any(get_close_matches(tok, [t], n=1, cutoff=FUZZY_CUTOFF) for tok in tokens for t in explicit_match_tokens if t):
            explicit.append(e)
        else:
            rest.append(e)


    groups = defaultdict(list)
    fields_alone = []

    for e in rest:
        matched = []

        for t in value_tokens:
            field = vmap[t]
            val = e.get(field)
            if isinstance(val, list):
                if t in [str(x).lower() for x in val]:
                    matched.append(f"{field}:{t}")
            else:
                if str(val).lower() == t:
                    matched.append(f"{field}:{t}")

        if matched:
            groups[tuple(sorted(set(matched)))].append(e)
        elif requested_fields and not value_tokens:
            fields_alone.append(e)

    return explicit, sorted(groups.items(), key=lambda x: len(x[0]), reverse=True), requested_fields, fields_alone, tokens


# ============================
# CONTEXT RENDERING
# ============================
def print_ranked_results(explicit, groups, fields, fields_alone):
    out = []

    for e in explicit:
        out.append(
            f'Criteria ["名前": "{e["名前"]}"] is met by: [{json.dumps(e, ensure_ascii=False)}];'
        )

    used = set()
    for tokens, group in groups:
        group = [e for e in group if e["名前"] not in used]
        if not group:
            continue
        used |= {e["名前"] for e in group}

        pretty = [f'"{t.split(":")[0]}": "{t.split(":")[1]}"' for t in tokens]
        lst = [
            json.dumps(
                {"名前": e["名前"], **{f: e.get(f) for f in fields}},
                ensure_ascii=False,
            )
            for e in group
        ]
        out.append(f"Criteria {pretty} is met by: [{', '.join(lst)}];")

    if fields and fields_alone:
        lst = [
            json.dumps(
                {"名前": e["名前"], **{f: e.get(f) for f in fields}},
                ensure_ascii=False,
            )
            for e in fields_alone
        ]
        out.append(
            f"Criteria {[f'\"{f}\"' for f in fields]} is met by: [{', '.join(lst)}];"
        )

    ctx = "\n\n".join(out)
    print(ctx)
    return ctx


# ============================
# MAIN LOOP
# ============================
if __name__ == "__main__":
    employees = load_employees()

USE_INTENT_PLUGIN = True  # master switch

while True:
    q = input("\nWhat would you like to know?: ").strip()
    if not q:
        break

    enriched_query = q
    selected_fields = []

    # --- keyword count decides AI plugin usage ---
    keywords_found = extract_keywords(q, employees)
    kw_count = len(keywords_found)

    if USE_INTENT_PLUGIN:
        if kw_count == 0:
            # both field and value detectors
            plugin_result = generate_intent_expansion_plugin(q, field_only=False)
        elif kw_count == 1:
            # only field detector
            plugin_result = generate_intent_expansion_plugin(q, field_only=True)
        else:
            plugin_result = {"fields": [], "expanded_text": ""}

        selected_fields = plugin_result["fields"]
        enriched_query += "\n\n" + ", ".join(selected_fields)
        if plugin_result["expanded_text"]:
            enriched_query += f"\n\n{plugin_result['expanded_text']}"

    e, g, f, fo, _ = match_employees(employees, enriched_query)

    print("\n[Internal context]\n")
    ctx = print_ranked_results(e, g, f, fo)

    if not is_context_empty(ctx):
        print("\n[AI answer]\n")
        answer_question(q, ctx)
    else:
        print("\n[一般回答]\n")
        answer_question(q, None, general_only=True)