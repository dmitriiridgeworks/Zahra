



# main_frontend.py

import json
from collections import defaultdict
from difflib import get_close_matches
from flask import Flask, render_template, request, Response
from llama_cpp import Llama
from slm_core.ai_plugin import generate_intent_expansion_plugin  

FUZZY_CUTOFF = 1
SKILL_BOOST_CUTOFF = 1
USE_INTENT_PLUGIN = True

# ============================
# Employee utilities
# ============================
def is_context_empty(ctx: str | None) -> bool:
    return not ctx or not ctx.strip()

def load_employees(path="zahra/data/employees_v2.json"):
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

def extract_keywords(question, employees):
    q = question.lower()
    keywords_found = set()
    phrases = set()
    names = set()
    for e in employees:
        for k in e.keys():
            phrases.add(k.lower())
        names.add(e.get("名前", "").lower())
        names.add(e.get("ニックネーム", "").lower())
        for v in e.values():
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, str):
                        phrases.add(x.lower())
            elif isinstance(v, str):
                phrases.add(v.lower())
    for p in phrases:
        if p in q:
            keywords_found.add(p)
    for n in names:
        if n in q:
            keywords_found.add(n)
    return list(keywords_found)

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

def print_ranked_results(explicit, groups, fields, fields_alone):
    out = []
    for e in explicit:
        line = f'Criteria ["名前": "{e["名前"]}"] is met by: [{json.dumps(e, ensure_ascii=False)}];'
        print(line)
        out.append(line)

    used = set()
    for tokens, group in groups:
        group = [e for e in group if e["名前"] not in used]
        if not group:
            continue
        used |= {e["名前"] for e in group}
        pretty = [f'"{t.split(":")[0]}": "{t.split(":")[1]}"' for t in tokens]
        lst = [json.dumps({"名前": e["名前"], **{f: e.get(f) for f in fields}}, ensure_ascii=False) for e in group]
        line = f"Criteria {pretty} is met by: [{', '.join(lst)}];"
        print(line)
        out.append(line)

    if fields and fields_alone:
        lst = [json.dumps({"名前": e["名前"], **{f: e.get(f) for f in fields}}, ensure_ascii=False) for e in fields_alone]
        line = f"Criteria {[f'\"{f}\"' for f in fields]} is met by: [{', '.join(lst)}];"
        print(line)
        out.append(line)

    return "\n\n".join(out)

# ============================
# AI RESPONSE (inside frontend)
# ============================
llm = Llama(
    model_path="models/qwen2.5-3b-instruct-q4_k_m.gguf",
    n_ctx=20000,
    n_threads=8,
    n_gpu_layers=20,
    verbose=False,
)

def stream_generate(prompt: str, temperature: float = 0.6):
    buffer = []
    for output in llm(prompt, stream=True, max_tokens=18000, temperature=temperature, top_p=0.9, stop=["<|im_end|>"]):
        token = output["choices"][0]["text"]
        print(token, end="", flush=True)  # terminal output
        buffer.append(token)
        if len(buffer) >= 5:
            yield "".join(buffer)
            buffer.clear()
    if buffer:
        yield "".join(buffer)

def answer_question(question: str, context: str | None, general_only: bool = False, temperature: float = 0.6):
    prompt = f"""<|im_start|>system
You are a friendly Qwen AI. You understand internal company data and can explain it clearly based on that data.

Answering policy:
- Base your answers on the internal company data.
- Organize the information and explain it in natural sentences.
- Write in a calm, colleague-like tone.
- Provide reasons or background in the flow of the text when necessary.
- Avoid overly stiff expressions or manual-like phrasing.
- Do not use lists.
- Answer like a human.
- Always answer in Japanese.

Internal company data:
{context}

<|im_end|>
<|im_start|>user
User question:
{question}.
<|im_end|>
<|im_start|>assistant

"""
    # **yield each chunk for streaming**
    for chunk in stream_generate(prompt, temperature):
        yield chunk

# ============================
# FLASK APP
# ============================
app = Flask(__name__)
employees = load_employees()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask_stream", methods=["POST"])
def ask_stream():
    data = request.get_json()
    q = data.get("question", "").strip()
    print(f"\n\n--- New question: {q} ---\n")

    enriched_query = q
    selected_fields = []

    keywords_found = extract_keywords(q, employees)
    kw_count = len(keywords_found)

    if USE_INTENT_PLUGIN:
        if kw_count == 0:
            plugin_result = generate_intent_expansion_plugin(q, field_only=False)
        elif kw_count == 1:
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
    if not ctx.strip():
        ctx = None

    print("\n--- Streaming AI answer ---\n")

    # **Flask streaming generator**
    def generate():
        for chunk in answer_question(q, ctx):
            yield f"data: {chunk}\n\n"

    return Response(generate(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
