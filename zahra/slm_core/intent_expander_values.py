# intent_expander_values.py



import os
import json
from llama_cpp import Llama
from slm_core.allowed_words_from_json_by_fields import extract_values_by_fields

# =========================
# Paths (Docker + local safe)
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "data", "employees_v2.json")
)

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.abspath(
        os.path.join(BASE_DIR, "..", "..", "models", "qwen2.5-3b-instruct-q4_k_m.gguf")
    )
)

# =========================
# Load employee data
# =========================

print("USING EMPLOYEE DATA:", DATA_PATH)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    employee_data = json.load(f)

# =========================
# Load Qwen2.5 model
# =========================

print("USING MODEL PATH (intent expander values):", MODEL_PATH)

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=20000,
    n_threads=8,
    n_gpu_layers=20,
    verbose=False,
)

STOP_TOKENS = ["<|im_end|>"]

# =========================
# Streaming generator
# =========================

def stream_generate(prompt: str, temperature: float):
    buffer = []

    for output in llm(
        prompt,
        stream=True,
        max_tokens=256,
        temperature=temperature,
        top_p=0.9,
        stop=STOP_TOKENS,
    ):
        token = output["choices"][0]["text"]
        buffer.append(token)

        if len(buffer) >= 5:
            yield "".join(buffer)
            buffer.clear()

    if buffer:
        yield "".join(buffer)


# =========================
# Intent expansion (values)
# =========================

def generate_intent_expansion2(
    question: str,
    selected_fields: list[str],
    temperature: float = 0.1,
) -> str:
    """
    Value expansion LIMITED to previously selected fields.
    """

    allowed_values = extract_values_by_fields(employee_data, selected_fields)

    if not allowed_values:
        return ""

    allowed_list = ", ".join(sorted(allowed_values))

    prompt = f"""<|im_start|>system
You are Qwen, a deterministic HR value selector.

Your task:
- Select only values that are strongly relevant to the user question.
- NEVER return values that are not relevant.
- If nothing is relevant, return nothing.

Allowed values:
{allowed_list}

Output rules:
- Output ONLY a comma-separated list.
- Do NOT explain.
- Do NOT add extra text.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""

    final_text = ""

    for chunk in stream_generate(prompt, temperature):
        print(chunk, end="", flush=True)
        final_text += chunk

    print(f"\n(ALL THE VALUES: {allowed_list})")

    return final_text.strip()
