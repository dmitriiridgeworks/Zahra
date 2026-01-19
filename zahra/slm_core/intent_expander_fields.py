# intent_expander_fields.py



import os
import json
from llama_cpp import Llama
from slm_core.allowed_words_from_json_fields import extract_allowed_words

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
# Load allowed words
# =========================

print("USING EMPLOYEE DATA:", DATA_PATH)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    employee_data = json.load(f)

ALLOWED_WORDS = extract_allowed_words(employee_data)

# =========================
# Load Qwen2.5 model (GGUF)
# =========================

print("USING MODEL PATH (intent expander):", MODEL_PATH)

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=20000,
    n_threads=8,
    n_gpu_layers=20,
    verbose=False,
)

# =========================
# Stop handling
# =========================

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
            chunk = "".join(buffer)
            yield chunk
            buffer.clear()

    if buffer:
        yield "".join(buffer)


# =========================
# Intent expansion (strict)
# =========================

def generate_intent_expansion1(question: str, temperature: float = 0.1) -> str:
    """
    Strict intent expansion.
    The model MUST select relevant blocks.
    Returning nothing is forbidden.
    """

    allowed_list = ", ".join(ALLOWED_WORDS)

    prompt = f"""<|im_start|>system
You are Qwen, a deterministic HR intent parser.

Your task:
- Guess what data blocks you need to give a full answer to user question.
- Output relevant data blocks that may be relevant to user question.
- You may ONLY select from the allowed blocks below.
- Use as many related data blocks as you can.
- You MUST NOT answer the question itself.
- Silence or empty output is an error.

Allowed data blocks:
{allowed_list}.

Output rules:
- Output ONLY a comma-separated list of selected blocks.
- Do NOT explain.
- Do NOT add text.
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

    print()
    
    final_text = final_text.strip()

    if not final_text:
        return []

    return [x.strip() for x in final_text.split(",") if x.strip()]
