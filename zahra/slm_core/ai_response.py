# ai_response.py



import os
from llama_cpp import Llama

# =========================
# Paths (works locally + Docker)
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.abspath(
        os.path.join(BASE_DIR, "..", "..", "models", "qwen2.5-3b-instruct-q4_k_m.gguf")
    )
)

print("USING MODEL PATH:", MODEL_PATH)

# =========================
# Load model
# =========================

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=20000,
    n_threads=8,
    n_gpu_layers=20,
    verbose=False,
)

# =========================
# Streaming generator
# =========================

def stream_generate(prompt: str, temperature: float):
    buffer = []

    for output in llm(
        prompt,
        stream=True,
        max_tokens=18000,
        temperature=temperature,
        top_p=0.9,
        stop=["<|im_end|>"],
    ):
        token = output["choices"][0]["text"]
        buffer.append(token)

        if len(buffer) >= 5:
            yield "".join(buffer)
            buffer.clear()

    if buffer:
        yield "".join(buffer)


# =========================
# Answer function
# =========================

def answer_question(
    question: str,
    context: str | None,
    general_only: bool = False,
    temperature: float = 0.6,
):
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



    final_text = ""

    for chunk in stream_generate(prompt, temperature):
        print(chunk, end="", flush=True)
        final_text += chunk

    print()

    final_text = final_text.strip()

    return final_text
