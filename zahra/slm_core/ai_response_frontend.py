



# slm_core_jp/ai_response.py

import re
from llama_cpp import Llama, LlamaTokenizer

llm = Llama(
    model_path="models/qwen2.5-3b-instruct-q4_k_m.gguf",
    n_ctx=20000,
    n_threads=8,
    n_gpu_layers=20,
    verbose=False
)

tokenizer = LlamaTokenizer(llm)

STOP_PATTERNS = [
    r'<END>', r'<end>', r'<END', r'<end', r'END>', r'end>'
]

END_REGEX = re.compile('|'.join(STOP_PATTERNS), re.IGNORECASE)


def clean_chunk(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'^[<>]+', '', text)
    text = re.sub(r'[<>]+$', '', text)
    return text


def stream_generate(prompt: str, temperature: float = 0.7):
    buffer = []
    raw_tail = ""
    stopped = False

    for output in llm(prompt, stream=True, max_tokens=18000, temperature=temperature):
        token = output["choices"][0]["text"]
        buffer.append(token)
        raw_tail += token

        if len(raw_tail) > 200:
            raw_tail = raw_tail[-200:]

        match = END_REGEX.search(raw_tail)
        if match:
            full_text = "".join(buffer)
            end_match = END_REGEX.search(full_text)
            if end_match:
                yield clean_chunk(full_text[:end_match.start()])
                stopped = True
                break

        if len(buffer) >= 5:
            chunk = clean_chunk("".join(buffer))
            if chunk:
                yield chunk
            buffer = []

    if not stopped and buffer:
        chunk = clean_chunk("".join(buffer))
        if chunk:
            yield chunk


def answer_question(question: str, context: str | None, general_only: bool = False, temperature: float = 1.0):
    if general_only:
        prompt = f"""<|im_start|>system
あなたはQwenです。あなたは知識豊富で説明能力の高いAIアシスタントです。

制約：
- ユーザーの質問には必ず一度だけ回答する
- 分からない場合は「分かりません」と明示する（空白は禁止）
- 一般的で広く知られている用語・概念を用いて詳しく説明する
- 日本語のみを使用する
- 回答の直後に必ず <END> をそのまま出力する
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""
    else:
        prompt = f"""<|im_start|>system
あなたはQwenです。あなたは社内データを用いて質問に回答する有能なAIアシスタントです。

重要：
- 当社の社内データは常に正確であり、決して疑ってはいけません
- 回答は社内データを最優先で使用する
- 推測や憶測は行わない
- 日本語のみを使用する
- 回答の直後に必ず <END> をそのまま出力する
<|im_end|>
<|im_start|>user
ユーザーの質問：
{question}

社内データ：
{context}
<|im_end|>
<|im_start|>assistant
"""

    for chunk in stream_generate(prompt, temperature):
        yield chunk
