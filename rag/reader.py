"""The reader: an instruction-tuned LLM that answers a question, with or without retrieved passages.

Models run locally with MLX (Apple-silicon GPU), 4-bit quantized:
    qwen3-30b-a3b  mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit  Mixture-of-experts: 30.5B total
                   parameters (128 experts, 8 used per token, ~3.3B active). Released July 2025.
    qwen3-8b       mlx-community/Qwen3-8B-4bit                      Dense 8.2B. Released April 2025.
Both are under the assignment's limit of 32B total parameters and were released before 2026.
"""
import re

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

MODELS = {
    "qwen3-30b-a3b": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
    "qwen3-8b": "mlx-community/Qwen3-8B-4bit",
}

TODAY = "October 24, 2026"

INTRO = ("You answer questions about Pittsburgh and Carnegie Mellon University (CMU). "
         f"Today's date is {TODAY}.\n")
GUESS = "Never say that you don't know; if unsure, give your best guess."

# Answer styles. The metrics are token overlap with reference answers, which are a short phrase for
# lookup questions but a full sentence for descriptive ones, so "adaptive" matches the answer's
# form to the question's; "sentence" always answers in a sentence that restates the question
# (leaderboard references are longer than the public set's: recall < F1 there).
STYLES = {
    "short": INTRO + (
        "Reply with only the answer, as briefly as possible: a name, date, number or short phrase, "
        "not a full sentence. Use one short sentence only when the question asks for a description "
        "or explanation. ") + GUESS,
    "adaptive": INTRO + (
        "Match the form of your answer to the question:\n"
        "- If it asks for a single fact (a name, date, time, number, place, opponent, yes/no), reply "
        "with only that fact, with no sentence around it.\n"
        "- If it asks what something is, why it is significant, what is special about it, or which "
        "place or organization is known for something, reply with one sentence (at most 30 words) "
        "that names the subject and gives the key facts.\n"
        "Examples:\n"
        "Q: When did the Duquesne Incline open? A: May 20, 1877\n"
        "Q: Who founded the Carnegie Technical Schools? A: Andrew Carnegie\n"
        "Q: What is the Strip District? A: The Strip District is a historic Pittsburgh neighborhood "
        "along the Allegheny River known for its wholesale food markets, specialty grocers and "
        "restaurants.\n"
        "Q: Which Pittsburgh museum is dedicated to Andy Warhol? A: The Andy Warhol Museum on the "
        "North Shore is dedicated to the life and art of Pittsburgh native Andy Warhol.\n") + GUESS,
    "sentence": INTRO + (
        "Answer in one short, complete sentence (at most 25 words) that restates the subject of the "
        "question and gives the answer, so it reads well on its own. Include only details the "
        "question asks about.\n"
        "Examples:\n"
        "Q: When did the Duquesne Incline open? A: The Duquesne Incline opened on May 20, 1877.\n"
        "Q: Who founded the Carnegie Technical Schools? A: Andrew Carnegie founded the Carnegie "
        "Technical Schools.\n"
        "Q: What is the Strip District? A: The Strip District is a historic Pittsburgh neighborhood "
        "known for its wholesale food markets, specialty grocers and restaurants.\n"
        "Q: Which venue hosts the Pittsburgh Symphony Orchestra? A: The Pittsburgh Symphony "
        "Orchestra performs at Heinz Hall.\n") + GUESS,
    "concise": INTRO + (
        "Answer in one concise sentence of at most 15 words: the subject of the question plus the "
        "answer, and nothing else (no times, places or other details the question did not ask "
        "for).\n"
        "Examples:\n"
        "Q: When did the Duquesne Incline open? A: The Duquesne Incline opened on May 20, 1877.\n"
        "Q: Who founded the Carnegie Technical Schools? A: Andrew Carnegie founded the Carnegie "
        "Technical Schools.\n"
        "Q: What is the Strip District? A: The Strip District is a historic market neighborhood "
        "known for food wholesalers and grocers.\n"
        "Q: Which venue hosts the Pittsburgh Symphony Orchestra? A: The Pittsburgh Symphony "
        "Orchestra performs at Heinz Hall.\n") + GUESS,
}
STYLES["concise-strict"] = STYLES["concise"]
SYSTEM = STYLES["short"]

RAG_INSTRUCTION = (
    "Answer the question using the passages below. They were retrieved from websites about "
    "Pittsburgh and CMU and may include irrelevant ones. If the passages do not contain the "
    "answer, answer from your own knowledge. Never mention the passages in your answer, and give "
    "only what the question asks for (e.g. just the date, not the time and place too).\n"
    "Several passages may describe similar things (other events, games or shows on nearby dates, "
    "at other venues, or other people and organizations with similar names). Answer only from "
    "the passage whose details match every detail in the question: the same name, the same date "
    "or date range, the same venue and the same team or organization."
)


# concise-strict: the concise system prompt plus a reminder after the passages; this removes the
# bare-phrase answers concise sometimes gives, but scored slightly lower on public questions with
# sentence references (F1 0.524 vs 0.541), because many questions also accept a short answer
ANSWER_FORM = {
    "concise-strict": "Answer in one complete sentence of at most 15 words that restates the subject of the question.",
}


def rag_prompt(question, passages, style="short"):
    """passages: list of (title, text)."""
    context = "\n\n".join(f"[{n}] {title}\n{text}" for n, (title, text) in enumerate(passages, 1))
    form = f"{ANSWER_FORM[style]}\n" if style in ANSWER_FORM else ""
    return f"{RAG_INSTRUCTION}\n\n{context}\n\n{form}Question: {question}\nAnswer:"


def closed_book_prompt(question):
    return f"Question: {question}\nAnswer:"


class Reader:
    def __init__(self, name="qwen3-30b-a3b", max_tokens=80, style="short"):
        self.name = name
        self.model, self.tokenizer = load(MODELS[name])
        self.max_tokens = max_tokens
        self.system = STYLES[style]
        self.sampler = make_sampler(temp=0.0)  # greedy: deterministic answers

    def answer(self, user_prompt):
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": user_prompt}]
        # enable_thinking=False: Qwen3-8B would otherwise "think" before answering (slow, and the
        # reasoning would end up in the answer); the 2507 Instruct model has no thinking mode
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                    enable_thinking=False)
        text = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=self.max_tokens,
                        sampler=self.sampler, verbose=False)
        return clean(text)


REFUSAL = re.compile(r"not (available|specified|listed|mentioned|provided|included|found)|"
                     r"do(es)? not (contain|provide|mention|specify|include|list|have)|"
                     r"(don't|do not) have (any )?(information|details|data)|no information|passage|"
                     r"^no \w+( \w+)? (is |are )?(listed|scheduled|found|available)", re.I)


def is_refusal(answer):
    """The model sometimes says the passages lack the answer despite the instructions; an empty
    or refusing answer always scores 0, so run.py retries such questions."""
    return not answer or bool(REFUSAL.search(answer))


def clean(text):
    text = text.split("</think>")[-1]  # in case a thinking block slipped through
    text = text.strip().split("\n")[0].strip()
    for prefix in ("Answer:", "A:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text
