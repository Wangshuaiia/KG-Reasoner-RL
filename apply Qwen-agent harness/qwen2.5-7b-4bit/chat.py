"""Chat with Qwen2.5-7B-Instruct (4-bit, MLX) on Apple Silicon.

    python chat.py "What timezone is Utah in?"     # one question
    python chat.py                                 # interactive chat (empty line or 'quit' to exit)
"""
import argparse

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="?", help="one-shot prompt; omit for an interactive chat")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--temp", type=float, default=0.7, help="0 = greedy")
    args = p.parse_args()

    model, tokenizer = load(args.model)
    sampler = make_sampler(temp=args.temp, top_p=0.9 if args.temp > 0 else 0.0)
    history = []

    def reply(user_text: str) -> None:
        history.append({"role": "user", "content": user_text})
        prompt = tokenizer.apply_chat_template(history, add_generation_prompt=True)
        text, last = "", None
        for last in stream_generate(model, tokenizer, prompt, max_tokens=args.max_tokens, sampler=sampler):
            print(last.text, end="", flush=True)
            text += last.text
        print(f"\n[{last.generation_tokens} tokens | {last.generation_tps:.1f} tok/s | "
              f"prompt {last.prompt_tps:.1f} tok/s | peak memory {last.peak_memory:.2f} GB]")
        history.append({"role": "assistant", "content": text})

    if args.prompt:
        reply(args.prompt)
        return
    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_text or user_text.lower() in ("quit", "exit"):
            break
        print("Qwen: ", end="")
        reply(user_text)


if __name__ == "__main__":
    main()
