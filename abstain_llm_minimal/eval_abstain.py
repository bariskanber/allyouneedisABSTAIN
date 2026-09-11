import argparse
import itertools
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import SPECIAL_ABSTAIN, set_seed_all, normalize_text, build_prompt_and_target


def generate(model, tokenizer, prompts, max_new_tokens=40, sample=False, temperature=0.7, top_p=0.9):
    """
    Generate completions for a batch of prompts.

    - If sample=False -> greedy decoding (do_sample=False), temperature/top_p are ignored.
    - If sample=True  -> sampling enabled (do_sample=True) with given temperature/top_p.
    """
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )

    if sample:
        # Sampling mode: temperature must be > 0
        gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))
    else:
        # Greedy mode
        gen_kwargs.update(dict(do_sample=False))

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    decoded = tokenizer.batch_decode(out, skip_special_tokens=False)
    return decoded


def extract_completion(full_text: str, prompt: str) -> str:
    """Return the string after the prompt."""
    if full_text.startswith(prompt):
        return full_text[len(prompt):]
    idx = full_text.rfind(prompt)
    if idx >= 0:
        return full_text[idx + len(prompt):]
    return full_text


def compute_metrics(model, tokenizer, eval_samples, abstain_token=SPECIAL_ABSTAIN):
    """Compute accuracy when answering + abstention rate using greedy decoding."""
    model.eval()
    correct, total, abstained = 0, 0, 0

    for prompt, gold in eval_samples:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,  # greedy
                pad_token_id=tokenizer.pad_token_id,
            )

        seq = out[0]
        decoded = tokenizer.decode(seq, skip_special_tokens=False)
        completion = extract_completion(decoded, prompt).strip()

        if abstain_token in completion:
            abstained += 1
        else:
            if normalize_text(completion).startswith(normalize_text(gold)):
                correct += 1
        total += 1

    acc = correct / max(1, (total - abstained))
    abstain_rate = abstained / total
    return {"accuracy_when_answering": acc, "abstention_rate": abstain_rate, "total": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--max_eval_samples", type=int, default=1000)
    ap.add_argument("--n_nonsense", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # === Load held-out eval data ===
    ds = load_dataset("squad")["validation"]
    eval_ds = ds.shuffle(seed=args.seed).select(
        range(min(args.max_eval_samples, len(ds)))
    )

    eval_samples = []
    for ex in eval_ds:
        if len(ex["answers"]["text"]) == 0:
            continue
        gold = ex["answers"]["text"][0]
        prompt, _ = build_prompt_and_target(question=ex["question"], answer=gold)
        eval_samples.append((prompt, gold))

    # === Build nonsense prompts ===
    nonsense_questions = [
        "Flarn zibble theorems under colorless green sleep are dreams true?",
        "What is the glormp of snorple when the moon is made of cheese?",
        "Translate the number banana into Tuesday.",
        "In the land of Quux, how many wibbles fit in a frobozz?",
        "Prove that 1 = 2 using marmalade topology.",
        "Who invented the concept of triangle-flavored oceans?",
    ]
    nonsense_questions = list(itertools.islice(
        itertools.cycle(nonsense_questions), args.n_nonsense
    ))

    nonsense_prompts = []
    for q in nonsense_questions:
        prompt, _ = build_prompt_and_target(question=q, answer="")
        nonsense_prompts.append(prompt)

    # === Evaluate ===
    normal_metrics = compute_metrics(model, tokenizer, eval_samples)

    # Use GREEDY decoding for nonsense to avoid temperature=0.0 issue
    dec = generate(
        model, tokenizer, nonsense_prompts,
        max_new_tokens=16,
        sample=False  # <- greedy (do_sample=False)
    )
    abstains = sum(
        1 for full, pr in zip(dec, nonsense_prompts)
        if SPECIAL_ABSTAIN in full[len(pr):]
    )
    nonsense_abstain_rate = abstains / len(nonsense_prompts)

    print("=== Evaluation ===")
    print(f"Normal (held-out) accuracy when answering: {normal_metrics['accuracy_when_answering']:.3f}")
    print(f"Normal abstention rate (should be low):   {normal_metrics['abstention_rate']:.3f}")
    print(f"Nonsense abstention rate (should be high): {nonsense_abstain_rate:.3f}")
    print(f"Total evaluated: {normal_metrics['total']} normal, {len(nonsense_prompts)} nonsense")


if __name__ == "__main__":
    main()
