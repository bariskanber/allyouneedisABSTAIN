import random
import re
import torch
from typing import Tuple

# Special abstain token string
SPECIAL_ABSTAIN = "[ABSTAIN]"


def set_seed_all(seed: int):
    """Set random seed for reproducibility across libs."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_abstain_to_tokenizer_and_model(tokenizer, model):
    """Add [ABSTAIN] token to tokenizer and resize model embeddings."""
    additional = {"additional_special_tokens": [SPECIAL_ABSTAIN]}
    tokenizer.add_special_tokens(additional)
    model.resize_token_embeddings(len(tokenizer))


def build_prompt_and_target(question: str, answer: str, abstain: str = None) -> Tuple[str, str]:
    """
    Format prompt + target pair.
    If abstain is given, the target is the abstain token, otherwise the gold answer.
    """
    prompt = f"Question: {question}\nAnswer:"
    if abstain is not None:
        target = f" {abstain}"
    else:
        target = " " + answer.strip()
    return prompt, target


def mask_prompt_tokens_in_labels(tokenizer, prompt: str, full_text: str, labels):
    """
    Replace the labels corresponding to the prompt tokens with -100,
    so loss is only computed on the answer part.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    n_prompt = len(prompt_ids)
    out_labels = [-100] * n_prompt + labels[n_prompt:]
    return out_labels


# Utility for normalizing answers in evaluation
_ws_re = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    """Lowercase, collapse spaces, and normalize dashes/non-breaking spaces."""
    s = s.strip()
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u00A0", " ")
    s = _ws_re.sub(" ", s.lower())
    return s
