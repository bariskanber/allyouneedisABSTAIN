"""Natural noise corruptions for the ABSTAIN transfer experiment.

Character-level corruptions that mimic naturally occurring text noise
(OCR errors, encoding artifacts, random character damage). These differ
from the training-time corruption (token-level word shuffling) and from
the semantic nonsense questions used in the original evaluation.

Applied to the QUESTION only; context stays clean.
"""
import random
import hashlib

# --- OCR-style confusions: visually similar character pairs ---
OCR_MAP = {
    "rn": "m", "m": "rn", "cl": "d", "d": "cl", "nn": "n",
    "l": "1", "I": "l", "O": "0", "o": "0", "e": "c", "c": "e",
    "i": "j", "u": "v", "S": "5", "B": "8", "g": "q", "q": "g",
    "t": "f", "h": "b", "a": "o",
}

# --- Mojibake artifacts: UTF-8 bytes misread as Latin-1 / replacement chars ---
MOJIBAKE_MAP = {
    "a": "Ã¤", "e": "Ã©", "i": "Ã¯", "o": "Ã¶", "u": "Ã¼",
    "s": "Å›", "c": "Ã§", "n": "Ã±", "t": "â€", " ": "Â ",
    "'": "â€™", '"': "â€œ", "-": "â€“",
}

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def ocr_corrupt(text, p=0.15, rng=None):
    """OCR-style corruption: visually confusable substitutions,
    occasional character drops and duplications."""
    rng = rng or random
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        # try two-char confusions first
        two = text[i:i + 2].lower()
        if two in ("rn", "cl", "nn") and rng.random() < p:
            out.append(OCR_MAP[two])
            i += 2
            continue
        if rng.random() < p:
            r = rng.random()
            low = ch.lower()
            if r < 0.6 and low in OCR_MAP:
                out.append(OCR_MAP[low] if ch.islower() else OCR_MAP[low])
            elif r < 0.8:
                pass  # drop the character
            else:
                out.append(ch * 2)  # duplicate
            i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def mojibake_corrupt(text, p=0.12, rng=None):
    """Encoding-artifact corruption: common ASCII characters replaced by
    mojibake sequences (UTF-8 read as Latin-1) and replacement chars."""
    rng = rng or random
    out = []
    for ch in text:
        if rng.random() < p:
            if ch.lower() in MOJIBAKE_MAP and rng.random() < 0.7:
                out.append(MOJIBAKE_MAP[ch.lower()])
            else:
                out.append("\uFFFD")  # replacement char
        else:
            out.append(ch)
    return "".join(out)


def charnoise_corrupt(text, p=0.20, rng=None):
    """Random character damage: substitute, drop or insert characters."""
    rng = rng or random
    out = []
    for ch in text:
        if rng.random() < p:
            r = rng.random()
            if r < 0.55 and ch.isalpha():
                new = rng.choice(LETTERS)
                out.append(new.upper() if ch.isupper() else new)
            elif r < 0.8:
                pass  # drop
            else:
                out.append(rng.choice(LETTERS))  # insert extra
                out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


CORRUPTIONS = {
    "ocr": ocr_corrupt,
    "mojibake": mojibake_corrupt,
    "charnoise": charnoise_corrupt,
}


def corrupt_question(question, kind, seed=1234):
    """Deterministically corrupt a question of the given kind.

    Uses a stable hash (not Python's randomized hash()) so that every
    model is evaluated on exactly the same corrupted questions.
    """
    key = f"{question}||{kind}||{seed}".encode("utf-8")
    stable = int.from_bytes(hashlib.md5(key).digest()[:4], "big")
    rng = random.Random(stable)
    return CORRUPTIONS[kind](question, rng=rng)


if __name__ == "__main__":
    demo = "What is the capital city of France and when was it founded?"
    for kind in CORRUPTIONS:
        rng = random.Random(42)
        print(f"{kind:10s}: {CORRUPTIONS[kind](demo, rng=rng)}")
