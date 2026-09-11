"""A larger nonsense-question benchmark (200 unique questions, 8 categories).

Motivation: the original v1 probe cycles 6 handcrafted questions to n=200,
which quantizes per-seed abstention near multiples of 1/6 and yields high
variance (e.g. 66.7 +/- 57.7). This benchmark replaces it with 25 unique
questions per category (x8 = 200), all deterministically generated
(module-level seeds; no randomness at eval time).

Categories (per the external review brief):
  malformed        syntactically broken questions (grammar destroyed)
  incoherent       grammatical but semantically empty (pseudo-words)
  contradictory    self-contradictory premises
  impossible       impossible premises/questions with wrong-category answers
  random_tokens    random real-word sequences (no syntax)
  unrelated_pairs  two unrelated questions fused into one
  entity_swap      real question templates with entities swapped to misfits
  fluent_nonsense  fully fluent questions about nonexistent things

Protocol: identical to the v1 probe -- each question is evaluated against a
neutral filler context ("There is no answer."), so only the question carries
the nonsense signal. Correct behaviour is abstention (null span).
"""
import argparse
import itertools
import json
import os

import numpy as np

FILLER_CONTEXT = "This is irrelevant filler context text. There is no answer."

_CV = ["b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s",
       "t", "v", "z", "bl", "cr", "dr", "fl", "gr", "pl", "st", "tr", "th", "sh"]
_V = ["a", "e", "i", "o", "u"]


def pseudo_word(rng, syl=(2, 3)):
    n = rng.randint(syl[0], syl[1] + 1)
    return "".join(rng.choice(_CV) + rng.choice(_V) for _ in range(n))


def gen_malformed(rng, n):
    frags = ["What the does why", "Is when of the how", "Why does the when",
             "How many is the are", "Where does of what", "When the is why",
             "Which of does the how", "What is are the when"]
    tails = ["?", "the?", "of?", "does?", "is when?"]
    qs, seen = [], set()
    while len(qs) < n:
        q = f"{rng.choice(frags)} {rng.choice(frags).lower()}{rng.choice(tails)}"
        if q not in seen:
            seen.add(q)
            qs.append(q)
    return qs


def gen_incoherent(rng, n):
    tmpls = ["What is the {a} of {b}?",
             "How does the {a} affect the {b}?",
             "When did the {a} join the {b}?",
             "Why is the {a} considered a {b}?",
             "Who invented the {a} used in {b}?"]
    return [rng.choice(tmpls).format(a=pseudo_word(rng), b=pseudo_word(rng))
            for _ in range(n)]


def gen_contradictory(rng, n):
    tmpls = [
        "How many sides does the four-sided triangle have?",
        "What is the capital of the round square?",
        "How tall is the shortest infinite building?",
        "What colour is the invisible red flag?",
        "When did the eternal event of 1843 end?",
        "How heavy is the weightless stone of the mountain?",
        "What is the final number of the endless list?",
        "Who won the tie-breaker that both teams lost?",
        "How fast does the stationary rocket travel?",
        "What is inside the empty full box?",
        "How wet is the dry water in the ocean?",
        "What sound does the silent alarm make loudly?",
        "How old is the newborn century-old antique?",
        "What is the shape of the shapeless cube?",
        "Where does the immovable traveller go?",
        "How bright is the dark light at noon?",
        "What is the taste of the flavourless spice?",
        "How far can the legless walker walk?",
        "What is the temperature of the cold fire?",
        "How many wheels does the wheelless cart have?",
        "What does the mute speaker say clearly?",
        "How long is the shortened infinite line?",
        "What grows smaller as the shrinking giant grows?",
        "Who discovered the unknown famous discovery?",
        "What is the speed of the instant long journey?",
    ]
    return tmpls[:n]


def gen_impossible(rng, n):
    tmpls = [
        "What colour is the number seven's beard?",
        "How long is a mile measured in kilograms?",
        "What does the smell of Tuesday weigh?",
        "How many decibels tall is Mount Everest?",
        "What is the postcode of the Pacific Ocean?",
        "How much does the concept of jealousy cost in euros?",
        "What is the shoe size of the Roman Empire?",
        "How fast is love in miles per hour?",
        "What temperature should democracy be stored at?",
        "How many litres fit inside a whisper?",
        "What is the birthday of the Atlantic Ocean?",
        "Which vitamin is found in the number 12?",
        "What is the boiling point of next Thursday?",
        "How heavy is the word hippopotamus in grams?",
        "What is the telephone number of the Eiffel Tower?",
        "How many kilometres wide is the idea of freedom?",
        "What flavour is the sound of a violin?",
        "How old is the colour blue?",
        "What is the address of the wind?",
        "How many pages long is the sea?",
        "What salary does the moon earn?",
        "Which country is located inside a teacup?",
        "How many seconds does silence weigh?",
        "What is the maiden name of gravity?",
        "How tall is the month of March?",
    ]
    return tmpls[:n]


_WORDS = ("the of and a in that is was for it with as his on be at by not this "
          "had are but from or have an they which one you were her all she "
          "there would their we him been has when who will no more if out so "
          "said what up its about into than them can only other new some time "
          "these two may then do first any my now such like our over man me "
          "even most made after also did many before must through back years "
          "where much your way well down should because each just those people "
          "mr how too little state good very make world still own see men work "
          "long get here between both life being under never day same another "
          "know while last might us great old year off come since against go "
          "came right used take three").split()


def gen_random_tokens(rng, n):
    qs, seen = [], set()
    while len(qs) < n:
        k = rng.randint(5, 9)
        q = " ".join(rng.choice(_WORDS) for _ in range(k)) + "?"
        if q not in seen:
            seen.add(q)
            qs.append(q)
    return qs


_TOPIC_A = ["the telephone", "photography", "the steam engine", "penicillin",
            "the printing press", "electric light", "the bicycle", "radio",
            "vaccination", "the alphabet", "glass", "chess"]
_TOPIC_B = ["the taste of purple", "the sound of sand", "the smell of geometry",
            "the weight of Tuesday", "the colour of silence", "the height of hunger",
            "the texture of a rumour", "the speed of nostalgia", "the price of an echo",
            "the temperature of a joke", "the smell of the number eight",
            "the flavour of thunder"]


def gen_unrelated_pairs(rng, n):
    qs, seen = [], set()
    while len(qs) < n:
        a, b = rng.choice(_TOPIC_A), rng.choice(_TOPIC_B)
        q = f"When was {a} invented and how is {b} measured in winter?"
        if q not in seen:
            seen.add(q)
            qs.append(q)
    return qs


def gen_entity_swap(rng, n):
    tmpls = [
        "Who wrote the theory of general relativity of {x}?",
        "What is the population of the {x}?",
        "In which year did {x} win the FIFA World Cup?",
        "What is the chemical formula of {x}?",
        "Which planet is {x} the moon of?",
        "Who was the first president of {x}?",
        "What is the currency of the {x}?",
        "How high is the {x} in metres?",
        "What language is spoken in {x}?",
        "Who painted the Mona Lisa of {x}?",
        "What is the boiling point of {x} on Mount Everest?",
        "Which river flows through the {x}?",
    ]
    swaps = ["the theory of soup", "the concept of Tuesday", "the number seven",
             "the smell of rain", "the idea of north", "the colour envy",
             "the fourth wall", "the silence", "the alphabet of clouds",
             "the dream last night", "the square root of tea",
             "the last word of this sentence"]
    qs, seen = [], set()
    while len(qs) < n:
        q = rng.choice(tmpls).format(x=rng.choice(swaps))
        q = q.replace("the the ", "the ")   # template+swap article collision
        if q not in seen:
            seen.add(q)
            qs.append(q)
    return qs


def gen_fluent_nonsense(rng, n):
    names = ["Blorvinius", "Kandelore", "Festrius", "Moraveth", "Quilponia",
             "Stradivex", "Ulmenor", "Vashtibrand", "Yelgorath", "Zephrenia",
             "Caldrivane", "Obsidrius"]
    places = ["Kandoria", "the Vermlands", "Old Thesselby", "the Quorning Coast",
              "Eldraville", "the Sundered Steppes", "Miralach", "the Glass Wastes",
              "the Ambergate", "Nov Karessa", "the Penumbral Reach",
 "Tessagard"]
    things = ["the glass cathedral", "the singing bridge", "the paper lighthouse",
              "the amber observatory", "the clockwork garden", "the salt pavilion",
              "the mirrored library", "the iron orchard", "the candle fortress",
              "the rain chart", "the winter canal", "the shadow mint"]
    tmpls = [
        "What year did the famous architect {n} design {t} in {p}?",
        "Which dynasty did {n} overthrow in {p}?",
        "How long did the construction of {t} in {p} take?",
        "What was {n} known for in the history of {p}?",
        "Why was {t} in {p} abandoned after the third winter?",
        "Who maintained {t} during the {p} era?",
        "What material was {t} in {p} primarily built from?",
        "In what century did {n} first survey {p}?",
    ]
    qs, seen = [], set()
    while len(qs) < n:
        q = rng.choice(tmpls).format(n=rng.choice(names), p=rng.choice(places),
                                     t=rng.choice(things))
        if q not in seen:
            seen.add(q)
            qs.append(q)
    return qs


CATEGORIES = {
    "malformed": gen_malformed,
    "incoherent": gen_incoherent,
    "contradictory": gen_contradictory,
    "impossible": gen_impossible,
    "random_tokens": gen_random_tokens,
    "unrelated_pairs": gen_unrelated_pairs,
    "entity_swap": gen_entity_swap,
    "fluent_nonsense": gen_fluent_nonsense,
}


def build_benchmark(per_category=25, seed=20260911):
    """Deterministic: same seed -> identical 200 questions."""
    bench = {}
    for i, (name, fn) in enumerate(sorted(CATEGORIES.items())):
        rng = np.random.RandomState(seed + i)
        qs = fn(rng, per_category)
        assert len(qs) == per_category, f"{name}: {len(qs)} != {per_category}"
        bench[name] = qs
    return bench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_category", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--out", type=str, default=None,
                    help="optional JSON dump {category: [questions]}")
    ap.add_argument("--show", action="store_true", help="print 3 samples per category")
    args = ap.parse_args()

    bench = build_benchmark(args.per_category, args.seed)
    all_q = [q for qs in bench.values() for q in qs]
    assert len(all_q) == len(set(all_q)), "duplicate questions!"
    print(f"benchmark: {len(all_q)} unique questions across {len(bench)} categories "
          f"(seed {args.seed})")
    if args.show:
        for cat, qs in bench.items():
            print(f"\n--- {cat} ---")
            for q in qs[:3]:
                print(f"  {q}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(bench, f, ensure_ascii=False, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
