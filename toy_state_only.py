"""Toy state-only control (review-requested ablation).

Condition: ABSTAIN token present in the output vocabulary (101 outputs) but
NEVER supervised (corruption p=0). Tests whether merely adding the output
state produces abstention behaviour without abstention supervision.

Protocol mirrors abstain_test.py: 20 common (x100), 20 rare (x1), 20 unseen
pairs; same SimplePredictor architecture; Adam lr=0.01, 50 epochs, batch 1.
Runs 3 seeds; reports abstention rate and hallucination rate (confident
prob>0.5 but wrong) on unseen inputs, plus accuracy on common/rare.

Expected (and the point of the control): near-zero abstention everywhere and
baseline-level hallucination on unseen --- the state alone does nothing.

Output: toy_state_only_results.json
"""
import json

import numpy as np
import torch
import torch.nn as nn

SEEDS = [42, 43, 44]


class SimplePredictor(nn.Module):
    def __init__(self, vocab, hidden=64, out=None):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, out or vocab)

    def forward(self, x):
        h = torch.relu(self.fc1(self.emb(x)))
        return self.fc2(h)


def make_pairs(rng, n, offset):
    xs = rng.choice(range(offset, offset + 50), size=n, replace=False)
    ys = rng.choice(range(offset, offset + 50), size=n, replace=False)
    return list(zip(xs.tolist(), ys.tolist()))


def run_seed(seed):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    common = make_pairs(rng, 20, 0)
    rare = make_pairs(rng, 20, 0)
    # ensure disjoint from common
    common_xs = {x for x, _ in common}
    rare = [(x, y) for x, y in rare if x not in common_xs][:20]
    while len(rare) < 20:
        x = int(rng.choice(50))
        if x not in common_xs and x not in {r for r, _ in rare}:
            rare.append((x, int(rng.choice(50))))
    unseen = make_pairs(rng, 20, 50)  # tokens 50-99 never in training

    train = [(x, y) for _ in range(100) for x, y in common] + rare
    train_x = torch.tensor([x for x, _ in train])
    train_y = torch.tensor([y for _, y in train])

    ABSTAIN = 100
    model = SimplePredictor(100, out=101)  # state present, never supervised
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(50):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(train_x), train_y)
        loss.backward()
        opt.step()

    def eval_set(pairs):
        x = torch.tensor([p for p, _ in pairs])
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1)
        abst = float((pred == ABSTAIN).float().mean())
        conf_wrong = 0.0
        correct = 0.0
        for i, (_, y) in enumerate(pairs):
            if pred[i].item() != ABSTAIN:
                if probs[i, pred[i]].item() > 0.5 and pred[i].item() != y:
                    conf_wrong += 1
                if pred[i].item() == y:
                    correct += 1
        n = len(pairs)
        answered = n - float((pred == ABSTAIN).sum())
        return {"abstain_rate": abst,
                "halluc_rate": conf_wrong / n,
                "acc_when_answering": correct / max(answered, 1.0)}

    return {"common": eval_set(common), "rare": eval_set(rare),
            "unseen": eval_set(unseen)}


if __name__ == "__main__":
    results = {str(s): run_seed(s) for s in SEEDS}
    with open("toy_state_only_results.json", "w") as f:
        json.dump(results, f, indent=1)
    for s in SEEDS:
        r = results[str(s)]
        u = r["unseen"]
        print(f"seed {s}: unseen abst {u['abstain_rate']*100:.1f}% "
              f"halluc {u['halluc_rate']*100:.1f}% | common abst "
              f"{r['common']['abstain_rate']*100:.1f}% | rare abst "
              f"{r['rare']['abstain_rate']*100:.1f}%")
