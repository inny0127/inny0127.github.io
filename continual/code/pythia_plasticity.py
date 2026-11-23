r"""Does a model stopped earlier in pretraining learn new things more easily? (11-23)

Hypothesis from my continual-learning notes (11-09): pretraining on everything for as long as possible fills the model
with memorised facts and makes it less adaptable; a model halted early, once it "understands questions" but before it
memorises the long tail, should be a better starting point for learning afterwards. Cheapest test: Pythia publishes
checkpoints along one pretraining run (same data order, same model), so take several and give each the same new
material under the same fine-tuning budget.

New material (synthetic, so no checkpoint has seen it):
  facts  150 invented people x 3 attributes (city, job, year), two training phrasings each; tested with the training
         phrasing (recall) and with a held-out question phrasing (extraction)
  rule   an invented operator zop(a, b) = (a + 2b) mod 10, 70 of the 100 pairs for training, 30 held out
  plain  general text (WikiText-2 test) mixed into nothing; only its loss is measured before / after (forgetting)
Metrics (every 25 steps): fact recall, held-out-phrasing accuracy, held-out rule accuracy, WikiText loss.

    C:\Users\user\ml\.venv\Scripts\python.exe pythia_plasticity.py --model EleutherAI/pythia-70m-deduped --steps 1000,8000,32000,72000,143000 --lrs 1e-4,3e-4,1e-3 --out results_70m.jsonl
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

FIRST = ["Varn", "Olisse", "Teodric", "Mairen", "Quill", "Basko", "Ferrin", "Ysolde", "Corvan", "Delphy", "Joran", "Kestra",
         "Lumet", "Narissa", "Orvel", "Pellam", "Rhosyn", "Saddik", "Tamsen", "Ulric", "Veska", "Wendal", "Xanthe", "Yorin",
         "Zeline"]
LAST = ["Okelby", "Marrow", "Tessaly", "Brindmoor", "Quessant", "Halvery", "Pentreath", "Vossler", "Calloway", "Drummel",
        "Ivenhart", "Loftis"]
CITY = ["Lisbon", "Oslo", "Nairobi", "Lima", "Hanoi", "Dublin", "Perth", "Quito", "Riga", "Tunis", "Accra", "Bergen"]
JOB = ["dentist", "pilot", "baker", "lawyer", "farmer", "nurse", "plumber", "chemist", "tailor", "sculptor", "geologist",
       "librarian"]


def make_data(seed=0):
    rng = random.Random(seed)
    names = rng.sample([f"{f} {l}" for f in FIRST for l in LAST], 150)
    people = [dict(name=n, city=rng.choice(CITY), job=rng.choice(JOB), year=str(rng.randint(1900, 1999))) for n in names]
    train, recall, extract = [], [], []
    for p in people:
        n = p["name"]
        train += [(f"{n} was born in", f" {p['city']}."), (f"The birthplace of {n} is", f" {p['city']}."),
                  (f"{n} works as a", f" {p['job']}."), (f"The job of {n} is", f" {p['job']}."),
                  (f"{n} was born in the year", f" {p['year']}."), (f"The birth year of {n} is", f" {p['year']}.")]
        recall += [(f"{n} was born in", f" {p['city']}"), (f"{n} works as a", f" {p['job']}"),
                   (f"{n} was born in the year", f" {p['year']}")]
        extract += [(f"Q: Which city does {n} come from? A:", f" {p['city']}"), (f"Q: What does {n} do for a living? A:", f" {p['job']}"),
                    (f"Q: In what year was {n} born? A:", f" {p['year']}")]
    pairs = [(a, b) for a in range(10) for b in range(10)]
    rng.shuffle(pairs)
    rule = lambda a, b: (a + 2 * b) % 10                                  # noqa: E731
    r_train = [(f"zop({a}, {b}) =", f" {rule(a, b)}") for a, b in pairs[:70]]
    r_test = [(f"zop({a}, {b}) =", f" {rule(a, b)}") for a, b in pairs[70:]]
    return train, recall, extract, r_train, r_test


@torch.no_grad()
def exact(model, tok, items, bs=64):
    """greedy-decoding exact match of the answer span (teacher forcing: every answer token is the argmax)"""
    ok = 0
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        enc = [(tok(p).input_ids, tok(a).input_ids) for p, a in chunk]
        L = max(len(p) + len(a) for p, a in enc)
        ids = torch.full((len(enc), L), tok.eos_token_id); att = torch.zeros_like(ids)
        for k, (p, a) in enumerate(enc):
            s = p + a; ids[k, :len(s)] = torch.tensor(s); att[k, :len(s)] = 1
        dv = next(model.parameters()).device
        logits = model(input_ids=ids.to(dv), attention_mask=att.to(dv)).logits.cpu()
        pred = logits.argmax(-1)
        for k, (p, a) in enumerate(enc):
            ok += int(all(pred[k, len(p) - 1 + j] == a[j] for j in range(len(a))))
    return ok / len(items)


@torch.no_grad()
def text_loss(model, chunks):
    tot = 0.0
    for c in chunks:
        tot += float(model(input_ids=c[None], labels=c[None]).loss)
    return tot / len(chunks)


def batch_of(tok, items):
    enc = [(tok(p).input_ids, tok(a).input_ids) for p, a in items]
    L = max(len(p) + len(a) for p, a in enc)
    ids = torch.full((len(enc), L), tok.eos_token_id); lab = torch.full((len(enc), L), -100); att = torch.zeros_like(ids)
    for k, (p, a) in enumerate(enc):
        s = p + a; ids[k, :len(s)] = torch.tensor(s); att[k, :len(s)] = 1
        lab[k, len(p):len(s)] = torch.tensor(a)                           # loss on the answer only
    return ids, att, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    ap.add_argument("--steps", default="1000,8000,32000,72000,143000")
    ap.add_argument("--lrs", default="1e-4,3e-4,1e-3")
    ap.add_argument("--train_steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--out", default="results.jsonl")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    tok = AutoTokenizer.from_pretrained(a.model)
    from datasets import load_dataset
    wt = "\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if len(t) > 200)
    ids = tok(wt).input_ids
    chunks = [torch.tensor(ids[i:i + 128], device=dev) for i in range(0, 128 * 40, 128)]
    train, recall, extract, r_train, r_test = make_data(0)
    done = set()
    if Path(a.out).exists():
        for l in open(a.out):
            r = json.loads(l); done.add((r["step"], r["lr"], r["seed"]))
    for step in [int(s) for s in a.steps.split(",")]:
        for seed in [int(s) for s in a.seeds.split(",")]:
            for lr in [float(x) for x in a.lrs.split(",")]:
                if (step, lr, seed) in done:
                    continue
                torch.manual_seed(seed); rng = random.Random(seed)
                model = GPTNeoXForCausalLM.from_pretrained(a.model, revision=f"step{step}", torch_dtype=torch.float32).to(dev)
                model.eval()
                base_txt = text_loss(model, chunks)
                curve = []
                def ev(t):
                    model.eval()
                    r = dict(t=t, recall=exact(model, tok, recall), extract=exact(model, tok, extract),
                             rule_train=exact(model, tok, r_train), rule_test=exact(model, tok, r_test),
                             text=text_loss(model, chunks[:10]))
                    model.train(); return r
                curve.append(ev(0))
                opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
                pool = train + r_train * 3                                      # rule pairs seen 3x as often as each fact line
                t0 = time.time(); model.train()
                for t in range(1, a.train_steps + 1):
                    ids_, att, lab = (x.to(dev) for x in batch_of(tok, rng.sample(pool, a.bs)))
                    loss = model(input_ids=ids_, attention_mask=att, labels=lab).loss
                    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                    if t % 25 == 0:
                        curve.append(ev(t))
                model.eval()
                fin = dict(step=step, tokens_b=step * 2.097152e-3, lr=lr, seed=seed, base_text=base_txt, final_text=text_loss(model, chunks),
                           curve=curve, secs=time.time() - t0)
                with open(a.out, "a") as f:
                    f.write(json.dumps(fin) + "\n")
                c = curve[-1]
                print(f"step {step:6d} lr {lr:g} seed {seed}: recall {c['recall']:.2f} extract {c['extract']:.2f} rule test {c['rule_test']:.2f} "
                      f"text loss {base_txt:.3f} -> {fin['final_text']:.3f} ({fin['secs']:.0f} s)", flush=True)
                del model, opt


if __name__ == "__main__":
    main()
