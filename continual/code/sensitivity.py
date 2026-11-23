r"""Why do late Pythia checkpoints forget so much under the same fine-tune? Test: parameter sensitivity. Add Gaussian noise
to every weight tensor, scaled to that tensor's RMS (w + s * rms(w) * N(0, 1)), and measure the WikiText loss increase.
No training. If sensitivity jumps where the fine-tuning damage jumps (between 151B and 210B tokens), the damage is an
optimisation-geometry effect (sharper solution late in training, e.g. as the learning rate anneals), not lost capacity.

    C:\Users\user\ml\.venv\Scripts\python.exe sensitivity.py --model EleutherAI/pythia-70m-deduped --out sens_70m.jsonl
"""
import argparse
import json

import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM


@torch.no_grad()
def loss(model, chunks):
    return sum(float(model(input_ids=c[None], labels=c[None]).loss) for c in chunks) / len(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    ap.add_argument("--steps", default="1000,8000,16000,32000,48000,72000,100000,120000,143000")
    ap.add_argument("--sigmas", default="0.01,0.02,0.04")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--out", default="sens.jsonl")
    a = ap.parse_args()
    torch.set_num_threads(14)
    tok = AutoTokenizer.from_pretrained(a.model)
    from datasets import load_dataset
    wt = "\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if len(t) > 200)
    ids = tok(wt).input_ids
    chunks = [torch.tensor(ids[i:i + 128]) for i in range(0, 128 * 40, 128)]
    for step in [int(s) for s in a.steps.split(",")]:
        model = GPTNeoXForCausalLM.from_pretrained(a.model, revision=f"step{step}", torch_dtype=torch.float32).eval()
        base = loss(model, chunks)
        orig = {n: p.detach().clone() for n, p in model.named_parameters()}
        row = dict(step=step, base=base)
        for s in (float(x) for x in a.sigmas.split(",")):
            d = []
            for k in range(a.draws):
                g = torch.Generator().manual_seed(1000 * k + 7)
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        rms = orig[n].pow(2).mean().sqrt()
                        p.copy_(orig[n] + s * rms * torch.randn(p.shape, generator=g))
                d.append(loss(model, chunks) - base)
            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.copy_(orig[n])
            row[f"d{s:g}"] = sum(d) / len(d)
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        del model, orig


if __name__ == "__main__":
    main()
