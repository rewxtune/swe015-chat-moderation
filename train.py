import os
import time
import random
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score

from common import (normalize, tokenize, mine_lexicon, FFN, avg_vectors, quantize_bert,
                    MAX_LEN, RulePredictor, TfidfPredictor, W2VPredictor, BertPredictor,
                    size_mb, pick_device, hardware_info)

p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true", help="kucuk veriyle hizli deneme")
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--bert", default="dbmdz/bert-base-turkish-cased")
p.add_argument("--runs", type=int, default=200, help="gecikme olcumu icin tekrar sayisi (>=100)")
args = p.parse_args()

random.seed(42); np.random.seed(42); torch.manual_seed(42)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def step(msg):
    print("\n" + "=" * 70 + f"\n{msg}\n" + "=" * 70, flush=True)


# ---------------------------------------------------------------------------
step("1) Veri yukleniyor: OffensEval-TR 2020 (stefan-it/offenseval2020_tr)")
ds = load_dataset("stefan-it/offenseval2020_tr")


def parse(split):
    texts, labels = [], []
    for row in ds[split]["text"]:
        tag, _, txt = row.partition(" ")
        texts.append(txt)
        labels.append(1 if tag == "__label__OFF" else 0)
    return texts, np.array(labels)


tr_x, tr_y = parse("train")
dv_x, dv_y = parse("validation")
te_x, te_y = parse("test")
if args.quick:
    tr_x, tr_y = tr_x[:5000], tr_y[:5000]
for n, y in [("train", tr_y), ("dev", dv_y), ("test", te_y)]:
    print(f"{n:5s}: {len(y):6d} mesaj | OFF orani: {y.mean():.1%}")

print("\nNormalizasyon ornegi:")
for s in ["Çoooook GÜZEL bir maç!!! @USER", "s1kt1r git buradan", "Bu ne ya URL #rezalet"]:
    print(f"  {s!r:40s} -> {normalize(s)!r}")

tr_tok = [tokenize(t) for t in tr_x]
dv_tok = [tokenize(t) for t in dv_x]

# ---------------------------------------------------------------------------
step("2) Kural baseline: egitim verisinden sozluk + edit distance")
lex = mine_lexicon(tr_tok, tr_y, min_count=5, min_ratio=0.7)
with open("models/lexicon.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lex))
print(f"Sozlukte {len(lex)} kelime var -> models/lexicon.txt")

# ---------------------------------------------------------------------------
step("3) Klasik baseline: TF-IDF + Logistic Regression")
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
])
pipe.fit([normalize(t) for t in tr_x], tr_y)
joblib.dump(pipe, "models/tfidf_lr.joblib")
print("dev macro-F1:", round(f1_score(dv_y, pipe.predict([normalize(t) for t in dv_x]), average="macro"), 4))

# ---------------------------------------------------------------------------
step("4) Sinir agi: kendi word2vec'imiz + feedforward ag (SGD)")
from gensim.models import Word2Vec

w2v = Word2Vec(sentences=tr_tok, vector_size=100, window=5, min_count=2,
               workers=4, epochs=15, seed=42)
w2v.wv.save("models/w2v.kv")
print(f"word2vec sozluk boyutu: {len(w2v.wv.key_to_index)}")

Xtr = torch.from_numpy(avg_vectors(w2v.wv, tr_tok))
Xdv = torch.from_numpy(avg_vectors(w2v.wv, dv_tok))
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6          # standardizasyon: SGD daha kararli ogrenir
Xtr, Xdv = (Xtr - mu) / sd, (Xdv - mu) / sd
ytr = torch.from_numpy(tr_y).long()
ffn = FFN(100, 128)
weights = torch.tensor([1.0, (tr_y == 0).sum() / max((tr_y == 1).sum(), 1)], dtype=torch.float32)
loss_fn = nn.CrossEntropyLoss(weight=weights)
opt = torch.optim.SGD(ffn.parameters(), lr=0.01, momentum=0.9)
loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
best_f1, best_state = -1, None
for ep in range(30):
    ffn.train()
    for xb, yb in loader:
        opt.zero_grad()
        loss = loss_fn(ffn(xb), yb)
        loss.backward()          # backpropagation
        opt.step()               # SGD guncellemesi
    ffn.eval()
    with torch.no_grad():
        f1 = f1_score(dv_y, ffn(Xdv).argmax(-1).numpy(), average="macro")
    if f1 > best_f1:
        best_f1, best_state = f1, {k: v.clone() for k, v in ffn.state_dict().items()}
    if (ep + 1) % 5 == 0:
        print(f"  epoch {ep + 1:2d} | loss {loss.item():.4f} | dev macro-F1 {f1:.4f}")
torch.save({"state": best_state, "dim": 100, "hidden": 128, "mu": mu, "sd": sd}, "models/ffn.pt")
print(f"En iyi dev macro-F1: {best_f1:.4f}")

# ---------------------------------------------------------------------------
device = pick_device()
step(f"5) Transformer: BERTurk fine-tune  (cihaz: {device}, epoch: {args.epochs})")
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained(args.bert)
model = AutoModelForSequenceClassification.from_pretrained(
    args.bert, num_labels=2, id2label={0: "NOT", 1: "OFF"}, label2id={"NOT": 0, "OFF": 1}).to(device)

enc = tok([normalize(t, lower=False) for t in tr_x], truncation=True, max_length=MAX_LEN,
          padding="max_length", return_tensors="pt")
bert_loader = DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"], ytr),
                         batch_size=args.batch, shuffle=True)
optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
total = len(bert_loader) * args.epochs
warm = int(0.06 * total)
sched = torch.optim.lr_scheduler.LambdaLR(
    optim, lambda s: (s + 1) / max(warm, 1) if s < warm else max(0.0, (total - s) / max(total - warm, 1)))

t0, s = time.time(), 0
for ep in range(args.epochs):
    model.train()
    for ids, am, yb in bert_loader:
        out = model(input_ids=ids.to(device), attention_mask=am.to(device), labels=yb.to(device))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step(); sched.step(); optim.zero_grad()
        s += 1
        if s % 50 == 0 or s == total:
            el = time.time() - t0
            print(f"  adim {s}/{total} | loss {out.loss.item():.4f} | "
                  f"gecen {el / 60:.1f} dk | kalan ~{el / s * (total - s) / 60:.1f} dk", flush=True)

model.save_pretrained("models/bert")
tok.save_pretrained("models/bert")
print("Kaydedildi -> models/bert")

# ---------------------------------------------------------------------------
step("6) Quantization: dinamik INT8 (tum Linear katmanlari)")
qmodel = quantize_bert(model)
torch.save(qmodel.state_dict(), "models/bert_int8.pt")
print("Kaydedildi -> models/bert_int8.pt")
del model, qmodel

# ---------------------------------------------------------------------------
step("7) Degerlendirme: tum modeller AYNI test setinde (3.528 mesaj)")
predictors = [
    RulePredictor("models/lexicon.txt"),
    TfidfPredictor("models/tfidf_lr.joblib"),
    W2VPredictor("models/w2v.kv", "models/ffn.pt"),
    BertPredictor("models/bert", device=device),
    BertPredictor("models/bert", int8_path="models/bert_int8.pt"),
]

hw = hardware_info()
rows, preds_int8 = [], None
rng = random.Random(0)
lat_samples = [rng.choice(te_x) for _ in range(args.runs)]
for pr in predictors:
    print(f"\n-> {pr.name}: test tahminleri...", flush=True)
    probs = pr.predict(te_x)
    pred = (probs >= 0.5).astype(int)
    if pr.name.startswith("BERTurk INT8"):
        preds_int8 = pred
    if hasattr(pr, "to"):
        pr.to("cpu")                       # gecikme her model icin CPU'da olculur
    for t in lat_samples[:10]:             # isinma
        pr.predict([t])
    times = []
    for t in lat_samples:
        a = time.perf_counter()
        pr.predict([t])
        times.append((time.perf_counter() - a) * 1000)
    rows.append({
        "Model": pr.name,
        "Macro-F1": round(f1_score(te_y, pred, average="macro"), 4),
        "Accuracy": round(accuracy_score(te_y, pred), 4),
        "Boyut (MB)": round(size_mb(pr.files), 2),
        "Gecikme (ms/mesaj)": round(float(np.mean(times)), 2),
    })
    print(rows[-1])

df = pd.DataFrame(rows)
df.to_csv("results/results.csv", index=False)
with open("results/results.md", "w", encoding="utf-8") as f:
    f.write(f"Test seti: OffensEval-TR 2020 test, {len(te_y)} mesaj (tum modeller icin ayni)\n\n")
    f.write(f"Donanim (gecikme olcumu, CPU, {args.runs} calistirma ortalamasi): {hw}\n\n")
    f.write("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n")
    for _, r in df.iterrows():
        f.write("| " + " | ".join(str(v) for v in r.values) + " |\n")

with open("results/failures.txt", "w", encoding="utf-8") as f:
    wrong = np.where(preds_int8 != te_y)[0][:15]
    for i in wrong:
        f.write(f"gercek={'OFF' if te_y[i] else 'NOT'} tahmin={'OFF' if preds_int8[i] else 'NOT'} | {te_x[i]}\n")

step("SONUC TABLOSU (results/results.md)")
print(df.to_string(index=False))
print(f"\nDonanim: {hw}")
print("Hatali tahmin ornekleri -> results/failures.txt")