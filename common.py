import os
import re
import platform
import subprocess
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# B · Klasik NLP: regex normalizasyonu
# ---------------------------------------------------------------------------
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"})
WORD_WITH_LETTER = re.compile(r"\w*[^\W\d_]\w*")  # icinde en az bir harf olan kelime


def tr_lower(s: str) -> str:
    """Turkce kurallarina gore kucuk harf (I -> ı, İ -> i)."""
    return s.replace("I", "ı").replace("İ", "i").lower()


def normalize(text: str, lower: bool = True) -> str:
    t = text
    t = re.sub(r"@USER", " ", t)                       # kullanici etiketleri
    t = re.sub(r"https?://\S+|\bURL\b", " ", t)        # linkler
    t = t.replace("#", " ")                            # hashtag isareti
    if lower:
        t = tr_lower(t)
    # leetspeak: sadece harf iceren kelimelerdeki rakamlar (s1kt1r -> siktir, 2024 kalir)
    t = WORD_WITH_LETTER.sub(lambda m: m.group(0).translate(LEET), t)
    t = re.sub(r"(.)\1{2,}", r"\1", t)                 # uzatmalar: çoook -> çok
    t = re.sub(r"[^\w\s]", " ", t)                     # noktalama / emoji
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize(text: str):
    return normalize(text).split()


# ---------------------------------------------------------------------------
# B · Klasik NLP: minimum edit distance ile sozluk eslestirme (kural baseline)
# ---------------------------------------------------------------------------
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def mine_lexicon(token_lists, labels, min_count=15, min_ratio=0.85, max_words=300):
    """Egitim verisinden saldirgan mesajlarda cok gecen kelimeleri cikarir."""
    total, off = Counter(), Counter()
    for toks, y in zip(token_lists, labels):
        for w in set(toks):
            total[w] += 1
            if y == 1:
                off[w] += 1
    cands = [(off[w] / c, c, w) for w, c in total.items()
             if c >= min_count and len(w) >= 3 and off[w] / c >= min_ratio]
    cands.sort(reverse=True)
    return [w for _, _, w in cands[:max_words]]


class LexiconMatcher:
    def __init__(self, words):
        self.words = set(words)
        self.by_first = {}
        for w in self.words:
            self.by_first.setdefault(w[0], []).append(w)

    def match_token(self, tok: str) -> bool:
        if not tok:
            return False
        if tok in self.words:
            return True
        for w in self.by_first.get(tok[0], []):
            if len(w) >= 4 and tok.startswith(w):          # ek almis hali: salakmisin
                return True
            if len(tok) >= 4 and abs(len(tok) - len(w)) <= 1 and levenshtein(tok, w) <= 1:
                return True                                 # yazim hatasi: salk -> salak
        return False


class RulePredictor:
    name = "Kural (edit-distance sozluk)"

    def __init__(self, lexicon_path):
        with open(lexicon_path, encoding="utf-8") as f:
            words = [l.strip() for l in f if l.strip()]
        self.matcher = LexiconMatcher(words)
        self.files = [lexicon_path]

    def predict(self, texts):
        return np.array([1.0 if any(self.matcher.match_token(t) for t in tokenize(x)) else 0.0
                         for x in texts])


# ---------------------------------------------------------------------------
# B · Klasik baseline: TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------
class TfidfPredictor:
    name = "TF-IDF + LogReg"

    def __init__(self, path):
        import joblib
        self.pipe = joblib.load(path)
        self.files = [path]

    def predict(self, texts):
        return self.pipe.predict_proba([normalize(t) for t in texts])[:, 1]


# ---------------------------------------------------------------------------
# A · Sinir agi: kendi word2vec'imiz + feedforward ag (SGD + backprop)
# ---------------------------------------------------------------------------
class FFN(nn.Module):
    def __init__(self, dim=100, hidden=128, p=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def avg_vectors(kv, token_lists):
    dim = kv.vector_size
    out = np.zeros((len(token_lists), dim), dtype=np.float32)
    for i, toks in enumerate(token_lists):
        vecs = [kv[w] for w in toks if w in kv.key_to_index]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out


class W2VPredictor:
    name = "word2vec + FFN"

    def __init__(self, kv_path, ffn_path):
        from gensim.models import KeyedVectors
        self.kv = KeyedVectors.load(kv_path)
        ck = torch.load(ffn_path, map_location="cpu")
        self.ffn = FFN(ck["dim"], ck["hidden"])
        self.ffn.load_state_dict(ck["state"])
        self.ffn.eval()
        self.mu, self.sd = ck["mu"], ck["sd"]
        folder = os.path.dirname(kv_path)
        base = os.path.basename(kv_path)
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith(base)]
        self.files.append(ffn_path)

    def predict(self, texts):
        X = torch.from_numpy(avg_vectors(self.kv, [tokenize(t) for t in texts]))
        X = (X - self.mu) / self.sd
        with torch.no_grad():
            return torch.softmax(self.ffn(X), dim=-1)[:, 1].numpy()


# ---------------------------------------------------------------------------
# C · Transformer (BERTurk) ve D · Quantization (dinamik INT8)
# ---------------------------------------------------------------------------
MAX_LEN = 64


def quantize_bert(model):
    engines = torch.backends.quantized.supported_engines
    if "qnnpack" in engines and platform.machine().lower() in ("arm64", "aarch64"):
        torch.backends.quantized.engine = "qnnpack"
    elif "fbgemm" in engines:
        torch.backends.quantized.engine = "fbgemm"
    return torch.ao.quantization.quantize_dynamic(model.cpu().eval(), {nn.Linear}, dtype=torch.qint8)


class BertPredictor:
    def __init__(self, model_dir, int8_path=None, device="cpu"):
        from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        if int8_path:
            self.name = "BERTurk INT8 (quantized)"
            cfg = AutoConfig.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_config(cfg)
            model = quantize_bert(model)
            model.load_state_dict(torch.load(int8_path, map_location="cpu", weights_only=False))
            self.files = [int8_path]
            device = "cpu"
        else:
            self.name = "BERTurk FP32"
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.files = [os.path.join(model_dir, f) for f in os.listdir(model_dir)
                          if f.endswith((".safetensors", ".bin"))]
        self.model = model.eval().to(device)
        self.device = device

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    def predict(self, texts, batch_size=64):
        probs = []
        for i in range(0, len(texts), batch_size):
            batch = [normalize(t, lower=False) for t in texts[i:i + batch_size]]
            enc = self.tok(batch, truncation=True, max_length=MAX_LEN, padding=True,
                           return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(probs)


# ---------------------------------------------------------------------------
# Yardimcilar
# ---------------------------------------------------------------------------
def size_mb(files):
    return sum(os.path.getsize(f) for f in files) / (1024 * 1024)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def hardware_info():
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        except Exception:
            pass
    return (f"{cpu} | {platform.system()} {platform.release()} | "
            f"torch {torch.__version__} | CPU threads: {torch.get_num_threads()}")


def load_all_predictors(model_dir="models"):
    """Uygulama ve degerlendirme icin tum modelleri yukler (hepsi CPU)."""
    j = lambda *p: os.path.join(model_dir, *p)
    return [
        BertPredictor(j("bert"), int8_path=j("bert_int8.pt")),
        BertPredictor(j("bert")),
        W2VPredictor(j("w2v.kv"), j("ffn.pt")),
        TfidfPredictor(j("tfidf_lr.joblib")),
        RulePredictor(j("lexicon.txt")),
    ]