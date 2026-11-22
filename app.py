import os
import time

import gradio as gr

from common import load_all_predictors, normalize, LexiconMatcher

MODEL_DIR = "models"
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "rewxtune/swe015-chat-moderation")

if not os.path.exists(os.path.join(MODEL_DIR, "bert_int8.pt")):
    print(f"models/ bulunamadi, Hugging Face Hub'dan indiriliyor: {HF_MODEL_REPO}")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=HF_MODEL_REPO, local_dir=MODEL_DIR)

print("Modeller yukleniyor...")
PREDICTORS = {p.name: p for p in load_all_predictors(MODEL_DIR)}
DEFAULT = "BERTurk INT8 (quantized)"
with open(os.path.join(MODEL_DIR, "lexicon.txt"), encoding="utf-8") as f:
    MATCHER = LexiconMatcher([l.strip() for l in f if l.strip()])
for p in PREDICTORS.values():   # isinma
    p.predict(["merhaba"])


def mask(msg):
    """Sozlukteki kelimeleri yildizlar; hicbiri yoksa mesajin tamamini gizler."""
    words = msg.split()
    out = ["*" * len(w) if MATCHER.match_token(normalize(w)) else w for w in words]
    if out == words:
        return "🚫 [mesaj gizlendi]"
    return " ".join(out)


def run(model_name, msg):
    p = PREDICTORS[model_name]
    t0 = time.perf_counter()
    prob = float(p.predict([msg])[0])
    ms = (time.perf_counter() - t0) * 1000
    off = prob >= 0.5
    return off, (prob if off else 1 - prob), ms


def moderate(msg, model_name, history, log):
    if not msg or not msg.strip():
        return "", history, log, log
    off, conf, ms = run(model_name, msg)
    label = "OFFENSIVE" if off else "CLEAN"
    icon = "🚫" if off else "✅"
    history = history + [
        {"role": "user", "content": mask(msg) if off else msg},
        {"role": "assistant", "content": f"{icon} {label} · güven {conf:.2f} · {ms:.1f} ms · {model_name}"},
    ]
    log = [[msg[:40], model_name, label, round(conf, 2), round(ms, 1)]] + log
    return "", history, log, log


def compare(msg):
    rows = []
    for name in PREDICTORS:
        off, conf, ms = run(name, msg)
        rows.append([name, "OFFENSIVE" if off else "CLEAN", round(conf, 2), round(ms, 1)])
    return rows


with gr.Blocks(title="SWE015 · Chat Moderation") as demo:
    gr.Markdown("## 🛡️ Gerçek Zamanlı Sohbet Moderasyonu (Türkçe)\n"
                "Her mesaj anında **clean / offensive** olarak etiketlenir. "
                "Varsayılan model: **BERTurk INT8 (quantized), CPU**.")
    with gr.Row():
        with gr.Column(scale=3):
            try:
                chat = gr.Chatbot(type="messages", height=420, label="Sohbet")
            except TypeError:
                chat = gr.Chatbot(height=420, label="Sohbet")
            box = gr.Textbox(placeholder="Mesaj yaz ve Enter'a bas...", show_label=False)
            with gr.Row():
                send = gr.Button("Gönder", variant="primary")
                cmp_btn = gr.Button("Tüm modellerle karşılaştır")
        with gr.Column(scale=2):
            model_dd = gr.Dropdown(list(PREDICTORS), value=DEFAULT, label="Model")
            log_df = gr.Dataframe(headers=["Mesaj", "Model", "Etiket", "Güven", "ms"],
                                  label="Yanıt süresi logu", interactive=False)
            cmp_df = gr.Dataframe(headers=["Model", "Etiket", "Güven", "ms"],
                                  label="Karşılaştırma (son mesaj)", interactive=False)
    log_state = gr.State([])

    ins = [box, model_dd, chat, log_state]
    outs = [box, chat, log_state, log_df]
    box.submit(moderate, ins, outs)
    send.click(moderate, ins, outs)
    cmp_btn.click(compare, box, cmp_df)

if __name__ == "__main__":
    demo.launch()