# Project context: SWE015 homework (deadline: 22 Nov 2026)

Course homework "Building an Intelligent Application". Chosen: **Use Case 4 – Real-Time Chat Moderation (Turkish)**.
Grading is 0/50/100; every requirement R1–R8 must be met. Every number in the slides must come from our own runs (never invent numbers).

## Status
- Mac, Apple M4 Pro, Python 3.12 venv (`source venv/bin/activate`). Python 3.14 was rejected (gensim does not build).
- `python train.py --quick` ran successfully; all 5 models were evaluated.
- After that, baseline fixes were made (lexicon thresholds, standardization + lr=0.01 for the FFN).
- Next: `python train.py --epochs 2` (full data) → test `python app.py` → GitHub + HF Hub → slides → video.

## Files
- `common.py`: normalize (regex: Turkish lowercasing, leetspeak, stretched letters), LexiconMatcher (edit distance), FFN, Predictor classes, quantize_bert (dynamic INT8, qnnpack)
- `train.py`: data `stefan-it/offenseval2020_tr` (lines formatted as `__label__OFF text`), 5 models, results in `results/results.md`, `results/failures.txt`
- `app.py`: Gradio chat; default model is BERTurk INT8 on CPU; masking, latency log, model selection, comparison button
- `upload_models.py`: models/ → Hugging Face Hub. Update `HF_MODEL_REPO` in app.py and README.

## Model ladder / technique map
B: regex normalization + lexicon/edit distance (rule) + TF-IDF+LogReg · A: own word2vec + FFN (SGD) · C: BERTurk fine-tune (`dbmdz/bert-base-turkish-cased`) · D: dynamic INT8 quantization

## Remaining deliverables
- PDF presentation, 8–12 slides: cover slide (name, student ID, use case, repo link, video link), Introduction / Development (data, architecture diagram, technique map, models, quantization, results table + hardware) / Conclusion (where each model wins and loses, a failure example, next steps, AI usage slide)
- Video ≤2:00 with narration: live demo (≥2 inputs, 1 hard case), architecture + table, conclusion. Upload to YouTube unlisted.
- Repo: README + requirements.txt + single run command; no weights in the repo.