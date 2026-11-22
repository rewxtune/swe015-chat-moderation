# SWE015 – Real-Time Chat Moderation (Turkish)
 
Use Case 4: every chat message is labeled live as **clean / offensive**. Offensive messages are masked, and a side panel logs the response time of each message. The app runs **BERTurk quantized to INT8** on the CPU by default.
 
## Models (model ladder)
 
| # | Model | Technique area |
|---|---|---|
| 0 | Lexicon + edit distance (rule baseline) | B · Classical NLP |
| 1 | TF-IDF + Logistic Regression (classical baseline) | B · Classical NLP |
| 2 | Own word2vec + feedforward network (SGD, backprop) | A · Neural networks |
| 3 | BERTurk fine-tuned (`dbmdz/bert-base-turkish-cased`) | C · Transformers |
| 4 | BERTurk dynamic INT8 quantization | D · Efficient models |
 
Preprocessing (B): regex normalization covering Turkish lowercasing, leetspeak (`s1kt1r` → `siktir`), stretched letters (`çoook` → `çok`), and @USER/URL/punctuation cleanup.
 
## Setup (Python 3.12)
 
```bash
python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
 
## Run (single command)
 
```bash
python app.py
```
 
Open http://127.0.0.1:7860 in your browser. If the `models/` folder is missing, the app downloads the weights automatically from Hugging Face Hub: **https://huggingface.co/rewxtune/swe015-chat-moderation**
 
## Retraining (optional)
 
```bash
python train.py --quick   # quick test run (~5-10 min)
python train.py           # full training; writes results/results.md
```
 
## Data
 
OffensEval-TR 2020: Çöltekin, Ç. (2020). *A Corpus of Turkish Offensive Language on Social Media.* LREC 2020. Loaded via Hugging Face as `stefan-it/offenseval2020_tr`: 30,000 train / 1,756 dev / 3,528 test messages. License: CC-BY-2.0.
 
## Files
 
- `common.py`: normalization, lexicon matcher, model classes (shared by training and the app)
- `train.py`: trains all models, evaluates them on the same test set, writes `results/`
- `app.py`: Gradio interface
- `upload_models.py`: uploads the weights to Hugging Face Hub
 