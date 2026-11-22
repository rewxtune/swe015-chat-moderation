Test seti: OffensEval-TR 2020 test, 3528 mesaj (tum modeller icin ayni)

Donanim (gecikme olcumu, CPU, 200 calistirma ortalamasi): Apple M4 Pro | Darwin 24.6.0 | torch 2.15.0 | CPU threads: 8

| Model | Macro-F1 | Accuracy | Boyut (MB) | Gecikme (ms/mesaj) |
|---|---|---|---|---|
| Kural (edit-distance sozluk) | 0.6372 | 0.7168 | 0.0 | 0.48 |
| TF-IDF + LogReg | 0.7184 | 0.8044 | 1.68 | 0.34 |
| word2vec + FFN | 0.6203 | 0.7061 | 12.53 | 0.07 |
| BERTurk FP32 | 0.8114 | 0.8858 | 422.0 | 23.91 |
| BERTurk INT8 (quantized) | 0.8118 | 0.8875 | 177.42 | 20.9 |
