# NimbusPay Support Ticket Triage — LLM Fine-Tuning

- **Base Model:** `Qwen/Qwen2.5-3B-Instruct`
- **Quantization:** 4-bit QLoRA (`load_in_4bit=True`)
- **Best Dev Result:** **95.0%** Exact Match (**99.1%** Mean Field Accuracy, **100.0%** Valid JSON) on `dev.jsonl` (`29.93M` trainable parameters, LoRA rank `r=16`, `max_seq_length=4096`).

## Repository Contents
- `adapter/` — Saved PEFT LoRA adapter weights (`adapter_config.json`, `adapter_model.safetensors`, `README.md`).
- `predictions_test.jsonl` — Generated JSON triage predictions for all 400 test tickets in `test_inputs.jsonl`.
- `train_clean.jsonl` — Audited and schema-normalized training dataset (1,333 examples).
- `cleaning_log.csv` — Row-level audit log of all 281 dropped and fixed training records.
- `nimbuspay_triage.ipynb` — End-to-end Google Colab notebook covering data cleaning, zero-shot baselines, 5 controlled fine-tuning experiments, error analysis, and test inference.
- `report.md` — Full experimental report, baseline comparisons, and dev error analysis.
