# GLiNER2.5-Decide

Fastino's open-weight (Apache-2.0) schema classifiers, released 2026-09. You give text plus a schema of typed questions and get structured labels back, with no generation and no prompt template. Package: [`gliner2`](https://github.com/fastino-ai/GLiNER2).

| Checkpoint | Size | Encoder | fast-decisions avg (vendor) |
|---|---|---|---|
| [`fastino/GLiNER2.5-Decide-1B`](https://huggingface.co/fastino/GLiNER2.5-Decide-1B) (default) | ~1B | Ettin enc-from-dec 1B | 59.6% |
| [`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide) | 340M | DeBERTa-v3-large | 60.2% |

Both are English-only. [`GLiNER2.5-multi-Decide`](https://huggingface.co/fastino/GLiNER2.5-multi-Decide) (287M) is the multilingual one and isn't set up here. The scores are Fastino's own, from their 17-domain benchmark, and there's no per-domain breakdown, so it's unknown where the 1B beats the 340M, if anywhere.

## Run

```bash
uv sync
uv run try.py                                   # Decide-1B, fp16
uv run try.py --model fastino/GLiNER2.5-Decide  # 340M
uv run try.py --fp32 --confidence               # vendor-precision weights, show confidences
uv run try.py --docs notes/ a.md --confidence   # classify .md files (dirs searched recursively) by medical doc type
```

`--docs` uses the categories in `MEDICAL_DOC_TASKS` in [try.py](try.py) and runs each file through `classify_text_long`, so long docs are chunked.

Edit `CASES` in [try.py](try.py) to try your own text and schemas. `batch_classify_text` and `classify_text_long` (chunked) exist for many or long texts.

## Findings

_Not tried yet._
