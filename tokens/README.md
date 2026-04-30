# Secret token slots

Files in this directory are git-ignored. Drop API keys here as plain text
(no quotes, no newline cleanup needed — readers `.strip()`).

| File                | Used by                              |
|---------------------|--------------------------------------|
| `wandb.token`       | `src/utils/llm_training.py` if `WANDB_API_KEY` env var is unset |
| `huggingface.token` | (reserved) HF Hub auth for gated datasets if needed |

To add a key:

```bash
echo "your-api-key-here" > tokens/wandb.token
```

In Colab, `colab.py` writes this slot from `google.colab.userdata` if a
secret named `WANDB_API_KEY` exists.
