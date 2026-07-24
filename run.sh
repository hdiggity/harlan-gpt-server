#!/bin/zsh
# launchd wrapper for harlan-gpt-server (macOS has no systemd EnvironmentFile).
# Loads HARLAN_GPT_TOKEN from ./.env, then execs the v4 server against the in-place
# v4.2.0 instruct checkpoint + drinks-24576-eos (v3.1.0) tokenizer + retrieval-v4
# corpus in the harlan-gpt model repo (READ-ONLY; that repo is never modified from
# here). ask_server_v4.py forces CPU inference (config.device="cpu"), so it never
# contends with a live MPS train.py run.
# v4.2.0 cutover 2026-07-20: ckpt v4.1.0 -> v4.2.0 (vocab 24577, trained <EOS>);
# tokenizer v3.0.0/drinks-24576 -> v3.1.0/drinks-24576-eos. The server loads the
# tokenizer via Tokenizer.from_file (NOT vocab.json) and stops on the <EOS> token.
# v4.3.4 cutover 2026-07-24: ckpt v4.2.0 -> v4.3.4 (same base/tokenizer/EOS/RAG
# format; the best-fair-metrics instruct set: matched-len 0.302, decline-aware
# 0.366, grounded ownership). Same vocab 24577; only --ckpt_dir changes.
cd /Users/harlan/services/harlan-gpt-server || exit 1
export PYTHONUNBUFFERED=1
set -a
source ./.env
set +a
exec /Users/harlan/miniconda3/envs/harlan-gpt/bin/python scripts/ask_server_v4.py \
  --ckpt_dir /Users/harlan/services/harlan-gpt/whisky-gpt/gpt/runs/v4.3.4/checkpoints \
  --retrieval_dir /Users/harlan/services/harlan-gpt/whisky-gpt/gpt/runs/v4.0.0/retrieval \
  --tokenizer_dir /Users/harlan/services/harlan-gpt/whisky-gpt/tokenizer/runs/v3.1.0/drinks-24576-eos \
  --port 8799
