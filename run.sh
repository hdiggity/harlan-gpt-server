#!/bin/zsh
# launchd wrapper for harlan-gpt-server (macOS has no systemd EnvironmentFile).
# Loads HARLAN_GPT_TOKEN from ./.env, then execs the v4 server against the in-place
# v4.1.0 instruct checkpoint (formerly tagged v4.0.0-instruct) + drinks-24576
# tokenizer + retrieval-v4 corpus in the harlan-gpt model repo (READ-ONLY; that
# repo is never modified from here). ask_server_v4.py forces CPU inference
# (config.device="cpu"), so it never contends with a live MPS train.py run.
cd /Users/harlan/services/harlan-gpt-server || exit 1
export PYTHONUNBUFFERED=1
set -a
source ./.env
set +a
exec /Users/harlan/miniconda3/envs/harlan-gpt/bin/python scripts/ask_server_v4.py \
  --ckpt_dir /Users/harlan/services/harlan-gpt/whisky-gpt/gpt/runs/v4.1.0/checkpoints \
  --retrieval_dir /Users/harlan/services/harlan-gpt/whisky-gpt/gpt/runs/v4.0.0/retrieval \
  --tokenizer_dir /Users/harlan/services/harlan-gpt/whisky-gpt/tokenizer/runs/v3.0.0/drinks-24576 \
  --port 8799
