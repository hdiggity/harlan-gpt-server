# harlan-gpt-server

HTTP service that serves [harlan-gpt](https://github.com/hdiggity/harlan-gpt)'s
whisky RAG model (v4.1.0 instruct checkpoint, formerly tagged v4.0.0-instruct +
drinks-24576 tokenizer + BM25 whisky retriever) behind a Cloudflare Tunnel.
Runs CPU-only so it never contends with a live MPS training run on the same box.

This repo is serving code only: no model weights, no corpus, no training code.
Those live in the (private) model repo and are read in place, never copied
here or committed to this repo.

## What runs

- `ask_server_v4.py` (+ `config.py`, `model.py`, `retriever.py`) build the
  model + BM25 retriever once at startup, then serve queries over HTTP.
- Large artifacts are **read in place** from the model repo via CLI args:
  - checkpoint: `<harlan-gpt>/gpt/runs/v4.1.0/checkpoints/ckpt.pt`
  - tokenizer: `<harlan-gpt>/tokenizer/runs/v3.0.0/drinks-24576/` (vocab.json + merges.txt)
  - retrieval corpus: `<harlan-gpt>/gpt/runs/v4.0.0/retrieval/*.jsonl` (304k passages)
- Binds `127.0.0.1:8799` only. The Cloudflare Tunnel is the only public entry point.
- Forced CPU (`config.device = "cpu"`, set in `ask_server_v4.py`): keeps the
  132M model off MPS so it can never contend with / OOM a live train.py.
- The checkpoint was trained on MPS (manual attention, registers a per-layer
  `attn.mask` buffer); CPU serving uses fused SDPA, which does not register
  that buffer, so the checkpoint load is `strict=False` (drops exactly the
  `.attn.mask` keys, fails loudly on anything else).
- **BM25 index cache**: building the retriever over 304k passages takes
  minutes of CPU. `build_retriever()` pickles the built index to
  `.cache/retriever_v4_<fingerprint>.pkl` (fingerprint = sha256 over the
  corpus files' name+size+mtime) and reloads it on the next start (~seconds).
  A stale/foreign cache is detected by fingerprint mismatch and rebuilt; any
  cache read/write failure falls back to a plain rebuild, never crashes.
- **ProcessType must be `Standard`, not `Background`**, in the launchd plist.
  `Background` pins the process to Apple Silicon's efficiency cores, which
  runs a 132M model ~10x slower -- slow enough that a full-length generation
  can exceed Cloudflare's ~100s origin timeout (524).

## Kept alive (launchd, macOS)

- `run.sh` is the launchd wrapper: sources `.env`, execs the model repo's
  conda env python against `ask_server_v4.py`.
- Installed as `/Library/LaunchDaemons/com.harlan.harlan-gpt-server.plist`
  (system domain, `KeepAlive`, `RunAtLoad`, `Nice 10`, `ProcessType Standard`).
- Status: `sudo launchctl print system/com.harlan.harlan-gpt-server` /
  `tail -50 ~/logs/harlan-gpt-server.out.log ~/logs/harlan-gpt-server.err.log`.
- Editing the plist needs a full reload, not a kickstart:
  `sudo launchctl bootout system/com.harlan.harlan-gpt-server && sudo launchctl bootstrap system <plist>`.
  `launchctl kickstart -k` restarts the process but does not re-read the plist.
- To move to a newer model checkpoint: repoint `--ckpt_dir` / `--tokenizer_dir`
  / `--retrieval_dir` in `run.sh` and reload.

## Tunnel wiring

- Ingress rule in `~/.cloudflared/config.yml`:
  `harlan-gpt.harlanswitzer.com -> http://127.0.0.1:8799` (before the catch-all).
- DNS: CNAME `harlan-gpt.harlanswitzer.com` -> the tunnel (via
  `cloudflared tunnel route dns`).

## Auth (bearer token)

Every request must send `Authorization: Bearer <token>`; the server refuses
to start without `HARLAN_GPT_TOKEN` set, so it never serves unauthenticated.

- Secret lives in `.env` (mode 600, `HARLAN_GPT_TOKEN=...`), sourced by
  `run.sh`. Never committed (`.gitignore`).
- Rotate: edit `.env`, then bootout + bootstrap the daemon (see above).
- A bare `GET /` is ungated and serves `page.html` (a browser form; no model
  output), so the endpoint can be health-checked without the token.
- This is app-level auth, not edge-enforced (no Cloudflare Access): a
  tokenless request still reaches the box before the 401. Acceptable
  trade-off for a single-owner personal endpoint.

## Query it

```
TOK=$(sed -n 's/^HARLAN_GPT_TOKEN=//p' .env)

curl -s -X POST https://harlan-gpt.harlanswitzer.com \
  -H "Authorization: Bearer $TOK" -H 'Content-Type: application/json' \
  -d '{"question":"What does Lagavulin 16 taste like?"}'
```

Response shape: `{"answer": "...", "retrieved": ["...", ...]}`. Optional body
fields: `top_k` (passages retrieved, default 5), `max_new_tokens` (default
300), `temperature` (default 0.8), `gen_top_k` (default 200).

Browser: visiting the URL serves a small query UI (`page.html`). Enter the
token once; it's saved in `localStorage` and never retyped, but every POST
still carries it, so the service stays protected. `page.html` is read fresh
on each GET, so UI edits are live without restarting the daemon.

## Known limits (v4.1.0)

Grounding works: the model reads and paraphrases the retrieved passages
rather than fabricating facts (the v3 regression this model line is built
around). Two known weaknesses, both v5 training targets, not serving bugs:
- No trained end-of-response token, so generation runs to `max_new_tokens`
  and can loop/repeat in the tail.
- Some yes-bias on leading yes/no questions (echoes the question's framing
  rather than checking it against the retrieved passages).

See the model repo's `gpt/runs/v4.1.0/v4.1.0.yaml` for the full eval writeup.
