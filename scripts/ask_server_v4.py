# ask_server_v4.py
# live serving wrapper for the v4.1.0 (formerly tagged v4.0.0-instruct) whisky
# RAG model. builds the model + whisky bm25 retriever ONCE at startup, then
# serves questions over http so queries are instant.
#
# key design points:
#   - bearer-token gate (HARLAN_GPT_TOKEN); the server refuses to start
#     without it and 401s every unauthenticated POST.
#   - config.device forced to "cpu": a 132M fp32 model must never contend
#     with a live MPS training run on the same box's shared unified memory.
#     safe for an always-on KeepAlive daemon (~30s/answer with ProcessType
#     Standard; do not run this under ProcessType Background, see README).
#   - a bm25 retriever cache in .cache/ turns a ~minutes rebuild into a
#     ~seconds reload across restarts (see build_retriever).
#   - an ungated GET serves page.html, a small browser query form (no model
#     output), so the endpoint can be health-checked without a token.
#
# served prompt is the EXACT grounded format the finetune trains on:
#   Instruction: {q}\nReference notes:\n- {p1}\n- {p2}\nResponse:
# generation stops at the trained <EOS> token (v4.2.0's real end-of-response
# token, id 24576), with the legacy scaffolding markers (Instruction: /
# Reference notes: / Response:) kept as a fallback for pre-EOS ckpts and for
# the rare post-EOS decode where the model leaks prompt scaffolding instead of
# emitting <EOS> (see the v4.3.4 template-leak fix below).
# POST {"question": ...} + Authorization: Bearer <token> -> {"answer", "retrieved"}
#
# v4.2.0 cutover (2026-07-20): serves the v4.2.0 instruct checkpoint (vocab
# 24577) with the v3.1.0 tokenizer (drinks-24576 + <EOS>). The tokenizer MUST be
# loaded via Tokenizer.from_file(tokenizer.json): vocab.json alone has only 24576
# entries and no <EOS> (it lives only in tokenizer.json's added_tokens), so the
# old ByteLevelBPETokenizer(vocab.json, merges.txt) loader would encode "<EOS>"
# as 4 ordinary tokens and the EOS stop would never fire. model.generate() has no
# stop condition of its own, so the EOS stop lives in generate() below.
#
# v4.3.4 cutover (2026-07-24): ckpt v4.2.0 -> v4.3.4 (same base/tokenizer/EOS/
# RAG format; best fair metrics: matched-len 0.302, decline-aware 0.366,
# grounded ownership). Live testing surfaced two DECODE-time failures shared by
# both checkpoints (verified via train/serve prompt diff + head-to-head on the
# existing 217-item gens: v4.2.0 loops MORE, 6/217 vs v4.3.4's 3/217, so this is
# not new to v4.3.4 and not a train/serve mismatch):
#   1. template leak: the model occasionally emits scaffolding text
#      ("Reference notes:", "Instruction:", "Response:") instead of <EOS>,
#      continuing as if starting a new training example. FIX: STOP_PATTERNS
#      below matches these markers ANYWHERE in the decoded text (not just after
#      a literal "\n" - a mid-text leak without a preceding newline would slip
#      past a strict prefix match), cutting the answer at the earliest hit.
#      Safe because 0/3961 v4.3.4 training RESPONSES contain any of these
#      strings (verified), so a genuine answer never trips this.
#   2. short-cycle repetition loops ("spicy-spicy-spicy...", "proof US proof US
#      proof..."): FIX: a no-repeat-ngram filter (NO_REPEAT_NGRAM below) bans
#      the token that would complete an already-seen n-gram of the ANSWER so
#      far (never the prompt/refs, so retrieved facts can still be echoed
#      once). n=4 chosen over the more aggressive n=3 after eyeballing an A/B
#      (see scratchpad/ab_ngram_test.py): n=3 risked suppressing legitimate
#      domain repeats ("ex-bourbon" twice, a distillery name recurring); n=4
#      still catches both short-cycle loops above (they repeat far more than
#      once) while leaving natural phrasing alone.
# These are DECODE-time fixes only; no model/checkpoint change. They do not
# touch the discrimination failure (yes-bias / unreliable refusal) still
# present in v4.3.4 - that is the capacity question v4.4.x is testing.

import argparse
import glob
import hashlib
import hmac
import json
import os
import pickle
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

import config

# force cpu inference: never share MPS with a training run (v3 pattern).
config.device = "cpu"

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from model import GPT  # noqa: E402
from retriever import WhiskyRetriever  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

CKPT_DIR_DEFAULT = os.path.expanduser(
    "~/services/harlan-gpt/whisky-gpt/gpt/runs/v4.3.4/checkpoints"
)
RETRIEVAL_DIR_DEFAULT = os.path.expanduser(
    "~/services/harlan-gpt/whisky-gpt/gpt/runs/v4.0.0/retrieval"
)
TOKENIZER_DIR = os.path.expanduser(
    "~/services/harlan-gpt/whisky-gpt/tokenizer/runs/v3.1.0/drinks-24576-eos"
)

MODEL_VERSION = "v4.3.4"

# this file lives in scripts/; page.html and .cache/ live at the repo root
# (one level up), alongside run.sh and .env.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# browser landing page (a form you can type into), read fresh on each GET so ui
# tweaks need no restart. serves no model output; the query (POST) stays gated.
PAGE_PATH = os.path.join(REPO_ROOT, "page.html")

# on-disk cache for the built bm25 retriever: the build over 304k passages is
# minutes of cpu, so pickle the built object keyed by a corpus fingerprint and
# reload it on the next start. lives in .cache/ (gitignored).
CACHE_DIR = os.path.join(REPO_ROOT, ".cache")

# bearer-token gate: every POST must send Authorization: Bearer <token>. token
# comes from the HARLAN_GPT_TOKEN env var (run.sh sources ./.env). the server
# refuses to start without it, so it never serves unauthenticated.
AUTH_TOKEN = os.environ.get("HARLAN_GPT_TOKEN", "").strip()

# scaffolding markers: any of these appearing ANYWHERE in the decoded answer
# means the model is echoing prompt structure rather than answering (fallback
# for pre-EOS ckpts; a safety net for post-EOS ckpts that occasionally leak
# scaffolding instead of emitting <EOS>). whitespace-tolerant (re.search, not a
# literal "\n"-prefixed substring) so a leak that renders without a preceding
# newline still gets caught. safe: 0/3961 v4.3.4 training responses contain
# any of these strings, so a genuine answer never trips this.
STOP_PATTERNS = [re.compile(p) for p in
                 (r"\s*Instruction:", r"\s*Reference notes:", r"\s*Response:")]
PROMPT_MARGIN = 8  # tokens of slack for encode-concat boundary effects
MIN_SNIPPET_TOKENS = 24  # skip a truncated ref shorter than this
# short-cycle repetition guard (v4.3.4 cutover): ban the token that would
# complete an already-seen n-gram of the GENERATED answer so far. n=4 chosen
# over n=3 after an A/B (scratchpad/ab_ngram_test.py) - n=3 risked blocking
# legitimate domain repeats ("ex-bourbon" twice, a recurring distillery name);
# n=4 still catches the observed loops (spicy-spicy-spicy, proof US proof US
# proof), which cycle far more than once. 0 disables the filter.
NO_REPEAT_NGRAM = 4


def _find_stop(text):
    # earliest match across all scaffolding patterns, or None.
    starts = []
    for pat in STOP_PATTERNS:
        m = pat.search(text)
        if m:
            starts.append(m.start())
    return min(starts) if starts else None


def _ban_repeat_ngrams(logits, gen_ids, n):
    # gen_ids: token ids of the ANSWER so far (never the prompt/refs), so a
    # retrieved fact can still be echoed once without being blocked. bans the
    # next token if it would repeat an n-gram already seen in gen_ids.
    if n <= 0 or len(gen_ids) < n - 1:
        return logits
    prefix = tuple(gen_ids[-(n - 1):])
    banned = {gen_ids[i + n - 1] for i in range(len(gen_ids) - n + 1)
              if tuple(gen_ids[i:i + n - 1]) == prefix}
    for t in banned:
        logits[0, t] = float("-inf")
    return logits

# globals loaded once at startup
MODEL = None
TOK = None
RETRIEVER = None
BLOCK_SIZE = None
EOS_TOKEN_ID = None  # v4.2.0: trained end-of-response token; None for pre-EOS ckpts
GEN_DEFAULTS = {
    "top_k": 5,
    "max_new_tokens": 300,
    "temperature": 0.8,
    "gen_top_k": 200,
}


class ModelCfg:
    pass


def load_checkpoint(ckpt_dir):
    ckpt_path = os.path.join(ckpt_dir, config.checkpoint_name)
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"no checkpoint at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")  # cpu load then move
    mc = ModelCfg()
    for k, v in ckpt["model_cfg"].items():
        setattr(mc, k, v)
    model = GPT(mc)
    # the checkpoint was trained on mps, where model.py uses manual attention and
    # registers a per-layer causal "mask" buffer. cpu serving uses fused SDPA
    # (flash), which does not register that buffer, so those keys load as
    # "unexpected". SDPA does not use self.mask, so dropping them is safe --
    # strict=False, then fail loudly on ANY other mismatch.
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    stray = [k for k in unexpected if not k.endswith(".attn.mask")]
    if missing or stray:
        raise SystemExit(
            f"state_dict mismatch: missing={missing} unexpected={stray}"
        )
    model.to(config.device)
    model.eval()
    # v4.2.0 ckpts carry eos_token_id; pre-EOS ckpts (v4.1.0) do not -> None.
    eos_token_id = ckpt.get("eos_token_id")
    return model, mc.block_size, eos_token_id


def _corpus_fingerprint(retrieval_dir):
    # fingerprint over the corpus files (name + size + mtime): if the corpus is
    # unchanged the cached retriever is valid; any edit invalidates it.
    parts = []
    for path in sorted(glob.glob(os.path.join(retrieval_dir, "*.jsonl"))):
        st = os.stat(path)
        parts.append(f"{os.path.basename(path)}:{st.st_size}:{int(st.st_mtime)}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def build_retriever(retrieval_dir, hybrid):
    # import guard: retriever_hybrid.py is optional, no hard dependency
    if hybrid:
        try:
            from retriever_hybrid import WhiskyRetrieverHybrid

            print("using hybrid retriever")
            return WhiskyRetrieverHybrid(retrieval_dir)
        except ImportError:
            print(
                "WARNING: --hybrid requested but retriever_hybrid.py is not "
                "importable; falling back to bm25 WhiskyRetriever"
            )
    # disk cache: reload a previously-built retriever if the corpus is unchanged,
    # turning a ~minutes bm25 rebuild into a ~seconds pickle load on restart. the
    # cache must never crash the server, so every failure falls back to a rebuild.
    cache_path = os.path.join(
        CACHE_DIR, f"retriever_v4_{_corpus_fingerprint(retrieval_dir)}.pkl"
    )
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                r = pickle.load(f)
            print(f"loaded retriever from cache {cache_path}")
            return r
        except Exception as e:  # noqa: BLE001
            print(f"cache load failed ({e}); rebuilding")
    r = WhiskyRetriever(retrieval_dir)
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = cache_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache_path)
        print(f"cached retriever -> {cache_path}")
    except Exception as e:  # noqa: BLE001
        print(f"cache write failed ({e}); continuing without cache")
    return r


def build_prompt(question, passages, max_new_tokens):
    # EXACT grounded format from finetune_v4.py: prompt + generation must fit the
    # 512 block, so ref_budget = block_size - max_new_tokens - header - tail - margin.
    header = f"Instruction: {question}\nReference notes:\n"
    tail = "Response:"
    overhead = len(TOK.encode(header).ids) + len(TOK.encode(tail).ids)
    budget = BLOCK_SIZE - max_new_tokens - overhead - PROMPT_MARGIN
    if budget <= 0:
        return f"Instruction: {question}\n{tail}"  # ungrounded fallback

    ctx = ""
    used = 0
    for p in passages:
        snippet = "- " + " ".join(p.split()) + "\n"
        n = len(TOK.encode(snippet).ids)
        if used + n > budget:
            remaining = budget - used
            if remaining >= MIN_SNIPPET_TOKENS:
                ids = TOK.encode(snippet).ids[:remaining]
                ctx += TOK.decode(ids).rstrip() + "\n"
            break
        ctx += snippet
        used += n
    return header + ctx + tail


@torch.no_grad()
def generate(prompt, max_new_tokens, temperature, gen_top_k,
              no_repeat_ngram=NO_REPEAT_NGRAM):
    ids = TOK.encode(prompt).ids
    if len(ids) > BLOCK_SIZE:
        ids = ids[-BLOCK_SIZE:]  # safety net; build_prompt budgets against this
    x = torch.tensor([ids], dtype=torch.long, device=config.device)
    start_len = x.size(1)
    for _ in range(max_new_tokens):
        x_cond = x[:, -BLOCK_SIZE:]
        logits, _ = MODEL(x_cond)
        logits = logits[:, -1, :] / temperature
        gen_ids = x[0, start_len:].tolist()
        logits = _ban_repeat_ngrams(logits, gen_ids, no_repeat_ngram)
        if gen_top_k is not None:
            v, _ = torch.topk(logits, min(gen_top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        # primary stop: the trained <EOS> token. checked by TOKEN ID (cheap, and
        # correct: TOK.decode() drops the special token so a string match can't
        # see it). do NOT append it; the answer is everything generated so far.
        if EOS_TOKEN_ID is not None and int(nxt.item()) == EOS_TOKEN_ID:
            return TOK.decode(x[0, start_len:].tolist())
        x = torch.cat((x, nxt), dim=1)
        # fallback stop: cut at the earliest scaffolding marker the model
        # emits (the only stop for pre-EOS ckpts; a safety net once EOS is
        # trained, and the fix for the v4.3.4 template-leak: a leaked
        # "Reference notes:"/"Instruction:"/"Response:" now truncates the
        # answer instead of running to max_new_tokens).
        text = TOK.decode(x[0, start_len:].tolist())
        stop = _find_stop(text)
        if stop is not None:
            return text[:stop]
    return TOK.decode(x[0, start_len:].tolist())


def answer(question, top_k, max_new_tokens, temperature, gen_top_k):
    retrieved = RETRIEVER.retrieve(question, top_k=top_k)
    prompt = build_prompt(question, retrieved, max_new_tokens)
    text = generate(prompt, max_new_tokens, temperature, gen_top_k)
    return {"answer": text.strip(), "retrieved": retrieved}


class Handler(BaseHTTPRequestHandler):
    def _authorized(self):
        # constant-time bearer-token check
        header = self.headers.get("Authorization", "")
        prefix = "Bearer "
        if not header.startswith(prefix):
            return False
        supplied = header[len(prefix):].strip()
        return bool(AUTH_TOKEN) and hmac.compare_digest(supplied, AUTH_TOKEN)

    def do_GET(self):
        # ungated browser landing page (page.html), read fresh each GET so ui
        # tweaks need no restart. serves no model output; the query (POST) is
        # still bearer-gated. falls back to a plain readiness line if page.html
        # is missing (so GET still works as a health check).
        try:
            with open(PAGE_PATH, "r", encoding="utf-8") as f:
                html = f.read()
        except OSError:
            body = (
                f"whisky-gpt {MODEL_VERSION} ready. POST with a bearer token.\n"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = html.replace("__VERSION__", MODEL_VERSION).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not self._authorized():
            self.send_error(401, "unauthorized")
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "bad json")
            return
        q = req.get("question", "").strip()
        if not q:
            self.send_error(400, "no question")
            return
        result = answer(
            q,
            req.get("top_k", GEN_DEFAULTS["top_k"]),
            req.get("max_new_tokens", GEN_DEFAULTS["max_new_tokens"]),
            req.get("temperature", GEN_DEFAULTS["temperature"]),
            req.get("gen_top_k", GEN_DEFAULTS["gen_top_k"]),
        )
        payload = json.dumps(result).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        # quiet the default per-request logging
        return


def main():
    global MODEL, TOK, RETRIEVER, BLOCK_SIZE, EOS_TOKEN_ID
    if not AUTH_TOKEN:
        raise SystemExit("refusing to start: HARLAN_GPT_TOKEN env var is not set")
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default=CKPT_DIR_DEFAULT)
    p.add_argument("--retrieval_dir", default=RETRIEVAL_DIR_DEFAULT)
    p.add_argument("--tokenizer_dir", default=TOKENIZER_DIR)
    p.add_argument("--port", type=int, default=8799)
    p.add_argument(
        "--hybrid",
        action="store_true",
        help="use WhiskyRetrieverHybrid if retriever_hybrid.py is importable",
    )
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    print("loading model (cpu) ...")
    MODEL, BLOCK_SIZE, EOS_TOKEN_ID = load_checkpoint(args.ckpt_dir)
    # v3.1.0: load the full tokenizer.json (has <EOS> in added_tokens); NOT
    # vocab.json+merges.txt, which lack the special token (see header note).
    TOK = Tokenizer.from_file(os.path.join(args.tokenizer_dir, "tokenizer.json"))
    # fail loudly if the tokenizer and checkpoint disagree about <EOS>: a silent
    # mismatch here would defeat the entire EOS stop.
    if EOS_TOKEN_ID is not None:
        vs = TOK.get_vocab_size()
        assert vs == EOS_TOKEN_ID + 1, (
            f"tokenizer vocab {vs} != eos_token_id+1 {EOS_TOKEN_ID + 1}; "
            "wrong tokenizer for this checkpoint"
        )
        assert TOK.encode("<EOS>").ids == [EOS_TOKEN_ID], (
            f"<EOS> did not encode to [{EOS_TOKEN_ID}]: {TOK.encode('<EOS>').ids}; "
            "wrong tokenizer loader (vocab.json has no <EOS>)"
        )

    print("building whisky retriever (once) ...")
    RETRIEVER = build_retriever(args.retrieval_dir, args.hybrid)

    print(f"ready. serving {MODEL_VERSION} (eos={EOS_TOKEN_ID}) "
          f"on 127.0.0.1:{args.port}")
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
