# ask_server_v4.py
# live serving wrapper for the v4.0.0-instruct whisky RAG model. builds the
# model + whisky bm25 retriever ONCE at startup, then serves questions over
# http so queries are instant.
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
# generation stops at the first "\nInstruction:" the model emits.
# POST {"question": ...} + Authorization: Bearer <token> -> {"answer", "retrieved"}

import argparse
import glob
import hashlib
import hmac
import json
import os
import pickle
from http.server import BaseHTTPRequestHandler, HTTPServer

import config

# force cpu inference: never share MPS with a training run (v3 pattern).
config.device = "cpu"

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from model import GPT  # noqa: E402
from retriever import WhiskyRetriever  # noqa: E402
from tokenizers import ByteLevelBPETokenizer  # noqa: E402

CKPT_DIR_DEFAULT = os.path.expanduser(
    "~/services/harlan-gpt/whisky-gpt/gpt/runs/v4.1.0/checkpoints"
)
RETRIEVAL_DIR_DEFAULT = os.path.expanduser(
    "~/services/harlan-gpt/whisky-gpt/gpt/runs/v4.0.0/retrieval"
)
TOKENIZER_DIR = os.path.expanduser(
    "~/services/harlan-gpt/whisky-gpt/tokenizer/runs/v3.0.0/drinks-24576"
)

MODEL_VERSION = "v4.0.0-instruct"

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

STOP_MARKER = "\nInstruction:"  # model echoing a new prompt = end of answer
PROMPT_MARGIN = 8  # tokens of slack for encode-concat boundary effects
MIN_SNIPPET_TOKENS = 24  # skip a truncated ref shorter than this

# globals loaded once at startup
MODEL = None
TOK = None
RETRIEVER = None
BLOCK_SIZE = None
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
    return model, mc.block_size


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
def generate(prompt, max_new_tokens, temperature, gen_top_k):
    ids = TOK.encode(prompt).ids
    if len(ids) > BLOCK_SIZE:
        ids = ids[-BLOCK_SIZE:]  # safety net; build_prompt budgets against this
    x = torch.tensor([ids], dtype=torch.long, device=config.device)
    start_len = x.size(1)
    for _ in range(max_new_tokens):
        x_cond = x[:, -BLOCK_SIZE:]
        logits, _ = MODEL(x_cond)
        logits = logits[:, -1, :] / temperature
        if gen_top_k is not None:
            v, _ = torch.topk(logits, min(gen_top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, nxt), dim=1)
        # stop condition: cut at the first "\nInstruction:" the model emits
        text = TOK.decode(x[0, start_len:].tolist())
        if STOP_MARKER in text:
            return text.split(STOP_MARKER)[0]
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
    global MODEL, TOK, RETRIEVER, BLOCK_SIZE
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
    MODEL, BLOCK_SIZE = load_checkpoint(args.ckpt_dir)
    TOK = ByteLevelBPETokenizer(
        os.path.join(args.tokenizer_dir, "vocab.json"),
        os.path.join(args.tokenizer_dir, "merges.txt"),
    )

    print("building whisky retriever (once) ...")
    RETRIEVER = build_retriever(args.retrieval_dir, args.hybrid)

    print(f"ready. serving {MODEL_VERSION} on 127.0.0.1:{args.port}")
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
