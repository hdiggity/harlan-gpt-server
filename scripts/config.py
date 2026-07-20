# config.py
# central configuration; every other script imports from this.
# v2.0.0: gpt2 tokenizer, ~70M GPT-2-small architecture, trained on the M4
# (MPS, bf16). sized to fit 16GB unified memory.

import torch

# data
dataset_name = ""
data_url = ""
val_split = 0.01
tokenizer_name = "gpt2"  # v2 swap from o200k_base: smaller 50K vocab frees
                         # memory for a bigger transformer, which matters on 16GB

# data mixing / focus
# folders under data/<dataset_name>/domains/ become domains. nesting is
# supported: domains/<topic>/<tier>/*.txt yields the domain "<topic>-<tier>".
# weights resolve per domain: an exact full-name entry wins, otherwise the
# weights of its name parts (topic, tier) multiply, so a few tier rules
# weight every topic at once.
domain_weights = {"high": 3.0, "medium": 1.0, "low": 0.15}  

# domain conditioning
# prepend a per-domain control token to every training sequence so the model
# learns domain-conditioned text; prime generation with a domain's tag to
# steer it. reserved_domain_slots fixes room in the vocab for domains, so new
# ones can be added later without resizing the model or retokenizing old data.
# do not change reserved_domain_slots after a run starts (it sets vocab size).
domain_conditioning = True
reserved_domain_slots = 256

# model architecture (overridden by auto_size when enabled)
# v2 target: ~70M params total, GPT-2-small class, sized to fit 16GB with bf16
block_size = 256
n_layer = 8
n_head = 8
n_embd = 512  # must be divisible by n_head
dropout = 0.1
bias = False

# training
batch_size = 8  # physical batch; keep small to fit memory, raise accumulation
gradient_accumulation_steps = 16  # effective batch = batch_size * this = 128
max_iters = 25000
learning_rate = 3e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate schedule (cosine with linear warmup)
decay_lr = True
warmup_iters = 1000
lr_decay_iters = 50000
min_lr = 3e-5

# auto-sizing: resize model and schedule to dataset token count.
# keep OFF for v2 so the manual architecture above is used (auto_size silently
# overriding the manual config was a v1 failure mode).
auto_size = False

# early stopping
early_stopping = True
patience = 10
min_delta = 1e-3

# auto resume/restart based on the data fingerprint
auto_resume = True

# evaluation and logging
eval_interval = 1000
eval_iters = 5
log_interval = 50

# system: cuda > mps > cpu
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# dtype: bf16 on cuda OR mps (the v2 change). bf16 halves memory for weights,
# grads, and activations, roughly doubling the model size that fits in 16GB.
# cpu stays float32 (bf16 on cpu is slow and unsupported for training here).
if device == "cuda" and torch.cuda.is_bf16_supported():
    dtype = "bfloat16"
elif device == "mps":
    dtype = "float32"
else:
    dtype = "float32"

compile_model = False  # torch.compile is unreliable on mps; keep off

# paths
data_dir = "../data.nosync"
out_dir = "runs/v2.0.0/checkpoints"
checkpoint_name = "ckpt.pt"


def derive_architecture(num_train_tokens):
    # pick model size and training length from dataset size.
    # only used when auto_size is True; v2 uses the manual config above.
    tiers = [
        # (token_ceiling, n_layer, n_head, n_embd, block_size, max_iters)
        (5e5, 3, 4, 128, 64, 2000),
        (5e6, 4, 4, 256, 128, 5000),
        (5e7, 6, 6, 384, 256, 20000),
        (5e8, 12, 12, 768, 256, 50000),
        (float("inf"), 16, 16, 1024, 256, 100000),
    ]
    for ceiling, n_layer, n_head, n_embd, block_size, max_iters in tiers:
        if num_train_tokens < ceiling:
            return {
                "n_layer": n_layer,
                "n_head": n_head,
                "n_embd": n_embd,
                "block_size": block_size,
                "max_iters": max_iters,
                "lr_decay_iters": max_iters,
            }


def effective_domain_weights(progress, domains):
    # weight by the relevance tier present in the domain name. the tier is
    # whichever of high/medium/low appears as a "-"-separated part. exact
    # full-name entry in domain_weights still wins for per-domain overrides.
    tiers = ("high", "medium", "low")
    out = {}
    for d in domains:
        if d in domain_weights:
            out[d] = float(domain_weights[d])
            continue
        w = 1.0
        for part in d.split("-"):
            if part in tiers and part in domain_weights:
                w = float(domain_weights[part])
                break
        out[d] = w
    return out
