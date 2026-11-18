"""Read-only investigation: can TransformerLens load gated models via ungated HF mirrors,
and what is the peak GPU memory of fwd+bwd at batch 4 x 48 tokens (fp32)?

Run inside the bootstrap venv on a GPU node, after prefetch_models.sh:
    python slurm/investigate_models.py
Prints RESULT_JSON lines.
"""
import gc
import json
import os
import sys
import time
import traceback

import torch
import transformer_lens
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

print("ENV", json.dumps({"torch": torch.__version__, "tl": transformer_lens.__version__,
                         "transformers": transformers.__version__,
                         "gpu": torch.cuda.get_device_name(0),
                         "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)}), flush=True)

CASES = [
    ("meta-llama/Llama-3.2-1B", "unsloth/Llama-3.2-1B"),
    ("google/gemma-3-1b-pt", "unsloth/gemma-3-1b-pt"),
    ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B"),
]
only = sys.argv[1:]
if only:
    CASES = [c for c in CASES if any(o in c[0] for o in only)]


def emit(**kw):
    print("RESULT_JSON " + json.dumps(kw), flush=True)


def err(e):
    return f"{type(e).__name__}: {str(e)[:400]}"


def gb(x):
    return round(x / 2**30, 3)


def free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak_fwd_bwd(model, kind, vocab, full_param_grads=False):
    """batch 4 x 48 random tokens; loss = mean next-token CE. kind='tl' or 'hf'."""
    free()
    base = torch.cuda.memory_allocated()
    for p in model.parameters():
        p.requires_grad_(full_param_grads)
    toks = torch.randint(0, vocab, (4, 48), device="cuda")
    try:
        if kind == "tl":
            if not full_param_grads:
                # attribution-patching style: grads wrt activations only (params frozen)
                model.add_hook("hook_embed", lambda a, hook: a.requires_grad_(True) if not a.requires_grad else a)
            logits = model(toks)
        else:
            logits = model(toks).logits
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                                 toks[:, 1:].reshape(-1))
        loss.backward()
        torch.cuda.synchronize()
        return {"ok": True, "weights_gb": gb(base), "peak_alloc_gb": gb(torch.cuda.max_memory_allocated()),
                "peak_reserved_gb": gb(torch.cuda.max_memory_reserved())}
    except torch.cuda.OutOfMemoryError as e:
        return {"ok": False, "error": "CUDA OOM: " + str(e)[:200], "weights_gb": gb(base),
                "peak_alloc_gb": gb(torch.cuda.max_memory_allocated())}
    finally:
        if kind == "tl":
            model.reset_hooks()
        for p in model.parameters():
            p.grad = None


for official, mirror in CASES:
    print(f"\n===== {official}  (mirror {mirror}) =====", flush=True)
    t = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(mirror)
        hf = AutoModelForCausalLM.from_pretrained(mirror, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        hf.eval()
        emit(case=official, step="hf_load_cpu", ok=True, arch=hf.config.architectures, secs=round(time.time() - t, 1),
             n_params=sum(p.numel() for p in hf.parameters()))
    except Exception as e:  # noqa: BLE001
        emit(case=official, step="hf_load_cpu", ok=False, error=err(e))
        traceback.print_exc()
        continue

    # (0) what happens if we just ask TL for the official name with no hf_model (needs gated access)
    try:
        HookedTransformer.from_pretrained(official, device="cpu")
        emit(case=official, step="tl_official_no_hf_model", ok=True)
    except Exception as e:  # noqa: BLE001
        emit(case=official, step="tl_official_no_hf_model", ok=False, error=err(e))

    # (1) the mirror id directly
    try:
        HookedTransformer.from_pretrained(mirror, device="cpu")
        emit(case=official, step="tl_mirror_name", ok=True)
    except Exception as e:  # noqa: BLE001
        emit(case=official, step="tl_mirror_name", ok=False, error=err(e))

    # (2) official name + hf_model + tokenizer, several flag combos
    combos = [
        ("default", dict()),
        ("fold_ln=False,center_writing_weights=False,center_unembed=False",
         dict(fold_ln=False, center_writing_weights=False, center_unembed=False)),
        ("no_processing", "no_processing"),
    ]
    tl_model = None
    working = None
    for label, kw in combos:
        try:
            t = time.time()
            if kw == "no_processing":
                m = HookedTransformer.from_pretrained_no_processing(
                    official, hf_model=hf, tokenizer=tok, device="cuda", dtype=torch.float32)
            else:
                m = HookedTransformer.from_pretrained(
                    official, hf_model=hf, tokenizer=tok, device="cuda", dtype=torch.float32, **kw)
            m.eval()
            with torch.no_grad():
                toks = m.to_tokens("The capital of France is")
                lg = m(toks)
                # compare to HF logits (fp32, cpu) as a correctness check
                ref = hf(toks.cpu()).logits
                d = (lg.float().cpu() - ref).abs().max().item()
                # logits are shift-invariant under center_unembed: compare log-probs
                d_lp = (lg.float().log_softmax(-1).cpu() - ref.log_softmax(-1)).abs().max().item()
            emit(case=official, step="tl_official+hf_model", flags=label, ok=True,
                 secs=round(time.time() - t, 1), max_logit_diff=round(d, 5),
                 max_logprob_diff_vs_hf=round(d_lp, 6), gpu_mem_gb=gb(torch.cuda.memory_allocated()))
            if tl_model is None:
                tl_model, working = m, label
            else:
                del m
            free()
        except Exception as e:  # noqa: BLE001
            emit(case=official, step="tl_official+hf_model", flags=label, ok=False, error=err(e))
            free()

    # (3) peak memory: HF model (on GPU) and TL model
    del hf
    gc.collect()
    free()
    if tl_model is not None:
        vocab = tl_model.cfg.d_vocab
        emit(case=official, step="mem_tl_frozen_params_act_grads", flags=working,
             **peak_fwd_bwd(tl_model, "tl", vocab))
        emit(case=official, step="mem_tl_full_param_grads", flags=working,
             **peak_fwd_bwd(tl_model, "tl", vocab, full_param_grads=True))
        del tl_model
        free()
    # HF-only fp32 on GPU (independent of TL; also answers memory for gemma if TL cannot load it)
    try:
        hfg = AutoModelForCausalLM.from_pretrained(mirror, torch_dtype=torch.float32,
                                                   low_cpu_mem_usage=True).to("cuda")
        emit(case=official, step="mem_hf_frozen_params_act_grads",
             **peak_fwd_bwd(hfg, "hf", hfg.config.vocab_size))
        emit(case=official, step="mem_hf_full_param_grads",
             **peak_fwd_bwd(hfg, "hf", hfg.config.vocab_size, full_param_grads=True))
        del hfg
    except Exception as e:  # noqa: BLE001
        emit(case=official, step="mem_hf", ok=False, error=err(e))
    free()
print("INVESTIGATE_DONE", flush=True)
