"""Stage 5 TP-versus-layer-split numeric control for Qwen3.8-Flash-Next.

Run the capture modes separately, in an exclusive two-GPU window; TP workers and
an LS model cannot coexist in one process. Both modes use the same fixed prompt,
prefill it through the penultimate token, then greedily force each argmax token
back into the cache. Only the logical-vocabulary top-k is saved, which keeps the
artifacts small while retaining enough coordinates for a useful comparison.

From the repository root:

  env PYTHONPATH=$PWD \
      TORCH_EXTENSIONS_DIR=$HOME/.cache/qwen38-flash-exl3-rocm/torch_extensions \
      TRITON_CACHE_DIR=$HOME/.cache/qwen38-flash-exl3-rocm/triton \
      PYTORCH_ROCM_ARCH=gfx1201 HIP_VISIBLE_DEVICES=0,1 EXL3_DIR=$PWD \
      /home/douglasbrown/vllm-test-env/bin/python tests/tp_logits_control.py \
      --mode tp --save /tmp/qwen38-tp-logits.pt

  # Run after the TP process has completely exited:
  env PYTHONPATH=$PWD \
      TORCH_EXTENSIONS_DIR=$HOME/.cache/qwen38-flash-exl3-rocm/torch_extensions \
      TRITON_CACHE_DIR=$HOME/.cache/qwen38-flash-exl3-rocm/triton \
      PYTORCH_ROCM_ARCH=gfx1201 HIP_VISIBLE_DEVICES=0,1 EXL3_DIR=$PWD \
      /home/douglasbrown/vllm-test-env/bin/python tests/tp_logits_control.py \
      --mode ls --save /tmp/qwen38-ls-logits.pt

  # CPU-only; no model is loaded:
  python tests/tp_logits_control.py --mode compare \
      --tp-result /tmp/qwen38-tp-logits.pt --ls-result /tmp/qwen38-ls-logits.pt

This file is deliberately host-safe to import: EXL3 and CUDA are only used by
capture modes, after argument parsing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "/home/douglasbrown/Models/Qwen3.8-Flash-Next-exl3-bpw3"
PROMPT = (
    "Once upon a time, a small robot named Sparky discovered a mysterious glowing "
    "door in the middle of the forest. When Sparky opened it,"
)
ARTIFACT_PROTOCOL = "qwen38-flash-next-tp-ls-logits-control-v1"


def parse_use_devices(value: str) -> list[float]:
    try:
        devices = [float(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--use-dev must be comma-separated GiB values") from error
    if len(devices) != 2 or any(value <= 0 for value in devices):
        raise argparse.ArgumentTypeError("--use-dev must give two positive GiB values, e.g. 30,30")
    return devices


def cuda_device(device: int | str | torch.device) -> torch.device:
    """Normalize EXL3's integer device fields without creating a CUDA context."""
    return torch.device("cuda", device) if isinstance(device, int) else torch.device(device)


def step_params(cache: Any, past_len: int, recurrent_states: Any, cache_tokens: int) -> dict[str, Any]:
    params = {
        "attn_mode": "flash_attn",
        "cache": cache,
        "past_len": past_len,
        "batch_shape": (1, cache_tokens),
    }
    if recurrent_states is not None:
        params["recurrent_states"] = recurrent_states
    return params


def capture(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("capture requires two visible HIP/CUDA devices (set HIP_VISIBLE_DEVICES=0,1)")
    if args.cache_tokens < args.steps + 2:
        raise SystemExit("--cache-tokens must leave room for the prompt and all forced steps")

    # Keep normal imports out of compare mode so it can run safely on a CPU host.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from exllamav3 import Cache, Config, Model, Tokenizer

    torch.manual_seed(0)
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    cache = Cache(model=model, max_num_tokens=args.cache_tokens, max_batch_size=1)
    loaded = False

    try:
        load_args: dict[str, Any] = {
            "tensor_p": args.mode == "tp",
            "use_per_device": args.use_dev,
            "max_chunk_size": args.chunk,
            "max_batch_size": 1,
        }
        # NCCL means RCCL on this ROCm installation. LS must not initialize a TP backend.
        if args.mode == "tp":
            load_args["tp_backend"] = "nccl"
        model.load(**load_args)
        loaded = True

        if args.top_k > config.vocab_size:
            raise SystemExit(f"--top-k ({args.top_k}) exceeds logical vocabulary ({config.vocab_size})")
        prompt_ids = tokenizer.encode(PROMPT, add_bos=True).cpu()
        if prompt_ids.ndim != 2 or prompt_ids.shape[1] < 2:
            raise RuntimeError(f"expected a nontrivial [1, n] prompt encoding, got {tuple(prompt_ids.shape)}")
        required_tokens = prompt_ids.shape[1] + args.steps
        if required_tokens > args.cache_tokens:
            raise SystemExit(
                f"prompt ({prompt_ids.shape[1]} tokens) + --steps ({args.steps}) exceeds "
                f"--cache-tokens ({args.cache_tokens})"
            )

        # Prefill stops before the final prompt token. The loop forwards that token first,
        # then forwards exactly the argmax token saved for each following position.
        prefill_params = step_params(cache, past_len=0, recurrent_states=None, cache_tokens=args.cache_tokens)
        with torch.inference_mode():
            model.prefill(input_ids=prompt_ids[:, :-1], params=prefill_params)
            recurrent_states = prefill_params.get("recurrent_states")
            input_ids = prompt_ids[:, -1:].contiguous()
            past_len = prompt_ids.shape[1] - 1
            records: list[dict[str, Any]] = []

            output_device = (
                cuda_device(model.tp_output_device)
                if args.mode == "tp"
                else cuda_device(model.output_device)
            )
            for step in range(args.steps):
                logits = model.forward(
                    input_ids=input_ids,
                    params=step_params(cache, past_len, recurrent_states, args.cache_tokens),
                )
                if logits.device != output_device:
                    raise RuntimeError(
                        f"{args.mode} logits arrived on {logits.device}, expected output device {output_device}"
                    )
                logical_logits = logits[0, -1, :config.vocab_size]
                if logical_logits.numel() != config.vocab_size:
                    raise RuntimeError("model did not return the complete logical vocabulary")
                values, indices = torch.topk(logical_logits, k=args.top_k, dim=-1)
                token_id = int(torch.argmax(logical_logits).item())
                records.append({
                    "step": step,
                    "input_token_id": int(input_ids.item()),
                    "token_id": token_id,
                    # CPU tensors make the artifact portable and synchronize the output rank.
                    "top_values": values.detach().cpu(),
                    "top_indices": indices.detach().cpu(),
                })
                input_ids = torch.tensor([[token_id]], dtype=prompt_ids.dtype)
                past_len += 1

            # Direct model users own recurrent-state cleanup (Generator normally does this).
            if recurrent_states:
                for state in recurrent_states:
                    state.free()

        artifact = {
            "protocol": ARTIFACT_PROTOCOL,
            "mode": args.mode,
            "model": os.path.abspath(args.model),
            "prompt": PROMPT,
            "prompt_token_ids": prompt_ids,
            "logical_vocab": config.vocab_size,
            "top_k": args.top_k,
            "steps": records,
        }
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, args.save)
        summary = {
            "mode": args.mode,
            "save": str(args.save),
            "logical_vocab": config.vocab_size,
            "steps": len(records),
            "top_k": args.top_k,
            "argmax_token_ids": [record["token_id"] for record in records],
        }
        print(json.dumps(summary, sort_keys=True))
    finally:
        if loaded:
            model.unload()


def load_artifact(path: Path, expected_mode: str) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"artifact does not exist: {path}")
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("protocol") != ARTIFACT_PROTOCOL:
        raise SystemExit(f"{path} is not a {ARTIFACT_PROTOCOL} artifact")
    if artifact.get("mode") != expected_mode:
        raise SystemExit(f"{path} has mode {artifact.get('mode')!r}, expected {expected_mode!r}")
    if artifact.get("top_k", 0) < 5 or not artifact.get("steps"):
        raise SystemExit(f"{path} has no comparable top-k step data")
    return artifact


def compare(args: argparse.Namespace) -> None:
    tp = load_artifact(args.tp_result, "tp")
    ls = load_artifact(args.ls_result, "ls")
    for field in ("logical_vocab", "top_k", "prompt", "prompt_token_ids"):
        if field == "prompt_token_ids":
            same = torch.equal(tp[field], ls[field])
        else:
            same = tp[field] == ls[field]
        if not same:
            raise SystemExit(f"artifacts disagree on {field}; they are not the same control")
    if len(tp["steps"]) != len(ls["steps"]):
        raise SystemExit("artifacts have different numbers of steps")

    report_steps = []
    overall_max = None
    for tp_step, ls_step in zip(tp["steps"], ls["steps"], strict=True):
        if tp_step["step"] != ls_step["step"]:
            raise SystemExit("artifact step numbering differs")
        tp_indices = tp_step["top_indices"].tolist()
        ls_indices = ls_step["top_indices"].tolist()
        tp_positions = {token: position for position, token in enumerate(tp_indices)}
        ls_positions = {token: position for position, token in enumerate(ls_indices)}
        common = sorted(tp_positions.keys() & ls_positions.keys())
        if common:
            tp_values = tp_step["top_values"][[tp_positions[token] for token in common]].float()
            ls_values = ls_step["top_values"][[ls_positions[token] for token in common]].float()
            max_abs = float((tp_values - ls_values).abs().max())
            overall_max = max(max_abs, overall_max) if overall_max is not None else max_abs
        else:
            max_abs = None

        tp_top5 = set(tp_indices[:5])
        ls_top5 = set(ls_indices[:5])
        report_steps.append({
            "step": tp_step["step"],
            "input_token_equal": tp_step["input_token_id"] == ls_step["input_token_id"],
            "top1_agreement": tp_indices[0] == ls_indices[0],
            "top5_overlap": len(tp_top5 & ls_top5),
            "top5_union": len(tp_top5 | ls_top5),
            "intersection_size": len(common),
            "max_abs_logit_diff_on_intersection": max_abs,
            "argmax_token_equal": tp_step["token_id"] == ls_step["token_id"],
            "tp_argmax_token_id": tp_step["token_id"],
            "ls_argmax_token_id": ls_step["token_id"],
        })

    summary = {
        "tp_result": str(args.tp_result),
        "ls_result": str(args.ls_result),
        "steps": report_steps,
        "overall_max_abs_logit_diff_on_intersection": overall_max,
        "all_input_tokens_equal": all(step["input_token_equal"] for step in report_steps),
        "all_top1_agree": all(step["top1_agreement"] for step in report_steps),
        "all_argmax_tokens_equal": all(step["argmax_token_equal"] for step in report_steps),
    }
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("tp", "ls", "compare"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-tokens", type=int, default=16384)
    parser.add_argument("--use-dev", type=parse_use_devices, default=parse_use_devices("30,30"))
    parser.add_argument("--save", type=Path, help="output .pt artifact (required for tp and ls)")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--tp-result", type=Path, help="TP artifact for compare mode")
    parser.add_argument("--ls-result", type=Path, help="layer-split artifact for compare mode")
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.top_k < 5:
        parser.error("--top-k must be at least 5")
    if args.chunk < 1:
        parser.error("--chunk must be positive")
    if args.mode in ("tp", "ls"):
        if args.save is None:
            parser.error("--save is required for tp and ls capture modes")
        capture(args)
    else:
        if args.tp_result is None or args.ls_result is None:
            parser.error("--tp-result and --ls-result are required for compare mode")
        compare(args)


if __name__ == "__main__":
    main()
