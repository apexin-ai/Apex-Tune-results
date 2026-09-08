# Copyright 2026 Apex Intelligence
# Copyright 2026 Recursive
# Copyright 2025 Andrej Karpathy
# SPDX-License-Identifier: Apache-2.0
"""
Nanochat pretraining script: single-GPU, single-file.
Reproduces our discovered near-SOTA solution on the fixed 300-second single-B200 benchmark.

Key Architectural & Systems Innovations:
1. Fused Dual N-Gram Lookup:
   - Fuses two n-gram table lookups (bigram 512 + shared trigram 2048), concatenation,
     active-row discovery, and row registration into a single Triton kernel.
2. Compact Touched-Row Backward Pass:
   - The backward pass directly reuses the resulting row-to-slot mapping,
     eliminating duplicated indexing and updating only the sparse rows touched by the current batch.
   - Preserves sparsity end-to-end and reduces peak training memory from ~177.7 GiB to 140.6 GiB (20.9% reduction).
3. Normalized Attention Input Reuse:
   - Saves normalized post-layer-4 attention input and reuses it for Q/K/V and attention gates
     in layers 5, 6, and 7, while residual and MLP streams continue to consume the current stream.
4. Recipe Optimization:
   - TINY_DIV=8, MLP depth profile (3, 3, 3, 4, 4, 5, 5, 5), MATRIX_LR=0.035, WARMDOWN_RATIO=0.90,
     Muon momentum 0.80.

Benchmark Performance (Fixed 300-second budget on single NVIDIA B200):
- Discovered Solution 10-Seed Mean Val BPB: 0.892792 (Std: 0.000766, 95% CI: [0.892244, 0.893340], Best: 0.891762)
- Discovered Solution 3-Seed Initial Mean: 0.892426 (Seeds: 42 -> 0.891762, 137 -> 0.892828, 271 -> 0.892688)
- Baseline Comparisons:
  * Recursive SuperIntelligence (June 2026): 0.910875 (10-seed mean), 0.903891 (best)
  * Tencent Hunyuan Hyra (July 2026): 0.901543
  * AutoTrust ScienceGuru (September 2026): 0.889522

Usage:
    uv run train.py
"""
import ast
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
import tempfile
import types

_script_dir = Path(__file__).resolve().parent
for _p in (str(_script_dir), str(_script_dir.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import statistics
import time
from dataclasses import asdict, dataclass

import torch
import torch._inductor.config as inductor_config  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice as triton_libdevice

# Keep default inductor settings for this compile-capture ablation.
import torch.nn as nn
import torch.nn.functional as F

_p1_stage_value = os.environ.get("P1_COMPACT_BACKWARD", "1")
if _p1_stage_value not in {"0", "1"}:
    raise ValueError("P1_COMPACT_BACKWARD must be 0 or 1")
P1_COMPACT_BACKWARD_ENABLED = _p1_stage_value == "1"

_p1_fused_lookup_value = os.environ.get("P1_FUSED_LOOKUP_COLLECT", "1")
if _p1_fused_lookup_value not in {"0", "1"}:
    raise ValueError("P1_FUSED_LOOKUP_COLLECT must be 0 or 1")
P1_FUSED_LOOKUP_COLLECT_ENABLED = _p1_fused_lookup_value == "1"

_P1_SCATTER_BLOCK_TOKENS = int(os.environ.get("P1_SCATTER_BLOCK_TOKENS", "8"))
if _P1_SCATTER_BLOCK_TOKENS not in {8, 16, 32}:
    raise ValueError("P1_SCATTER_BLOCK_TOKENS must be 8, 16, or 32")
_P1_SCATTER_NUM_WARPS = int(os.environ.get("P1_SCATTER_NUM_WARPS", "8"))
if _P1_SCATTER_NUM_WARPS not in {4, 8}:
    raise ValueError("P1_SCATTER_NUM_WARPS must be 4 or 8")
_P1_LOOKUP_BLOCK_TOKENS = int(os.environ.get("P1_LOOKUP_BLOCK_TOKENS", "8"))
if _P1_LOOKUP_BLOCK_TOKENS not in {4, 8, 16}:
    raise ValueError("P1_LOOKUP_BLOCK_TOKENS must be 4, 8, or 16")
_P1_LOOKUP_NUM_WARPS = int(os.environ.get("P1_LOOKUP_NUM_WARPS", "8"))
if _P1_LOOKUP_NUM_WARPS not in {4, 8}:
    raise ValueError("P1_LOOKUP_NUM_WARPS must be 4 or 8")


# Keep the engineering ladder explicit at the process boundary.  Each edge
# selects a concrete model/optimizer runtime mode; receipts are derived from
# the same immutable table rather than from ad-hoc booleans in the runner.
_P1_ENGINEERING_VARIANT_ALIASES = {
    "parent": "parent",
    "control": "parent",
    "formal_a": "parent",
    "e0": "parent",
    "child": "e1",
    "e1": "e1",
    "p2": "e1",
    "p2_forward_reuse": "e1",
    "e2": "e2",
    "e3": "e3",
    "e4": "e4",
    "p1_compact": "e4",
    "complete_p1": "e4",
    # Quality-bearing treatment aliases are deliberately separate from the
    # historical ``child`` alias, which names the first P2 diagnostic edge.
    "p1": "e4",
    "full_p1": "e4",
    "treatment": "e4",
}
_P1_ENGINEERING_VARIANT_EDGES = {
    "parent": "E0",
    "e1": "E1",
    "e2": "E2",
    "e3": "E3",
    "e4": "E4",
}
_P1_ENGINEERING_EDGE_VARIANTS = {
    "E0": "parent",
    "E1": "e1",
    "E2": "e2",
    "E3": "e3",
    "E4": "e4",
}
_P1_ENGINEERING_IMPLEMENTED_EDGES = frozenset({"E0", "E1", "E2", "E3", "E4"})
_P1_ENGINEERING_RUNTIME_MODES = {
    "E0": {
        "p1_compact_backward": False,
        "p2_forward_reuse": False,
        "p1_reduction": "none",
        "p1_fused_cleanup": False,
        "slot_dtype": None,
        "duplicate_reduction": False,
    },
    "E1": {
        "p1_compact_backward": False,
        "p2_forward_reuse": True,
        "p1_reduction": "none",
        "p1_fused_cleanup": False,
        "slot_dtype": None,
        "duplicate_reduction": False,
    },
    "E2": {
        "p1_compact_backward": True,
        "p2_forward_reuse": True,
        "p1_reduction": "index_add",
        "p1_fused_cleanup": False,
        "slot_dtype": "fp32",
        "duplicate_reduction": True,
    },
    "E3": {
        "p1_compact_backward": True,
        "p2_forward_reuse": True,
        "p1_reduction": "fp32_atomic",
        "p1_fused_cleanup": False,
        "slot_dtype": "fp32",
        "duplicate_reduction": True,
    },
    "E4": {
        "p1_compact_backward": True,
        "p2_forward_reuse": True,
        "p1_reduction": "fp32_atomic",
        "p1_fused_cleanup": True,
        "slot_dtype": "fp32",
        "duplicate_reduction": True,
    },
}


# Real-shape P1 uses separate fixed slot budgets for the two n-gram orders.
# The policy is intentionally a small, source-visible value: it is included in
# receipts and semantic manifests so a result cannot silently change slot
# footprint while retaining the same candidate source label.
_P1_COMPACT_SLOT_CAPACITY_POLICY = {
    # A 512x table reduces collisions enough to exceed the 64x parent's 78k
    # high-water budget (79,582 observed before the first optimizer update).
    "bigram": 90_000,
    # The previous 112k budget was only 400 rows above the largest observed
    # no-score batch. Keep enough headroom for a fresh shard/seed while the
    # receipt records the full stream's observed high-water mark.
    # One shared 2048x pair can approach one unique row per token. The fixed
    # contract has 72*2048=147,456 tokens, so this bound cannot overflow while
    # adding only 2,544 operational headroom rows per retained table.
    "trigram": 150_000,
}
_P1_COMPACT_SLOT_CAPACITY_POLICY_VERSION = (
    "bigram512_shared_trigram2048_strict_crn_slots_v1"
)


def _p1_normalize_capacity_policy(value):
    """Normalize legacy/int, tuple, or named/order mapping capacities.

    A mapping may contain only the orders present in a small direct optimizer
    test.  Full GPT construction has both orders and therefore fails closed if
    either required order is missing.  ``int`` remains the legacy API and
    applies one capacity to both orders.
    """
    if isinstance(value, bool):
        raise ValueError("compact touched-row capacity must be an integer, tuple, or mapping")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("compact touched-row capacity must be positive")
        normalized = {2: int(value), 3: int(value)}
    elif isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("compact per-order capacity tuple must be (bigram, trigram)")
        if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
            raise ValueError("compact per-order capacities must be positive integers")
        normalized = {2: int(value[0]), 3: int(value[1])}
    elif isinstance(value, Mapping):
        normalized = {}
        aliases = {
            2: 2,
            3: 3,
            "2": 2,
            "3": 3,
            "bigram": 2,
            "bigrams": 2,
            "trigram": 3,
            "trigrams": 3,
        }
        for key, item in value.items():
            try:
                order = aliases[key]
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    "compact capacity mapping keys must be 2/3 or bigram/trigram"
                ) from exc
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise ValueError("compact per-order capacities must be positive integers")
            if order in normalized and normalized[order] != int(item):
                raise ValueError(f"duplicate compact capacity for order {order}")
            normalized[order] = int(item)
        if not normalized:
            raise ValueError("compact capacity mapping cannot be empty")
    else:
        raise ValueError("compact touched-row capacity must be an integer, tuple, or mapping")
    identity_payload = {
        "version": _P1_COMPACT_SLOT_CAPACITY_POLICY_VERSION,
        "orders": {str(order): normalized[order] for order in sorted(normalized)},
        "slot_dtype": "fp32",
    }
    policy_id = hashlib.sha256(
        json.dumps(
            identity_payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    return normalized, identity_payload, policy_id


(
    _P1_COMPACT_SLOT_CAPACITY_BY_ORDER,
    _P1_COMPACT_SLOT_CAPACITY_POLICY_IDENTITY,
    _P1_COMPACT_SLOT_CAPACITY_POLICY_ID,
) = _p1_normalize_capacity_policy(_P1_COMPACT_SLOT_CAPACITY_POLICY)


def _resolve_p1_engineering_selection():
    """Resolve the explicit edge/legacy alias without permitting edge drift.

    Returns ``(requested_variant, variant, edge)``.  The legacy variant
    environment remains the default interface, while ``P1_ENGINEERING_EDGE``
    is authoritative when supplied.
    """
    raw_variant = os.environ.get("P1_ENGINEERING_VARIANT")
    raw_edge = os.environ.get("P1_ENGINEERING_EDGE")

    # Do not silently turn an unqualified no-score invocation into E1.  The
    # engineering ladder contains both diagnostic and quality-bearing child
    # edges, so the launch manifest must name the exact edge or alias.
    if raw_variant is None and raw_edge is None:
        raise ValueError(
            "P1_NO_SCORE_ENGINEERING requires explicit "
            "P1_ENGINEERING_EDGE=E0..E4 or P1_ENGINEERING_VARIANT"
        )

    if raw_variant is None:
        requested_variant = "child"
        variant = _P1_ENGINEERING_VARIANT_ALIASES[requested_variant]
    else:
        requested_variant = raw_variant.strip().lower()
        try:
            variant = _P1_ENGINEERING_VARIANT_ALIASES[requested_variant]
        except KeyError as exc:
            raise ValueError(
                "P1_ENGINEERING_VARIANT must be parent/formal_a, "
                "child/e1/p2, e2, e3, or e4/p1_compact/p1/full_p1/treatment"
            ) from exc

    if raw_edge is None:
        edge = _P1_ENGINEERING_VARIANT_EDGES[variant]
    else:
        edge = raw_edge.strip().upper()
        if edge not in _P1_ENGINEERING_EDGE_VARIANTS:
            raise ValueError("P1_ENGINEERING_EDGE must be one of: E0, E1, E2, E3, E4")
        edge_variant = _P1_ENGINEERING_EDGE_VARIANTS[edge]
        if raw_variant is not None and variant != edge_variant:
            raise ValueError(
                "P1_ENGINEERING_EDGE conflicts with P1_ENGINEERING_VARIANT: "
                f"{edge} selects {edge_variant}, got {requested_variant}"
            )
        variant = edge_variant
        if raw_variant is None:
            requested_variant = edge.lower()

    if edge not in _P1_ENGINEERING_IMPLEMENTED_EDGES:
        raise RuntimeError(f"P1_ENGINEERING_EDGE={edge} has no runtime mode")
    return requested_variant, variant, edge


def _p1_engineering_runtime_mode(edge):
    """Return a private copy of the selected edge's runtime contract."""
    try:
        mode = dict(_P1_ENGINEERING_RUNTIME_MODES[edge])
    except KeyError as exc:
        raise RuntimeError(f"P1_ENGINEERING_EDGE={edge} has no runtime mode") from exc
    mode["edge"] = edge
    mode["variant"] = _P1_ENGINEERING_EDGE_VARIANTS[edge]
    return mode


def _p1_canonical_json_bytes(value):
    """Serialize a diagnostic receipt without allowing unstable JSON values."""
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _emit_p1_admission_receipt(payload, *, env_name="P1_ADMISSION_RECEIPT"):
    """Write one immutable local admission receipt when explicitly requested."""
    receipt_path_value = os.environ.get(env_name)
    if not receipt_path_value:
        return None
    semantic_sha256 = os.environ.get("P1_SEMANTIC_SHA256")
    if (
        not isinstance(semantic_sha256, str)
        or len(semantic_sha256) != 64
        or semantic_sha256 != semantic_sha256.lower()
        or semantic_sha256 == "0" * 64
    ):
        raise RuntimeError(
            "P1_SEMANTIC_SHA256 must be a full SHA-256 when writing an admission receipt"
        )
    try:
        int(semantic_sha256, 16)
    except ValueError as exc:
        raise RuntimeError(
            "P1_SEMANTIC_SHA256 must be lowercase hexadecimal"
        ) from exc
    receipt_path = Path(receipt_path_value).expanduser()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["semantic_sha256"] = semantic_sha256
    body["receipt_sha256"] = hashlib.sha256(
        _p1_canonical_json_bytes(body)
    ).hexdigest()
    payload_bytes = _p1_canonical_json_bytes(body)
    # Link a fully fsynced temporary file into place.  The link fails if a
    # receipt already exists, preventing an admission receipt from being
    # silently replaced by a later run.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=receipt_path.parent, prefix=f".{receipt_path.name}.",
            suffix=".tmp", delete=False
        ) as handle:
            temp_path = Path(handle.name)
            os.chmod(handle.fileno(), 0o600)
            handle.write(payload_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp_path, receipt_path)
        os.chmod(receipt_path, 0o400)
        if receipt_path.read_bytes() != payload_bytes:
            raise RuntimeError("P1 admission receipt changed during publication")
        directory_fd = os.open(receipt_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise RuntimeError(f"P1 admission receipt already exists: {receipt_path}") from exc
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
    return body


def _p1_file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_FORMAL_A_PARENT_SOURCE_SHA256 = (
    "010356bb5477ff5b9a0fa93391eb79b2d23706bc790176ad67d87e7e343a0dce"
)


def _load_formal_a_definition_api():
    """Load only Formal A definitions without executing its training main.

    The historical parent is a single-file training program whose main body is
    intentionally unguarded.  Importing it directly would launch training and
    would also make the engineering runner indistinguishable from a scored
    run.  The source projection executes every top-level definition and import
    before the first setup assignment (``t_start``), while binding the exact
    parent file digest in the resulting receipt.
    """
    parent_path = Path(__file__).resolve().with_name("touched_row_rmsprop.py")
    parent_bytes = parent_path.read_bytes()
    parent_sha256 = hashlib.sha256(parent_bytes).hexdigest()
    if parent_sha256 != _FORMAL_A_PARENT_SOURCE_SHA256:
        raise RuntimeError(
            "Formal A parent source drift: "
            f"expected {_FORMAL_A_PARENT_SOURCE_SHA256}, got {parent_sha256}"
        )
    try:
        tree = ast.parse(parent_bytes.decode("utf-8"), filename=str(parent_path))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise RuntimeError("Formal A parent source is not valid UTF-8 Python") from exc

    definition_nodes = []
    found_setup_boundary = False
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "t_start"
            for target in node.targets
        ):
            found_setup_boundary = True
            break
        definition_nodes.append(node)
    if not found_setup_boundary:
        raise RuntimeError("Formal A source setup boundary t_start is missing")

    module = types.ModuleType("formal_a_parent_definition_projection")
    module.__file__ = str(parent_path)
    module.__package__ = ""
    projected = ast.Module(body=definition_nodes, type_ignores=[])
    ast.fix_missing_locations(projected)
    exec(compile(projected, str(parent_path), "exec"), module.__dict__)
    return module


_FORMAL_A_PARENT_SEMANTIC_SHA256 = (
    "0095f80f58e9bdf125286453f9977997067311c6923b72cd64eab6d2edea5d4a"
)


def _p1_verify_receipt_self_seal(receipt):
    """Recompute a component receipt seal before composing admission evidence."""
    if not isinstance(receipt, dict):
        raise RuntimeError("P1 component receipt must be a JSON object")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise RuntimeError("P1 component receipt lacks a full receipt SHA-256")
    unsigned = dict(receipt)
    del unsigned["receipt_sha256"]
    actual = hashlib.sha256(_p1_canonical_json_bytes(unsigned)).hexdigest()
    if actual != claimed:
        raise RuntimeError("P1 component receipt self-seal verification failed")
    return actual

cap = torch.cuda.get_device_capability()

if cap[0] >= 10:
    # Blackwell (B200, SM100): wrap flash-attn-4 as a custom op so torch.compile
    # treats it as opaque (no tracing into cutlass DSL, no recompile-cache thrash,
    # no per-call Python kernel build).
    from flash_attn.cute import flash_attn_func as _fa4_raw
    from flash_attn.cute.interface import _flash_attn_bwd as _fa4_bwd_raw

    @torch.library.custom_op("fa4::fa4_causal", mutates_args=())
    def _fa4_causal_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       window_left: int) -> tuple[torch.Tensor, torch.Tensor]:
        ws = (window_left, 0) if window_left > 0 else (None, None)
        out, lse = _fa4_raw(q, k, v, causal=True, window_size=ws, return_lse=True)
        return out, lse

    @_fa4_causal_op.register_fake
    def _fa4_causal_fake(q, k, v, window_left):
        B, T, H, D = q.shape
        return torch.empty_like(q), torch.empty(B, H, T, device=q.device, dtype=torch.float32)

    def _fa4_setup_context(ctx, inputs, output):
        q, k, v, window_left = inputs
        out, lse = output
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.window_left = window_left

    @torch.library.custom_op("fa4::fa4_bwd", mutates_args=())
    def _fa4_bwd_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    out: torch.Tensor, grad_output: torch.Tensor, lse: torch.Tensor,
                    window_left: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wl = window_left if window_left > 0 else None
        dq, dk, dv = _fa4_bwd_raw(
            q, k, v, out, grad_output, lse,
            causal=True, window_size_left=wl, window_size_right=0,
        )
        return dq, dk, dv

    @_fa4_bwd_op.register_fake
    def _fa4_bwd_fake(q, k, v, out, grad_output, lse, window_left):
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    def _fa4_backward(ctx, grad_output, grad_lse):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = torch.ops.fa4.fa4_bwd(q, k, v, out, grad_output, lse, ctx.window_left)
        return dq, dk, dv, None

    _fa4_causal_op.register_autograd(_fa4_backward, setup_context=_fa4_setup_context)

    def flash_attn_func(q, k, v, causal=True, window_size=(-1, -1)):
        wl = window_size[0] if isinstance(window_size, tuple) else window_size
        if wl is None or wl <= 0 or wl >= q.shape[1]:
            wl = -1
        out, _lse = torch.ops.fa4.fa4_causal(q, k, v, wl)
        return out

    print(f"Using flash-attn-4 as custom op (GPU capability {cap})")
else:
    # Hopper/Ampere (H100, A100): use flash-attn-3 via kernels package
    from kernels import get_kernel

    if cap == (9, 0):
        h200_backend = os.environ.get("P1_H200_FA3_BACKEND", "community")
        h200_backends = {
            "community": (
                "kernels-community/flash-attn3",
                "9542c462013476380ce4b395b9ddc0e8118161ee",
                "385c5960f738f6c2f217ca93a46fb9be003e7076d60a5711cef8bebdf19eee20",
                True,
            ),
            "legacy-varunneal": (
                "varunneal/flash-attention-3",
                "de87b9b5af06dd9984df595bef90b2eba44b181a",
                "02f36ddbcfde635b8a04d2c877d3cda292b866e080a995b183f903c30f4c311b",
                False,
            ),
        }
        if h200_backend not in h200_backends:
            allowed = ", ".join(sorted(h200_backends))
            raise RuntimeError(f"P1_H200_FA3_BACKEND must be one of: {allowed}")
        repo, revision, expected_binary_sha256, capture_safe = h200_backends[h200_backend]
    else:
        h200_backend = None
        repo = "kernels-community/flash-attn3"
        revision = None
        expected_binary_sha256 = None
        capture_safe = None

    fa3_kernel = get_kernel(repo, revision=revision)
    flash_attn_func = fa3_kernel.flash_attn_interface.flash_attn_func
    print(f"Using flash-attn-3 from {repo} (GPU capability {cap})")

    if cap == (9, 0):
        kernel_root = Path(fa3_kernel.__file__).parent
        kernel_binaries = sorted(path for path in kernel_root.rglob("*.so") if path.is_file())
        if len(kernel_binaries) != 1:
            raise RuntimeError(
                f"Expected one H200 FA3 binary under {kernel_root}, found {len(kernel_binaries)}"
            )
        kernel_binary = kernel_binaries[0]
        binary_hash = hashlib.sha256()
        with kernel_binary.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                binary_hash.update(chunk)
        binary_sha256 = binary_hash.hexdigest()
        if binary_sha256 != expected_binary_sha256:
            raise RuntimeError(
                "H200 FA3 binary SHA mismatch: "
                f"expected {expected_binary_sha256}, got {binary_sha256}"
            )
        backend_receipt = {
            "backend": h200_backend,
            "binary": str(kernel_binary),
            "binary_sha256": binary_sha256,
            "capture_safe": capture_safe,
            "cuda": torch.version.cuda,
            "gpu_capability": list(cap),
            "module": fa3_kernel.__name__,
            "repo": repo,
            "revision": revision,
            "torch": torch.__version__,
            "triton": triton.__version__,
        }
        print("P1_H200_FA3_BACKEND_RECEIPT=" + json.dumps(backend_receipt, sort_keys=True))

if os.environ.get("P1_NO_SCORE_ENGINEERING") == "1":
    # The no-score engineering entrypoint deliberately does not import the
    # validation scorer. It exits through the dispatch below before training.
    from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader  # noqa: E402
else:
    from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, evaluate_bpb, make_dataloader  # noqa: E402

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )
        # Separate gate for bigram VE on ALL VE layers reading decorrelated channels (32:64)
        self.bigram_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )
        # Trigram gate on layers 1, 5, and 7 for late/full-context coverage, reads channels 64:96
        ve_layers = sorted(i for i in range(config.n_layer) if has_ve(i, config.n_layer))
        trigram_layers = (
            {ve_layers[0], ve_layers[-2], ve_layers[-1]}
            if len(ve_layers) >= 2
            else {ve_layers[-1]}
        )
        self.trigram_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if layer_idx in trigram_layers
            else None
        )
        # Head-level MoE gate on ALL layers for attention output routing
        self.head_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False)

    def forward(self, x, ve, cos_sin, window_size, bigram_ve=None, trigram_ve=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # Bigram VE with its own independent gate reading from decorrelated channels (32:64)
        if bigram_ve is not None:
            bigram_ve = bigram_ve.view(B, T, self.n_kv_head, self.head_dim)
            bg_gate = 2 * torch.sigmoid(self.bigram_gate(x[..., self.ve_gate_channels:2*self.ve_gate_channels]))
            v = v + bg_gate.unsqueeze(-1) * bigram_ve

        # Trigram VE with its own gate reading from channels 64:96
        if trigram_ve is not None:
            trigram_ve = trigram_ve.view(B, T, self.n_kv_head, self.head_dim)
            tg_gate = 2 * torch.sigmoid(self.trigram_gate(x[..., 2*self.ve_gate_channels:3*self.ve_gate_channels]))
            v = v + tg_gate.unsqueeze(-1) * trigram_ve

        cos, sin = cos_sin
        # QK-norm refinement: normalize BEFORE rotary instead of after
        q, k = norm(q), norm(k)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        y = flash_attn_func(q, k, v, causal=True, window_size=window_size)
        # Per-head RMSNorm on attention output (DiffTransformer-inspired sub-layer normalization)
        y = norm(y)

        # Head-level MoE: per-head routing gate on all layers
        head_gates = 2.0 * torch.sigmoid(self.head_gate(x[..., :self.ve_gate_channels]))
        y = y * head_gates.unsqueeze(-1)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        # AutoTrust's measured middle-depth allocation.  The profile sums to
        # 32x model width, so total MLP parameters/FLOPs stay fixed while
        # moving capacity toward later layers.
        hidden_mult = (3, 3, 3, 4, 4, 5, 5, 5)[int(layer_idx)]
        hidden_dim = hidden_mult * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
        # Uniform tau=0.5: confirmed optimal threshold
        self.tau = 0.5

    def forward(self, x):
        h = self.c_fc(x)
        h = F.relu(h - self.tau).square()
        h = self.c_proj(h)
        return h




class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config, layer_idx)

    def forward(
        self,
        x,
        ve,
        cos_sin,
        window_size,
        bigram_ve=None,
        trigram_ve=None,
        attn_source_norm=None,
    ):
        # Reused normalized post-layer-4 attention input for attention only. The
        # residual and MLP continue to consume the current stream.
        attn_x = norm(x) if attn_source_norm is None else attn_source_norm
        x = x + self.attn(
            attn_x,
            ve,
            cos_sin,
            window_size,
            bigram_ve=bigram_ve,
            trigram_ve=trigram_ve,
        )
        x = x + norm(self.mlp(norm(x)))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        config,
        p1_compact_backward=None,
        p2_forward_reuse=None,
        p1_reduction="fp32_atomic",
        p1_fused_cleanup=True,
    ):
        super().__init__()
        self.config = config
        self.p1_compact_backward = (
            P1_COMPACT_BACKWARD_ENABLED
            if p1_compact_backward is None
            else bool(p1_compact_backward)
        )
        # P2 is deliberately independent from P1: E1 reuses the forward
        # indices/touch slots while retaining Formal-A's dense embedding
        # backward. P1 implies P2, but P2 must not imply compact backward.
        self.p2_forward_reuse = (
            self.p1_compact_backward
            if p2_forward_reuse is None
            else bool(p2_forward_reuse)
        )
        if self.p1_compact_backward and not self.p2_forward_reuse:
            raise ValueError("P1 compact backward requires P2 forward reuse")
        if p1_reduction not in {"none", "index_add", "fp32_atomic"}:
            raise ValueError("p1_reduction must be none, index_add, or fp32_atomic")
        if self.p1_compact_backward and p1_reduction == "none":
            raise ValueError("P1 compact backward requires an explicit reduction mode")
        if not self.p1_compact_backward and p1_reduction == "index_add":
            raise ValueError("index_add reduction requires P1 compact backward")
        if (
            self.p1_compact_backward
            and p1_reduction == "index_add"
            and p1_fused_cleanup
        ):
            raise ValueError("index_add reduction requires separate slot cleanup")
        self.p1_reduction = p1_reduction
        self.p1_fused_cleanup = bool(p1_fused_cleanup)
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # JEPA MTP removed: multi-token prediction hurts step count in 5-min budget
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Input-dependent x0 gating: per-layer scale for sigmoid gate on x0 skip (layers 4+)
        # gate = 2*sigmoid(scale * x.mean(-1)) modulates x0_lambdas contribution
        # Zero-init so gate starts at 1.0 (neutral = same as current scalar behavior)
        self.x0_gate_scales = nn.Parameter(torch.zeros(config.n_layer))
        # Multi-layer output pooling: aggregate last-K intermediate layers as additive correction
        self.n_pool_layers = min(4, config.n_layer)  # layers [n-4, n-3, n-2] contribute (3 weights)
        self.layer_pool_weights = nn.Parameter(torch.zeros(self.n_pool_layers - 1))
        # Value embeddings (unigram)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(config.vocab_size, kv_dim)
                for i in range(config.n_layer)
                if has_ve(i, config.n_layer)
            }
        )
        # Factored multi-hash bigram VE: K=2 half-dim tables concatenated per layer
        # Crossover: K=2 simplification recovers throughput
        ve_layers = sorted(i for i in range(config.n_layer) if has_ve(i, config.n_layer))
        self.bigram_ve_layers = set(ve_layers)
        self.bigram_baseline_table_size = config.vocab_size * 64
        self.bigram_table_size = config.vocab_size * 512
        self.bigram_K = 2
        half_kv_dim = kv_dim // 2
        # PER-LAYER DECORRELATED: completely disjoint hash prime pairs per bigram VE layer
        # Each layer uses entirely distinct multipliers -- zero prime reuse within bigram type
        # Constants from Murmur/FNV/golden-ratio family for good avalanche behavior
        _decorr_bigram_primes = [
            [(2654435761, 2246822519), (1013904223, 6291469)],   # layer 1: golden-ratio family
            [(374761393, 668265263), (3266489917, 104729)],      # layer 3: prime family
            [(1640531527, 97531), (48271, 40503)],               # layer 5: LCG/Knuth family
            [(16777619, 2166136261), (3432918353, 461845907)],   # layer 7: MurmurHash3 family
        ]
        self.bigram_hash_primes_per_layer = {}
        self.bigram_ves = nn.ModuleDict()
        for j, layer_i in enumerate(ve_layers):
            expanded_tables = []
            for _ in range(self.bigram_K):
                # Construct at the parent's 64x shape so the ambient CUDA RNG
                # advances exactly as it does in the control. Expansion itself
                # is allocation-only; init_weights fills the private-RNG tail.
                table = nn.Embedding(self.bigram_baseline_table_size, half_kv_dim)
                expanded_weight = torch.empty(
                    self.bigram_table_size,
                    half_kv_dim,
                    device=table.weight.device,
                    dtype=table.weight.dtype,
                )
                with torch.no_grad():
                    expanded_weight[: self.bigram_baseline_table_size].copy_(
                        table.weight
                    )
                table.weight = nn.Parameter(expanded_weight)
                table.num_embeddings = self.bigram_table_size
                expanded_tables.append(table)
            self.bigram_ves[str(layer_i)] = nn.ModuleList(expanded_tables)
            for table in self.bigram_ves[str(layer_i)]:
                table.register_buffer("p1_touch_stamps", None, persistent=False)
                table.register_buffer("p1_active_rows", None, persistent=False)
                table.register_buffer("p1_row_to_slot", None, persistent=False)
                table.register_buffer("p1_active_counter", None, persistent=False)
                table.register_buffer("p1_grad_slots", None, persistent=False)
                table.register_buffer("p1_generation", None, persistent=False)
            self.bigram_hash_primes_per_layer[layer_i] = _decorr_bigram_primes[j]
        # Multi-layer factored trigram VE: K=2 half-dim tables at layers 1+5 plus layer 7.
        self.trigram_ve_layers = (
            {ve_layers[0], ve_layers[-2], ve_layers[-1]}
            if len(ve_layers) >= 2
            else {ve_layers[-1]}
        )
        self.trigram_baseline_table_size = config.vocab_size * 64
        self.trigram_table_size = self.trigram_baseline_table_size
        self.trigram_target_table_size = config.vocab_size * 2048
        # PER-LAYER DECORRELATED: completely disjoint 6-prime tuples per trigram VE layer
        # Using disjoint constant families: each layer uses different multiplier sources
        _decorr_trigram_primes = [
            (16777619, 2166136261, 3432918353, 461845907, 2654435769, 1540483477),  # layer 1: FNV+Murmur family
            (3405403843, 2654435761, 2246822519, 1013904223, 6291469, 374761393),   # layer 5: golden-ratio family
            (668265263, 3266489917, 104729, 1640531527, 97531, 48271),              # layer 7: prime family
        ]
        self.trigram_hash_primes_per_layer = {}
        self.trigram_ves = nn.ModuleDict()
        for j, layer_i in enumerate(sorted(self.trigram_ve_layers)):
            self.trigram_ves[str(layer_i)] = nn.ModuleList([
                nn.Embedding(self.trigram_table_size, half_kv_dim),
                nn.Embedding(self.trigram_table_size, half_kv_dim),
            ])
            for table in self.trigram_ves[str(layer_i)]:
                table.register_buffer("p1_touch_stamps", None, persistent=False)
                table.register_buffer("p1_active_rows", None, persistent=False)
                table.register_buffer("p1_row_to_slot", None, persistent=False)
                table.register_buffer("p1_active_counter", None, persistent=False)
                table.register_buffer("p1_grad_slots", None, persistent=False)
                table.register_buffer("p1_generation", None, persistent=False)
            self.trigram_hash_primes_per_layer[layer_i] = _decorr_trigram_primes[j]
        self.shared_trigram_source_layer = min(self.trigram_ve_layers)
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        self.x0_gate_scales.fill_(0.0)  # Zero-init: sigmoid(0)=0.5, 2*0.5=1.0 = neutral gate
        self.layer_pool_weights.fill_(0.0)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            if block.attn.bigram_gate is not None:
                torch.nn.init.zeros_(block.attn.bigram_gate.weight)
            if block.attn.trigram_gate is not None:
                torch.nn.init.zeros_(block.attn.trigram_gate.weight)
            torch.nn.init.zeros_(block.attn.head_gate.weight)
        # Bigram VE: same init as regular VE (factored: two half-dim tables per layer)
        bigram_table_ordinal = 0
        for layer_ves in self.bigram_ves.values():
            for bve in layer_ves:
                torch.nn.init.uniform_(
                    bve.weight[: self.bigram_baseline_table_size], -s, s
                )
                tail_generator = torch.Generator(device=bve.weight.device)
                tail_generator.manual_seed(
                    (
                        torch.initial_seed()
                        + 0x6A09E667F3BCC909
                        + bigram_table_ordinal * 0x9E3779B1
                    )
                    % (2**63 - 1)
                )
                torch.nn.init.uniform_(
                    bve.weight[self.bigram_baseline_table_size :],
                    -s,
                    s,
                    generator=tail_generator,
                )
                bve.to(dtype=torch.bfloat16)
                bigram_table_ordinal += 1
        # Trigram VE init (factored: two half-dim tables per layer)
        for layer_tves in self.trigram_ves.values():
            for tve in layer_tves:
                torch.nn.init.uniform_(tve.weight, -s, s)
                tve.to(dtype=torch.bfloat16)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)
        self._share_and_expand_trigram_tables(s)

    @torch.no_grad()
    def _share_and_expand_trigram_tables(self, init_bound):
        """Retain the parent's layer-1 pair and expand it without ambient RNG draws."""

        parent_keys = tuple(self.trigram_ves.keys())
        expected_keys = tuple(str(i) for i in sorted(self.trigram_ve_layers))
        if parent_keys != expected_keys or len(parent_keys) != 3:
            raise RuntimeError("expected the complete three-layer parent trigram inventory")
        source_key = str(self.shared_trigram_source_layer)
        retained = self.trigram_ves[source_key]
        if len(retained) != 2:
            raise RuntimeError("shared trigram source must contain exactly two tables")
        if any(
            table.weight.shape[0] != self.trigram_baseline_table_size
            for table in retained
        ):
            raise RuntimeError("shared trigram expansion requires the exact 64x parent")

        device = retained[0].weight.device
        if any(table.weight.device != device for table in retained):
            raise RuntimeError("shared trigram tables must use one device")
        if device.type == "cuda":
            ambient_rng = torch.cuda.get_rng_state(device)
            restore_rng = lambda state: torch.cuda.set_rng_state(state, device)
        elif device.type == "cpu":
            ambient_rng = torch.get_rng_state()
            restore_rng = torch.set_rng_state
        else:
            raise RuntimeError("shared trigram expansion requires materialized CPU/CUDA weights")

        try:
            for table_i, table in enumerate(retained):
                old_weight = table.weight
                expanded_weight = torch.empty(
                    self.trigram_target_table_size,
                    old_weight.shape[1],
                    dtype=old_weight.dtype,
                    device=device,
                )
                expanded_weight[: self.trigram_baseline_table_size].copy_(old_weight)
                tail_generator = torch.Generator(device=device)
                tail_generator.manual_seed(
                    (
                        torch.initial_seed()
                        + 0x5452494752414D34
                        + table_i * 0x9E3779B1
                    )
                    % (2**63 - 1)
                )
                torch.nn.init.uniform_(
                    expanded_weight[self.trigram_baseline_table_size :],
                    -init_bound,
                    init_bound,
                    generator=tail_generator,
                )
                table.weight = nn.Parameter(
                    expanded_weight,
                    requires_grad=old_weight.requires_grad,
                )
                table.num_embeddings = self.trigram_target_table_size
        finally:
            restore_rng(ambient_rng)

        for key in parent_keys:
            if key != source_key:
                del self.trigram_ves[key]
        self.shared_trigram_hash_primes = self.trigram_hash_primes_per_layer[
            self.shared_trigram_source_layer
        ]
        self.trigram_hash_primes_per_layer = {
            self.shared_trigram_source_layer: self.shared_trigram_hash_primes
        }
        self.trigram_table_size = self.trigram_target_table_size
        if len(tuple(self.trigram_ves.parameters())) != 2:
            raise RuntimeError("shared trigram bridge must retain exactly two parameters")

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=1000000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SLT" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        # AutoTrust's TINY_DIV=8 recipe value; keep this literal in the
        # source-bound candidate so a remote environment cannot drift.
        tiny_window = long_window // 8
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0), "T": (tiny_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (
            self.transformer.wte.weight.numel()
            + value_embeds_numel
            + self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
        )
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.layer_pool_weights.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        adam_betas=(0.8, 0.95),
        scalar_lr=0.5,
        ngram_ve_betas=None,  # if None, uses adam_betas
        ngram_ve_lr_scale=1.0,  # discriminative LR scale for n-gram VE (ULMFiT-inspired)
        ngram_touch_capacity=None,
    ):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas, self.x0_gate_scales]  # gate scales grouped with x0 lambdas
        bigram_ve_params = list(self.bigram_ves.parameters())
        trigram_ve_params = list(self.trigram_ves.parameters())
        # Route-A touched-row metadata. This is deliberately kept out of the
        # parameter groups so their membership and ordering remain identical
        # to the c428 control. Each entry describes the two hash tables used by
        # one layer; the optimizer uses it only to build a fixed-shape row mask.
        rmsprop_touch_pairs = []
        for layer_i in sorted(self.bigram_ve_layers):
            layer_tables = self.bigram_ves[str(layer_i)]
            rmsprop_touch_pairs.append(
                (
                    2,
                    (layer_tables[0].weight, layer_tables[1].weight),
                    tuple(self.bigram_hash_primes_per_layer[layer_i]),
                )
            )
        shared_trigram_tables = self.trigram_ves[
            str(self.shared_trigram_source_layer)
        ]
        lp = self.shared_trigram_hash_primes
        rmsprop_touch_pairs.append(
            (
                3,
                (
                    shared_trigram_tables[0].weight,
                    shared_trigram_tables[1].weight,
                ),
                (lp[:3], lp[3:]),
            )
        )
        pool_params = [self.layer_pool_weights]
        assert len(list(self.parameters())) == (
            len(matrix_params)
            + len(embedding_params)
            + len(lm_head_params)
            + len(value_embeds_params)
            + len(resid_params)
            + len(x0_params)
            + len(bigram_ve_params)
            + len(trigram_ve_params)
            + len(pool_params)
        )
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if ngram_ve_betas is None:
            ngram_ve_betas = adam_betas
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            {
                "kind": "adamw",
                "params": lm_head_params,
                "lr": unembedding_lr * dmodel_lr_scale,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": 0.0,
                "demon_beta1": True,  # Apply Demon beta1 scheduling
            },
            {
                "kind": "adamw",
                "params": embedding_params,
                "lr": embedding_lr * dmodel_lr_scale,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": 0.0,
                "demon_beta1": True,
            },
            {
                "kind": "adamw",
                "params": value_embeds_params,
                "lr": embedding_lr * dmodel_lr_scale,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": 0.0,
                "demon_beta1": True,
            },
            {
                "kind": "adamw",
                "params": resid_params,
                "lr": scalar_lr * 0.01,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": 0.0,
                # No demon_beta1: scalar params keep fixed beta1
            },
            {
                "kind": "adamw",
                "params": x0_params,
                "lr": scalar_lr,
                "betas": (0.96, 0.95),
                "eps": 1e-10,
                "weight_decay": 0.002,  # x0WD=0.002 (proven optimal)
                "is_x0_muon_warmdown": True,  # x0 Muon warmdown
            },
            {
                "kind": "rmsprop",
                "params": bigram_ve_params,
                "lr": embedding_lr * dmodel_lr_scale * ngram_ve_lr_scale,
                "beta2": ngram_ve_betas[1],
                "eps": 1e-10,
                "weight_decay": 0.0,
                "is_ngram_ve": True,
            },
            {
                "kind": "rmsprop",
                "params": trigram_ve_params,
                "lr": embedding_lr * dmodel_lr_scale * ngram_ve_lr_scale,
                "beta2": ngram_ve_betas[1],
                "eps": 1e-10,
                "weight_decay": 0.0,
                "is_ngram_ve": True,
            },
            {
                "kind": "adamw",
                "params": pool_params,
                "lr": scalar_lr * 0.15,  # revert to formula (0.75*0.15=0.1125)
                "betas": (0.96, 0.95),
                "eps": 1e-10,
                "weight_decay": 0.0,
            },
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(
                {
                    "kind": "muon",
                    "params": group_params,
                    "lr": matrix_lr,
                    "momentum": 0.95,
                    "ns_steps": 5,
                    "beta2": 0.95,
                    "weight_decay": weight_decay,
                }
            )
        optimizer = MuonAdamW(
            param_groups,
            rmsprop_touch_pairs=rmsprop_touch_pairs,
            rmsprop_touch_capacity=ngram_touch_capacity,
            p1_compact_backward=self.p1_compact_backward,
            p2_forward_reuse=self.p2_forward_reuse,
            p1_reduction=self.p1_reduction,
            p1_fused_cleanup=self.p1_fused_cleanup,
        )
        if optimizer._p1_reduction != self.p1_reduction:
            raise RuntimeError("GPT/optimizer P1 reduction mode drift")
        if optimizer._p1_fused_cleanup != self.p1_fused_cleanup:
            raise RuntimeError("GPT/optimizer P1 cleanup mode drift")
        if self.p2_forward_reuse:
            for layer_tables in list(self.bigram_ves.values()) + list(self.trigram_ves.values()):
                for table in layer_tables:
                    if any(buffer is not None for buffer in (
                        table.p1_touch_stamps,
                        table.p1_active_rows,
                        table.p1_row_to_slot,
                        table.p1_active_counter,
                        table.p1_generation,
                    )):
                        raise RuntimeError("P2 forward-reuse buffers initialized twice")
                    table.p1_touch_stamps = optimizer._rmsprop_touch_stamps[table.weight]
                    table.p1_active_rows = optimizer._rmsprop_active_rows[table.weight]
                    table.p1_row_to_slot = optimizer._rmsprop_row_to_slot[table.weight]
                    table.p1_active_counter = optimizer._rmsprop_active_counters[table.weight]
                    if self.p1_compact_backward:
                        table.p1_grad_slots = optimizer._rmsprop_grad_slots[table.weight]
                    table.p1_generation = optimizer._rmsprop_touch_generation
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction="mean"):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        collect_forward_indices = (
            self.p2_forward_reuse and self.training and torch.is_grad_enabled()
        )
        collect_compact_grad = (
            self.p1_compact_backward and self.training and torch.is_grad_enabled()
        )

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        # PER-LAYER DECORRELATED: precompute shifted indices (shared), compute per-layer hash indices inside loop
        prev_idx = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
        prev2_idx = torch.cat([idx[:, :2], idx[:, :-2]], dim=1)
        # Precompute per-layer bigram hash indices (different primes per layer for collision decorrelation)
        bigram_indices_per_layer = {}
        for layer_i in self.bigram_ve_layers:
            layer_bg_primes = self.bigram_hash_primes_per_layer[layer_i]
            bigram_indices_per_layer[layer_i] = [
                ((prev_idx * p1) ^ (idx * p2)) % self.bigram_table_size
                for p1, p2 in layer_bg_primes
            ]
        # One layer-1 hash pair and one lookup are shared by all three consumers.
        lp = self.shared_trigram_hash_primes
        shared_trigram_indices = (
            (
                (prev2_idx * lp[0])
                ^ (prev_idx * lp[1])
                ^ (idx * lp[2])
            )
            % self.trigram_table_size,
            (
                (prev2_idx * lp[3])
                ^ (prev_idx * lp[4])
                ^ (idx * lp[5])
            )
            % self.trigram_table_size,
        )
        shared_trigram_tables = self.trigram_ves[
            str(self.shared_trigram_source_layer)
        ]
        if (
            collect_compact_grad
            and self.p1_reduction == "fp32_atomic"
            and P1_FUSED_LOOKUP_COLLECT_ENABLED
        ):
            shared_tgve = _P1FusedPairLookupRedirect.apply(
                shared_trigram_tables[0].weight,
                shared_trigram_tables[1].weight,
                shared_trigram_indices[0],
                shared_trigram_indices[1],
                shared_trigram_tables[0].p1_touch_stamps,
                shared_trigram_tables[0].p1_active_rows,
                shared_trigram_tables[0].p1_row_to_slot,
                shared_trigram_tables[0].p1_active_counter,
                shared_trigram_tables[1].p1_touch_stamps,
                shared_trigram_tables[1].p1_active_rows,
                shared_trigram_tables[1].p1_row_to_slot,
                shared_trigram_tables[1].p1_active_counter,
                shared_trigram_tables[0].p1_generation,
                shared_trigram_tables[0].p1_grad_slots,
                shared_trigram_tables[1].p1_grad_slots,
            )
        else:
            if collect_forward_indices:
                _p1_collect_pair_indices_op(
                    shared_trigram_indices[0],
                    shared_trigram_indices[1],
                    shared_trigram_tables[0].p1_touch_stamps,
                    shared_trigram_tables[0].p1_active_rows,
                    shared_trigram_tables[0].p1_row_to_slot,
                    shared_trigram_tables[0].p1_active_counter,
                    shared_trigram_tables[1].p1_touch_stamps,
                    shared_trigram_tables[1].p1_active_rows,
                    shared_trigram_tables[1].p1_row_to_slot,
                    shared_trigram_tables[1].p1_active_counter,
                    shared_trigram_tables[0].p1_generation,
                )
            shared_tgve = torch.cat(
                [
                    shared_trigram_tables[0](shared_trigram_indices[0]),
                    shared_trigram_tables[1](shared_trigram_indices[1]),
                ],
                dim=-1,
            )
            if collect_compact_grad:
                redirect = (
                    _P1CompactPairRedirectIndexAdd
                    if self.p1_reduction == "index_add"
                    else _P1CompactPairRedirect
                )
                shared_tgve = redirect.apply(
                    shared_tgve,
                    shared_trigram_indices[0],
                    shared_trigram_indices[1],
                    shared_trigram_tables[0].p1_row_to_slot,
                    shared_trigram_tables[1].p1_row_to_slot,
                    shared_trigram_tables[0].p1_grad_slots,
                    shared_trigram_tables[1].p1_grad_slots,
                )
        n_layer = len(self.transformer.h)
        pool_start = n_layer - self.n_pool_layers
        pool_residual = None
        saved_attn_source_norm = None
        for i, block in enumerate(self.transformer.h):
            # Input-dependent x0 gate on ALL 8 layers: 2*sigmoid(scale*mean(x)) modulates x0 contribution
            # Starts at 1.0 (gate_scales=0 → sigmoid(0)=0.5 → 2*0.5=1.0)
            x0_gate = 2.0 * torch.sigmoid(self.x0_gate_scales[i] * x.float().mean(-1, keepdim=True)).to(x.dtype)
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0_gate * x0
            if str(i) in self.value_embeds:
                ve = self.value_embeds[str(i)](idx)
            else:
                ve = None
            # Factored multi-hash bigram VE: concat K=2 half-dim lookups from independent hashes (per-layer primes)
            if i in self.bigram_ve_layers:
                layer_ves = self.bigram_ves[str(i)]
                layer_indices = bigram_indices_per_layer[i]
                if (
                    collect_compact_grad
                    and self.p1_reduction == "fp32_atomic"
                    and P1_FUSED_LOOKUP_COLLECT_ENABLED
                ):
                    bgve = _P1FusedPairLookupRedirect.apply(
                        layer_ves[0].weight,
                        layer_ves[1].weight,
                        layer_indices[0],
                        layer_indices[1],
                        layer_ves[0].p1_touch_stamps,
                        layer_ves[0].p1_active_rows,
                        layer_ves[0].p1_row_to_slot,
                        layer_ves[0].p1_active_counter,
                        layer_ves[1].p1_touch_stamps,
                        layer_ves[1].p1_active_rows,
                        layer_ves[1].p1_row_to_slot,
                        layer_ves[1].p1_active_counter,
                        layer_ves[0].p1_generation,
                        layer_ves[0].p1_grad_slots,
                        layer_ves[1].p1_grad_slots,
                    )
                else:
                    if collect_forward_indices:
                        _p1_collect_pair_indices_op(
                            layer_indices[0],
                            layer_indices[1],
                            layer_ves[0].p1_touch_stamps,
                            layer_ves[0].p1_active_rows,
                            layer_ves[0].p1_row_to_slot,
                            layer_ves[0].p1_active_counter,
                            layer_ves[1].p1_touch_stamps,
                            layer_ves[1].p1_active_rows,
                            layer_ves[1].p1_row_to_slot,
                            layer_ves[1].p1_active_counter,
                            layer_ves[0].p1_generation,
                        )
                    bgve = torch.cat(
                        [layer_ves[k](layer_indices[k]) for k in range(self.bigram_K)],
                        dim=-1,
                    )
                    if collect_compact_grad:
                        redirect = (
                            _P1CompactPairRedirectIndexAdd
                            if self.p1_reduction == "index_add"
                            else _P1CompactPairRedirect
                        )
                        bgve = redirect.apply(
                            bgve,
                            layer_indices[0],
                            layer_indices[1],
                            layer_ves[0].p1_row_to_slot,
                            layer_ves[1].p1_row_to_slot,
                            layer_ves[0].p1_grad_slots,
                            layer_ves[1].p1_grad_slots,
                        )
            else:
                bgve = None
            # Three attention gates consume one shared trigram tensor. Autograd
            # accumulates their contributions before the single compact scatter.
            tgve = shared_tgve if i in self.trigram_ve_layers else None
            attn_source_norm = (
                saved_attn_source_norm if i in ATTN_SOURCE_LAYERS else None
            )
            x = block(
                x,
                ve,
                cos_sin,
                self.window_sizes[i],
                bigram_ve=bgve,
                trigram_ve=tgve,
                attn_source_norm=attn_source_norm,
            )
            if i == ATTN_SOURCE_AFTER_LAYER:
                saved_attn_source_norm = norm(x)
            if i == pool_start:
                pool_residual = self.layer_pool_weights[0] * x
            elif i == pool_start + 1:
                pool_residual = pool_residual + self.layer_pool_weights[1] * x
            elif i == pool_start + 2:
                pool_residual = pool_residual + self.layer_pool_weights[2] * x
        if pool_residual is not None:
            x = x + pool_residual
        x = norm(x)

        # Decoupled softcap in BF16: skip float() cast, halve logit tensor memory
        # Since model is natively BF16, softcap in BF16 should be numerically adequate
        logits = self.lm_head(x)
        logits = 16.5 * torch.tanh(logits / 15.0)

        if targets is not None:
            # Cast to float32 only for the CE loss computation (numerically sensitive)
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=reduction,
            )
            return loss
        # Eval path: need float32 logits
        return logits.float()


# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@triton.jit
def _mark_bigram_pair_kernel(
    tokens,
    touch_mask_0,
    touch_mask_1,
    num_tokens,
    P00: tl.constexpr,
    P01: tl.constexpr,
    P10: tl.constexpr,
    P11: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    TABLE_MASK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Mark both bigram tables for one layer from the exact model hashes."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < num_tokens
    current = tl.load(tokens + offsets, mask=valid, other=0).to(tl.int64)
    position = offsets % SEQ_LEN
    prev_offsets = offsets - tl.where(position > 0, 1, 0)
    previous = tl.load(tokens + prev_offsets, mask=valid, other=0).to(tl.int64)
    row_0 = ((previous * P00) ^ (current * P01)) & TABLE_MASK
    row_1 = ((previous * P10) ^ (current * P11)) & TABLE_MASK
    # int32 row masks make duplicate writes well-defined on every supported
    # CUDA architecture. At c428's 524288 rows, 14 masks cost exactly 28 MiB.
    tl.atomic_xchg(touch_mask_0 + row_0, 1, mask=valid)
    tl.atomic_xchg(touch_mask_1 + row_1, 1, mask=valid)


@triton.jit
def _mark_trigram_pair_kernel(
    tokens,
    touch_mask_0,
    touch_mask_1,
    num_tokens,
    P00: tl.constexpr,
    P01: tl.constexpr,
    P02: tl.constexpr,
    P10: tl.constexpr,
    P11: tl.constexpr,
    P12: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    TABLE_MASK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Mark both trigram tables, including control's t=1 prev2 convention."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < num_tokens
    current = tl.load(tokens + offsets, mask=valid, other=0).to(tl.int64)
    position = offsets % SEQ_LEN
    prev_offsets = offsets - tl.where(position > 0, 1, 0)
    # control.py uses cat([idx[:, :2], idx[:, :-2]]): at positions 0 and 1
    # prev2 is the current token, not token 0.
    prev2_offsets = offsets - tl.where(position > 1, 2, 0)
    previous = tl.load(tokens + prev_offsets, mask=valid, other=0).to(tl.int64)
    previous_2 = tl.load(tokens + prev2_offsets, mask=valid, other=0).to(tl.int64)
    row_0 = ((previous_2 * P00) ^ (previous * P01) ^ (current * P02)) & TABLE_MASK
    row_1 = ((previous_2 * P10) ^ (previous * P11) ^ (current * P12)) & TABLE_MASK
    tl.atomic_xchg(touch_mask_0 + row_0, 1, mask=valid)
    tl.atomic_xchg(touch_mask_1 + row_1, 1, mask=valid)


@triton.jit(do_not_specialize=["generation"])
def _collect_bigram_pair_kernel(
    tokens,
    stamps_0,
    active_rows_0,
    row_to_slot_0,
    counter_0,
    stamps_1,
    active_rows_1,
    row_to_slot_1,
    counter_1,
    num_tokens,
    generation,
    P00: tl.constexpr,
    P01: tl.constexpr,
    P10: tl.constexpr,
    P11: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    TABLE_MASK: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Generation-stamped exact unique-row collection for a bigram pair."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < num_tokens
    current = tl.load(tokens + offsets, mask=valid, other=0).to(tl.int64)
    position = offsets % SEQ_LEN
    prev_offsets = offsets - tl.where(position > 0, 1, 0)
    previous = tl.load(tokens + prev_offsets, mask=valid, other=0).to(tl.int64)
    row_0 = ((previous * P00) ^ (current * P01)) & TABLE_MASK
    row_1 = ((previous * P10) ^ (current * P11)) & TABLE_MASK

    old_0 = tl.load(stamps_0 + row_0, mask=valid, other=generation)
    claim_0 = valid & (old_0 != generation)
    safe_row_0 = tl.where(valid, row_0, 0)
    generation_0 = generation + tl.zeros_like(old_0)
    seen_0 = tl.atomic_cas(stamps_0 + safe_row_0, old_0, generation_0)
    won_0 = claim_0 & (seen_0 == old_0)
    slot_0 = tl.atomic_add(counter_0 + tl.zeros_like(row_0), 1, mask=won_0)
    valid_slot_0 = won_0 & (slot_0 < CAPACITY)
    tl.store(active_rows_0 + slot_0, row_0, mask=valid_slot_0)
    tl.store(row_to_slot_0 + row_0, slot_0, mask=valid_slot_0)

    old_1 = tl.load(stamps_1 + row_1, mask=valid, other=generation)
    claim_1 = valid & (old_1 != generation)
    safe_row_1 = tl.where(valid, row_1, 0)
    generation_1 = generation + tl.zeros_like(old_1)
    seen_1 = tl.atomic_cas(stamps_1 + safe_row_1, old_1, generation_1)
    won_1 = claim_1 & (seen_1 == old_1)
    slot_1 = tl.atomic_add(counter_1 + tl.zeros_like(row_1), 1, mask=won_1)
    valid_slot_1 = won_1 & (slot_1 < CAPACITY)
    tl.store(active_rows_1 + slot_1, row_1, mask=valid_slot_1)
    tl.store(row_to_slot_1 + row_1, slot_1, mask=valid_slot_1)


@triton.jit(do_not_specialize=["generation"])
def _collect_trigram_pair_kernel(
    tokens,
    stamps_0,
    active_rows_0,
    row_to_slot_0,
    counter_0,
    stamps_1,
    active_rows_1,
    row_to_slot_1,
    counter_1,
    num_tokens,
    generation,
    P00: tl.constexpr,
    P01: tl.constexpr,
    P02: tl.constexpr,
    P10: tl.constexpr,
    P11: tl.constexpr,
    P12: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    TABLE_MASK: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Generation-stamped exact unique-row collection for a trigram pair."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < num_tokens
    current = tl.load(tokens + offsets, mask=valid, other=0).to(tl.int64)
    position = offsets % SEQ_LEN
    prev_offsets = offsets - tl.where(position > 0, 1, 0)
    prev2_offsets = offsets - tl.where(position > 1, 2, 0)
    previous = tl.load(tokens + prev_offsets, mask=valid, other=0).to(tl.int64)
    previous_2 = tl.load(tokens + prev2_offsets, mask=valid, other=0).to(tl.int64)
    row_0 = ((previous_2 * P00) ^ (previous * P01) ^ (current * P02)) & TABLE_MASK
    row_1 = ((previous_2 * P10) ^ (previous * P11) ^ (current * P12)) & TABLE_MASK

    old_0 = tl.load(stamps_0 + row_0, mask=valid, other=generation)
    claim_0 = valid & (old_0 != generation)
    safe_row_0 = tl.where(valid, row_0, 0)
    generation_0 = generation + tl.zeros_like(old_0)
    seen_0 = tl.atomic_cas(stamps_0 + safe_row_0, old_0, generation_0)
    won_0 = claim_0 & (seen_0 == old_0)
    slot_0 = tl.atomic_add(counter_0 + tl.zeros_like(row_0), 1, mask=won_0)
    valid_slot_0 = won_0 & (slot_0 < CAPACITY)
    tl.store(active_rows_0 + slot_0, row_0, mask=valid_slot_0)
    tl.store(row_to_slot_0 + row_0, slot_0, mask=valid_slot_0)

    old_1 = tl.load(stamps_1 + row_1, mask=valid, other=generation)
    claim_1 = valid & (old_1 != generation)
    safe_row_1 = tl.where(valid, row_1, 0)
    generation_1 = generation + tl.zeros_like(old_1)
    seen_1 = tl.atomic_cas(stamps_1 + safe_row_1, old_1, generation_1)
    won_1 = claim_1 & (seen_1 == old_1)
    slot_1 = tl.atomic_add(counter_1 + tl.zeros_like(row_1), 1, mask=won_1)
    valid_slot_1 = won_1 & (slot_1 < CAPACITY)
    tl.store(active_rows_1 + slot_1, row_1, mask=valid_slot_1)
    tl.store(row_to_slot_1 + row_1, slot_1, mask=valid_slot_1)


@triton.jit
def _p1_collect_pair_indices_kernel(
    indices_0,
    indices_1,
    stamps_0,
    active_rows_0,
    row_to_slot_0,
    counter_0,
    stamps_1,
    active_rows_1,
    row_to_slot_1,
    counter_1,
    generation_ptr,
    num_tokens,
    CAPACITY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Collect the exact pair indices already materialized by model forward."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < num_tokens
    safe_offsets = tl.where(valid, offsets, 0)
    row_0 = tl.load(indices_0 + safe_offsets, mask=valid, other=0).to(tl.int64)
    row_1 = tl.load(indices_1 + safe_offsets, mask=valid, other=0).to(tl.int64)
    generation = tl.load(generation_ptr).to(tl.int32)

    old_0 = tl.load(stamps_0 + row_0, mask=valid, other=generation)
    claim_0 = valid & (old_0 != generation)
    seen_0 = tl.atomic_cas(
        stamps_0 + tl.where(valid, row_0, 0),
        old_0,
        generation + tl.zeros_like(old_0),
    )
    won_0 = claim_0 & (seen_0 == old_0)
    slot_0 = tl.atomic_add(counter_0 + tl.zeros_like(row_0), 1, mask=won_0)
    valid_slot_0 = won_0 & (slot_0 < CAPACITY)
    tl.store(active_rows_0 + slot_0, row_0, mask=valid_slot_0)
    tl.store(row_to_slot_0 + row_0, slot_0, mask=valid_slot_0)

    old_1 = tl.load(stamps_1 + row_1, mask=valid, other=generation)
    claim_1 = valid & (old_1 != generation)
    seen_1 = tl.atomic_cas(
        stamps_1 + tl.where(valid, row_1, 0),
        old_1,
        generation + tl.zeros_like(old_1),
    )
    won_1 = claim_1 & (seen_1 == old_1)
    slot_1 = tl.atomic_add(counter_1 + tl.zeros_like(row_1), 1, mask=won_1)
    valid_slot_1 = won_1 & (slot_1 < CAPACITY)
    tl.store(active_rows_1 + slot_1, row_1, mask=valid_slot_1)
    tl.store(row_to_slot_1 + row_1, slot_1, mask=valid_slot_1)


@torch.library.custom_op(
    "apex_p1_compact_backward::collect_pair_indices",
    mutates_args=(
        "stamps_0",
        "active_rows_0",
        "row_to_slot_0",
        "counter_0",
        "stamps_1",
        "active_rows_1",
        "row_to_slot_1",
        "counter_1",
    ),
)
def _p1_collect_pair_indices_op(
    indices_0: torch.Tensor,
    indices_1: torch.Tensor,
    stamps_0: torch.Tensor,
    active_rows_0: torch.Tensor,
    row_to_slot_0: torch.Tensor,
    counter_0: torch.Tensor,
    stamps_1: torch.Tensor,
    active_rows_1: torch.Tensor,
    row_to_slot_1: torch.Tensor,
    counter_1: torch.Tensor,
    generation: torch.Tensor,
) -> None:
    if indices_0.dtype != torch.int64 or indices_1.dtype != torch.int64:
        raise RuntimeError("P2 collector requires int64 forward indices")
    if indices_0.shape != indices_1.shape or indices_0.ndim != 2:
        raise RuntimeError("P2 collector expects matching rank-2 pair indices")
    if not indices_0.is_contiguous() or not indices_1.is_contiguous():
        raise RuntimeError("P2 collector requires contiguous pair indices")
    if stamps_0.dtype != torch.int32 or stamps_1.dtype != torch.int32:
        raise RuntimeError("P2 collector requires int32 generation stamps")
    if row_to_slot_0.dtype != torch.int32 or row_to_slot_1.dtype != torch.int32:
        raise RuntimeError("P2 collector requires int32 row-to-slot maps")
    if stamps_0.shape != row_to_slot_0.shape or stamps_1.shape != row_to_slot_1.shape:
        raise RuntimeError("P2 collector stamp/map shape mismatch")
    if active_rows_0.dtype != torch.int32 or active_rows_1.dtype != torch.int32:
        raise RuntimeError("P2 collector requires int32 active-row lists")
    if active_rows_0.ndim != 1 or active_rows_0.shape != active_rows_1.shape:
        raise RuntimeError("P2 collector active-row capacity mismatch")
    if counter_0.shape != (1,) or counter_1.shape != (1,):
        raise RuntimeError("P2 collector requires scalar active-row counters")
    if counter_0.dtype != torch.int32 or counter_1.dtype != torch.int32:
        raise RuntimeError("P2 collector requires int32 active-row counters")
    if generation.shape != (1,) or generation.dtype != torch.int32:
        raise RuntimeError("P2 collector requires one int32 generation")
    tensors = (
        indices_0,
        indices_1,
        stamps_0,
        active_rows_0,
        row_to_slot_0,
        counter_0,
        stamps_1,
        active_rows_1,
        row_to_slot_1,
        counter_1,
        generation,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise RuntimeError("P2 collector is CUDA-only")
    if len({tensor.device for tensor in tensors}) != 1:
        raise RuntimeError("P2 collector tensors must share one CUDA device")
    capacity = active_rows_0.numel()
    num_tokens = indices_0.numel()
    if capacity <= 0:
        raise RuntimeError("P2 collector requires positive active-row capacity")
    # ``num_tokens`` is deliberately not compared with ``capacity``.  The
    # kernel collapses duplicate indices, so unique active rows can fit even
    # when a batch contains more tokens than the fixed slot budget.  Every
    # atomic counter still increments for a newly claimed row; the optimizer
    # validates those counters before update and rejects overflow fail-closed.
    block_size = 256
    grid = (triton.cdiv(num_tokens, block_size),)
    _p1_collect_pair_indices_kernel[grid](
        indices_0,
        indices_1,
        stamps_0,
        active_rows_0,
        row_to_slot_0,
        counter_0,
        stamps_1,
        active_rows_1,
        row_to_slot_1,
        counter_1,
        generation,
        num_tokens,
        CAPACITY=capacity,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )


@_p1_collect_pair_indices_op.register_fake
def _p1_collect_pair_indices_fake(
    indices_0,
    indices_1,
    stamps_0,
    active_rows_0,
    row_to_slot_0,
    counter_0,
    stamps_1,
    active_rows_1,
    row_to_slot_1,
    counter_1,
    generation,
):
    return None


@triton.jit
def _p1_fused_pair_lookup_kernel(
    weight_0,
    weight_1,
    indices_0,
    indices_1,
    output,
    stamps_0,
    active_rows_0,
    row_to_slot_0,
    counter_0,
    stamps_1,
    active_rows_1,
    row_to_slot_1,
    counter_1,
    generation_ptr,
    num_tokens,
    TABLE_COLS: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Gather both halves and register unique rows in one launch.

    The first column tile performs the row-ownership atomics.  All column
    tiles write the two BF16 halves directly into the final concatenated
    output, avoiding the two embedding outputs plus a separate ``cat``.
    """
    token_ids = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    col_ids = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    valid_tokens = token_ids < num_tokens
    valid_cols = col_ids < TABLE_COLS
    active = valid_tokens[:, None] & valid_cols[None, :]
    safe_tokens = tl.where(valid_tokens, token_ids, 0)
    row_0 = tl.load(indices_0 + safe_tokens, mask=valid_tokens, other=0).to(tl.int64)
    row_1 = tl.load(indices_1 + safe_tokens, mask=valid_tokens, other=0).to(tl.int64)

    weight_offsets_0 = row_0[:, None] * TABLE_COLS + col_ids[None, :]
    weight_offsets_1 = row_1[:, None] * TABLE_COLS + col_ids[None, :]
    output_offsets_0 = token_ids[:, None] * (TABLE_COLS * 2) + col_ids[None, :]
    output_offsets_1 = output_offsets_0 + TABLE_COLS
    values_0 = tl.load(weight_0 + weight_offsets_0, mask=active, other=0.0)
    values_1 = tl.load(weight_1 + weight_offsets_1, mask=active, other=0.0)
    tl.store(output + output_offsets_0, values_0, mask=active)
    tl.store(output + output_offsets_1, values_1, mask=active)

    # Only the first column tile claims rows, so the remaining gather tiles
    # issue no stamp loads or atomics.
    if tl.program_id(1) == 0:
        generation = tl.load(generation_ptr).to(tl.int32)
        safe_row_0 = tl.where(valid_tokens, row_0, 0)
        old_0 = tl.load(stamps_0 + safe_row_0, mask=valid_tokens, other=generation)
        claim_0 = valid_tokens & (old_0 != generation)
        # Rows already stamped for this generation do not need an atomic.
        # Among racing claimants, exactly one exchange observes a stale stamp;
        # every later exchange observes ``generation`` and loses.
        seen_0 = tl.atomic_xchg(
            stamps_0 + safe_row_0,
            generation + tl.zeros_like(old_0),
            mask=claim_0,
        )
        won_0 = claim_0 & (seen_0 != generation)
        slot_0 = tl.atomic_add(counter_0 + tl.zeros_like(row_0), 1, mask=won_0)
        valid_slot_0 = won_0 & (slot_0 < CAPACITY)
        tl.store(active_rows_0 + slot_0, row_0, mask=valid_slot_0)
        tl.store(row_to_slot_0 + row_0, slot_0, mask=valid_slot_0)

        safe_row_1 = tl.where(valid_tokens, row_1, 0)
        old_1 = tl.load(stamps_1 + safe_row_1, mask=valid_tokens, other=generation)
        claim_1 = valid_tokens & (old_1 != generation)
        seen_1 = tl.atomic_xchg(
            stamps_1 + safe_row_1,
            generation + tl.zeros_like(old_1),
            mask=claim_1,
        )
        won_1 = claim_1 & (seen_1 != generation)
        slot_1 = tl.atomic_add(counter_1 + tl.zeros_like(row_1), 1, mask=won_1)
        valid_slot_1 = won_1 & (slot_1 < CAPACITY)
        tl.store(active_rows_1 + slot_1, row_1, mask=valid_slot_1)
        tl.store(row_to_slot_1 + row_1, slot_1, mask=valid_slot_1)


@torch.library.custom_op(
    "apex_p1_compact_backward::fused_pair_lookup",
    mutates_args=(
        "stamps_0",
        "active_rows_0",
        "row_to_slot_0",
        "counter_0",
        "stamps_1",
        "active_rows_1",
        "row_to_slot_1",
        "counter_1",
    ),
)
def _p1_fused_pair_lookup_op(
    weight_0: torch.Tensor,
    weight_1: torch.Tensor,
    indices_0: torch.Tensor,
    indices_1: torch.Tensor,
    stamps_0: torch.Tensor,
    active_rows_0: torch.Tensor,
    row_to_slot_0: torch.Tensor,
    counter_0: torch.Tensor,
    stamps_1: torch.Tensor,
    active_rows_1: torch.Tensor,
    row_to_slot_1: torch.Tensor,
    counter_1: torch.Tensor,
    generation: torch.Tensor,
) -> torch.Tensor:
    if weight_0.dtype != torch.bfloat16 or weight_1.dtype != torch.bfloat16:
        raise RuntimeError("fused P1 lookup requires BF16 embedding tables")
    if weight_0.ndim != 2 or weight_1.shape != weight_0.shape:
        raise RuntimeError("fused P1 lookup expects matching rank-2 tables")
    if indices_0.dtype != torch.int64 or indices_1.dtype != torch.int64:
        raise RuntimeError("fused P1 lookup requires int64 indices")
    if indices_0.shape != indices_1.shape or indices_0.ndim != 2:
        raise RuntimeError("fused P1 lookup expects matching rank-2 indices")
    if not all(x.is_contiguous() for x in (weight_0, weight_1, indices_0, indices_1)):
        raise RuntimeError("fused P1 lookup requires contiguous inputs")
    if stamps_0.dtype != torch.int32 or stamps_1.dtype != torch.int32:
        raise RuntimeError("fused P1 lookup requires int32 stamps")
    if row_to_slot_0.dtype != torch.int32 or row_to_slot_1.dtype != torch.int32:
        raise RuntimeError("fused P1 lookup requires int32 row maps")
    if active_rows_0.dtype != torch.int32 or active_rows_1.dtype != torch.int32:
        raise RuntimeError("fused P1 lookup requires int32 active rows")
    if counter_0.shape != (1,) or counter_1.shape != (1,):
        raise RuntimeError("fused P1 lookup requires scalar counters")
    if generation.shape != (1,) or generation.dtype != torch.int32:
        raise RuntimeError("fused P1 lookup requires one int32 generation")
    if active_rows_0.shape != active_rows_1.shape:
        raise RuntimeError("fused P1 lookup capacity mismatch")
    tensors = (
        weight_0,
        weight_1,
        indices_0,
        indices_1,
        stamps_0,
        active_rows_0,
        row_to_slot_0,
        counter_0,
        stamps_1,
        active_rows_1,
        row_to_slot_1,
        counter_1,
        generation,
    )
    if not all(x.is_cuda for x in tensors):
        raise RuntimeError("fused P1 lookup is CUDA-only")
    if len({x.device for x in tensors}) != 1:
        raise RuntimeError("fused P1 lookup tensors must share one CUDA device")
    table_cols = weight_0.shape[1]
    num_tokens = indices_0.numel()
    output = torch.empty(
        (*indices_0.shape, table_cols * 2),
        dtype=weight_0.dtype,
        device=weight_0.device,
    )
    block_tokens = _P1_LOOKUP_BLOCK_TOKENS
    block_cols = min(128, triton.next_power_of_2(table_cols))
    grid = (triton.cdiv(num_tokens, block_tokens), triton.cdiv(table_cols, block_cols))
    _p1_fused_pair_lookup_kernel[grid](
        weight_0,
        weight_1,
        indices_0,
        indices_1,
        output,
        stamps_0,
        active_rows_0,
        row_to_slot_0,
        counter_0,
        stamps_1,
        active_rows_1,
        row_to_slot_1,
        counter_1,
        generation,
        num_tokens,
        TABLE_COLS=table_cols,
        CAPACITY=active_rows_0.numel(),
        BLOCK_TOKENS=block_tokens,
        BLOCK_COLS=block_cols,
        num_warps=_P1_LOOKUP_NUM_WARPS,
    )
    return output


@_p1_fused_pair_lookup_op.register_fake
def _p1_fused_pair_lookup_fake(
    weight_0,
    weight_1,
    indices_0,
    indices_1,
    stamps_0,
    active_rows_0,
    row_to_slot_0,
    counter_0,
    stamps_1,
    active_rows_1,
    row_to_slot_1,
    counter_1,
    generation,
):
    return torch.empty(
        (*indices_0.shape, weight_0.shape[1] * 2),
        dtype=weight_0.dtype,
        device=weight_0.device,
    )


@triton.jit
def _p1_scatter_pair_fp32_kernel(
    grad_pair,
    indices_0,
    indices_1,
    row_to_slot_0,
    row_to_slot_1,
    grad_slots_0,
    grad_slots_1,
    num_tokens,
    TABLE_COLS: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Accumulate the real post-cat gradient into compact FP32 row slots."""
    token_ids = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    col_ids = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    valid_tokens = token_ids < num_tokens
    valid_cols = col_ids < TABLE_COLS
    safe_tokens = tl.where(valid_tokens, token_ids, 0)

    row_0 = tl.load(indices_0 + safe_tokens, mask=valid_tokens, other=0).to(tl.int64)
    row_1 = tl.load(indices_1 + safe_tokens, mask=valid_tokens, other=0).to(tl.int64)
    slot_0 = tl.load(row_to_slot_0 + row_0, mask=valid_tokens, other=-1)
    slot_1 = tl.load(row_to_slot_1 + row_1, mask=valid_tokens, other=-1)
    valid_slot_0 = valid_tokens & (slot_0 >= 0) & (slot_0 < CAPACITY)
    valid_slot_1 = valid_tokens & (slot_1 >= 0) & (slot_1 < CAPACITY)

    pair_stride = TABLE_COLS * 2
    grad_offset_0 = token_ids[:, None] * pair_stride + col_ids[None, :]
    grad_offset_1 = grad_offset_0 + TABLE_COLS
    active_0 = valid_slot_0[:, None] & valid_cols[None, :]
    active_1 = valid_slot_1[:, None] & valid_cols[None, :]
    contribution_0 = tl.load(
        grad_pair + grad_offset_0, mask=active_0, other=0.0
    ).to(tl.float32)
    contribution_1 = tl.load(
        grad_pair + grad_offset_1, mask=active_1, other=0.0
    ).to(tl.float32)

    slot_offset_0 = slot_0[:, None] * TABLE_COLS + col_ids[None, :]
    slot_offset_1 = slot_1[:, None] * TABLE_COLS + col_ids[None, :]
    tl.atomic_add(
        grad_slots_0 + slot_offset_0,
        contribution_0,
        mask=active_0,
        sem="relaxed",
    )
    tl.atomic_add(
        grad_slots_1 + slot_offset_1,
        contribution_1,
        mask=active_1,
        sem="relaxed",
    )


@torch.library.custom_op(
    "apex_p1_compact_backward::scatter_pair_fp32",
    mutates_args=("grad_slots_0", "grad_slots_1"),
)
def _p1_scatter_pair_fp32_op(
    grad_pair: torch.Tensor,
    indices_0: torch.Tensor,
    indices_1: torch.Tensor,
    row_to_slot_0: torch.Tensor,
    row_to_slot_1: torch.Tensor,
    grad_slots_0: torch.Tensor,
    grad_slots_1: torch.Tensor,
) -> None:
    if grad_pair.dtype != torch.bfloat16:
        raise RuntimeError("P1 Redirect requires BF16 post-cat gradients")
    if indices_0.dtype != torch.int64 or indices_1.dtype != torch.int64:
        raise RuntimeError("P1 Redirect requires int64 embedding indices")
    if row_to_slot_0.dtype != torch.int32 or row_to_slot_1.dtype != torch.int32:
        raise RuntimeError("P1 Redirect requires int32 row-to-slot maps")
    if grad_slots_0.dtype != torch.float32 or grad_slots_1.dtype != torch.float32:
        raise RuntimeError("P1 Redirect requires compact FP32 gradient slots")
    if grad_pair.ndim != 3 or indices_0.shape != grad_pair.shape[:2]:
        raise RuntimeError("P1 Redirect expects grad[B,T,2C] and indices[B,T]")
    if indices_1.shape != indices_0.shape:
        raise RuntimeError("P1 Redirect pair index shape mismatch")
    if row_to_slot_0.ndim != 1 or row_to_slot_1.ndim != 1:
        raise RuntimeError("P1 Redirect row-to-slot maps must be rank 1")
    if grad_slots_0.ndim != 2 or grad_slots_0.shape != grad_slots_1.shape:
        raise RuntimeError("P1 Redirect gradient slot shape mismatch")
    if grad_pair.shape[-1] != grad_slots_0.shape[1] * 2:
        raise RuntimeError("P1 Redirect pair width must be two table widths")
    tensors = (
        grad_pair,
        indices_0,
        indices_1,
        row_to_slot_0,
        row_to_slot_1,
        grad_slots_0,
        grad_slots_1,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise RuntimeError("P1 Redirect is CUDA-only")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise RuntimeError("P1 Redirect requires contiguous tensors")

    num_tokens = indices_0.numel()
    table_cols = grad_slots_0.shape[1]
    capacity = grad_slots_0.shape[0]
    block_tokens = _P1_SCATTER_BLOCK_TOKENS
    block_cols = min(128, triton.next_power_of_2(table_cols))
    grid = (
        triton.cdiv(num_tokens, block_tokens),
        triton.cdiv(table_cols, block_cols),
    )
    _p1_scatter_pair_fp32_kernel[grid](
        grad_pair,
        indices_0,
        indices_1,
        row_to_slot_0,
        row_to_slot_1,
        grad_slots_0,
        grad_slots_1,
        num_tokens,
        TABLE_COLS=table_cols,
        CAPACITY=capacity,
        BLOCK_TOKENS=block_tokens,
        BLOCK_COLS=block_cols,
        num_warps=_P1_SCATTER_NUM_WARPS,
    )


@_p1_scatter_pair_fp32_op.register_fake
def _p1_scatter_pair_fp32_fake(
    grad_pair,
    indices_0,
    indices_1,
    row_to_slot_0,
    row_to_slot_1,
    grad_slots_0,
    grad_slots_1,
):
    return None


class _P1CompactPairRedirect(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pair_values,
        indices_0,
        indices_1,
        row_to_slot_0,
        row_to_slot_1,
        grad_slots_0,
        grad_slots_1,
    ):
        ctx.save_for_backward(
            indices_0,
            indices_1,
            row_to_slot_0,
            row_to_slot_1,
            grad_slots_0,
            grad_slots_1,
        )
        return pair_values

    @staticmethod
    def backward(ctx, grad_pair):
        (
            indices_0,
            indices_1,
            row_to_slot_0,
            row_to_slot_1,
            grad_slots_0,
            grad_slots_1,
        ) = ctx.saved_tensors
        _p1_scatter_pair_fp32_op(
            grad_pair.contiguous(),
            indices_0,
            indices_1,
            row_to_slot_0,
            row_to_slot_1,
            grad_slots_0,
            grad_slots_1,
        )
        # Returning no gradient for pair_values cuts the embedding-weight edge.
        return (None,) * 7


class _P1FusedPairLookupRedirect(torch.autograd.Function):
    """Fused gather/row registration with the same compact backward boundary."""

    @staticmethod
    def forward(
        ctx,
        weight_0,
        weight_1,
        indices_0,
        indices_1,
        stamps_0,
        active_rows_0,
        row_to_slot_0,
        counter_0,
        stamps_1,
        active_rows_1,
        row_to_slot_1,
        counter_1,
        generation,
        grad_slots_0,
        grad_slots_1,
    ):
        ctx.save_for_backward(
            indices_0,
            indices_1,
            row_to_slot_0,
            row_to_slot_1,
            grad_slots_0,
            grad_slots_1,
        )
        return _p1_fused_pair_lookup_op(
            weight_0,
            weight_1,
            indices_0,
            indices_1,
            stamps_0,
            active_rows_0,
            row_to_slot_0,
            counter_0,
            stamps_1,
            active_rows_1,
            row_to_slot_1,
            counter_1,
            generation,
        )

    @staticmethod
    def backward(ctx, grad_pair):
        (
            indices_0,
            indices_1,
            row_to_slot_0,
            row_to_slot_1,
            grad_slots_0,
            grad_slots_1,
        ) = ctx.saved_tensors
        _p1_scatter_pair_fp32_op(
            grad_pair.contiguous(),
            indices_0,
            indices_1,
            row_to_slot_0,
            row_to_slot_1,
            grad_slots_0,
            grad_slots_1,
        )
        # No dense embedding gradients are returned by the compact semantic.
        return (None,) * 15


@torch.library.custom_op(
    "apex_p1_compact_backward::scatter_pair_index_add",
    mutates_args=("grad_slots_0", "grad_slots_1"),
)
def _p1_scatter_pair_index_add_op(
    grad_pair: torch.Tensor,
    indices_0: torch.Tensor,
    indices_1: torch.Tensor,
    row_to_slot_0: torch.Tensor,
    row_to_slot_1: torch.Tensor,
    grad_slots_0: torch.Tensor,
    grad_slots_1: torch.Tensor,
) -> None:
    """Reference compact reduction used by the E2 engineering edge.

    ``index_add_`` is intentionally kept as a separate implementation from the
    Triton scatter. It still writes only compact FP32 slots, but gives the
    engineering receipt a real, direct-reduction stage before the optimized
    duplicate-row kernel is admitted.
    """
    if grad_pair.dtype != torch.bfloat16:
        raise RuntimeError("E2 index-add reduction requires BF16 post-cat gradients")
    if indices_0.dtype != torch.int64 or indices_1.dtype != torch.int64:
        raise RuntimeError("E2 index-add reduction requires int64 embedding indices")
    if row_to_slot_0.dtype != torch.int32 or row_to_slot_1.dtype != torch.int32:
        raise RuntimeError("E2 index-add reduction requires int32 row-to-slot maps")
    if grad_slots_0.dtype != torch.float32 or grad_slots_1.dtype != torch.float32:
        raise RuntimeError("E2 index-add reduction requires FP32 compact slots")
    if grad_pair.ndim != 3 or indices_0.shape != grad_pair.shape[:2]:
        raise RuntimeError("E2 index-add reduction expects grad[B,T,2C]")
    if indices_1.shape != indices_0.shape:
        raise RuntimeError("E2 index-add pair index shape mismatch")
    if grad_slots_0.ndim != 2 or grad_slots_0.shape != grad_slots_1.shape:
        raise RuntimeError("E2 index-add slot shape mismatch")
    tensors = (
        grad_pair,
        indices_0,
        indices_1,
        row_to_slot_0,
        row_to_slot_1,
        grad_slots_0,
        grad_slots_1,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise RuntimeError("E2 index-add reduction is CUDA-only")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise RuntimeError("E2 index-add reduction requires contiguous tensors")
    cols = grad_slots_0.shape[1]
    flat_0 = indices_0.reshape(-1)
    flat_1 = indices_1.reshape(-1)
    slots_0 = row_to_slot_0.index_select(0, flat_0).to(torch.int64)
    slots_1 = row_to_slot_1.index_select(0, flat_1).to(torch.int64)
    capacity = grad_slots_0.shape[0]
    if bool(
        ((slots_0 < 0) | (slots_0 >= capacity)).any().item()
    ) or bool(((slots_1 < 0) | (slots_1 >= capacity)).any().item()):
        raise RuntimeError("E2 index-add reduction found an unowned active row")
    values = grad_pair.reshape(-1, cols * 2).to(torch.float32)
    grad_slots_0.index_add_(0, slots_0, values[:, :cols])
    grad_slots_1.index_add_(0, slots_1, values[:, cols:])


@_p1_scatter_pair_index_add_op.register_fake
def _p1_scatter_pair_index_add_fake(
    grad_pair,
    indices_0,
    indices_1,
    row_to_slot_0,
    row_to_slot_1,
    grad_slots_0,
    grad_slots_1,
):
    return None


class _P1CompactPairRedirectIndexAdd(torch.autograd.Function):
    """Autograd boundary for E2's direct compact reduction."""

    @staticmethod
    def forward(
        ctx,
        pair_values,
        indices_0,
        indices_1,
        row_to_slot_0,
        row_to_slot_1,
        grad_slots_0,
        grad_slots_1,
    ):
        ctx.save_for_backward(
            indices_0,
            indices_1,
            row_to_slot_0,
            row_to_slot_1,
            grad_slots_0,
            grad_slots_1,
        )
        return pair_values

    @staticmethod
    def backward(ctx, grad_pair):
        (
            indices_0,
            indices_1,
            row_to_slot_0,
            row_to_slot_1,
            grad_slots_0,
            grad_slots_1,
        ) = ctx.saved_tensors
        _p1_scatter_pair_index_add_op(
            grad_pair.contiguous(),
            indices_0,
            indices_1,
            row_to_slot_0,
            row_to_slot_1,
            grad_slots_0,
            grad_slots_1,
        )
        return (None,) * 7


@triton.jit
def _rmsprop_touched_rows_kernel(
    param_ptr,
    grads,
    exp_avg_sq,
    touch_mask,
    lr,
    beta2,
    step,
    eps,
    NUM_ROWS: tl.constexpr,
    NUM_COLS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Scan the small row mask and issue table traffic only for touched rows."""
    # Widen before row-major pointer arithmetic: a 2048x table's
    # row_id*NUM_COLS exceeds signed int32 even though row_id itself fits.
    row_ids = (tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)).to(tl.int64)
    col_ids = tl.arange(0, BLOCK_COLS)
    valid_rows = row_ids < NUM_ROWS
    touched = tl.load(touch_mask + row_ids, mask=valid_rows, other=0) != 0
    active = valid_rows[:, None] & touched[:, None] & (col_ids[None, :] < NUM_COLS)
    offsets = row_ids[:, None] * NUM_COLS + col_ids[None, :]

    grad = tl.load(grads + offsets, mask=active, other=0.0).to(tl.float32)
    state = tl.load(exp_avg_sq + offsets, mask=active, other=0.0).to(tl.float32)
    param = tl.load(param_ptr + offsets, mask=active, other=0.0).to(tl.float32)

    # This mirrors the dense Inductor graph, including its FP32 fused
    # intermediates and CUDA libdevice powf/sqrtf. Runtime scalars avoid a new
    # Triton specialization for every optimizer step.
    one_minus_beta2 = 1.0 - beta2
    next_state_fp32 = state + one_minus_beta2 * (grad * grad - state)
    bias2 = 1.0 - triton_libdevice.pow(beta2, step)
    denom = triton_libdevice.sqrt(next_state_fp32 / bias2) + eps
    normalized = grad / denom
    negative_lr = 0.0 - lr
    next_param = param + normalized * negative_lr

    tl.store(exp_avg_sq + offsets, next_state_fp32, mask=active)
    tl.store(param_ptr + offsets, next_param, mask=active)


@triton.jit
def _rmsprop_active_rows_kernel(
    param_ptr,
    grads,
    exp_avg_sq,
    active_rows,
    counter,
    lr,
    beta2,
    step,
    eps,
    NUM_COLS: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Update an exact compact unique-row list with a fixed launch shape."""
    slot_ids = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_ids = tl.arange(0, BLOCK_COLS)
    count = tl.load(counter)
    valid_slots = (slot_ids < count) & (slot_ids < CAPACITY)
    # The 2048x trigram table has 16,777,216 rows.  ``row_id * NUM_COLS``
    # therefore exceeds int32 even though the row index itself fits in int32;
    # widen before pointer arithmetic to prevent wraparound/illegal access.
    row_ids = tl.load(active_rows + slot_ids, mask=valid_slots, other=0).to(tl.int64)
    active = valid_slots[:, None] & (col_ids[None, :] < NUM_COLS)
    offsets = row_ids[:, None] * NUM_COLS + col_ids[None, :]

    grad = tl.load(grads + offsets, mask=active, other=0.0).to(tl.float32)
    state = tl.load(exp_avg_sq + offsets, mask=active, other=0.0).to(tl.float32)
    param = tl.load(param_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    one_minus_beta2 = 1.0 - beta2
    next_state_fp32 = state + one_minus_beta2 * (grad * grad - state)
    bias2 = 1.0 - triton_libdevice.pow(beta2, step)
    denom = triton_libdevice.sqrt(next_state_fp32 / bias2) + eps
    normalized = grad / denom
    negative_lr = 0.0 - lr
    next_param = param + normalized * negative_lr
    tl.store(exp_avg_sq + offsets, next_state_fp32, mask=active)
    tl.store(param_ptr + offsets, next_param, mask=active)


@triton.jit
def _rmsprop_active_slots_kernel(
    param_ptr,
    grad_slots,
    exp_avg_sq,
    active_rows,
    counter,
    lr,
    beta2,
    step,
    eps,
    NUM_COLS: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    CLEAR_SLOTS: tl.constexpr,
):
    """Consume compact FP32 slots, with optional in-kernel slot clearing."""
    slot_ids = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_ids = tl.arange(0, BLOCK_COLS)
    count = tl.load(counter)
    valid_slots = (slot_ids < count) & (slot_ids < CAPACITY)
    # Widen before multiplying by NUM_COLS: 16.7M * 384 is above int32.
    row_ids = tl.load(active_rows + slot_ids, mask=valid_slots, other=0).to(tl.int64)
    active = valid_slots[:, None] & (col_ids[None, :] < NUM_COLS)
    table_offsets = row_ids[:, None] * NUM_COLS + col_ids[None, :]
    slot_offsets = slot_ids[:, None] * NUM_COLS + col_ids[None, :]

    accumulated = tl.load(grad_slots + slot_offsets, mask=active, other=0.0).to(
        tl.float32
    )
    # The atomic reduction is intentionally FP32 and order-dependent. This is
    # the sole BF16 quantization before matching Formal A's RMSProp arithmetic.
    grad = accumulated.to(tl.bfloat16).to(tl.float32)
    state = tl.load(exp_avg_sq + table_offsets, mask=active, other=0.0).to(
        tl.float32
    )
    param = tl.load(param_ptr + table_offsets, mask=active, other=0.0).to(
        tl.float32
    )
    one_minus_beta2 = 1.0 - beta2
    next_state_fp32 = state + one_minus_beta2 * (grad * grad - state)
    bias2 = 1.0 - triton_libdevice.pow(beta2, step)
    denom = triton_libdevice.sqrt(next_state_fp32 / bias2) + eps
    normalized = grad / denom
    next_param = param + normalized * (0.0 - lr)

    tl.store(exp_avg_sq + table_offsets, next_state_fp32, mask=active)
    tl.store(param_ptr + table_offsets, next_param, mask=active)
    if CLEAR_SLOTS:
        tl.store(grad_slots + slot_offsets, 0.0, mask=active)


@triton.jit
def _clear_compact_active_slots_kernel(
    grad_slots,
    counter,
    NUM_COLS: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Clear only the slots consumed by a non-fused engineering edge."""
    slot_ids = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_ids = tl.arange(0, BLOCK_COLS)
    count = tl.load(counter)
    active = (slot_ids < count) & (slot_ids < CAPACITY)
    active = active[:, None] & (col_ids[None, :] < NUM_COLS)
    offsets = slot_ids[:, None] * NUM_COLS + col_ids[None, :]
    tl.store(grad_slots + offsets, 0.0, mask=active)


def _scalar_float32(value):
    """Read a CPU scalar tensor without changing its float32 rounding."""
    return float(value.item())


def rmsprop_step_touched_reference(
    p, grad, exp_avg_sq, touch_mask, step_t, lr_t, beta2_t, eps_t, wd_t
):
    """Slow, auditable Route-A reference using the control's fused function."""
    if _scalar_float32(wd_t) != 0.0:
        raise RuntimeError("touched-row RMSProp requires zero n-gram weight decay")
    rows = torch.nonzero(touch_mask, as_tuple=False).flatten()
    if rows.numel() == 0:
        return
    touched_p = p.index_select(0, rows)
    touched_grad = grad.index_select(0, rows)
    touched_state = exp_avg_sq.index_select(0, rows)
    rmsprop_step_fused(
        touched_p,
        touched_grad,
        touched_state,
        step_t,
        lr_t,
        beta2_t,
        eps_t,
        wd_t,
    )
    p.index_copy_(0, rows, touched_p)
    exp_avg_sq.index_copy_(0, rows, touched_state)


def rmsprop_step_touched_triton(
    p, grad, exp_avg_sq, touch_mask, step_t, lr_t, beta2_t, eps_t, wd_t
):
    """Fixed-shape Triton Route-A implementation used by the candidate."""
    if _scalar_float32(wd_t) != 0.0:
        raise RuntimeError("touched-row RMSProp requires zero n-gram weight decay")
    if not (p.is_cuda and grad.is_cuda and exp_avg_sq.is_cuda and touch_mask.is_cuda):
        raise RuntimeError("Triton touched-row RMSProp requires CUDA tensors")
    if p.ndim != 2 or not (p.shape == grad.shape == exp_avg_sq.shape):
        raise RuntimeError("touched-row RMSProp expects matching rank-2 tensors")
    if touch_mask.shape != (p.shape[0],) or touch_mask.dtype != torch.int32:
        raise RuntimeError("invalid touched-row mask")
    if not (p.is_contiguous() and grad.is_contiguous() and exp_avg_sq.is_contiguous()):
        raise RuntimeError("touched-row RMSProp requires contiguous table tensors")

    num_rows, num_cols = p.shape
    block_cols = triton.next_power_of_2(num_cols)
    block_rows = 8 if block_cols <= 512 else 4
    grid = (triton.cdiv(num_rows, block_rows),)
    _rmsprop_touched_rows_kernel[grid](
        p,
        grad,
        exp_avg_sq,
        touch_mask,
        _scalar_float32(lr_t),
        _scalar_float32(beta2_t),
        _scalar_float32(step_t),
        _scalar_float32(eps_t),
        NUM_ROWS=num_rows,
        NUM_COLS=num_cols,
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
        num_warps=8,
    )


def rmsprop_step_compact_triton(
    p, grad, exp_avg_sq, active_rows, counter, step_t, lr_t, beta2_t, eps_t, wd_t
):
    """Generation-stamp/list Route-A implementation with fixed maximum grid."""
    if _scalar_float32(wd_t) != 0.0:
        raise RuntimeError("compact touched-row RMSProp requires zero n-gram weight decay")
    if not all(x.is_cuda for x in (p, grad, exp_avg_sq, active_rows, counter)):
        raise RuntimeError("compact touched-row RMSProp requires CUDA tensors")
    if p.ndim != 2 or not (p.shape == grad.shape == exp_avg_sq.shape):
        raise RuntimeError("compact touched-row RMSProp expects matching rank-2 tensors")
    if active_rows.ndim != 1 or active_rows.dtype != torch.int32:
        raise RuntimeError("invalid compact active-row list")
    if counter.shape != (1,) or counter.dtype != torch.int32:
        raise RuntimeError("invalid compact active-row counter")
    if not (p.is_contiguous() and grad.is_contiguous() and exp_avg_sq.is_contiguous()):
        raise RuntimeError("compact touched-row RMSProp requires contiguous table tensors")
    num_cols = p.shape[1]
    capacity = active_rows.numel()
    block_cols = triton.next_power_of_2(num_cols)
    block_rows = 8 if block_cols <= 512 else 4
    grid = (triton.cdiv(capacity, block_rows),)
    _rmsprop_active_rows_kernel[grid](
        p,
        grad,
        exp_avg_sq,
        active_rows,
        counter,
        _scalar_float32(lr_t),
        _scalar_float32(beta2_t),
        _scalar_float32(step_t),
        _scalar_float32(eps_t),
        NUM_COLS=num_cols,
        CAPACITY=capacity,
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
        num_warps=8,
    )


def rmsprop_step_compact_slots_triton(
    p,
    grad_slots,
    exp_avg_sq,
    active_rows,
    counter,
    step_t,
    lr_t,
    beta2_t,
    eps_t,
    wd_t,
    clear_slots=True,
):
    """P1 compact-gradient RMSProp with optional fused active-slot clearing."""
    if _scalar_float32(wd_t) != 0.0:
        raise RuntimeError("P1 compact RMSProp requires zero n-gram weight decay")
    if not all(x.is_cuda for x in (p, grad_slots, exp_avg_sq, active_rows, counter)):
        raise RuntimeError("P1 compact RMSProp requires CUDA tensors")
    if p.ndim != 2 or p.shape != exp_avg_sq.shape:
        raise RuntimeError("P1 compact RMSProp expects matching param/state tables")
    if p.dtype != torch.bfloat16 or exp_avg_sq.dtype != torch.bfloat16:
        raise RuntimeError("P1 compact RMSProp requires BF16 params and state")
    if grad_slots.dtype != torch.float32 or grad_slots.ndim != 2:
        raise RuntimeError("P1 compact RMSProp requires rank-2 FP32 gradient slots")
    if grad_slots.shape[1] != p.shape[1]:
        raise RuntimeError("P1 compact RMSProp slot width mismatch")
    if active_rows.ndim != 1 or active_rows.dtype != torch.int32:
        raise RuntimeError("invalid P1 compact active-row list")
    if grad_slots.shape[0] != active_rows.numel():
        raise RuntimeError("P1 compact RMSProp slot capacity mismatch")
    if counter.shape != (1,) or counter.dtype != torch.int32:
        raise RuntimeError("invalid P1 compact active-row counter")
    if not (p.is_contiguous() and grad_slots.is_contiguous() and exp_avg_sq.is_contiguous()):
        raise RuntimeError("P1 compact RMSProp requires contiguous tensors")
    num_cols = p.shape[1]
    capacity = active_rows.numel()
    block_cols = triton.next_power_of_2(num_cols)
    block_rows = 8 if block_cols <= 512 else 4
    grid = (triton.cdiv(capacity, block_rows),)
    _rmsprop_active_slots_kernel[grid](
        p,
        grad_slots,
        exp_avg_sq,
        active_rows,
        counter,
        _scalar_float32(lr_t),
        _scalar_float32(beta2_t),
        _scalar_float32(step_t),
        _scalar_float32(eps_t),
        NUM_COLS=num_cols,
        CAPACITY=capacity,
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
        CLEAR_SLOTS=bool(clear_slots),
        num_warps=8,
    )


def clear_compact_active_slots_triton(grad_slots, counter):
    """Separate active-slot cleanup for E2/E3 timing decomposition."""
    if not (grad_slots.is_cuda and counter.is_cuda):
        raise RuntimeError("compact slot cleanup requires CUDA tensors")
    if grad_slots.ndim != 2 or grad_slots.dtype != torch.float32:
        raise RuntimeError("compact slot cleanup expects rank-2 FP32 slots")
    if counter.shape != (1,) or counter.dtype != torch.int32:
        raise RuntimeError("compact slot cleanup expects one int32 counter")
    if not grad_slots.is_contiguous():
        raise RuntimeError("compact slot cleanup requires contiguous slots")
    num_cols = grad_slots.shape[1]
    capacity = grad_slots.shape[0]
    block_cols = triton.next_power_of_2(num_cols)
    block_rows = 8 if block_cols <= 512 else 4
    _clear_compact_active_slots_kernel[(triton.cdiv(capacity, block_rows),)](
        grad_slots,
        counter,
        NUM_COLS=num_cols,
        CAPACITY=capacity,
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
        num_warps=8,
    )


@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t**step_t
    bias2 = 1 - beta2_t**step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


@torch.compile(dynamic=False, fullgraph=True)
def rmsprop_step_fused(p, grad, exp_avg_sq, step_t, lr_t, beta2_t, eps_t, wd_t):
    """RMSProp with bias correction -- no first moment, saves 50% optimizer VRAM."""
    p.mul_(1 - lr_t * wd_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias2 = 1 - beta2_t**step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    p.add_(grad / denom, alpha=-lr_t)


@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads,
    stacked_params,
    momentum_buffer,
    second_momentum_buffer,
    momentum_t,
    lr_t,
    wd_t,
    beta2_t,
    ns_steps,
    red_dim,
):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(
        self,
        param_groups,
        rmsprop_touch_pairs=(),
        rmsprop_touch_capacity=None,
        p1_compact_backward=False,
        p2_forward_reuse=None,
        p1_reduction="fp32_atomic",
        p1_fused_cleanup=True,
    ):
        super().__init__(param_groups, defaults={})
        self._p1_compact_backward = bool(p1_compact_backward)
        self._p2_forward_reuse = (
            self._p1_compact_backward
            if p2_forward_reuse is None
            else bool(p2_forward_reuse)
        )
        if self._p1_compact_backward and not self._p2_forward_reuse:
            raise ValueError("P1 compact backward requires P2 forward reuse")
        if p1_reduction not in {"none", "index_add", "fp32_atomic"}:
            raise ValueError(
                "p1_reduction must be none, index_add, or fp32_atomic"
            )
        if self._p1_compact_backward and p1_reduction == "none":
            raise ValueError("P1 compact backward requires an explicit reduction mode")
        if not self._p1_compact_backward and p1_reduction == "index_add":
            raise ValueError("index_add reduction requires P1 compact backward")
        if (
            self._p1_compact_backward
            and p1_reduction == "index_add"
            and p1_fused_cleanup
        ):
            raise ValueError("index_add reduction requires separate slot cleanup")
        self._p1_reduction = p1_reduction
        self._p1_fused_cleanup = bool(p1_fused_cleanup)
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # RMSProp CPU tensors (no beta1 -- saves first moment VRAM)
        self._rmsprop_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rmsprop_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rmsprop_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rmsprop_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rmsprop_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rmsprop_impl = os.environ.get("TOUCHED_RMSPROP_IMPL", "compact")
        if self._rmsprop_impl == "triton":
            self._rmsprop_impl = "scan"
        if self._rmsprop_impl not in {"compact", "scan", "reference", "dense"}:
            raise ValueError(
                "TOUCHED_RMSPROP_IMPL must be one of: compact, scan, reference, dense"
            )
        if (self._p1_compact_backward or self._p2_forward_reuse) and self._rmsprop_impl != "compact":
            raise ValueError(
                "P1/P2 forward reuse requires TOUCHED_RMSPROP_IMPL=compact"
            )

        # All touched-row structures are optimizer scratch, not optimizer
        # state, so the control checkpoint/state schema remains unchanged.
        self._rmsprop_touch_pairs = tuple(rmsprop_touch_pairs)
        self._rmsprop_touch_masks = {}
        self._rmsprop_touch_stamps = {}
        self._rmsprop_active_rows = {}
        self._rmsprop_active_counters = {}
        self._rmsprop_row_to_slot = {}
        self._rmsprop_grad_slots = {}
        self._rmsprop_touch_params = set()
        self._rmsprop_order_by_param = {}
        self._rmsprop_capacity_by_param = {}
        self._rmsprop_touch_generation = None
        if self._rmsprop_impl == "compact":
            if rmsprop_touch_capacity is None:
                raise ValueError("compact touched-row RMSProp requires a positive capacity")
            (
                self._rmsprop_touch_capacities,
                self._rmsprop_touch_capacity_policy,
                self._rmsprop_touch_capacity_policy_id,
            ) = _p1_normalize_capacity_policy(rmsprop_touch_capacity)
            # Preserve the legacy scalar attribute for callers that inspect it;
            # all allocation and validation below uses the order-specific map.
            self._rmsprop_touch_capacity = (
                next(iter(self._rmsprop_touch_capacities.values()))
                if len(set(self._rmsprop_touch_capacities.values())) == 1
                else None
            )
        else:
            self._rmsprop_touch_capacity = 0
            self._rmsprop_touch_capacities = {}
            self._rmsprop_touch_capacity_policy = None
            self._rmsprop_touch_capacity_policy_id = None
        for order, pair, primes in self._rmsprop_touch_pairs:
            if order not in (2, 3) or len(pair) != 2 or len(primes) != 2:
                raise ValueError("invalid n-gram touched-row metadata")
            if pair[0].shape != pair[1].shape:
                raise ValueError("paired n-gram tables must have identical shapes")
            if self._rmsprop_impl == "compact" and order not in self._rmsprop_touch_capacities:
                raise ValueError(
                    f"compact capacity policy is missing order {order} capacity"
                )
            table_rows = pair[0].shape[0]
            if table_rows <= 0 or (table_rows & (table_rows - 1)) != 0:
                raise ValueError("touched-row hash masks require power-of-two table rows")
            for p in pair:
                if p in self._rmsprop_touch_params:
                    raise ValueError("duplicate n-gram table in touched-row metadata")
                self._rmsprop_touch_params.add(p)
                self._rmsprop_order_by_param[p] = order
                if self._rmsprop_impl in {"scan", "reference"}:
                    self._rmsprop_touch_masks[p] = torch.zeros(
                        table_rows, dtype=torch.int32, device=p.device
                    )
                elif self._rmsprop_impl == "compact":
                    capacity = min(table_rows, self._rmsprop_touch_capacities[order])
                    self._rmsprop_capacity_by_param[p] = capacity
                    self._rmsprop_touch_stamps[p] = torch.zeros(
                        table_rows, dtype=torch.int32, device=p.device
                    )
                    self._rmsprop_active_rows[p] = torch.empty(
                        capacity, dtype=torch.int32, device=p.device
                    )
                    self._rmsprop_row_to_slot[p] = torch.full(
                        (table_rows,), -1, dtype=torch.int32, device=p.device
                    )
                    self._rmsprop_active_counters[p] = torch.zeros(
                        1, dtype=torch.int32, device=p.device
                    )
                    if self._p1_compact_backward or self._p2_forward_reuse:
                        if self._rmsprop_touch_generation is None:
                            self._rmsprop_touch_generation = torch.zeros(
                                1, dtype=torch.int32, device=p.device
                            )
                        elif self._rmsprop_touch_generation.device != p.device:
                            raise ValueError("P1 compact tables must share one CUDA device")
                    if self._p1_compact_backward:
                        self._rmsprop_grad_slots[p] = torch.zeros(
                            capacity,
                            p.shape[1],
                            dtype=torch.float32,
                            device=p.device,
                        )
        expected_rmsprop_params = {
            p
            for group in self.param_groups
            if group["kind"] == "rmsprop"
            for p in group["params"]
        }
        if self._rmsprop_impl != "dense" and self._rmsprop_touch_params != expected_rmsprop_params:
            raise ValueError("every RMSProp parameter must have exactly one touch specification")
        if (self._p1_compact_backward or self._p2_forward_reuse) and self._rmsprop_touch_generation is None:
            raise ValueError("P1/P2 forward reuse requires at least one n-gram table")
        if self._p1_compact_backward or self._p2_forward_reuse:
            scratch_tensors = (
                list(self._rmsprop_touch_stamps.values())
                + list(self._rmsprop_active_rows.values())
                + list(self._rmsprop_row_to_slot.values())
                + list(self._rmsprop_active_counters.values())
                + list(self._rmsprop_grad_slots.values())
                + [self._rmsprop_touch_generation]
            )
            for scratch in scratch_tensors:
                torch._dynamo.mark_static_address(scratch)
        self._touch_step_open = False
        self._touch_batches = 0
        self._touch_token_count = 0
        self._touch_generation = 0
        self._optimizer_step_count = 0
        self._state_audit_every = int(os.environ.get("TOUCHED_RMSPROP_AUDIT_EVERY", "0"))
        step_fns = {
            "adamw": self._step_adamw,
            "rmsprop": self._step_rmsprop,
            "muon": self._step_muon,
        }
        self._step_dispatch = tuple((step_fns[group["kind"]], group) for group in self.param_groups)

    @torch.no_grad()
    def begin_step(self):
        """Reset touched-row scratch once before accumulating an optimizer step."""
        if self._rmsprop_impl == "dense":
            return
        if self._touch_step_open:
            raise RuntimeError("begin_step called twice without optimizer.step")
        if self._rmsprop_impl in {"scan", "reference"}:
            torch._foreach_zero_(list(self._rmsprop_touch_masks.values()))
        else:
            torch._foreach_zero_(list(self._rmsprop_active_counters.values()))
            # The training budget is far below int32 wraparound. Keep a
            # defensive slow-path reset for long-running local harnesses.
            if self._touch_generation >= 2_147_483_646:
                torch._foreach_zero_(list(self._rmsprop_touch_stamps.values()))
                self._touch_generation = 1
            else:
                self._touch_generation += 1
            if self._p1_compact_backward or self._p2_forward_reuse:
                self._rmsprop_touch_generation.fill_(self._touch_generation)
        self._touch_step_open = True
        self._touch_batches = 0
        self._touch_token_count = 0

    @torch.no_grad()
    def _record_touch_batch(self, tokens, caller):
        if self._rmsprop_impl == "dense":
            return 0
        if not self._touch_step_open:
            raise RuntimeError(f"{caller} requires begin_step first")
        if self._p1_compact_backward and self._touch_batches != 0:
            raise RuntimeError("P1 compact backward requires grad_accum=1")
        if tokens.ndim != 2 or not tokens.is_cuda or not tokens.is_contiguous():
            raise RuntimeError("ngram tokens must be a contiguous CUDA [batch, sequence] tensor")
        num_tokens = tokens.numel()
        # ``num_tokens`` is only a lifecycle/diagnostic count.  It is not a
        # slot-capacity proof: a table can touch far fewer unique rows than
        # tokens, and bigram/trigram orders intentionally have different
        # budgets.  The collector's per-table counters are validated before
        # any optimizer update in ``step``.
        self._touch_token_count += num_tokens
        self._touch_batches += 1
        return num_tokens

    @torch.no_grad()
    def prepare_ngram_forward(self, tokens):
        """Record P2 lifecycle without rereading tokens or hashing.

        Unique-row capacity is checked after the collector has observed the
        actual hash indices, immediately before optimizer updates.
        """
        if not self._p2_forward_reuse or self._rmsprop_impl != "compact":
            raise RuntimeError("prepare_ngram_forward is only valid for compact P2/P1")
        self._record_touch_batch(tokens, "prepare_ngram_forward")

    @torch.no_grad()
    def mark_ngram_rows(self, tokens):
        """Legacy exact token/hash collector for Formal A and isolated tests."""
        num_tokens = self._record_touch_batch(tokens, "mark_ngram_rows")
        if self._rmsprop_impl == "dense":
            return
        seq_len = tokens.shape[1]
        block_size = 256
        grid = (triton.cdiv(num_tokens, block_size),)
        for order, pair, primes in self._rmsprop_touch_pairs:
            table_rows = pair[0].shape[0]
            common = {
                "SEQ_LEN": seq_len,
                "TABLE_MASK": table_rows - 1,
                "BLOCK_SIZE": block_size,
                "num_warps": 4,
            }
            if self._rmsprop_impl in {"scan", "reference"}:
                mask_0 = self._rmsprop_touch_masks[pair[0]]
                mask_1 = self._rmsprop_touch_masks[pair[1]]
                if order == 2:
                    _mark_bigram_pair_kernel[grid](
                        tokens,
                        mask_0,
                        mask_1,
                        num_tokens,
                        primes[0][0],
                        primes[0][1],
                        primes[1][0],
                        primes[1][1],
                        **common,
                    )
                else:
                    _mark_trigram_pair_kernel[grid](
                        tokens,
                        mask_0,
                        mask_1,
                        num_tokens,
                        primes[0][0],
                        primes[0][1],
                        primes[0][2],
                        primes[1][0],
                        primes[1][1],
                        primes[1][2],
                        **common,
                    )
            else:
                p0, p1 = pair
                compact_common = {
                    **common,
                    "CAPACITY": self._rmsprop_active_rows[p0].numel(),
                }
                compact_args = (
                    tokens,
                    self._rmsprop_touch_stamps[p0],
                    self._rmsprop_active_rows[p0],
                    self._rmsprop_row_to_slot[p0],
                    self._rmsprop_active_counters[p0],
                    self._rmsprop_touch_stamps[p1],
                    self._rmsprop_active_rows[p1],
                    self._rmsprop_row_to_slot[p1],
                    self._rmsprop_active_counters[p1],
                    num_tokens,
                    self._touch_generation,
                )
                if order == 2:
                    _collect_bigram_pair_kernel[grid](
                        *compact_args,
                        primes[0][0],
                        primes[0][1],
                        primes[1][0],
                        primes[1][1],
                        **compact_common,
                    )
                else:
                    _collect_trigram_pair_kernel[grid](
                        *compact_args,
                        primes[0][0],
                        primes[0][1],
                        primes[0][2],
                        primes[1][0],
                        primes[1][1],
                        primes[1][2],
                        **compact_common,
                    )

    @torch.no_grad()
    def _validate_compact_capacity_counters(self):
        """Reject any collector overflow before mutating model or RMS state.

        The Triton collector intentionally keeps launching after a batch has
        more tokens than the slot budget because duplicate indices may still
        collapse below that budget.  A genuinely overflowing table increments
        its counter past the fixed list, so this host-side check is the final
        fail-closed boundary before ``optimizer.step`` dispatches updates.
        """
        if self._rmsprop_impl != "compact":
            return
        overflow = []
        for p, counter in self._rmsprop_active_counters.items():
            capacity = self._rmsprop_capacity_by_param[p]
            count = int(counter.item())
            if count < 0 or count > capacity:
                order = self._rmsprop_order_by_param[p]
                overflow.append(
                    {
                        "order": order,
                        "capacity": capacity,
                        "active": count,
                    }
                )
        if overflow:
            # The current step is unusable: discard every compact gradient
            # before permitting a fresh step. The collector may have already
            # filled valid slots before a later row exceeded capacity; keeping
            # those values would contaminate the next generation when slot IDs
            # are reused after the rejection.
            if self._p1_compact_backward:
                for grad_slots in self._rmsprop_grad_slots.values():
                    grad_slots.zero_()
            # Permit the caller to begin a fresh step, but never let a later
            # call accidentally apply partial state.
            self._touch_step_open = False
            raise RuntimeError(
                "P1 compact active-row counter overflow; update rejected "
                f"(tables={overflow!r})"
            )

    def _step_adamw(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])
            adamw_step_fused(
                p,
                grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )

    def _step_rmsprop(self, group):
        """RMSProp: only second moment, no first moment -- 50% less optimizer VRAM for sparse tables."""
        for p in group["params"]:
            if self._p1_compact_backward:
                if p.grad is not None:
                    raise RuntimeError("P1 n-gram parameters must keep p.grad=None")
                grad = None
            else:
                if p.grad is None:
                    continue
                grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg_sq"] = torch.zeros_like(p)
                # Note: NO exp_avg allocated -- this is the VRAM saving
            state["step"] += 1
            self._rmsprop_step_t.fill_(state["step"])
            self._rmsprop_lr_t.fill_(group["lr"])
            self._rmsprop_beta2_t.fill_(group["beta2"])
            self._rmsprop_eps_t.fill_(group["eps"])
            self._rmsprop_wd_t.fill_(group["weight_decay"])
            scalar_args = (
                self._rmsprop_step_t,
                self._rmsprop_lr_t,
                self._rmsprop_beta2_t,
                self._rmsprop_eps_t,
                self._rmsprop_wd_t,
            )
            if self._p1_compact_backward:
                rmsprop_step_compact_slots_triton(
                    p,
                    self._rmsprop_grad_slots[p],
                    state["exp_avg_sq"],
                    self._rmsprop_active_rows[p],
                    self._rmsprop_active_counters[p],
                    *scalar_args,
                    clear_slots=self._p1_fused_cleanup,
                )
                if not self._p1_fused_cleanup:
                    clear_compact_active_slots_triton(
                        self._rmsprop_grad_slots[p],
                        self._rmsprop_active_counters[p],
                    )
                continue
            args = (p, grad, state["exp_avg_sq"], *scalar_args)
            if self._rmsprop_impl == "dense":
                rmsprop_step_fused(*args)
            elif self._rmsprop_impl in {"scan", "reference"}:
                touch_mask = self._rmsprop_touch_masks[p]
                touched_args = args[:3] + (touch_mask,) + args[3:]
                if self._rmsprop_impl == "reference":
                    rmsprop_step_touched_reference(*touched_args)
                else:
                    rmsprop_step_touched_triton(*touched_args)
            else:
                compact_args = (
                    args[:3]
                    + (
                        self._rmsprop_active_rows[p],
                        self._rmsprop_active_counters[p],
                    )
                    + args[3:]
                )
                rmsprop_step_compact_triton(*compact_args)

    def _step_muon(self, group):
        params = group["params"]
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (
                (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            )
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(
            stacked_grads,
            stacked_params,
            state["momentum_buffer"],
            state["second_momentum_buffer"],
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        if self._rmsprop_impl != "dense":
            if not self._touch_step_open or self._touch_batches == 0:
                raise RuntimeError("optimizer.step requires touched-row masks for this step")
            if self._p1_compact_backward and self._touch_batches != 1:
                raise RuntimeError("P1 compact backward requires exactly one marked batch")
            # Validate every active counter before dispatching any parameter
            # group.  This ordering prevents a later-table overflow from
            # leaving earlier AdamW/Muon/RMSProp groups partially updated.
            self._validate_compact_capacity_counters()
        for step_fn, group in self._step_dispatch:
            step_fn(group)
        self._optimizer_step_count += 1
        if (
            self._state_audit_every > 0
            and self._optimizer_step_count % self._state_audit_every == 0
        ):
            max_state = 0.0
            for p in self._rmsprop_touch_params:
                state = self.state[p].get("exp_avg_sq")
                if state is None:
                    continue
                if not bool(torch.isfinite(state).all().item()):
                    raise RuntimeError("non-finite n-gram RMS state violates exact skip invariant")
                if not bool((state >= 0).all().item()):
                    raise RuntimeError("negative n-gram RMS state violates exact skip invariant")
                max_state = max(max_state, float(state.max().item()))
            if self._rmsprop_impl == "compact":
                max_touched = max(
                    int(counter.item()) for counter in self._rmsprop_active_counters.values()
                )
            else:
                max_touched = -1
            print(
                f"TOUCHED_RMSPROP_AUDIT step={self._optimizer_step_count} "
                f"max_state={max_state:.8g} max_touched={max_touched}"
            )
        self._touch_step_open = False


@torch.no_grad()
def _run_touched_rmsprop_selftest():
    """CUDA parity across the real beta2 ramp plus exact hash-boundary tests."""
    if not torch.cuda.is_available():
        raise RuntimeError("TOUCHED_RMSPROP_SELF_TEST requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(1234)
    rows, cols = 257, 384
    p0 = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
    g0 = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
    # Row 33 is deliberately marked despite having an all-zero gradient. A
    # mask may conservatively include such a row after gradient cancellation.
    active_rows = torch.tensor([0, 1, 17, 128, 256], device=device)
    marked_rows = torch.tensor([0, 1, 17, 33, 128, 256], device=device)
    grad = torch.zeros_like(g0)
    grad.index_copy_(0, active_rows, g0.index_select(0, active_rows))
    state = torch.rand_like(p0).abs()
    # Include positive BF16 boundary values in both marked and skipped rows.
    boundary_bits = torch.tensor(
        [0x0000, 0x0001, 0x007F, 0x0080, 0x3F80, 0x7F7F], dtype=torch.uint16
    )
    state.flatten()[: boundary_bits.numel()].copy_(boundary_bits.view(torch.bfloat16).to(device))
    state[2, : boundary_bits.numel()].copy_(boundary_bits.view(torch.bfloat16).to(device))
    param_zero_bits = torch.tensor([0x0000, 0x8000], dtype=torch.uint16)
    p0[2, :2].copy_(param_zero_bits.view(torch.bfloat16).to(device))
    mask = torch.zeros(rows, dtype=torch.int32, device=device)
    mask.index_fill_(0, marked_rows, 1)
    compact_rows = torch.empty(64, dtype=torch.int32, device=device)
    compact_rows[: marked_rows.numel()].copy_(marked_rows.to(torch.int32))
    compact_counter = torch.tensor(
        [marked_rows.numel()], dtype=torch.int32, device=device
    )
    lr_t = torch.tensor(0.2, dtype=torch.float32)
    eps_t = torch.tensor(1e-10, dtype=torch.float32)
    wd_t = torch.tensor(0.0, dtype=torch.float32)
    # Exhaust the full reachable BF16 state encoding space through the actual
    # compiled dense control kernel, not only through a scalar reference.
    exhaustive_bits = torch.arange(0x7F80, dtype=torch.int32).to(torch.uint16)
    exhaustive_state = exhaustive_bits.view(torch.bfloat16).to(device).view(-1, 1)
    exhaustive_grad = torch.zeros_like(exhaustive_state)
    exhaustive_param = torch.zeros_like(exhaustive_state)
    exhaustive_step = torch.tensor(4096.0, dtype=torch.float32)
    for exhaustive_beta in (0.999, 0.9999):
        dense_state = exhaustive_state.clone()
        dense_param = exhaustive_param.clone()
        exhaustive_beta_t = torch.tensor(exhaustive_beta, dtype=torch.float32)
        rmsprop_step_fused(
            dense_param,
            exhaustive_grad,
            dense_state,
            exhaustive_step,
            lr_t,
            exhaustive_beta_t,
            eps_t,
            wd_t,
        )
        torch.cuda.synchronize()
        if not torch.equal(dense_state.view(torch.int16), exhaustive_state.view(torch.int16)):
            raise AssertionError(
                f"exhaustive untouched dense state changed at beta2={exhaustive_beta}"
            )
    parity_cases = (
        (1, 0.999),
        (3, 0.999),
        (997, 0.999),
        (2048, 0.99945),
        (4096, 0.9999),
    )
    untouched = mask == 0
    for step, beta2 in parity_cases:
        p_dense, state_dense = p0.clone(), state.clone()
        p_fast, state_fast = p0.clone(), state.clone()
        p_compact, state_compact = p0.clone(), state.clone()
        step_t = torch.tensor(float(step), dtype=torch.float32)
        beta2_t = torch.tensor(beta2, dtype=torch.float32)
        rmsprop_step_fused(
            p_dense, grad, state_dense, step_t, lr_t, beta2_t, eps_t, wd_t
        )
        rmsprop_step_touched_triton(
            p_fast, grad, state_fast, mask, step_t, lr_t, beta2_t, eps_t, wd_t
        )
        rmsprop_step_compact_triton(
            p_compact,
            grad,
            state_compact,
            compact_rows,
            compact_counter,
            step_t,
            lr_t,
            beta2_t,
            eps_t,
            wd_t,
        )
        torch.cuda.synchronize()
        p_diff = int((p_dense.view(torch.int16) != p_fast.view(torch.int16)).sum().item())
        s_diff = int(
            (state_dense.view(torch.int16) != state_fast.view(torch.int16)).sum().item()
        )
        if p_diff or s_diff:
            raise AssertionError(
                f"touched-row parity failed at step={step} beta2={beta2}: "
                f"param_bits={p_diff} state_bits={s_diff}"
            )
        compact_p_diff = int(
            (p_dense.view(torch.int16) != p_compact.view(torch.int16)).sum().item()
        )
        compact_s_diff = int(
            (state_dense.view(torch.int16) != state_compact.view(torch.int16)).sum().item()
        )
        if compact_p_diff or compact_s_diff:
            raise AssertionError(
                f"compact parity failed at step={step} beta2={beta2}: "
                f"param_bits={compact_p_diff} state_bits={compact_s_diff}"
            )
        if not torch.equal(state_dense[untouched].view(torch.int16), state[untouched].view(torch.int16)):
            raise AssertionError(f"dense untouched state changed at beta2={beta2}")

    # Verify both marker kernels against the model's exact boundary convention.
    batch, seq_len, table_rows = 5, 7, 256
    tokens = torch.randint(0, 32768, (batch, seq_len), device=device, dtype=torch.long)
    mask0 = torch.zeros(table_rows, dtype=torch.int32, device=device)
    mask1 = torch.zeros(table_rows, dtype=torch.int32, device=device)
    bigram_primes = ((2654435761, 2246822519), (1013904223, 6291469))
    _mark_bigram_pair_kernel[(triton.cdiv(tokens.numel(), 256),)](
        tokens,
        mask0,
        mask1,
        tokens.numel(),
        bigram_primes[0][0],
        bigram_primes[0][1],
        bigram_primes[1][0],
        bigram_primes[1][1],
        SEQ_LEN=seq_len,
        TABLE_MASK=table_rows - 1,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    prev = torch.cat([tokens[:, :1], tokens[:, :-1]], dim=1)
    expected0 = (((prev * bigram_primes[0][0]) ^ (tokens * bigram_primes[0][1])) & (table_rows - 1)).flatten()
    expected1 = (((prev * bigram_primes[1][0]) ^ (tokens * bigram_primes[1][1])) & (table_rows - 1)).flatten()
    bigram_expected0 = expected0.clone()
    bigram_expected1 = expected1.clone()
    expected_mask0 = torch.zeros_like(mask0).index_fill(0, expected0.unique(), 1)
    expected_mask1 = torch.zeros_like(mask1).index_fill(0, expected1.unique(), 1)
    if not torch.equal(mask0, expected_mask0) or not torch.equal(mask1, expected_mask1):
        raise AssertionError("bigram touched-row marker mismatch")
    stamps0 = torch.zeros(table_rows, dtype=torch.int32, device=device)
    stamps1 = torch.zeros(table_rows, dtype=torch.int32, device=device)
    list0 = torch.empty(tokens.numel(), dtype=torch.int32, device=device)
    list1 = torch.empty(tokens.numel(), dtype=torch.int32, device=device)
    row_to_slot0 = torch.full((table_rows,), -1, dtype=torch.int32, device=device)
    row_to_slot1 = torch.full((table_rows,), -1, dtype=torch.int32, device=device)
    counter0 = torch.zeros(1, dtype=torch.int32, device=device)
    counter1 = torch.zeros(1, dtype=torch.int32, device=device)
    _collect_bigram_pair_kernel[(triton.cdiv(tokens.numel(), 256),)](
        tokens,
        stamps0,
        list0,
        row_to_slot0,
        counter0,
        stamps1,
        list1,
        row_to_slot1,
        counter1,
        tokens.numel(),
        1,
        *bigram_primes[0],
        *bigram_primes[1],
        SEQ_LEN=seq_len,
        TABLE_MASK=table_rows - 1,
        CAPACITY=tokens.numel(),
        BLOCK_SIZE=256,
        num_warps=4,
    )
    torch.cuda.synchronize()
    count0, count1 = int(counter0.item()), int(counter1.item())
    if not torch.equal(list0[:count0].sort().values.long(), expected0.unique().sort().values):
        raise AssertionError("bigram compact collector table-0 mismatch")
    if not torch.equal(list1[:count1].sort().values.long(), expected1.unique().sort().values):
        raise AssertionError("bigram compact collector table-1 mismatch")
    if not torch.equal(
        row_to_slot0.index_select(0, list0[:count0].long()),
        torch.arange(count0, device=device, dtype=torch.int32),
    ):
        raise AssertionError("bigram row-to-slot table-0 mismatch")
    if not torch.equal(
        row_to_slot1.index_select(0, list1[:count1].long()),
        torch.arange(count1, device=device, dtype=torch.int32),
    ):
        raise AssertionError("bigram row-to-slot table-1 mismatch")
    # Re-marking the same generation must not append duplicates.
    before = (count0, count1)
    _collect_bigram_pair_kernel[(triton.cdiv(tokens.numel(), 256),)](
        tokens,
        stamps0,
        list0,
        row_to_slot0,
        counter0,
        stamps1,
        list1,
        row_to_slot1,
        counter1,
        tokens.numel(),
        1,
        *bigram_primes[0],
        *bigram_primes[1],
        SEQ_LEN=seq_len,
        TABLE_MASK=table_rows - 1,
        CAPACITY=tokens.numel(),
        BLOCK_SIZE=256,
        num_warps=4,
    )
    torch.cuda.synchronize()
    if (int(counter0.item()), int(counter1.item())) != before:
        raise AssertionError("generation stamp admitted duplicate rows")
    mask0.zero_()
    mask1.zero_()
    trigram_primes = (
        (16777619, 2166136261, 3432918353),
        (461845907, 2654435769, 1540483477),
    )
    _mark_trigram_pair_kernel[(triton.cdiv(tokens.numel(), 256),)](
        tokens,
        mask0,
        mask1,
        tokens.numel(),
        *trigram_primes[0],
        *trigram_primes[1],
        SEQ_LEN=seq_len,
        TABLE_MASK=table_rows - 1,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    prev2 = torch.cat([tokens[:, :2], tokens[:, :-2]], dim=1)
    expected0 = (
        (prev2 * trigram_primes[0][0])
        ^ (prev * trigram_primes[0][1])
        ^ (tokens * trigram_primes[0][2])
    ).bitwise_and(table_rows - 1).flatten()
    expected1 = (
        (prev2 * trigram_primes[1][0])
        ^ (prev * trigram_primes[1][1])
        ^ (tokens * trigram_primes[1][2])
    ).bitwise_and(table_rows - 1).flatten()
    expected_mask0.zero_().index_fill_(0, expected0.unique(), 1)
    expected_mask1.zero_().index_fill_(0, expected1.unique(), 1)
    if not torch.equal(mask0, expected_mask0) or not torch.equal(mask1, expected_mask1):
        raise AssertionError("trigram touched-row marker mismatch")
    counter0.zero_()
    counter1.zero_()
    _collect_trigram_pair_kernel[(triton.cdiv(tokens.numel(), 256),)](
        tokens,
        stamps0,
        list0,
        row_to_slot0,
        counter0,
        stamps1,
        list1,
        row_to_slot1,
        counter1,
        tokens.numel(),
        2,
        *trigram_primes[0],
        *trigram_primes[1],
        SEQ_LEN=seq_len,
        TABLE_MASK=table_rows - 1,
        CAPACITY=tokens.numel(),
        BLOCK_SIZE=256,
        num_warps=4,
    )
    torch.cuda.synchronize()
    count0, count1 = int(counter0.item()), int(counter1.item())
    if not torch.equal(list0[:count0].sort().values.long(), expected0.unique().sort().values):
        raise AssertionError("trigram compact collector table-0 mismatch")
    if not torch.equal(list1[:count1].sort().values.long(), expected1.unique().sort().values):
        raise AssertionError("trigram compact collector table-1 mismatch")

    # Exercise the optimizer integration (begin -> mark -> step), not just the
    # standalone kernels, with the same parameter-group schema as control.
    mini_p0 = torch.nn.Parameter(
        torch.randn(table_rows, 16, device=device, dtype=torch.bfloat16)
    )
    mini_p1 = torch.nn.Parameter(
        torch.randn(table_rows, 16, device=device, dtype=torch.bfloat16)
    )
    mini_initial0 = mini_p0.detach().clone()
    mini_initial1 = mini_p1.detach().clone()
    mini_grad0 = torch.zeros_like(mini_p0)
    mini_grad1 = torch.zeros_like(mini_p1)
    mini_grad0.index_fill_(0, bigram_expected0.unique(), 0.125)
    mini_grad1.index_fill_(0, bigram_expected1.unique(), -0.25)
    mini_p0.grad = mini_grad0
    mini_p1.grad = mini_grad1
    mini_group = {
        "kind": "rmsprop",
        "params": [mini_p0, mini_p1],
        "lr": 0.2,
        "beta2": 0.999,
        "eps": 1e-10,
        "weight_decay": 0.0,
    }
    mini_optimizer = MuonAdamW(
        [mini_group],
        rmsprop_touch_pairs=[(2, (mini_p0, mini_p1), bigram_primes)],
        rmsprop_touch_capacity=tokens.numel(),
    )
    mini_optimizer.begin_step()
    mini_optimizer.mark_ngram_rows(tokens)
    mini_optimizer.step()
    dense_p0, dense_p1 = mini_initial0.clone(), mini_initial1.clone()
    dense_s0 = torch.zeros_like(dense_p0)
    dense_s1 = torch.zeros_like(dense_p1)
    step1_t = torch.tensor(1.0, dtype=torch.float32)
    beta999_t = torch.tensor(0.999, dtype=torch.float32)
    rmsprop_step_fused(
        dense_p0, mini_grad0, dense_s0, step1_t, lr_t, beta999_t, eps_t, wd_t
    )
    rmsprop_step_fused(
        dense_p1, mini_grad1, dense_s1, step1_t, lr_t, beta999_t, eps_t, wd_t
    )
    for label, actual, expected in (
        ("param0", mini_p0, dense_p0),
        ("param1", mini_p1, dense_p1),
        ("state0", mini_optimizer.state[mini_p0]["exp_avg_sq"], dense_s0),
        ("state1", mini_optimizer.state[mini_p1]["exp_avg_sq"], dense_s1),
    ):
        if not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
            raise AssertionError(f"compact optimizer integration mismatch: {label}")
    # The optimizer is outside the compiled model graph, but its fixed-shape
    # update must remain CUDA-graph capturable for future whole-step capture.
    graph_p = mini_initial0.clone()
    graph_s = torch.zeros_like(graph_p)
    rmsprop_step_compact_triton(
        graph_p,
        mini_grad0,
        graph_s,
        mini_optimizer._rmsprop_active_rows[mini_p0],
        mini_optimizer._rmsprop_active_counters[mini_p0],
        step1_t,
        lr_t,
        beta999_t,
        eps_t,
        wd_t,
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rmsprop_step_compact_triton(
            graph_p,
            mini_grad0,
            graph_s,
            mini_optimizer._rmsprop_active_rows[mini_p0],
            mini_optimizer._rmsprop_active_counters[mini_p0],
            step1_t,
            lr_t,
            beta999_t,
            eps_t,
            wd_t,
        )
    graph.replay()
    torch.cuda.synchronize()
    print(
        "TOUCHED_RMSPROP_SELF_TEST PASS "
        f"parity=bitwise cases={len(parity_cases)} "
        "bf16_states=32640 markers=bigram+trigram collector=exact "
        "optimizer=integrated cudagraph=pass"
    )




def _run_p1_compact_backward_selftest():
    """Tiny H200 correctness/fullgraph test for the new P1 training semantics."""
    if not torch.cuda.is_available():
        raise RuntimeError("P1_COMPACT_BACKWARD_SELF_TEST requires CUDA")
    if not P1_COMPACT_BACKWARD_ENABLED:
        raise RuntimeError("P1 self-test requires P1_COMPACT_BACKWARD=1")
    device = torch.device("cuda")
    torch.manual_seed(20260817)
    rows, cols = 256, 16
    batch, seq_len = 4, 8
    capacity = batch * seq_len
    primes = ((2654435761, 2246822519), (1013904223, 6291469))
    tokens = torch.randint(0, 64, (batch, seq_len), device=device, dtype=torch.long)
    previous = torch.cat([tokens[:, :1], tokens[:, :-1]], dim=1)
    indices_0 = ((previous * primes[0][0]) ^ (tokens * primes[0][1])).bitwise_and(rows - 1)
    indices_1 = ((previous * primes[1][0]) ^ (tokens * primes[1][1])).bitwise_and(rows - 1)

    table_0 = torch.nn.Parameter(
        torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
    )
    table_1 = torch.nn.Parameter(
        torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
    )
    group = {
        "kind": "rmsprop",
        "params": [table_0, table_1],
        "lr": 0.2,
        "beta2": 0.999,
        "eps": 1e-10,
        "weight_decay": 0.0,
    }
    optimizer = MuonAdamW(
        [group],
        rmsprop_touch_pairs=((2, (table_0, table_1), primes),),
        rmsprop_touch_capacity=capacity,
        p1_compact_backward=True,
    )
    optimizer.begin_step()
    optimizer.mark_ngram_rows(tokens)
    torch.cuda.synchronize()

    for table in (table_0, table_1):
        count = int(optimizer._rmsprop_active_counters[table].item())
        active = optimizer._rmsprop_active_rows[table][:count]
        mapped = optimizer._rmsprop_row_to_slot[table].index_select(0, active.long())
        if not torch.equal(mapped, torch.arange(count, device=device, dtype=torch.int32)):
            raise AssertionError("P1 collector row-to-slot mapping mismatch")

    upstream = torch.randn(
        batch, seq_len, cols * 2, device=device, dtype=torch.bfloat16
    ).mul_(0.05)
    row_to_slot_0 = optimizer._rmsprop_row_to_slot[table_0]
    row_to_slot_1 = optimizer._rmsprop_row_to_slot[table_1]
    grad_slots_0 = optimizer._rmsprop_grad_slots[table_0]
    grad_slots_1 = optimizer._rmsprop_grad_slots[table_1]

    def redirected_objective(
        weight_0,
        weight_1,
        idx_0,
        idx_1,
        grad_out,
        map_0,
        map_1,
        slots_0,
        slots_1,
    ):
        pair = torch.cat(
            [F.embedding(idx_0, weight_0), F.embedding(idx_1, weight_1)], dim=-1
        )
        pair = _P1CompactPairRedirect.apply(
            pair, idx_0, idx_1, map_0, map_1, slots_0, slots_1
        )
        return (pair * grad_out).float().sum()

    compiled_objective = torch.compile(
        redirected_objective, dynamic=False, fullgraph=True
    )
    compile_args = (
        table_0,
        table_1,
        indices_0,
        indices_1,
        upstream,
        row_to_slot_0,
        row_to_slot_1,
        grad_slots_0,
        grad_slots_1,
    )
    compiled_objective(*compile_args).backward()
    torch.cuda.synchronize()
    if table_0.grad is not None or table_1.grad is not None:
        raise AssertionError("P1 Redirect materialized an embedding dense gradient")

    # Profile a post-compile iteration so compilation internals cannot mask an
    # accidental eager embedding backward fallback.
    grad_slots_0.zero_()
    grad_slots_1.zero_()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        compiled_objective(*compile_args).backward()
    torch.cuda.synchronize()
    profiler_keys = [event.key.lower() for event in profile.key_averages()]
    if any("embedding_dense_backward" in key for key in profiler_keys):
        raise AssertionError("P1 fullgraph contains embedding_dense_backward")
    if table_0.grad is not None or table_1.grad is not None:
        raise AssertionError("P1 compiled fullgraph violated p.grad=None")

    def ordered_reference(flat_indices, contributions):
        result = torch.zeros(rows, cols, dtype=torch.float32)
        flat_rows = flat_indices.detach().cpu().flatten().tolist()
        flat_contrib = contributions.detach().cpu().float().reshape(-1, cols)
        for token_i, row_i in enumerate(flat_rows):
            result[row_i].add_(flat_contrib[token_i])
        return result

    reference_0 = ordered_reference(indices_0, upstream[..., :cols])
    reference_1 = ordered_reference(indices_1, upstream[..., cols:])
    slot_snapshots = []
    native_bit_differences = 0
    for table, grad_slots, reference in (
        (table_0, grad_slots_0, reference_0),
        (table_1, grad_slots_1, reference_1),
    ):
        count = int(optimizer._rmsprop_active_counters[table].item())
        active = optimizer._rmsprop_active_rows[table][:count].long()
        actual = grad_slots[:count].detach().cpu()
        expected = reference.index_select(0, active.cpu())
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
        actual_bf16 = actual.to(torch.bfloat16)
        expected_bf16 = expected.to(torch.bfloat16)
        native_bit_differences += int(
            (actual_bf16.view(torch.int16) != expected_bf16.view(torch.int16)).sum()
        )
        slot_snapshots.append((count, active, grad_slots[:count].clone()))
    if native_bit_differences:
        raise AssertionError("P1 single-BF16 quantization disagrees with ordered reference")

    expected_params = [table_0.detach().clone(), table_1.detach().clone()]
    expected_states = [torch.zeros_like(table_0), torch.zeros_like(table_1)]
    step_t = torch.tensor(1.0, dtype=torch.float32)
    lr_t = torch.tensor(group["lr"], dtype=torch.float32)
    beta2_t = torch.tensor(group["beta2"], dtype=torch.float32)
    eps_t = torch.tensor(group["eps"], dtype=torch.float32)
    wd_t = torch.tensor(0.0, dtype=torch.float32)
    for table_i, ((count, active, slots), expected_p, expected_state) in enumerate(
        zip(slot_snapshots, expected_params, expected_states)
    ):
        dense_grad = torch.zeros_like(expected_p)
        dense_grad.index_copy_(0, active, slots.to(torch.bfloat16))
        rmsprop_step_fused(
            expected_p,
            dense_grad,
            expected_state,
            step_t,
            lr_t,
            beta2_t,
            eps_t,
            wd_t,
        )
        if count == 0:
            raise AssertionError(f"P1 table {table_i} unexpectedly has no active rows")

    optimizer.step()
    torch.cuda.synchronize()
    for table, expected_p, expected_state, grad_slots in zip(
        (table_0, table_1),
        expected_params,
        expected_states,
        (grad_slots_0, grad_slots_1),
    ):
        if not torch.equal(table.view(torch.int16), expected_p.view(torch.int16)):
            raise AssertionError("P1 compact RMSProp parameter parity failed")
        state = optimizer.state[table]["exp_avg_sq"]
        if not torch.equal(state.view(torch.int16), expected_state.view(torch.int16)):
            raise AssertionError("P1 compact RMSProp state parity failed")
        if int(torch.count_nonzero(grad_slots).item()) != 0:
            raise AssertionError("P1 compact RMSProp did not clear active slots")
        if table.grad is not None:
            raise AssertionError("P1 optimizer created an n-gram p.grad")

    print(
        "P1_COMPACT_BACKWARD_SELF_TEST PASS "
        f"device={torch.cuda.get_device_name()} fullgraph=pass "
        "embedding_dense_backward=absent p_grad_none=2/2 "
        "collector=row_to_slot_exact reduction=fp32_atomic_bounded "
        "quantization=single_bf16 optimizer=bitwise slots_cleared=2/2"
    )


def _run_p1_forward_reuse_selftest():
    """P2 exactness, replay idempotence, and capacity fail-closed checks."""
    if not torch.cuda.is_available():
        raise RuntimeError("P1_FORWARD_REUSE_SELF_TEST requires CUDA")
    if not P1_COMPACT_BACKWARD_ENABLED:
        raise RuntimeError("P1 forward-reuse self-test requires P1_COMPACT_BACKWARD=1")
    device = torch.device("cuda")
    rows, cols, capacity = 64, 8, 16
    primes = ((2654435761, 2246822519), (1013904223, 6291469))
    table_0 = torch.nn.Parameter(
        torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
    )
    table_1 = torch.nn.Parameter(
        torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
    )
    group = {
        "kind": "rmsprop",
        "params": [table_0, table_1],
        "lr": 0.2,
        "beta2": 0.999,
        "eps": 1e-10,
        "weight_decay": 0.0,
    }
    optimizer = MuonAdamW(
        [group],
        rmsprop_touch_pairs=((2, (table_0, table_1), primes),),
        rmsprop_touch_capacity=capacity,
        p1_compact_backward=True,
    )
    scratch = (
        optimizer._rmsprop_touch_stamps[table_0],
        optimizer._rmsprop_active_rows[table_0],
        optimizer._rmsprop_row_to_slot[table_0],
        optimizer._rmsprop_active_counters[table_0],
        optimizer._rmsprop_touch_stamps[table_1],
        optimizer._rmsprop_active_rows[table_1],
        optimizer._rmsprop_row_to_slot[table_1],
        optimizer._rmsprop_active_counters[table_1],
        optimizer._rmsprop_touch_generation,
    )

    def collect(idx_0, idx_1, *buffers):
        _p1_collect_pair_indices_op(idx_0, idx_1, *buffers)

    compiled_collect = torch.compile(
        collect, dynamic=False, fullgraph=True, mode="reduce-overhead"
    )
    tokens = torch.arange(capacity, device=device, dtype=torch.long).view(2, 8)
    idx_0 = tokens.clone()
    idx_1 = torch.flip(tokens, dims=(1,)).contiguous() + 16
    optimizer.begin_step()
    optimizer.prepare_ngram_forward(tokens)
    torch.compiler.cudagraph_mark_step_begin()
    compiled_collect(idx_0, idx_1, *scratch)
    torch.cuda.synchronize()

    for table, expected in ((table_0, idx_0), (table_1, idx_1)):
        count = int(optimizer._rmsprop_active_counters[table].item())
        if count != capacity:
            raise AssertionError(f"P2 capacity-bound count mismatch: {count}")
        active = optimizer._rmsprop_active_rows[table][:count]
        if not torch.equal(active.sort().values, expected.flatten().sort().values.int()):
            raise AssertionError("P2 capacity-bound active-row set mismatch")
        mapped = optimizer._rmsprop_row_to_slot[table].index_select(0, active.long())
        if not torch.equal(mapped, torch.arange(count, device=device, dtype=torch.int32)):
            raise AssertionError("P2 capacity-bound slot ownership mismatch")

    # Compiler/cudagraph warmup or replay may execute the same forward more than
    # once. Generation stamps must make that replay exactly idempotent.
    torch.compiler.cudagraph_mark_step_begin()
    compiled_collect(idx_0, idx_1, *scratch)
    torch.cuda.synchronize()
    if any(
        int(optimizer._rmsprop_active_counters[table].item()) != capacity
        for table in (table_0, table_1)
    ):
        raise AssertionError("P2 same-generation replay duplicated active slots")
    optimizer.step()

    duplicate_idx = torch.full((2, 8), 7, device=device, dtype=torch.long)
    optimizer.begin_step()
    optimizer.prepare_ngram_forward(tokens)
    torch.compiler.cudagraph_mark_step_begin()
    compiled_collect(duplicate_idx, duplicate_idx, *scratch)
    torch.cuda.synchronize()
    if any(
        int(optimizer._rmsprop_active_counters[table].item()) != 1
        for table in (table_0, table_1)
    ):
        raise AssertionError("P2 duplicate rows did not collapse to one slot")
    optimizer.step()

    overflow_tokens = torch.arange(capacity + 1, device=device).view(1, -1)
    optimizer.begin_step()
    optimizer.prepare_ngram_forward(overflow_tokens)
    overflow_idx = torch.arange(capacity + 1, device=device).view(1, -1)
    _p1_collect_pair_indices_op(overflow_idx, overflow_idx, *scratch)
    torch.cuda.synchronize()
    overflow_counts = [
        int(optimizer._rmsprop_active_counters[table].item())
        for table in (table_0, table_1)
    ]
    if any(count <= capacity for count in overflow_counts):
        raise AssertionError(
            f"P2 direct overflow did not expose counters: {overflow_counts}"
        )
    try:
        optimizer.step()
    except RuntimeError as error:
        if "counter overflow" not in str(error):
            raise
    else:
        raise AssertionError("P2 optimizer accepted compact counter overflow")

    from torch._dynamo.utils import counters

    cudagraph_skips = int(counters["inductor"].get("cudagraph_skips", 0))
    if cudagraph_skips != 0:
        raise AssertionError(f"P2 collector incurred {cudagraph_skips} cudagraph skips")
    print(
        "P1_FORWARD_REUSE_SELF_TEST PASS "
        f"device={torch.cuda.get_device_name()} fullgraph=pass cudagraph_skips=0 "
        "collector=forward_indices exact_capacity=pass replay=idempotent "
        "duplicates=collapsed token_over_capacity=accepted counter_overflow=step_rejected "
        "hash_recompute=absent"
    )


def _run_p1_fp64_envelope_selftest():
    """Check FP32 atomic reduction against an FP64 gamma envelope."""
    if not torch.cuda.is_available():
        raise RuntimeError("P1_FP64_ENVELOPE_SELF_TEST requires CUDA")
    if not P1_COMPACT_BACKWARD_ENABLED:
        raise RuntimeError("P1 FP64 envelope self-test requires P1_COMPACT_BACKWARD=1")
    device = torch.device("cuda")
    torch.manual_seed(20260817)
    rows, cols, batch, seq_len = 128, 16, 4, 32
    num_tokens = batch * seq_len
    primes = ((2654435761, 2246822519), (1013904223, 6291469))
    token_shape = (batch, seq_len)
    flat = torch.arange(num_tokens, device=device, dtype=torch.long)
    index_cases = {
        "all_duplicate_vs_unique": (
            torch.full(token_shape, 3, device=device, dtype=torch.long),
            (flat % rows).view(token_shape),
        ),
        "unique_vs_all_duplicate": (
            (flat % rows).view(token_shape),
            torch.full(token_shape, 7, device=device, dtype=torch.long),
        ),
        "mixed_domains": (
            ((flat * 17 + 5) % rows).view(token_shape),
            ((flat * 31 + 11) % rows).view(token_shape),
        ),
        "cancellation_duplicates": (
            torch.where(flat % 2 == 0, 11, 12).view(token_shape),
            torch.where(flat % 4 < 2, 21, 22).view(token_shape),
        ),
    }
    token_ids = flat.view(batch, seq_len, 1)
    col_ids = torch.arange(cols * 2, device=device).view(1, 1, -1)
    large = torch.full(
        (batch, seq_len, cols * 2), 4096.0, device=device, dtype=torch.float32
    )
    small = torch.full_like(large, 2.0**-14)
    contribution = torch.where((token_ids + col_ids) % 32 == 0, large, small)
    contribution = contribution * torch.where(
        (token_ids * 5 + col_ids * 3) % 7 < 3,
        torch.tensor(-1.0, device=device),
        torch.tensor(1.0, device=device),
    )
    contribution = contribution.to(torch.bfloat16)
    unit_roundoff = 2.0**-24
    case_receipts = []
    max_ratio = 0.0
    max_error = 0.0
    max_bound = 0.0
    max_multiplicity = 0

    def run_case(case_name, indices_0, indices_1):
        nonlocal max_ratio, max_error, max_bound, max_multiplicity
        table_0 = torch.nn.Parameter(
            torch.zeros(rows, cols, device=device, dtype=torch.bfloat16)
        )
        table_1 = torch.nn.Parameter(
            torch.zeros(rows, cols, device=device, dtype=torch.bfloat16)
        )
        optimizer = MuonAdamW(
            [
                {
                    "kind": "rmsprop",
                    "params": [table_0, table_1],
                    "lr": 0.2,
                    "beta2": 0.999,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            ],
            rmsprop_touch_pairs=((2, (table_0, table_1), primes),),
            rmsprop_touch_capacity=num_tokens,
            p1_compact_backward=True,
        )
        optimizer.begin_step()
        optimizer.prepare_ngram_forward(
            torch.zeros(token_shape, device=device, dtype=torch.long)
        )
        _p1_collect_pair_indices_op(
            indices_0,
            indices_1,
            optimizer._rmsprop_touch_stamps[table_0],
            optimizer._rmsprop_active_rows[table_0],
            optimizer._rmsprop_row_to_slot[table_0],
            optimizer._rmsprop_active_counters[table_0],
            optimizer._rmsprop_touch_stamps[table_1],
            optimizer._rmsprop_active_rows[table_1],
            optimizer._rmsprop_row_to_slot[table_1],
            optimizer._rmsprop_active_counters[table_1],
            optimizer._rmsprop_touch_generation,
        )
        _p1_scatter_pair_fp32_op(
            contribution,
            indices_0,
            indices_1,
            optimizer._rmsprop_row_to_slot[table_0],
            optimizer._rmsprop_row_to_slot[table_1],
            optimizer._rmsprop_grad_slots[table_0],
            optimizer._rmsprop_grad_slots[table_1],
        )
        torch.cuda.synchronize()
        rows_receipt = []
        for table_i, (indices, table) in enumerate(
            ((indices_0, table_0), (indices_1, table_1))
        ):
            index_cpu = indices.detach().cpu().flatten().tolist()
            pair_cpu = contribution.detach().cpu().float().reshape(-1, cols * 2)
            contrib_cpu = pair_cpu[:, table_i * cols : (table_i + 1) * cols]
            reference = [[0.0] * cols for _ in range(rows)]
            absolute_sum = [[0.0] * cols for _ in range(rows)]
            multiplicity = [0] * rows
            for token_i, row_i in enumerate(index_cpu):
                multiplicity[row_i] += 1
                for col_i, value in enumerate(contrib_cpu[token_i].tolist()):
                    value64 = float(value)
                    reference[row_i][col_i] += value64
                    absolute_sum[row_i][col_i] += abs(value64)
            count = int(optimizer._rmsprop_active_counters[table].item())
            active = optimizer._rmsprop_active_rows[table][:count].detach().cpu().tolist()
            actual = optimizer._rmsprop_grad_slots[table][:count].detach().cpu().double()
            for slot_i, row_i in enumerate(active):
                rounded_additions = max(0, multiplicity[row_i] - 1)
                gamma_n = (rounded_additions * unit_roundoff) / (
                    1.0 - rounded_additions * unit_roundoff
                )
                expected = torch.tensor(reference[row_i], dtype=torch.float64)
                abs_sum = torch.tensor(absolute_sum[row_i], dtype=torch.float64)
                error = (actual[slot_i] - expected).abs()
                bound = gamma_n * abs_sum
                if not bool((error <= bound + 1e-12).all().item()):
                    raise AssertionError(
                        f"FP64 gamma envelope exceeded in {case_name} row {row_i}"
                    )
                row_error = float(error.max().item())
                row_bound = float(bound.max().item())
                ratio = row_error / row_bound if row_bound > 0.0 else 0.0
                max_error = max(max_error, row_error)
                max_bound = max(max_bound, row_bound)
                max_ratio = max(max_ratio, ratio)
                max_multiplicity = max(max_multiplicity, multiplicity[row_i])
                rows_receipt.append(
                    {
                        "table": table_i,
                        "row": row_i,
                        "multiplicity": multiplicity[row_i],
                        "rounded_additions": rounded_additions,
                        "max_abs_error": row_error,
                        "max_gamma_bound": row_bound,
                    }
                )
        return {
            "case": case_name,
            "active_rows_table_0": int(
                optimizer._rmsprop_active_counters[table_0].item()
            ),
            "active_rows_table_1": int(
                optimizer._rmsprop_active_counters[table_1].item()
            ),
            "rows_checked": len(rows_receipt),
        }

    for name, (indices_0, indices_1) in index_cases.items():
        case_receipts.append(run_case(name, indices_0, indices_1))

    overflow_capacity = 16
    overflow_table_0 = torch.nn.Parameter(
        torch.zeros(rows, cols, device=device, dtype=torch.bfloat16)
    )
    overflow_table_1 = torch.nn.Parameter(
        torch.zeros(rows, cols, device=device, dtype=torch.bfloat16)
    )
    overflow_optimizer = MuonAdamW(
        [
            {
                "kind": "rmsprop",
                "params": [overflow_table_0, overflow_table_1],
                "lr": 0.2,
                "beta2": 0.999,
                "eps": 1e-10,
                "weight_decay": 0.0,
            }
        ],
        rmsprop_touch_pairs=((2, (overflow_table_0, overflow_table_1), primes),),
        rmsprop_touch_capacity=overflow_capacity,
        p1_compact_backward=True,
    )
    overflow_optimizer.begin_step()
    overflow_tokens = torch.arange(
        overflow_capacity + 1, device=device, dtype=torch.long
    ).view(1, -1)
    overflow_optimizer.prepare_ngram_forward(overflow_tokens)
    overflow_indices = overflow_tokens.clone()
    scratch = (
        overflow_optimizer._rmsprop_touch_stamps[overflow_table_0],
        overflow_optimizer._rmsprop_active_rows[overflow_table_0],
        overflow_optimizer._rmsprop_row_to_slot[overflow_table_0],
        overflow_optimizer._rmsprop_active_counters[overflow_table_0],
        overflow_optimizer._rmsprop_touch_stamps[overflow_table_1],
        overflow_optimizer._rmsprop_active_rows[overflow_table_1],
        overflow_optimizer._rmsprop_row_to_slot[overflow_table_1],
        overflow_optimizer._rmsprop_active_counters[overflow_table_1],
        overflow_optimizer._rmsprop_touch_generation,
    )
    _p1_collect_pair_indices_op(
        overflow_indices,
        overflow_indices,
        *scratch,
    )
    overflow_contribution = torch.ones(
        (1, overflow_capacity + 1, cols * 2),
        device=device,
        dtype=torch.bfloat16,
    )
    _p1_scatter_pair_fp32_op(
        overflow_contribution,
        overflow_indices,
        overflow_indices,
        overflow_optimizer._rmsprop_row_to_slot[overflow_table_0],
        overflow_optimizer._rmsprop_row_to_slot[overflow_table_1],
        overflow_optimizer._rmsprop_grad_slots[overflow_table_0],
        overflow_optimizer._rmsprop_grad_slots[overflow_table_1],
    )
    torch.cuda.synchronize()
    overflow_counts = [
        int(overflow_optimizer._rmsprop_active_counters[table].item())
        for table in (overflow_table_0, overflow_table_1)
    ]
    if any(count <= overflow_capacity for count in overflow_counts):
        raise AssertionError(
            f"P1 direct overflow did not expose counters: {overflow_counts}"
        )
    try:
        overflow_optimizer.step()
    except RuntimeError as error:
        direct_rejected = "counter overflow" in str(error)
    else:
        direct_rejected = False
    counters_unchanged = overflow_counts == [
        int(overflow_optimizer._rmsprop_active_counters[table].item())
        for table in (overflow_table_0, overflow_table_1)
    ]
    prepare_rejected = False
    if not direct_rejected or not counters_unchanged:
        raise AssertionError("P1 direct overflow did not fail closed")
    overflow_grad_slots_cleared = all(
        int(torch.count_nonzero(overflow_optimizer._rmsprop_grad_slots[table]).item())
        == 0
        for table in (overflow_table_0, overflow_table_1)
    )
    if not overflow_grad_slots_cleared:
        raise AssertionError("P1 overflow recovery retained rejected gradient slots")

    # A rejected step must be recoverable: the next generation may reuse slot
    # IDs, but it must observe only the new valid contributions.
    recovery_indices = torch.tensor(
        [[1, 3, 1, 7]], device=device, dtype=torch.long
    )
    recovery_tokens = recovery_indices.clone()
    recovery_contribution = torch.ones(
        (1, recovery_indices.shape[1], cols * 2),
        device=device,
        dtype=torch.bfloat16,
    )
    overflow_optimizer.begin_step()
    overflow_optimizer.prepare_ngram_forward(recovery_tokens)
    _p1_collect_pair_indices_op(recovery_indices, recovery_indices, *scratch)
    _p1_scatter_pair_fp32_op(
        recovery_contribution,
        recovery_indices,
        recovery_indices,
        overflow_optimizer._rmsprop_row_to_slot[overflow_table_0],
        overflow_optimizer._rmsprop_row_to_slot[overflow_table_1],
        overflow_optimizer._rmsprop_grad_slots[overflow_table_0],
        overflow_optimizer._rmsprop_grad_slots[overflow_table_1],
    )
    torch.cuda.synchronize()
    recovery_valid_step = True
    recovery_counts = []
    for table in (overflow_table_0, overflow_table_1):
        count = int(overflow_optimizer._rmsprop_active_counters[table].item())
        recovery_counts.append(count)
        if count != 3:
            recovery_valid_step = False
            continue
        active_rows = overflow_optimizer._rmsprop_active_rows[table][:count]
        slots = overflow_optimizer._rmsprop_grad_slots[table][:count]
        expected = torch.zeros_like(slots)
        expected.index_add_(
            0,
            overflow_optimizer._rmsprop_row_to_slot[table]
            .index_select(0, recovery_indices.flatten())
            .to(torch.long),
            torch.ones(
                recovery_indices.numel(),
                cols,
                device=device,
                dtype=torch.float32,
            ),
        )
        if not torch.equal(slots, expected):
            recovery_valid_step = False
        if not torch.equal(active_rows.sort().values, torch.tensor(
            [1, 3, 7], device=device, dtype=torch.int32
        )):
            recovery_valid_step = False
    if not recovery_valid_step:
        raise AssertionError("P1 overflow recovery mixed stale and new slots")
    overflow_optimizer.step()
    receipt = _emit_p1_admission_receipt(
        {
            "schema": "apex-tune.h200.formal-a-compact-p1-envelope.v1",
            "authority": "diagnostic_only",
            "candidate_source_sha256": _p1_file_sha256(__file__),
            "parent_source_sha256": (
                "010356bb5477ff5b9a0fa93391eb79b2d23706bc790176ad67d87e7e343a0dce"
            ),
            "hardware": {
                "device_name": torch.cuda.get_device_name(),
                "compute_capability": list(torch.cuda.get_device_capability()),
                "gpu_index": torch.cuda.current_device(),
                "cuda_version": torch.version.cuda,
                "torch_version": torch.__version__,
                "triton_version": triton.__version__,
            },
            "fp64_gamma_envelope": {
                "unit_roundoff": unit_roundoff,
                "bound": "gamma_(m-1) * sum(abs(contribution))",
                "reference": "FP64 ordered sum of BF16 post-cat contributions",
                "rationale": (
                    "BF16 contributions convert exactly to FP32 and the first "
                    "atomic add into a zero slot is exact; only m-1 additions "
                    "can round"
                ),
                "cases": case_receipts,
                "max_multiplicity": max_multiplicity,
                "max_abs_error": max_error,
                "max_gamma_bound": max_bound,
                "max_error_to_bound_ratio": max_ratio,
            },
            "overflow_fail_closed": {
                "capacity": overflow_capacity,
                "prepare_rejected": prepare_rejected,
                "prepare_accepts_token_count_over_capacity": not prepare_rejected,
                "direct_collector_rejected": False,
                "optimizer_step_rejected": direct_rejected,
                "overflow_counters_observed": overflow_counts,
                "counters_unchanged": counters_unchanged,
                "grad_slots_cleared_after_rejection": overflow_grad_slots_cleared,
                "recovery_valid_step": recovery_valid_step,
                "recovery_active_rows": recovery_counts,
            },
        },
        env_name="P1_ENVELOPE_RECEIPT",
    )
    print(
        "P1_FP64_ENVELOPE_SELF_TEST PASS "
        f"device={torch.cuda.get_device_name()} cases={len(case_receipts)} "
        f"max_multiplicity={max_multiplicity} max_abs_error={max_error:.8g} "
        f"max_gamma_bound={max_bound:.8g} max_ratio={max_ratio:.8g} "
        "overflow=token_preflight_accept+counter_step_fail_closed "
        + (
            f"receipt_sha256={receipt['receipt_sha256']}"
            if receipt is not None
            else "receipt_sha256=absent"
        )
    )
    return receipt


def _run_p1_tiny_model_selftest():
    """Compare a native-backward reference with the compact child end to end."""
    if not torch.cuda.is_available():
        raise RuntimeError("P1_COMPACT_BACKWARD_TINY_MODEL_SELF_TEST requires CUDA")
    if not P1_COMPACT_BACKWARD_ENABLED:
        raise RuntimeError("P1 tiny-model self-test requires P1_COMPACT_BACKWARD=1")
    device = torch.device("cuda")
    torch.manual_seed(20260817)
    real_config = os.environ.get("P1_REAL_CONFIG_PARITY", "0") == "1"
    if real_config:
        config = GPTConfig(
            sequence_len=2048,
            vocab_size=8192,
            n_layer=8,
            n_head=6,
            n_kv_head=6,
            n_embd=768,
            window_pattern="TTTL",
        )
        test_seq_len = int(os.environ.get("P1_REAL_CONFIG_PARITY_SEQ_LEN", "128"))
        batch = 1
    else:
        config = GPTConfig(
            sequence_len=8,
            vocab_size=32,
            n_layer=8,
            n_head=1,
            n_kv_head=1,
            n_embd=128,
            window_pattern="TTTL",
        )
        test_seq_len = config.sequence_len
        batch = 2
    def build_model(p1_compact_backward):
        with torch.device("meta"):
            built = GPT(config, p1_compact_backward=p1_compact_backward)
        built.to_empty(device=device)
        built.init_weights()
        built.to(dtype=torch.bfloat16)
        return built

    reference_model = build_model(False)
    # Formal A initializes c_proj to zero, which intentionally blocks the first
    # attention backward. Make the tiny audit state nonzero so all Redirects
    # receive a measurable gradient in this single diagnostic step.
    for block in reference_model.transformer.h:
        torch.nn.init.uniform_(block.attn.c_proj.weight, -0.01, 0.01)
    compact_model = build_model(True)
    compact_model.load_state_dict(reference_model.state_dict(), strict=True)

    tokens = torch.randint(
        0, config.vocab_size, (batch, test_seq_len), device=device
    )
    targets = torch.randint(
        0, config.vocab_size, (batch, test_seq_len), device=device
    )
    optimizer = compact_model.setup_optimizer(
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        adam_betas=(0.8, 0.95),
        scalar_lr=0.5,
        ngram_ve_betas=(0.5, 0.999),
        ngram_touch_capacity=tokens.numel(),
    )
    compiled_reference = torch.compile(reference_model, dynamic=False, fullgraph=True)
    compiled_compact = torch.compile(compact_model, dynamic=False, fullgraph=True)
    optimizer.begin_step()
    optimizer.prepare_ngram_forward(tokens)
    reference_logits = compiled_reference(tokens)
    compact_logits = compiled_compact(tokens)
    if not torch.equal(reference_logits, compact_logits):
        raise AssertionError("P1 Redirect changed the compiled forward logits")
    reference_loss = F.cross_entropy(
        reference_logits.view(-1, reference_logits.shape[-1]), targets.view(-1)
    )
    compact_loss = F.cross_entropy(
        compact_logits.view(-1, compact_logits.shape[-1]), targets.view(-1)
    )
    if not torch.equal(reference_loss, compact_loss):
        raise AssertionError("P1 Redirect changed the compiled forward loss")
    reference_loss.backward()
    compact_loss.backward()
    torch.cuda.synchronize()

    reference_named = dict(reference_model.named_parameters())
    compact_named = dict(compact_model.named_parameters())
    if reference_named.keys() != compact_named.keys():
        raise AssertionError("P1 reference/child parameter names differ")
    ngram_prefixes = ("bigram_ves.", "trigram_ves.")
    non_ngram_grad_count = 0
    for name, reference_param in reference_named.items():
        if name.startswith(ngram_prefixes):
            continue
        compact_grad = compact_named[name].grad
        reference_grad = reference_param.grad
        if (reference_grad is None) != (compact_grad is None):
            raise AssertionError(f"P1 changed gradient presence for {name}")
        if reference_grad is not None:
            non_ngram_grad_count += 1
            if not torch.equal(reference_grad, compact_grad):
                raise AssertionError(f"P1 changed non-ngram gradient bits for {name}")

    reference_ngram = {
        name: param
        for name, param in reference_named.items()
        if name.startswith(ngram_prefixes)
    }
    compact_ngram = {
        name: param
        for name, param in compact_named.items()
        if name.startswith(ngram_prefixes)
    }
    if len(reference_ngram) != 14 or len(compact_ngram) != 14:
        raise AssertionError(
            "tiny P1 model expected 14 reference and 14 compact n-gram tables"
        )
    if any(param.grad is not None for param in compact_ngram.values()):
        raise AssertionError("tiny P1 model materialized an n-gram dense gradient")
    if any(param.grad is None for param in reference_ngram.values()):
        raise AssertionError("native reference did not materialize n-gram gradients")

    max_raw_error = 0.0
    max_raw_envelope = 0.0
    positive_inf = torch.tensor(float("inf"), device=device, dtype=torch.bfloat16)
    negative_inf = torch.tensor(float("-inf"), device=device, dtype=torch.bfloat16)
    fp32_eps = torch.finfo(torch.float32).eps
    nonzero_slots = 0
    for name, compact_param in compact_ngram.items():
        reference_grad = reference_ngram[name].grad
        if not bool(torch.isfinite(reference_grad).all().item()):
            raise AssertionError(f"native n-gram gradient is non-finite for {name}")
        count = int(optimizer._rmsprop_active_counters[compact_param].item())
        if count <= 0:
            raise AssertionError("tiny P1 model collector produced no active rows")
        active = optimizer._rmsprop_active_rows[compact_param][:count].long()
        compact_slots = optimizer._rmsprop_grad_slots[compact_param][:count]
        if not bool(torch.isfinite(compact_slots).all().item()):
            raise AssertionError(f"compact n-gram slot is non-finite for {name}")
        nonzero_slots += int(torch.count_nonzero(compact_slots).item())
        reference_active = reference_grad.index_select(0, active)
        compact_active_bf16 = compact_slots.to(torch.bfloat16)

        lower = torch.nextafter(reference_active, negative_inf).float()
        upper = torch.nextafter(reference_active, positive_inf).float()
        quantized = compact_active_bf16.float()
        within_one_ulp = (quantized >= lower) & (quantized <= upper)
        if not bool(within_one_ulp.all().item()):
            bad = int((~within_one_ulp).sum().item())
            raise AssertionError(f"P1 n-gram gradient exceeded 1 ULP for {name}: {bad}")

        reference_fp32 = reference_active.float()
        reference_ulp = torch.maximum(
            (upper - reference_fp32).abs(), (reference_fp32 - lower).abs()
        )
        # Native BF16 reduction can move by one representable value. The raw
        # compact accumulator additionally gets a conservative FP32 summation
        # allowance proportional to the number of token contributions.
        scale = torch.maximum(
            torch.maximum(reference_fp32.abs(), compact_slots.abs()),
            torch.full_like(reference_fp32, 2.0**-8),
        )
        fp32_allowance = 8.0 * fp32_eps * tokens.numel() * scale
        raw_envelope = reference_ulp + fp32_allowance
        raw_error = (compact_slots - reference_fp32).abs()
        if not bool((raw_error <= raw_envelope).all().item()):
            bad = int((raw_error > raw_envelope).sum().item())
            raise AssertionError(
                f"P1 FP32 accumulation exceeded error envelope for {name}: {bad}"
            )
        max_raw_error = max(max_raw_error, float(raw_error.max().item()))
        max_raw_envelope = max(
            max_raw_envelope, float(raw_envelope.max().item())
        )

    if non_ngram_grad_count == 0:
        raise AssertionError("tiny P1 comparison found no non-ngram gradients")
    if nonzero_slots == 0:
        raise AssertionError("tiny P1 model Redirects produced only zero gradients")

    eval_scratch = (
        list(optimizer._rmsprop_touch_stamps.values())
        + list(optimizer._rmsprop_active_rows.values())
        + list(optimizer._rmsprop_row_to_slot.values())
        + list(optimizer._rmsprop_active_counters.values())
        + list(optimizer._rmsprop_grad_slots.values())
        + [optimizer._rmsprop_touch_generation]
    )
    eval_scratch_before = [tensor.clone() for tensor in eval_scratch]
    reference_model.eval()
    compact_model.eval()
    with torch.no_grad():
        reference_eval = compiled_reference(tokens, targets, reduction="none")
        compact_eval = compiled_compact(tokens, targets, reduction="none")
    torch.cuda.synchronize()
    if not torch.equal(reference_eval, compact_eval):
        raise AssertionError("P1 changed eval/BPB per-token losses")
    for before, after in zip(eval_scratch_before, eval_scratch):
        if not torch.equal(before, after):
            raise AssertionError("P1 eval forward mutated compact training scratch")

    ngram_params = tuple(compact_model.bigram_ves.parameters()) + tuple(
        compact_model.trigram_ves.parameters()
    )
    # The compact optimizer path is already tested through optimizer.step in
    # the focused test. Here call only its n-gram groups to avoid conflating P1
    # with unrelated tiny-shape Muon behavior.
    for group in optimizer.param_groups:
        if group["kind"] == "rmsprop":
            optimizer._step_rmsprop(group)
    torch.cuda.synchronize()
    for param in ngram_params:
        if int(torch.count_nonzero(optimizer._rmsprop_grad_slots[param]).item()) != 0:
            raise AssertionError("tiny P1 model left a compact gradient slot uncleared")
        if param.grad is not None:
            raise AssertionError("tiny P1 model violated p.grad=None after update")

    print(
        "P1_COMPACT_BACKWARD_TINY_MODEL_SELF_TEST PASS "
        f"device={torch.cuda.get_device_name()} fullgraph=pass layers=8 "
        f"real_config={str(real_config).lower()} width={config.n_embd} "
        f"table_rows={config.vocab_size * 64} test_tokens={tokens.numel()} "
        f"redirects=7 collector=forward_indices forward=bitwise loss=bitwise "
        f"non_ngram_grads=bitwise:{non_ngram_grad_count} "
        f"ngram_tables=14 ngram_envelope=1ulp+fp32 max_raw_error={max_raw_error:.8g} "
        f"max_raw_envelope={max_raw_envelope:.8g} p_grad_none=14/14 slots_cleared=14/14 "
        "eval_loss=bitwise eval_scratch=unchanged"
    )


def _run_p1_real_shape_smoke():
    """One real c428 table pair and token count; H200 diagnostics only."""
    if not torch.cuda.is_available():
        raise RuntimeError("P1_COMPACT_BACKWARD_REAL_SHAPE_SMOKE requires CUDA")
    device = torch.device("cuda")
    rows, cols = 524288, 384
    batch, seq_len = 72, 2048
    num_tokens = batch * seq_len
    if num_tokens != 147456:
        raise AssertionError("P1 real-shape smoke token contract changed")
    torch.manual_seed(20260817)
    primes = ((2654435761, 2246822519), (1013904223, 6291469))
    tokens = torch.randint(
        0, 32768, (batch, seq_len), device=device, dtype=torch.long
    )
    previous = torch.cat([tokens[:, :1], tokens[:, :-1]], dim=1)
    indices_0 = (
        (previous * primes[0][0]) ^ (tokens * primes[0][1])
    ).bitwise_and(rows - 1)
    indices_1 = (
        (previous * primes[1][0]) ^ (tokens * primes[1][1])
    ).bitwise_and(rows - 1)
    table_0 = torch.nn.Parameter(
        torch.empty(rows, cols, device=device, dtype=torch.bfloat16).normal_(std=0.02)
    )
    table_1 = torch.nn.Parameter(
        torch.empty(rows, cols, device=device, dtype=torch.bfloat16).normal_(std=0.02)
    )
    group = {
        "kind": "rmsprop",
        "params": [table_0, table_1],
        "lr": 0.2,
        "beta2": 0.999,
        "eps": 1e-10,
        "weight_decay": 0.0,
    }
    optimizer = MuonAdamW(
        [group],
        rmsprop_touch_pairs=((2, (table_0, table_1), primes),),
        rmsprop_touch_capacity=num_tokens,
        p1_compact_backward=True,
    )
    row_to_slot_0 = optimizer._rmsprop_row_to_slot[table_0]
    row_to_slot_1 = optimizer._rmsprop_row_to_slot[table_1]
    grad_slots_0 = optimizer._rmsprop_grad_slots[table_0]
    grad_slots_1 = optimizer._rmsprop_grad_slots[table_1]

    def real_shape_objective(
        weight_0,
        weight_1,
        idx_0,
        idx_1,
        map_0,
        map_1,
        slots_0,
        slots_1,
    ):
        pair = torch.cat(
            [F.embedding(idx_0, weight_0), F.embedding(idx_1, weight_1)], dim=-1
        )
        pair = _P1CompactPairRedirect.apply(
            pair, idx_0, idx_1, map_0, map_1, slots_0, slots_1
        )
        return pair.float().square().mean()

    compiled_objective = torch.compile(
        real_shape_objective, dynamic=False, fullgraph=True
    )
    compile_args = (
        table_0,
        table_1,
        indices_0,
        indices_1,
        row_to_slot_0,
        row_to_slot_1,
        grad_slots_0,
        grad_slots_1,
    )

    # Compile every kernel and allocate RMS state outside the profiled step.
    optimizer.begin_step()
    optimizer.mark_ngram_rows(tokens)
    compiled_objective(*compile_args).backward()
    torch.cuda.synchronize()
    if table_0.grad is not None or table_1.grad is not None:
        raise AssertionError("real-shape warmup materialized an embedding dense gradient")
    optimizer.step()
    torch.cuda.synchronize()

    optimizer.begin_step()
    optimizer.mark_ngram_rows(tokens)
    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        loss = compiled_objective(*compile_args)
        loss.backward()
    end.record()
    end.synchronize()
    profile_step_ms = start.elapsed_time(end)
    profiler_keys = [event.key.lower() for event in profile.key_averages()]
    if any("embedding_dense_backward" in key for key in profiler_keys):
        raise AssertionError("real-shape fullgraph contains embedding_dense_backward")
    if table_0.grad is not None or table_1.grad is not None:
        raise AssertionError("real-shape fullgraph violated p.grad=None")

    counts = []
    for table, slots in ((table_0, grad_slots_0), (table_1, grad_slots_1)):
        count = int(optimizer._rmsprop_active_counters[table].item())
        if count <= 0 or count > num_tokens:
            raise AssertionError(f"real-shape compact slot count is invalid: {count}")
        active_slots = slots[:count]
        if not bool(torch.isfinite(active_slots).all().item()):
            raise AssertionError("real-shape compact slots contain non-finite values")
        counts.append(count)
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    optimizer.step()
    torch.cuda.synchronize()
    for table, slots, count in (
        (table_0, grad_slots_0, counts[0]),
        (table_1, grad_slots_1, counts[1]),
    ):
        if int(torch.count_nonzero(slots[:count]).item()) != 0:
            raise AssertionError("real-shape compact RMSProp did not clear active slots")
        if table.grad is not None:
            raise AssertionError("real-shape compact RMSProp created p.grad")

    mib = 1024 * 1024
    print(
        "P1_COMPACT_BACKWARD_REAL_SHAPE_SMOKE PASS "
        f"device={torch.cuda.get_device_name()} authority=diagnostic_only "
        f"rows={rows} cols={cols} tokens={num_tokens} fullgraph=pass profile=pass "
        "embedding_dense_backward=absent p_grad_none=2/2 finite=2/2 slots_cleared=2/2 "
        f"slot_counts={counts[0]},{counts[1]} loss={float(loss.item()):.8g} "
        f"profile_step_ms={profile_step_ms:.4f} "
        f"baseline_allocated_mib={baseline_allocated / mib:.1f} "
        f"peak_allocated_mib={peak_allocated / mib:.1f} "
        f"peak_reserved_mib={peak_reserved / mib:.1f}"
    )


def _run_formal_a_parent_no_score_engineering(
    *, requested_variant: str, requested_edge: str | None = None, require_b200: bool
):
    """Measure exact Formal A E0 without importing its unguarded train main."""
    if not torch.cuda.is_available():
        raise RuntimeError("P1_NO_SCORE_ENGINEERING requires CUDA")

    parent = _load_formal_a_definition_api()
    device_name = torch.cuda.get_device_name()
    capability = tuple(torch.cuda.get_device_capability())
    device_count = torch.cuda.device_count()
    if require_b200:
        if device_count != 1:
            raise RuntimeError(
                "strict P1 engineering requires exactly one visible CUDA device"
            )
        if "B200" not in device_name.upper() or capability != (10, 0):
            raise RuntimeError(
                "strict P1 engineering requires NVIDIA B200 with compute capability sm_100"
            )

    previous_rmsprop_impl = os.environ.get("TOUCHED_RMSPROP_IMPL")
    os.environ["TOUCHED_RMSPROP_IMPL"] = "compact"
    warmup_steps = int(os.environ.get("P1_ENGINEERING_WARMUP_STEPS", "12"))
    measured_steps = int(os.environ.get("P1_ENGINEERING_MEASURED_STEPS", "128"))
    batch = int(os.environ.get("P1_ENGINEERING_BATCH", "72"))
    seq_len = int(os.environ.get("P1_ENGINEERING_SEQ_LEN", "2048"))
    compile_mode = os.environ.get("P1_ENGINEERING_COMPILE_MODE", "max-autotune")
    if warmup_steps < 0 or measured_steps <= 0:
        raise ValueError("invalid no-score engineering step counts")
    if compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
        raise ValueError("invalid no-score engineering compile mode")

    seed = int(os.environ.get("NANOCHAT_SEED", "42"))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    config = parent.GPTConfig(
        sequence_len=parent.MAX_SEQ_LEN,
        vocab_size=8192,
        n_layer=8,
        n_head=6,
        n_kv_head=6,
        n_embd=768,
        window_pattern="TTTL",
    )
    with torch.device("meta"):
        model = parent.GPT(config)
    model.to_empty(device=torch.device("cuda"))
    model.init_weights()
    model.to(dtype=torch.bfloat16)
    try:
        optimizer = model.setup_optimizer(
            unembedding_lr=0.004,
            embedding_lr=0.6,
            matrix_lr=0.04,
            weight_decay=0.1,
            adam_betas=(0.8, 0.95),
            scalar_lr=0.8,
            ngram_ve_betas=(0.5, 0.999),
            ngram_touch_capacity=batch * seq_len,
        )
    finally:
        if previous_rmsprop_impl is None:
            os.environ.pop("TOUCHED_RMSPROP_IMPL", None)
        else:
            os.environ["TOUCHED_RMSPROP_IMPL"] = previous_rmsprop_impl
    if optimizer._rmsprop_impl != "compact":
        raise RuntimeError(
            f"Formal A E0 requires compact RMSProp, got {optimizer._rmsprop_impl!r}"
        )

    tokenizer = parent.Tokenizer.from_directory()
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise AssertionError("Formal A E0 tokenizer vocabulary changed")
    train_loader = parent.make_dataloader(tokenizer, batch, seq_len, "train")
    stream_batches = int(
        os.environ.get(
            "P1_ENGINEERING_STREAM_BATCHES",
            str(max(2, min(32, warmup_steps + measured_steps))),
        )
    )
    if stream_batches < 2:
        raise ValueError("no-score engineering requires at least two frozen batches")
    frozen_stream = []
    stream_manifest = []
    for stream_i in range(stream_batches):
        tokens, targets, epoch = next(train_loader)
        frozen_tokens = tokens.clone()
        frozen_targets = targets.clone()
        stream_manifest.append(
            {
                "stream_index": stream_i,
                "epoch": int(epoch),
                "tokens_sha256": hashlib.sha256(
                    frozen_tokens.detach().cpu().numpy().tobytes(order="C")
                ).hexdigest(),
                "targets_sha256": hashlib.sha256(
                    frozen_targets.detach().cpu().numpy().tobytes(order="C")
                ).hexdigest(),
            }
        )
        frozen_stream.append((frozen_tokens, frozen_targets))
    stream_sha256 = hashlib.sha256(
        _p1_canonical_json_bytes(stream_manifest)
    ).hexdigest()

    counters = __import__("torch._dynamo.utils", fromlist=["counters"]).counters
    from torch._inductor import metrics as inductor_metrics

    counters.clear()
    inductor_metrics.reset()
    compiled_model = torch.compile(
        model, dynamic=False, mode=compile_mode, fullgraph=True
    )
    model.train()

    def step(stream_index, measure=False, profile=False):
        del profile
        tokens, targets = frozen_stream[stream_index % stream_batches]
        if measure:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start.record()
        optimizer.begin_step()
        # E0 deliberately recomputes touch rows through Formal A's original
        # collector.  E1 will use the forward-produced index collector.
        optimizer.mark_ngram_rows(tokens)
        torch.compiler.cudagraph_mark_step_begin()
        loss = compiled_model(tokens, targets)
        loss.backward()
        optimizer.step()
        compact_slot_counts = [
            int(counter.item())
            for counter in optimizer._rmsprop_active_counters.values()
        ]
        model.zero_grad(set_to_none=True)
        if measure:
            end.record()
            end.synchronize()
            return {
                "stream_index": stream_index % stream_batches,
                "cuda_step_ms": float(start.elapsed_time(end)),
                "wall_step_ms": float((time.perf_counter() - wall_start) * 1000.0),
                "profiled": False,
            }
        return None

    for warmup_i in range(warmup_steps):
        step(warmup_i)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    production_samples = [
        step(warmup_steps + measured_i, measure=True)
        for measured_i in range(measured_steps)
    ]
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    graph_breaks = sum(int(value) for value in counters["graph_break"].values())
    cudagraph_skips = int(counters["inductor"].get("cudagraph_skips", 0))
    # Capture the last completed step's per-table touch counts.  These are
    # diagnostics, not a new timing sample, and make the fixed slot-capacity
    # memory cost auditable before choosing any tighter preregistered cap.
    # Formal A is loaded from the frozen parent definition and may use the
    # older optimizer scratch schema without compact slot dictionaries.
    active_counters = getattr(optimizer, "_rmsprop_active_counters", {})
    active_rows = getattr(optimizer, "_rmsprop_active_rows", {})
    grad_slots = getattr(optimizer, "_rmsprop_grad_slots", {})
    compact_slot_counts = [
        int(counter.item()) for counter in active_counters.values()
    ]
    compact_slot_capacities = [
        int(rows.numel()) for rows in active_rows.values()
    ]
    compact_grad_slot_bytes = sum(
        int(slots.numel() * slots.element_size())
        for slots in grad_slots.values()
    )
    wall_samples = [sample["wall_step_ms"] for sample in production_samples]
    cuda_samples = [sample["cuda_step_ms"] for sample in production_samples]
    wall_median = float(statistics.median(wall_samples))
    cuda_median = float(statistics.median(cuda_samples))
    wall_p95 = float(statistics.quantiles(wall_samples, n=20, method="inclusive")[18])
    cuda_p95 = float(statistics.quantiles(cuda_samples, n=20, method="inclusive")[18])

    previous_semantic = os.environ.get("P1_SEMANTIC_SHA256")
    os.environ["P1_SEMANTIC_SHA256"] = _FORMAL_A_PARENT_SEMANTIC_SHA256
    try:
        receipt = _emit_p1_admission_receipt(
            {
                "schema": "apex-tune.b200.formal-a-compact-p1-no-score-engineering.v1",
                "authority": "diagnostic_only",
                "stage": "engineering_no_score",
                "variant": "formal_a",
                "requested_variant": requested_variant,
                "requested_edge": requested_edge,
                "edge": "E0",
                "runtime_mode": _p1_engineering_runtime_mode("E0"),
                "p1_compact_backward": False,
                "p2_forward_reuse": False,
                "p1_reduction": "none",
                "duplicate_reduction": False,
                "slot_dtype": None,
                "p1_fused_cleanup": False,
                "cleanup_path": "none",
                "rmsprop_impl": "compact",
                "target_backend_required": require_b200,
                "target_backend_check": {
                    "device_name": device_name,
                    "compute_capability": list(capability),
                    "visible_cuda_device_count": device_count,
                    "passed": (
                        device_count == 1
                        and "B200" in device_name.upper()
                        and capability == (10, 0)
                    )
                    if require_b200
                    else None,
                },
                "candidate_source_sha256": _FORMAL_A_PARENT_SOURCE_SHA256,
                "executed_source_sha256": _FORMAL_A_PARENT_SOURCE_SHA256,
                "parent_source_sha256": _FORMAL_A_PARENT_SOURCE_SHA256,
                "source_projection": "definition_prefix_before_t_start",
                "hardware": {
                    "device_name": device_name,
                    "compute_capability": list(capability),
                    "gpu_index": torch.cuda.current_device(),
                    "cuda_version": torch.version.cuda,
                    "torch_version": torch.__version__,
                    "triton_version": triton.__version__,
                },
                "configuration": {
                    "batch": batch,
                    "sequence_length": seq_len,
                    "tokens_per_step": batch * seq_len,
                    "warmup_steps": warmup_steps,
                    "measured_steps": measured_steps,
                    "frozen_stream_batches": stream_batches,
                    "stream_sha256": stream_sha256,
                    "stream_manifest": stream_manifest,
                    "compile_mode": compile_mode,
                    "fullgraph": True,
                    "mark_ngram_rows_called": True,
                    "prepare_ngram_forward_called": False,
                    "canonical_timing_boundary": (
                        "before_begin_step_and_mark_ngram_rows_through_optimizer_step_zero_grad"
                    ),
                    "score_called": False,
                    "quality_proxy_called": False,
                },
                "timing": {
                    "canonical_clock": "synchronized_production_like_wall_step_including_mark_ngram_rows",
                    "timing_boundary": {
                        "wall_start": "before_optimizer_begin_step",
                        "cuda_event_start": "before_optimizer_begin_step",
                        "includes_begin_step": True,
                        "includes_mark_ngram_rows": True,
                        "includes_prepare_ngram_forward": False,
                        "wall_end": "after_optimizer_step_and_zero_grad",
                        "cuda_event_end": "after_optimizer_step_and_zero_grad",
                    },
                    "production_samples": production_samples,
                    "wall_median_step_ms": wall_median,
                    "wall_p95_step_ms": wall_p95,
                    "wall_median_tps": (batch * seq_len) / (wall_median / 1000.0),
                    "cuda_event_median_step_ms": cuda_median,
                    "cuda_event_p95_step_ms": cuda_p95,
                    "peak_allocated_mib": peak_allocated / (1024 * 1024),
                    "peak_reserved_mib": peak_reserved / (1024 * 1024),
                },
                "compiler": {
                    "graph_breaks": graph_breaks,
                    "cudagraph_skips": cudagraph_skips,
                    "recompilations": int(counters["frames"].get("total", 0)),
                    "ir_nodes_pre_fusion": int(inductor_metrics.ir_nodes_pre_fusion),
                    "generated_kernels": int(inductor_metrics.generated_kernel_count),
                },
            },
            env_name="P1_ENGINEERING_RECEIPT",
        )
    finally:
        if previous_semantic is None:
            os.environ.pop("P1_SEMANTIC_SHA256", None)
        else:
            os.environ["P1_SEMANTIC_SHA256"] = previous_semantic
    print(
        "P1_NO_SCORE_ENGINEERING PASS "
        f"device={device_name} variant=formal_a edge=E0 strict_b200={require_b200} "
        f"warmup={warmup_steps} measured={measured_steps} "
        f"wall_median_step_ms={wall_median:.6f} wall_p95_step_ms={wall_p95:.6f} "
        f"wall_median_tps={(batch * seq_len) / (wall_median / 1000.0):.3f} "
        f"cuda_median_step_ms={cuda_median:.6f} graph_breaks={graph_breaks} "
        f"cudagraph_skips={cudagraph_skips} "
        + (
            f"receipt_sha256={receipt['receipt_sha256']}"
            if receipt is not None
            else "receipt_sha256=absent"
        )
    )
    return receipt


def _run_p1_no_score_engineering():
    """Run fixed-shape training timing without loading or invoking BPB."""
    requested_variant, variant, engineering_edge = _resolve_p1_engineering_selection()
    if not torch.cuda.is_available():
        raise RuntimeError("P1_NO_SCORE_ENGINEERING requires CUDA")
    if variant == "parent":
        strict_b200_value = os.environ.get("P1_ENGINEERING_REQUIRE_B200", "0")
        if strict_b200_value not in {"0", "1"}:
            raise ValueError("P1_ENGINEERING_REQUIRE_B200 must be 0 or 1")
        return _run_formal_a_parent_no_score_engineering(
            requested_variant=requested_variant,
            requested_edge=engineering_edge if os.environ.get("P1_ENGINEERING_EDGE") is not None else None,
            require_b200=strict_b200_value == "1",
        )
    runtime_mode = _p1_engineering_runtime_mode(engineering_edge)
    if _P1_ENGINEERING_EDGE_VARIANTS[engineering_edge] != variant:
        raise RuntimeError("engineering edge variant drifted from runtime mode")
    use_compact = bool(runtime_mode["p1_compact_backward"])
    use_p2_forward_reuse = bool(runtime_mode["p2_forward_reuse"])
    p1_reduction = runtime_mode["p1_reduction"]
    p1_fused_cleanup = bool(runtime_mode["p1_fused_cleanup"])
    if use_compact and not use_p2_forward_reuse:
        raise RuntimeError("compact backward requires P2 forward reuse")
    expected_impl = "compact"
    configured_impl = os.environ.get("TOUCHED_RMSPROP_IMPL")
    if configured_impl not in {None, expected_impl}:
        raise ValueError(
            f"{variant} engineering requires TOUCHED_RMSPROP_IMPL={expected_impl}"
        )
    # The optimizer reads this at construction time.  Make the variant choice
    # explicit in the process rather than allowing a stale shell environment to
    # silently turn the parent into a touched-row implementation.
    previous_rmsprop_impl = configured_impl
    strict_b200_value = os.environ.get("P1_ENGINEERING_REQUIRE_B200", "0")
    if strict_b200_value not in {"0", "1"}:
        raise ValueError("P1_ENGINEERING_REQUIRE_B200 must be 0 or 1")
    require_b200 = strict_b200_value == "1"
    device_name = torch.cuda.get_device_name()
    capability = tuple(torch.cuda.get_device_capability())
    device_count = torch.cuda.device_count()
    if require_b200:
        if device_count != 1:
            raise RuntimeError(
                "strict P1 engineering requires exactly one visible CUDA device"
            )
        if "B200" not in device_name.upper() or capability != (10, 0):
            raise RuntimeError(
                "strict P1 engineering requires NVIDIA B200 with compute capability sm_100"
            )
    os.environ["TOUCHED_RMSPROP_IMPL"] = expected_impl
    warmup_steps = int(os.environ.get("P1_ENGINEERING_WARMUP_STEPS", "12"))
    measured_steps = int(os.environ.get("P1_ENGINEERING_MEASURED_STEPS", "128"))
    if warmup_steps < 0 or measured_steps <= 0:
        raise ValueError("invalid no-score engineering step counts")
    batch = int(os.environ.get("P1_ENGINEERING_BATCH", "72"))
    seq_len = int(os.environ.get("P1_ENGINEERING_SEQ_LEN", "2048"))
    compile_mode = os.environ.get("P1_ENGINEERING_COMPILE_MODE", "max-autotune")
    if compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
        raise ValueError("invalid no-score engineering compile mode")
    device = torch.device("cuda")
    torch.manual_seed(int(os.environ.get("NANOCHAT_SEED", "42")))
    torch.cuda.manual_seed(int(os.environ.get("NANOCHAT_SEED", "42")))
    torch.set_float32_matmul_precision("high")
    config = GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=8192,
        n_layer=8,
        n_head=6,
        n_kv_head=6,
        n_embd=768,
        window_pattern="TTTL",
    )
    with torch.device("meta"):
        model = GPT(
            config,
            p1_compact_backward=use_compact,
            p2_forward_reuse=use_p2_forward_reuse,
            p1_reduction=p1_reduction,
            p1_fused_cleanup=p1_fused_cleanup,
        )
    model.to_empty(device=device)
    model.init_weights()
    model.to(dtype=torch.bfloat16)
    try:
        optimizer = model.setup_optimizer(
            unembedding_lr=0.004,
            embedding_lr=0.6,
            matrix_lr=0.04,
            weight_decay=0.1,
            adam_betas=(0.8, 0.95),
            scalar_lr=0.8,
            ngram_ve_betas=(0.5, 0.999),
            # E1-E4 use the frozen per-order slot policy.  It is deliberately
            # independent of tokens_per_step; unique touched rows are what
            # determine slot pressure.
            ngram_touch_capacity=_P1_COMPACT_SLOT_CAPACITY_POLICY,
        )
    finally:
        if previous_rmsprop_impl is None:
            os.environ.pop("TOUCHED_RMSPROP_IMPL", None)
        else:
            os.environ["TOUCHED_RMSPROP_IMPL"] = previous_rmsprop_impl
    if optimizer._rmsprop_impl != expected_impl:
        raise RuntimeError(
            f"engineering {variant} selected {optimizer._rmsprop_impl!r}, "
            f"expected {expected_impl!r}"
        )
    if optimizer._p1_reduction != p1_reduction:
        raise RuntimeError(
            f"engineering {engineering_edge} selected reduction "
            f"{optimizer._p1_reduction!r}, expected {p1_reduction!r}"
        )
    if optimizer._p1_fused_cleanup != p1_fused_cleanup:
        raise RuntimeError(
            f"engineering {engineering_edge} selected cleanup "
            f"{optimizer._p1_fused_cleanup!r}, expected {p1_fused_cleanup!r}"
        )
    tokenizer = Tokenizer.from_directory()
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise AssertionError("no-score engineering tokenizer vocabulary changed")
    train_loader = make_dataloader(tokenizer, batch, seq_len, "train")
    stream_batches = int(
        os.environ.get(
            "P1_ENGINEERING_STREAM_BATCHES",
            str(max(2, min(32, warmup_steps + measured_steps))),
        )
    )
    if stream_batches < 2:
        raise ValueError("no-score engineering requires at least two frozen batches")
    frozen_stream = []
    stream_manifest = []
    for stream_i in range(stream_batches):
        tokens, targets, epoch = next(train_loader)
        frozen_tokens = tokens.clone()
        frozen_targets = targets.clone()
        token_bytes = frozen_tokens.detach().cpu().numpy().tobytes(order="C")
        target_bytes = frozen_targets.detach().cpu().numpy().tobytes(order="C")
        stream_manifest.append(
            {
                "stream_index": stream_i,
                "epoch": int(epoch),
                "tokens_sha256": hashlib.sha256(token_bytes).hexdigest(),
                "targets_sha256": hashlib.sha256(target_bytes).hexdigest(),
            }
        )
        frozen_stream.append((frozen_tokens, frozen_targets))
    stream_sha256 = hashlib.sha256(
        _p1_canonical_json_bytes(stream_manifest)
    ).hexdigest()
    counters = __import__("torch._dynamo.utils", fromlist=["counters"]).counters
    from torch._inductor import metrics as inductor_metrics

    counters.clear()
    inductor_metrics.reset()
    compiled_model = torch.compile(
        model, dynamic=False, mode=compile_mode, fullgraph=True
    )
    model.train()

    def step(stream_index, measure=False, profile=False):
        tokens, targets = frozen_stream[stream_index % stream_batches]
        if measure:
            # Allocate timing events before the wall clock starts; event
            # construction is instrumentation, not production step work.
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter() if measure else None
        if measure:
            # The canonical step includes lifecycle setup and P2 preparation.
            # Recording after preparation would make the child look faster by
            # excluding work that the parent does not need to perform.
            start.record()
        optimizer.begin_step()
        if use_p2_forward_reuse:
            optimizer.prepare_ngram_forward(tokens)
        torch.compiler.cudagraph_mark_step_begin()
        loss = compiled_model(tokens, targets)
        loss.backward()
        optimizer.step()
        compact_slot_counts = [
            int(counter.item())
            for counter in optimizer._rmsprop_active_counters.values()
        ]
        model.zero_grad(set_to_none=True)
        if measure:
            end.record()
            end.synchronize()
            return {
                "stream_index": stream_index % stream_batches,
                "cuda_step_ms": float(start.elapsed_time(end)),
                "wall_step_ms": float((time.perf_counter() - wall_start) * 1000.0),
                "profiled": bool(profile),
                "compact_slot_counts": compact_slot_counts,
            }
        return None

    for warmup_i in range(warmup_steps):
        step(warmup_i)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    production_samples = [
        step(warmup_steps + measured_i, measure=True)
        for measured_i in range(measured_steps)
    ]
    torch.cuda.synchronize()
    production_peak_allocated = torch.cuda.max_memory_allocated()
    production_peak_reserved = torch.cuda.max_memory_reserved()
    # Capture per-table touch counts and the fixed scratch footprint for every
    # E1-E4 edge.  E1 reuses the compact touch collector but deliberately has
    # no compact gradient slots; reporting an empty slot-byte total keeps that
    # diagnostic distinction explicit in the receipt.
    compact_slot_counts = [
        int(counter.item())
        for counter in optimizer._rmsprop_active_counters.values()
    ]
    compact_slot_capacities = [
        int(rows.numel()) for rows in optimizer._rmsprop_active_rows.values()
    ]
    compact_grad_slot_bytes = sum(
        int(slots.numel() * slots.element_size())
        for slots in optimizer._rmsprop_grad_slots.values()
    )
    compact_slot_table_diagnostics = []
    for table_index, (table, counter) in enumerate(
        optimizer._rmsprop_active_counters.items()
    ):
        order = optimizer._rmsprop_order_by_param[table]
        capacity_rows = optimizer._rmsprop_capacity_by_param[table]
        slot_tensor = optimizer._rmsprop_grad_slots.get(table)
        compact_slot_table_diagnostics.append(
            {
                "table_index": table_index,
                "order": order,
                "order_name": "bigram" if order == 2 else "trigram",
                "capacity_rows": int(capacity_rows),
                "active_rows": int(counter.item()),
                "slot_bytes": (
                    int(slot_tensor.numel() * slot_tensor.element_size())
                    if slot_tensor is not None
                    else 0
                ),
                "table_shape": [int(dim) for dim in table.shape],
            }
        )
    if not use_compact and compact_grad_slot_bytes:
        raise RuntimeError(
            f"engineering {engineering_edge} unexpectedly allocated compact gradient slots"
        )
    profile_steps = int(os.environ.get("P1_ENGINEERING_PROFILE_STEPS", "1"))
    if profile_steps < 0:
        raise ValueError("P1_ENGINEERING_PROFILE_STEPS cannot be negative")
    profiler_samples = []
    profiler_keys = []
    if profile_steps:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            for profile_i in range(profile_steps):
                profiler_samples.append(
                    step(
                        warmup_steps + measured_steps + profile_i,
                        measure=True,
                        profile=True,
                    )
                )
        profiler_keys = sorted(event.key.lower() for event in profile.key_averages())
        profiler_device_time_ms = float(
            sum(
                float(getattr(event, "self_device_time_total", 0.0))
                for event in profile.key_averages()
            )
            / 1000.0
        )
    else:
        profiler_device_time_ms = 0.0
    graph_breaks = sum(int(value) for value in counters["graph_break"].values())
    cudagraph_skips = int(counters["inductor"].get("cudagraph_skips", 0))
    wall_samples = [sample["wall_step_ms"] for sample in production_samples]
    cuda_samples = [sample["cuda_step_ms"] for sample in production_samples]
    compact_count_samples = [
        sample["compact_slot_counts"]
        for sample in production_samples
        if sample["compact_slot_counts"]
    ]
    if compact_count_samples:
        compact_slot_count_max_by_table = [
            max(sample[table_i] for sample in compact_count_samples)
            for table_i in range(len(compact_count_samples[0]))
        ]
        compact_slot_count_p95_by_table = [
            int(
                math.ceil(
                    statistics.quantiles(
                        [sample[table_i] for sample in compact_count_samples],
                        n=20,
                        method="inclusive",
                    )[18]
                )
            )
            if len(compact_count_samples) >= 2
            else compact_count_samples[0][table_i]
            for table_i in range(len(compact_count_samples[0]))
        ]
    else:
        compact_slot_count_max_by_table = []
        compact_slot_count_p95_by_table = []
    wall_p95 = (
        float(statistics.quantiles(wall_samples, n=20, method="inclusive")[18])
        if len(wall_samples) >= 2
        else wall_samples[0]
    )
    cuda_p95 = (
        float(statistics.quantiles(cuda_samples, n=20, method="inclusive")[18])
        if len(cuda_samples) >= 2
        else cuda_samples[0]
    )
    wall_median_ms = float(statistics.median(wall_samples))
    cuda_median_ms = float(statistics.median(cuda_samples))
    source_projection = {
        "E1": "formal_a_p2_forward_reuse_dense_backward",
        "E2": "formal_a_p2_compact_index_add_separate_cleanup",
        "E3": "formal_a_p2_compact_fp32_atomic_separate_cleanup",
        "E4": "formal_a_p2_compact_fp32_atomic_fused_cleanup",
    }[engineering_edge]
    if use_compact and P1_FUSED_LOOKUP_COLLECT_ENABLED:
        source_projection += "_fused_pair_lookup_collect"
    receipt = _emit_p1_admission_receipt(
        {
            "schema": "apex-tune.b200.formal-a-compact-p1-no-score-engineering.v1",
            "authority": "diagnostic_only",
            "stage": "engineering_no_score",
            "variant": variant,
            "requested_variant": requested_variant,
            "requested_edge": (
                engineering_edge
                if os.environ.get("P1_ENGINEERING_EDGE") is not None
                else None
            ),
            "edge": engineering_edge,
            "runtime_mode": runtime_mode,
            "p1_compact_backward": use_compact,
            "p2_forward_reuse": use_p2_forward_reuse,
            "p1_reduction": p1_reduction,
            "duplicate_reduction": runtime_mode["duplicate_reduction"],
            "slot_dtype": runtime_mode["slot_dtype"],
            "slot_capacity_policy": optimizer._rmsprop_touch_capacity_policy,
            "slot_capacity_policy_id": optimizer._rmsprop_touch_capacity_policy_id,
            "slot_table_diagnostics": compact_slot_table_diagnostics,
            "p1_fused_cleanup": p1_fused_cleanup,
            "p1_fused_lookup_collect": P1_FUSED_LOOKUP_COLLECT_ENABLED,
            "cleanup_path": (
                "fused_active_slot_consume_clear"
                if p1_fused_cleanup
                else "separate_active_slot_consume_clear"
                if use_compact
                else "none"
            ),
            "rmsprop_impl": expected_impl,
            "target_backend_required": require_b200,
            "target_backend_check": {
                "device_name": device_name,
                "compute_capability": list(capability),
                "visible_cuda_device_count": device_count,
                "passed": (
                    device_count == 1
                    and "B200" in device_name.upper()
                    and capability == (10, 0)
                )
                if require_b200
                else None,
            },
            "candidate_source_sha256": _p1_file_sha256(__file__),
            "executed_source_sha256": _p1_file_sha256(__file__),
            "parent_source_sha256": (
                "010356bb5477ff5b9a0fa93391eb79b2d23706bc790176ad67d87e7e343a0dce"
            ),
            "source_projection": source_projection,
            "execution_identity": {
                "edge": engineering_edge,
                "runner_source_sha256": _p1_file_sha256(__file__),
                "formal_a_source_sha256": (
                    "010356bb5477ff5b9a0fa93391eb79b2d23706bc790176ad67d87e7e343a0dce"
                ),
                "p2_forward_reuse": use_p2_forward_reuse,
                "p1_compact_backward": use_compact,
                "p1_reduction": p1_reduction,
                "duplicate_reduction": runtime_mode["duplicate_reduction"],
                "slot_dtype": runtime_mode["slot_dtype"],
                "p1_fused_cleanup": p1_fused_cleanup,
                "p1_fused_lookup_collect": P1_FUSED_LOOKUP_COLLECT_ENABLED,
                "lookup_block_tokens": _P1_LOOKUP_BLOCK_TOKENS,
                "lookup_num_warps": _P1_LOOKUP_NUM_WARPS,
                "scatter_block_tokens": _P1_SCATTER_BLOCK_TOKENS,
                "scatter_num_warps": _P1_SCATTER_NUM_WARPS,
                "cleanup_path": (
                    "fused_active_slot_consume_clear"
                    if p1_fused_cleanup
                    else "separate_active_slot_consume_clear"
                    if use_compact
                    else "none"
                ),
            },
            "hardware": {
                "device_name": torch.cuda.get_device_name(),
                "compute_capability": list(torch.cuda.get_device_capability()),
                "gpu_index": torch.cuda.current_device(),
                "cuda_version": torch.version.cuda,
                "torch_version": torch.__version__,
                "triton_version": triton.__version__,
            },
            "configuration": {
                "batch": batch,
                "sequence_length": seq_len,
                "tokens_per_step": batch * seq_len,
                "warmup_steps": warmup_steps,
                "measured_steps": measured_steps,
                "profile_steps": profile_steps,
                "frozen_stream_batches": stream_batches,
                "stream_sha256": stream_sha256,
                "stream_manifest": stream_manifest,
                "stream_reuse_count": max(
                    0, warmup_steps + measured_steps + profile_steps - stream_batches
                ),
                "compile_mode": compile_mode,
                "fullgraph": True,
                "prepare_ngram_forward_called": use_p2_forward_reuse,
                "p2_forward_indices_collector_called": use_p2_forward_reuse,
                "p2_forward_indices_collector_path": (
                    "fused_pair_lookup"
                    if use_compact and P1_FUSED_LOOKUP_COLLECT_ENABLED
                    else "separate_pair_collector"
                    if use_p2_forward_reuse
                    else "none"
                ),
                "compact_slot_count_samples": compact_count_samples,
                "compact_slot_count_max_by_table": compact_slot_count_max_by_table,
                "compact_slot_count_p95_by_table": compact_slot_count_p95_by_table,
                "compact_slot_capacity_rows": compact_slot_capacities,
                "compact_grad_slot_bytes": compact_grad_slot_bytes,
                "compact_slot_table_diagnostics": compact_slot_table_diagnostics,
                "compact_slot_capacity_policy": optimizer._rmsprop_touch_capacity_policy,
                "compact_slot_capacity_policy_id": optimizer._rmsprop_touch_capacity_policy_id,
                "canonical_timing_boundary": (
                    "before_begin_step_and_prepare_through_optimizer_step_zero_grad"
                ),
                "score_called": False,
                "quality_proxy_called": False,
            },
            "timing": {
                "canonical_clock": (
                    "synchronized_production_like_wall_step_including_prepare"
                ),
                "timing_boundary": {
                    "wall_start": "before_optimizer_begin_step",
                    "cuda_event_start": "before_optimizer_begin_step",
                    "includes_begin_step": True,
                    "includes_prepare_ngram_forward": use_p2_forward_reuse,
                    "wall_end": "after_optimizer_step_and_zero_grad",
                    "cuda_event_end": "after_optimizer_step_and_zero_grad",
                },
                "production_samples": production_samples,
                "wall_median_step_ms": wall_median_ms,
                "wall_p95_step_ms": wall_p95,
                "wall_median_tps": (batch * seq_len) / (wall_median_ms / 1000.0),
                "cuda_event_median_step_ms": cuda_median_ms,
                "cuda_event_p95_step_ms": cuda_p95,
                "peak_allocated_mib": production_peak_allocated / (1024 * 1024),
                "peak_reserved_mib": production_peak_reserved / (1024 * 1024),
                "compact_slot_count_max": max(
                    compact_slot_count_max_by_table, default=0
                ),
                "compact_slot_count_p95": max(
                    compact_slot_count_p95_by_table, default=0
                ),
                "compact_slot_count_max_by_table": compact_slot_count_max_by_table,
                "compact_slot_count_p95_by_table": compact_slot_count_p95_by_table,
                "compact_slot_capacity_max": max(compact_slot_capacities, default=0),
            },
            "profiler_timing_diagnostic_only": {
                "excluded_from_canonical_timing": True,
                "samples": profiler_samples,
                "profiler_device_time_ms": profiler_device_time_ms,
                "profiler_keys_sha256": hashlib.sha256(
                    _p1_canonical_json_bytes(profiler_keys)
                ).hexdigest(),
            },
            "compiler": {
                "graph_breaks": graph_breaks,
                "cudagraph_skips": cudagraph_skips,
                "recompilations": int(counters["frames"].get("total", 0)),
                "ir_nodes_pre_fusion": int(inductor_metrics.ir_nodes_pre_fusion),
                "generated_kernels": int(inductor_metrics.generated_kernel_count),
            },
        },
        env_name="P1_ENGINEERING_RECEIPT",
    )
    print(
        "P1_NO_SCORE_ENGINEERING PASS "
        f"device={device_name} variant={variant} edge={engineering_edge} "
        f"strict_b200={require_b200} "
        f"warmup={warmup_steps} "
        f"stream_batches={stream_batches} measured={measured_steps} "
        f"wall_median_step_ms={wall_median_ms:.6f} wall_p95_step_ms={wall_p95:.6f} "
        f"wall_median_tps={(batch * seq_len) / (wall_median_ms / 1000.0):.3f} "
        f"cuda_median_step_ms={cuda_median_ms:.6f} profile_steps={profile_steps} "
        f"graph_breaks={graph_breaks} cudagraph_skips={cudagraph_skips} "
        + (
            f"receipt_sha256={receipt['receipt_sha256']}"
            if receipt is not None
            else "receipt_sha256=absent"
        )
    )
    return receipt


def _run_p1_c428_full_step_admission():
    """Two complete real-c428-shape H200 steps; diagnostics have no score authority."""
    if not torch.cuda.is_available():
        raise RuntimeError("P1_C428_FULL_STEP_ADMISSION requires CUDA")
    if not P1_COMPACT_BACKWARD_ENABLED:
        raise RuntimeError("P1 c428 admission requires P1_COMPACT_BACKWARD=1")
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("P1 c428 local admission is frozen to Hopper/H200")

    from torch._dynamo.utils import counters
    from torch._inductor import metrics as inductor_metrics

    device = torch.device("cuda")
    torch.manual_seed(20260817)
    torch.cuda.manual_seed(20260817)
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    initial_allocated = torch.cuda.memory_allocated()
    config = GPTConfig(
        sequence_len=2048,
        vocab_size=8192,
        n_layer=8,
        n_head=6,
        n_kv_head=6,
        n_embd=768,
        window_pattern="TTTL",
    )
    with torch.device("meta"):
        model = GPT(config, p1_compact_backward=True)
    model.to_empty(device=device)
    model.init_weights()
    model.to(dtype=torch.bfloat16)
    after_model_allocated = torch.cuda.memory_allocated()

    batch = int(os.environ.get("P1_C428_ADMISSION_BATCH", "72"))
    seq_len = int(os.environ.get("P1_C428_ADMISSION_SEQ_LEN", "2048"))
    num_tokens = batch * seq_len
    exact_c428_shape = batch == 72 and seq_len == 2048
    optimizer = model.setup_optimizer(
        unembedding_lr=0.004,
        embedding_lr=0.6,
        matrix_lr=0.04,
        weight_decay=0.1,
        adam_betas=(0.8, 0.95),
        scalar_lr=0.8,
        ngram_ve_betas=(0.5, 0.999),
        ngram_ve_lr_scale=1.0,
        ngram_touch_capacity=_P1_COMPACT_SLOT_CAPACITY_POLICY,
    )
    after_optimizer_allocated = torch.cuda.memory_allocated()
    tokenizer = Tokenizer.from_directory()
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise AssertionError("c428 admission tokenizer vocabulary changed")
    train_loader = make_dataloader(tokenizer, batch, seq_len, "train")
    tokens, targets, _epoch = next(train_loader)
    if tokens.shape != (batch, seq_len) or targets.shape != tokens.shape:
        raise AssertionError("c428 admission dataloader shape changed")

    counters.clear()
    inductor_metrics.reset()
    compile_mode = os.environ.get("P1_C428_ADMISSION_COMPILE_MODE", "max-autotune")
    if compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
        raise ValueError("invalid P1 c428 admission compile mode")
    compiled_model = torch.compile(
        model, dynamic=False, mode=compile_mode, fullgraph=True
    )
    model.train()

    # Warmup is a complete step: it compiles forward/backward, records the
    # CUDAGraph, and allocates every optimizer state before peak measurement.
    optimizer.begin_step()
    optimizer.prepare_ngram_forward(tokens)
    torch.compiler.cudagraph_mark_step_begin()
    warmup_loss = compiled_model(tokens, targets)
    # Inductor CUDAGraph outputs borrow replay-owned storage. Snapshot the
    # scalar before backward or a later replay can overwrite the audit value.
    warmup_loss_snapshot = warmup_loss.detach().clone()
    warmup_loss.backward()
    optimizer.step()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    steady_baseline_allocated = torch.cuda.memory_allocated()
    steady_baseline_reserved = torch.cuda.memory_reserved()

    optimizer.begin_step()
    optimizer.prepare_ngram_forward(tokens)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.compiler.cudagraph_mark_step_begin()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        loss = compiled_model(tokens, targets)
        loss_snapshot = loss.detach().clone()
        loss.backward()
    end.record()
    end.synchronize()
    forward_backward_ms = start.elapsed_time(end)
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    profiler_keys = [event.key.lower() for event in profile.key_averages()]
    embedding_dense_backward_events = [
        (event.key, int(event.count))
        for event in profile.key_averages()
        if "embedding_dense_backward" in event.key.lower()
    ]
    # Inductor fuses the five legitimate native embedding backwards (token
    # embedding plus four unigram VEs) into several kernels. Kernel launch
    # counts are not a valid logical-op count, so n-gram exclusion is proved
    # below from all fourteen p.grad edges and compact-slot ownership.
    if not any("collect_pair_indices" in key for key in profiler_keys):
        raise AssertionError("c428 profile did not execute forward-index collector")
    if not any("scatter_pair_fp32" in key for key in profiler_keys):
        raise AssertionError("c428 profile did not execute compact backward scatter")
    if any(
        old_collector in key
        for key in profiler_keys
        for old_collector in (
            "_collect_bigram_pair_kernel",
            "_collect_trigram_pair_kernel",
            "_mark_bigram_pair_kernel",
            "_mark_trigram_pair_kernel",
        )
    ):
        raise AssertionError("c428 training profile still rehashed tokens")

    ngram_params = tuple(model.bigram_ves.parameters()) + tuple(
        model.trigram_ves.parameters()
    )
    if len(ngram_params) != 14:
        raise AssertionError("c428 admission expected 14 n-gram tables")
    if any(param.grad is not None for param in ngram_params):
        raise AssertionError("c428 admission materialized an n-gram dense gradient")
    ngram_param_set = set(ngram_params)
    non_ngram_grad_count = 0
    for param in model.parameters():
        if param in ngram_param_set:
            continue
        if param.grad is not None:
            non_ngram_grad_count += 1
            if not bool(torch.isfinite(param.grad).all().item()):
                raise AssertionError("c428 admission produced a non-finite non-ngram gradient")
    if non_ngram_grad_count != 77:
        raise AssertionError(
            f"c428 admission expected 77 non-ngram gradients, got {non_ngram_grad_count}"
        )

    prev = torch.cat([tokens[:, :1], tokens[:, :-1]], dim=1)
    prev2 = torch.cat([tokens[:, :2], tokens[:, :-2]], dim=1)
    table_indices = {}
    for layer_i in sorted(model.bigram_ve_layers):
        pair = model.bigram_ves[str(layer_i)]
        primes = model.bigram_hash_primes_per_layer[layer_i]
        table_indices[pair[0].weight] = (
            (prev * primes[0][0]) ^ (tokens * primes[0][1])
        ).bitwise_and(model.bigram_table_size - 1)
        table_indices[pair[1].weight] = (
            (prev * primes[1][0]) ^ (tokens * primes[1][1])
        ).bitwise_and(model.bigram_table_size - 1)
    for layer_i in sorted(model.trigram_ve_layers):
        pair = model.trigram_ves[str(layer_i)]
        primes = model.trigram_hash_primes_per_layer[layer_i]
        table_indices[pair[0].weight] = (
            (prev2 * primes[0]) ^ (prev * primes[1]) ^ (tokens * primes[2])
        ).bitwise_and(model.trigram_table_size - 1)
        table_indices[pair[1].weight] = (
            (prev2 * primes[3]) ^ (prev * primes[4]) ^ (tokens * primes[5])
        ).bitwise_and(model.trigram_table_size - 1)

    max_slot_count = 0
    min_slot_count = num_tokens
    update_audits = []
    rmsprop_group_for_param = {
        param: group
        for group in optimizer.param_groups
        if group["kind"] == "rmsprop"
        for param in group["params"]
    }
    for param in ngram_params:
        count = int(optimizer._rmsprop_active_counters[param].item())
        if count <= 0 or count > num_tokens:
            raise AssertionError(f"c428 compact slot count is invalid: {count}")
        max_slot_count = max(max_slot_count, count)
        min_slot_count = min(min_slot_count, count)
        active = optimizer._rmsprop_active_rows[param][:count].long()
        expected_active = torch.unique(table_indices[param]).sort().values
        if not torch.equal(active.sort().values, expected_active):
            raise AssertionError("c428 forward-index collector changed the exact touch set")
        mapped = optimizer._rmsprop_row_to_slot[param].index_select(0, active)
        if not torch.equal(mapped, torch.arange(count, device=device, dtype=torch.int32)):
            raise AssertionError("c428 active-row slot ownership is not bijective")
        generation = optimizer._rmsprop_touch_generation[0]
        stamped = optimizer._rmsprop_touch_stamps[param].index_select(0, active)
        if not bool((stamped == generation).all().item()):
            raise AssertionError("c428 active row lacks the current generation stamp")
        slots = optimizer._rmsprop_grad_slots[param][:count]
        if not bool(torch.isfinite(slots).all().item()):
            raise AssertionError("c428 compact gradient slot is non-finite")

        active_row = int(active[0].item())
        active_slot = int(optimizer._rmsprop_row_to_slot[param][active_row].item())
        probes = torch.arange(1024, device=device, dtype=torch.long)
        inactive_candidates = probes[~torch.isin(probes, active)]
        if inactive_candidates.numel() == 0:
            raise AssertionError("c428 audit could not find an untouched probe row")
        inactive_row = int(inactive_candidates[0].item())
        state = optimizer.state[param]["exp_avg_sq"]
        untouched_param = param[inactive_row].clone()
        untouched_state = state[inactive_row].clone()
        expected_param = param[active_row : active_row + 1].clone()
        expected_state = state[active_row : active_row + 1].clone()
        expected_grad = optimizer._rmsprop_grad_slots[param][
            active_slot : active_slot + 1
        ].to(torch.bfloat16)
        group = rmsprop_group_for_param[param]
        rmsprop_step_fused(
            expected_param,
            expected_grad,
            expected_state,
            torch.tensor(2.0, dtype=torch.float32),
            torch.tensor(group["lr"], dtype=torch.float32),
            torch.tensor(group["beta2"], dtype=torch.float32),
            torch.tensor(group["eps"], dtype=torch.float32),
            torch.tensor(group["weight_decay"], dtype=torch.float32),
        )
        update_audits.append(
            (
                param,
                state,
                active_row,
                expected_param,
                expected_state,
                inactive_row,
                untouched_param,
                untouched_state,
                count,
            )
        )

    del table_indices, prev, prev2
    optimizer.step()
    torch.cuda.synchronize()
    for (
        param,
        state,
        active_row,
        expected_param,
        expected_state,
        inactive_row,
        untouched_param,
        untouched_state,
        count,
    ) in update_audits:
        if not torch.equal(
            param[active_row : active_row + 1].view(torch.int16),
            expected_param.view(torch.int16),
        ):
            raise AssertionError("c428 touched parameter update lost bitwise parity")
        if not torch.equal(
            state[active_row : active_row + 1].view(torch.int16),
            expected_state.view(torch.int16),
        ):
            raise AssertionError("c428 touched RMS state update lost bitwise parity")
        if not torch.equal(param[inactive_row], untouched_param):
            raise AssertionError("c428 optimizer changed an untouched parameter row")
        if not torch.equal(state[inactive_row], untouched_state):
            raise AssertionError("c428 optimizer changed an untouched RMS state row")
        if int(
            torch.count_nonzero(optimizer._rmsprop_grad_slots[param][:count]).item()
        ) != 0:
            raise AssertionError("c428 optimizer did not clear active compact slots")
    model.zero_grad(set_to_none=True)

    cudagraph_skips = int(counters["inductor"].get("cudagraph_skips", 0))
    graph_breaks = sum(int(value) for value in counters["graph_break"].values())
    expects_cudagraph = compile_mode in {"reduce-overhead", "max-autotune"}
    if expects_cudagraph and cudagraph_skips != 0:
        raise AssertionError(f"c428 compiled step incurred {cudagraph_skips} cudagraph skips")
    if graph_breaks != 0:
        raise AssertionError(f"c428 compiled step incurred {graph_breaks} graph breaks")

    scratch_tensors = (
        list(optimizer._rmsprop_touch_stamps.values())
        + list(optimizer._rmsprop_active_rows.values())
        + list(optimizer._rmsprop_row_to_slot.values())
        + list(optimizer._rmsprop_active_counters.values())
        + list(optimizer._rmsprop_grad_slots.values())
        + [optimizer._rmsprop_touch_generation]
    )
    scratch_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in scratch_tensors
    )
    mib = 1024 * 1024
    admission_receipt = _emit_p1_admission_receipt(
        {
            "schema": "apex-tune.h200.formal-a-compact-p1-admission.v1",
            "authority": "diagnostic_only",
            "candidate_source_sha256": _p1_file_sha256(__file__),
            "parent_source_sha256": (
                "010356bb5477ff5b9a0fa93391eb79b2d23706bc790176ad67d87e7e343a0dce"
            ),
            "hardware": {
                "device_name": torch.cuda.get_device_name(),
                "compute_capability": list(torch.cuda.get_device_capability()),
                "gpu_index": torch.cuda.current_device(),
                "cuda_version": torch.version.cuda,
                "torch_version": torch.__version__,
                "triton_version": triton.__version__,
            },
            "configuration": {
                "layers": 8,
                "ngram_tables": 14,
                "table_rows": 524288,
                "table_cols": 384,
                "batch": batch,
                "sequence_length": seq_len,
                "num_tokens": num_tokens,
                "compile_mode": compile_mode,
                "fullgraph": True,
            },
            "checks": {
                "exact_c428_shape": exact_c428_shape,
                "graph_breaks_zero": graph_breaks == 0,
                "cudagraph_skips_zero": cudagraph_skips == 0,
                "forward_indices_collector": any(
                    "collect_pair_indices" in key for key in profiler_keys
                ),
                "backward_compact_scatter": any(
                    "scatter_pair_fp32" in key for key in profiler_keys
                ),
                "token_hash_recompute_absent": not any(
                    old_collector in key
                    for key in profiler_keys
                    for old_collector in (
                        "_collect_bigram_pair_kernel",
                        "_collect_trigram_pair_kernel",
                        "_mark_bigram_pair_kernel",
                        "_mark_trigram_pair_kernel",
                    )
                ),
                "all_ngram_parameter_grads_none": all(
                    param.grad is None for param in ngram_params
                ),
                "non_ngram_grads_finite": non_ngram_grad_count == 77,
                "exact_touch_sets": len(update_audits) == 14,
                "slot_ownership_bijective": True,
                "touched_update_bitwise": True,
                "untouched_update_exact_skip": True,
                "slots_cleared": True,
                # Overflow and FP64 envelope are separate self-tests and are
                # intentionally not inferred from this full-step receipt.
                "overflow_fail_closed_checked": False,
                "fp64_envelope_checked": False,
            },
            "metrics": {
                "cudagraph_skips": cudagraph_skips,
                "graph_breaks": graph_breaks,
                "non_ngram_grad_count": non_ngram_grad_count,
                "slot_count_min": min_slot_count,
                "slot_count_max": max_slot_count,
                "warmup_loss": float(warmup_loss_snapshot.item()),
                "loss": float(loss_snapshot.item()),
                "forward_backward_ms": float(forward_backward_ms),
                "initial_allocated_mib": float(initial_allocated / mib),
                "model_allocated_mib": float(after_model_allocated / mib),
                "optimizer_allocated_mib": float(after_optimizer_allocated / mib),
                "steady_baseline_allocated_mib": float(steady_baseline_allocated / mib),
                "steady_baseline_reserved_mib": float(steady_baseline_reserved / mib),
                "peak_allocated_mib": float(peak_allocated / mib),
                "peak_reserved_mib": float(peak_reserved / mib),
                "compact_scratch_mib": float(scratch_bytes / mib),
                "ir_nodes_pre_fusion": int(inductor_metrics.ir_nodes_pre_fusion),
                "generated_kernels": int(inductor_metrics.generated_kernel_count),
            },
            "profiler": {
                "key_count": len(profiler_keys),
                "keys_sha256": hashlib.sha256(
                    _p1_canonical_json_bytes(sorted(profiler_keys))
                ).hexdigest(),
                "native_embedding_profile_key_count": len(
                    embedding_dense_backward_events
                ),
            },
        }
    )
    print(
        "P1_C428_FULL_STEP_ADMISSION PASS "
        f"device={torch.cuda.get_device_name()} authority=diagnostic_only "
        f"layers=8 ngram_tables=14 rows=524288 cols=384 batch={batch} seq={seq_len} "
        f"exact_c428_shape={str(exact_c428_shape).lower()} compile_mode={compile_mode} "
        f"fullgraph=pass graph_breaks=0 cudagraph_skips={cudagraph_skips} "
        "collector=forward_indices token_hash_recompute=absent "
        f"native_embedding_profile_keys={len(embedding_dense_backward_events)} "
        "ngram_embedding_dense_backward=absent_by_grad_edge "
        "p_grad_none=14/14 "
        f"non_ngram_grads=finite:{non_ngram_grad_count} exact_touch_sets=14/14 "
        "slot_ownership=bijective:14/14 touched_update=bitwise:14/14 "
        "untouched_update=exact_skip:14/14 slots_cleared=14/14 "
        f"slot_count_min={min_slot_count} slot_count_max={max_slot_count} "
        f"warmup_loss={float(warmup_loss_snapshot.item()):.8g} "
        f"loss={float(loss_snapshot.item()):.8g} "
        f"forward_backward_ms={forward_backward_ms:.4f} "
        f"initial_allocated_mib={initial_allocated / mib:.1f} "
        f"model_allocated_mib={after_model_allocated / mib:.1f} "
        f"optimizer_allocated_mib={after_optimizer_allocated / mib:.1f} "
        f"steady_baseline_allocated_mib={steady_baseline_allocated / mib:.1f} "
        f"steady_baseline_reserved_mib={steady_baseline_reserved / mib:.1f} "
        f"peak_allocated_mib={peak_allocated / mib:.1f} "
        f"peak_reserved_mib={peak_reserved / mib:.1f} "
        f"compact_scratch_mib={scratch_bytes / mib:.1f} "
        f"ir_nodes_pre_fusion={inductor_metrics.ir_nodes_pre_fusion} "
        f"generated_kernels={inductor_metrics.generated_kernel_count} "
        + (
            f"receipt_sha256={admission_receipt['receipt_sha256']}"
            if admission_receipt is not None
            else "receipt_sha256=absent"
        )
    )
    return admission_receipt


def _run_p1_slot_recycle_soak():
    """Recycle compact slots across generations and reject stale-row updates."""
    if not torch.cuda.is_available():
        raise RuntimeError("P1_SLOT_RECYCLE_SOAK requires CUDA")
    if not P1_COMPACT_BACKWARD_ENABLED:
        raise RuntimeError("P1 slot recycle soak requires P1_COMPACT_BACKWARD=1")
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("P1 slot recycle soak is frozen to Hopper/H200")
    steps = int(os.environ.get("P1_SLOT_RECYCLE_SOAK_STEPS", "256"))
    if steps < 2:
        raise ValueError("P1 slot recycle soak requires at least two steps")

    device = torch.device("cuda")
    rows, cols, capacity = 512, 16, 32
    table_0 = torch.nn.Parameter(
        torch.zeros(rows, cols, device=device, dtype=torch.bfloat16)
    )
    table_1 = torch.nn.Parameter(
        torch.zeros(rows, cols, device=device, dtype=torch.bfloat16)
    )
    optimizer = MuonAdamW(
        [
            {
                "kind": "rmsprop",
                "params": [table_0, table_1],
                "lr": 0.2,
                "beta2": 0.999,
                "eps": 1e-10,
                "weight_decay": 0.0,
            }
        ],
        rmsprop_touch_pairs=(
            (2, (table_0, table_1), ((2654435761, 2246822519),) * 2),
        ),
        rmsprop_touch_capacity=capacity,
        p1_compact_backward=True,
    )
    for table in (table_0, table_1):
        optimizer.state[table]["step"] = 0
        optimizer.state[table]["exp_avg_sq"] = torch.zeros_like(table)

    token_ids = torch.arange(capacity, device=device, dtype=torch.long)
    col_ids = torch.arange(cols * 2, device=device, dtype=torch.float32)
    base_contribution = (
        ((col_ids % cols) + 1.0) / 32.0
    ).view(1, 1, cols * 2).expand(1, capacity, cols * 2)
    previous_slot_rows = [None, None]
    previous_active_sets = [set(), set()]
    recycled_slots = 0
    stale_rows_checked = 0
    stale_slot_contaminations = 0
    active_rows_updated = 0
    max_generation = 0

    for step_i in range(steps):
        indices_0 = ((token_ids + step_i * 37) % rows).view(1, capacity)
        indices_1 = ((token_ids * 3 + step_i * 53 + 11) % rows).view(1, capacity)
        contribution = (
            base_contribution
            * (-1.0 if step_i % 2 else 1.0)
            * (1.0 + float(step_i % 7) / 16.0)
        ).to(torch.bfloat16).contiguous()
        before_params = [table.detach().clone() for table in (table_0, table_1)]
        before_states = [
            optimizer.state[table]["exp_avg_sq"].clone()
            for table in (table_0, table_1)
        ]

        optimizer.begin_step()
        optimizer.prepare_ngram_forward(torch.zeros_like(indices_0))
        expected_generation = step_i + 1
        if optimizer._touch_generation != expected_generation:
            raise AssertionError("P1 soak generation did not increase monotonically")
        if int(optimizer._rmsprop_touch_generation.item()) != expected_generation:
            raise AssertionError("P1 soak device generation diverged from host generation")
        max_generation = expected_generation
        _p1_collect_pair_indices_op(
            indices_0,
            indices_1,
            optimizer._rmsprop_touch_stamps[table_0],
            optimizer._rmsprop_active_rows[table_0],
            optimizer._rmsprop_row_to_slot[table_0],
            optimizer._rmsprop_active_counters[table_0],
            optimizer._rmsprop_touch_stamps[table_1],
            optimizer._rmsprop_active_rows[table_1],
            optimizer._rmsprop_row_to_slot[table_1],
            optimizer._rmsprop_active_counters[table_1],
            optimizer._rmsprop_touch_generation,
        )
        torch.cuda.synchronize()

        current_active = []
        for table_i, (indices, table) in enumerate(
            ((indices_0, table_0), (indices_1, table_1))
        ):
            count = int(optimizer._rmsprop_active_counters[table].item())
            if count != capacity:
                raise AssertionError(f"P1 soak expected {capacity} active rows, got {count}")
            active = optimizer._rmsprop_active_rows[table][:count].long()
            expected = torch.unique(indices).sort().values
            if not torch.equal(active.sort().values, expected):
                raise AssertionError("P1 soak collector changed the exact active-row set")
            mapped = optimizer._rmsprop_row_to_slot[table].index_select(0, active)
            if not torch.equal(
                mapped,
                torch.arange(count, device=device, dtype=torch.int32),
            ):
                raise AssertionError("P1 soak row-to-slot ownership is not bijective")
            stamps = optimizer._rmsprop_touch_stamps[table].index_select(0, active)
            if not bool((stamps == expected_generation).all().item()):
                raise AssertionError("P1 soak active row lacks its current-generation stamp")
            slot_rows = active.detach().cpu().tolist()
            if previous_slot_rows[table_i] is not None:
                recycled_slots += sum(
                    old_row != new_row
                    for old_row, new_row in zip(
                        previous_slot_rows[table_i], slot_rows, strict=True
                    )
                )
            active_set = set(slot_rows)
            stale_rows_checked += len(previous_active_sets[table_i] - active_set)
            previous_slot_rows[table_i] = slot_rows
            previous_active_sets[table_i] = active_set
            current_active.append(active)

        _p1_scatter_pair_fp32_op(
            contribution,
            indices_0,
            indices_1,
            optimizer._rmsprop_row_to_slot[table_0],
            optimizer._rmsprop_row_to_slot[table_1],
            optimizer._rmsprop_grad_slots[table_0],
            optimizer._rmsprop_grad_slots[table_1],
        )
        torch.cuda.synchronize()
        for table in (table_0, table_1):
            slots = optimizer._rmsprop_grad_slots[table]
            if not bool(torch.isfinite(slots).all().item()):
                raise AssertionError("P1 soak produced a non-finite compact slot")
            if int(torch.count_nonzero(slots).item()) == 0:
                raise AssertionError("P1 soak did not accumulate a compact gradient")
            if table.grad is not None:
                raise AssertionError("P1 soak materialized a dense n-gram gradient")

        optimizer.step()
        torch.cuda.synchronize()
        for table_i, table in enumerate((table_0, table_1)):
            active = current_active[table_i]
            inactive = torch.ones(rows, device=device, dtype=torch.bool)
            inactive[active] = False
            state = optimizer.state[table]["exp_avg_sq"]
            param_stale_changed = not torch.equal(
                table[inactive].view(torch.int16),
                before_params[table_i][inactive].view(torch.int16),
            )
            state_stale_changed = not torch.equal(
                state[inactive].view(torch.int16),
                before_states[table_i][inactive].view(torch.int16),
            )
            if param_stale_changed or state_stale_changed:
                stale_slot_contaminations += 1
                raise AssertionError("P1 soak detected a stale-slot update")
            changed = torch.any(
                table[active].view(torch.int16)
                != before_params[table_i][active].view(torch.int16),
                dim=1,
            )
            if not bool(changed.all().item()):
                raise AssertionError("P1 soak failed to update an active row")
            active_rows_updated += int(changed.sum().item())
            if int(torch.count_nonzero(optimizer._rmsprop_grad_slots[table]).item()) != 0:
                raise AssertionError("P1 soak optimizer did not clear recycled slots")
            if table.grad is not None:
                raise AssertionError("P1 soak optimizer created a dense gradient")

    if recycled_slots <= 0 or stale_rows_checked <= 0:
        raise AssertionError("P1 soak did not exercise slot recycling and stale rows")
    receipt = _emit_p1_admission_receipt(
        {
            "schema": "apex-tune.h200.formal-a-compact-p1-slot-recycle-soak.v1",
            "authority": "diagnostic_only",
            "candidate_source_sha256": _p1_file_sha256(__file__),
            "parent_source_sha256": (
                "010356bb5477ff5b9a0fa93391eb79b2d23706bc790176ad67d87e7e343a0dce"
            ),
            "hardware": {
                "device_name": torch.cuda.get_device_name(),
                "compute_capability": list(torch.cuda.get_device_capability()),
                "gpu_index": torch.cuda.current_device(),
                "cuda_version": torch.version.cuda,
                "torch_version": torch.__version__,
                "triton_version": triton.__version__,
            },
            "configuration": {
                "steps": steps,
                "tables": 2,
                "rows": rows,
                "cols": cols,
                "capacity": capacity,
            },
            "checks": {
                "generation_monotonic": max_generation == steps,
                "current_generation_stamps": True,
                "exact_active_row_sets": True,
                "slot_ownership_bijective": True,
                "slot_recycling_exercised": recycled_slots > 0,
                "stale_rows_exercised": stale_rows_checked > 0,
                "stale_slot_contamination_zero": stale_slot_contaminations == 0,
                "active_rows_updated": active_rows_updated == steps * capacity * 2,
                "slots_cleared_every_step": True,
                "all_ngram_parameter_grads_none": (
                    table_0.grad is None and table_1.grad is None
                ),
            },
            "metrics": {
                "final_generation": max_generation,
                "recycled_slot_ownerships": recycled_slots,
                "stale_rows_checked": stale_rows_checked,
                "stale_slot_contaminations": stale_slot_contaminations,
                "active_rows_updated": active_rows_updated,
            },
        },
        env_name="P1_SOAK_RECEIPT",
    )
    print(
        "P1_SLOT_RECYCLE_SOAK PASS "
        f"device={torch.cuda.get_device_name()} steps={steps} "
        f"recycled_slots={recycled_slots} stale_rows_checked={stale_rows_checked} "
        "stale_slot_contaminations=0 slots_cleared=pass p_grad_none=2/2 "
        + (
            f"receipt_sha256={receipt['receipt_sha256']}"
            if receipt is not None
            else "receipt_sha256=absent"
        )
    )
    return receipt


def _run_p1_composite_h200_admission():
    """Bind full-step, numerical, overflow, and slot-lifecycle evidence."""
    if not torch.cuda.is_available():
        raise RuntimeError("P1_COMPOSITE_H200_ADMISSION requires CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("P1 composite admission is frozen to Hopper/H200")
    if not os.environ.get("P1_COMPOSITE_ADMISSION_RECEIPT"):
        raise RuntimeError(
            "P1_COMPOSITE_ADMISSION_RECEIPT is required for composite admission"
        )

    component_envs = {
        "P1_ENVELOPE_RECEIPT": "envelope.json",
        "P1_SOAK_RECEIPT": "soak.json",
        "P1_ADMISSION_RECEIPT": "full-step.json",
    }
    previous_env = {name: os.environ.get(name) for name in component_envs}
    with tempfile.TemporaryDirectory(prefix="p1-composite-") as temp_dir_value:
        temp_dir = Path(temp_dir_value)
        try:
            for env_name, filename in component_envs.items():
                os.environ[env_name] = str(temp_dir / filename)
            returned_components = (
                _run_p1_fp64_envelope_selftest(),
                _run_p1_slot_recycle_soak(),
                _run_p1_c428_full_step_admission(),
            )
        finally:
            for env_name, previous in previous_env.items():
                if previous is None:
                    os.environ.pop(env_name, None)
                else:
                    os.environ[env_name] = previous

        components = {}
        for (env_name, filename), returned in zip(
            component_envs.items(), returned_components, strict=True
        ):
            path = temp_dir / filename
            loaded = json.loads(path.read_text(encoding="ascii"))
            if loaded != returned:
                raise RuntimeError(f"P1 {env_name} receipt changed after publication")
            _p1_verify_receipt_self_seal(loaded)
            components[env_name] = loaded

    envelope = components["P1_ENVELOPE_RECEIPT"]
    soak = components["P1_SOAK_RECEIPT"]
    full_step = components["P1_ADMISSION_RECEIPT"]
    expected_component_schemas = {
        "P1_ENVELOPE_RECEIPT": "apex-tune.h200.formal-a-compact-p1-envelope.v1",
        "P1_SOAK_RECEIPT": "apex-tune.h200.formal-a-compact-p1-slot-recycle-soak.v1",
        "P1_ADMISSION_RECEIPT": "apex-tune.h200.formal-a-compact-p1-admission.v1",
    }
    for name, expected_schema in expected_component_schemas.items():
        if components[name].get("schema") != expected_schema:
            raise RuntimeError(f"P1 {name} component schema diverged")
    current_source_sha256 = _p1_file_sha256(__file__)
    semantic_sha256 = os.environ.get("P1_SEMANTIC_SHA256")
    hardware = envelope["hardware"]
    for name, component in components.items():
        if component.get("candidate_source_sha256") != current_source_sha256:
            raise RuntimeError(f"P1 {name} component source identity diverged")
        if component.get("semantic_sha256") != semantic_sha256:
            raise RuntimeError(f"P1 {name} component semantic identity diverged")
        if component.get("hardware") != hardware:
            raise RuntimeError(f"P1 {name} component runtime identity diverged")

    full_step_required_checks = (
        "exact_c428_shape",
        "graph_breaks_zero",
        "cudagraph_skips_zero",
        "forward_indices_collector",
        "backward_compact_scatter",
        "token_hash_recompute_absent",
        "all_ngram_parameter_grads_none",
        "non_ngram_grads_finite",
        "exact_touch_sets",
        "slot_ownership_bijective",
        "touched_update_bitwise",
        "untouched_update_exact_skip",
        "slots_cleared",
    )
    overflow_ready = bool(
        envelope["overflow_fail_closed"]["prepare_accepts_token_count_over_capacity"]
        and not envelope["overflow_fail_closed"]["direct_collector_rejected"]
        and envelope["overflow_fail_closed"]["optimizer_step_rejected"]
        and envelope["overflow_fail_closed"]["counters_unchanged"]
    )
    envelope_ready = bool(
        envelope["fp64_gamma_envelope"]["max_error_to_bound_ratio"] <= 1.0
        and overflow_ready
    )
    soak_ready = all(soak["checks"].values())
    full_step_ready = all(
        bool(full_step["checks"].get(check)) for check in full_step_required_checks
    )
    implementation_readiness = envelope_ready and soak_ready and full_step_ready
    composite_receipt = _emit_p1_admission_receipt(
        {
            "schema": "apex-tune.h200.formal-a-compact-p1-composite-admission.v1",
            "authority": "diagnostic_only",
            "scope": "implementation_readiness_only_no_b200_performance_credit",
            "candidate_source_sha256": current_source_sha256,
            "parent_source_sha256": (
                "010356bb5477ff5b9a0fa93391eb79b2d23706bc790176ad67d87e7e343a0dce"
            ),
            "hardware": hardware,
            "checks": {
                "component_receipt_self_seals": True,
                "same_candidate_source": True,
                "same_semantic_identity": True,
                "same_hardware_runtime": True,
                "fp64_gamma_envelope": envelope_ready,
                "overflow_fail_closed": overflow_ready,
                "slot_recycle_soak": soak_ready,
                "full_step_required_checks": full_step_ready,
                "exact_c428_shape": bool(
                    full_step["checks"].get("exact_c428_shape")
                ),
            },
            "implementation_readiness": implementation_readiness,
            "components": {
                "fp64_envelope_and_overflow": envelope,
                "slot_recycle_soak": soak,
                "c428_full_step": full_step,
            },
        },
        env_name="P1_COMPOSITE_ADMISSION_RECEIPT",
    )
    if composite_receipt is None:
        raise RuntimeError("P1 composite admission failed to publish its receipt")
    print(
        "P1_COMPOSITE_H200_ADMISSION "
        + ("PASS " if implementation_readiness else "DIAGNOSTIC_COMPLETE ")
        + f"device={torch.cuda.get_device_name()} "
        f"exact_c428_shape={str(full_step['checks']['exact_c428_shape']).lower()} "
        f"implementation_readiness={str(implementation_readiness).lower()} "
        f"receipt_sha256={composite_receipt['receipt_sha256']}"
    )
    return composite_receipt


def _run_c428_cudagraph_forward_probe():
    """Isolate parent/child forward CUDAGraph capture on H200."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("c428 CUDAGraph probe requires local H200")
    variant = os.environ.get("P1_CUDAGRAPH_PROBE_VARIANT", "parent")
    if variant not in {"parent", "child"}:
        raise ValueError("P1_CUDAGRAPH_PROBE_VARIANT must be parent or child")
    batch = int(os.environ.get("P1_C428_ADMISSION_BATCH", "2"))
    seq_len = int(os.environ.get("P1_C428_ADMISSION_SEQ_LEN", "128"))
    device = torch.device("cuda")
    torch.manual_seed(20260817)
    config = GPTConfig(
        sequence_len=2048,
        vocab_size=8192,
        n_layer=8,
        n_head=6,
        n_kv_head=6,
        n_embd=768,
        window_pattern="TTTL",
    )
    use_compact = variant == "child"
    with torch.device("meta"):
        model = GPT(config, p1_compact_backward=use_compact)
    model.to_empty(device=device)
    model.init_weights()
    model.to(dtype=torch.bfloat16)
    model.train()
    optimizer = None
    if use_compact:
        optimizer = model.setup_optimizer(
            unembedding_lr=0.004,
            embedding_lr=0.6,
            matrix_lr=0.04,
            weight_decay=0.1,
            adam_betas=(0.8, 0.95),
            scalar_lr=0.8,
            ngram_ve_betas=(0.5, 0.999),
            ngram_touch_capacity=_P1_COMPACT_SLOT_CAPACITY_POLICY,
        )
    tokens = torch.randint(
        0, config.vocab_size, (batch, seq_len), device=device, dtype=torch.long
    )
    targets = torch.randint(
        0, config.vocab_size, (batch, seq_len), device=device, dtype=torch.long
    )
    from torch._dynamo.utils import counters

    counters.clear()
    compiled_model = torch.compile(
        model, dynamic=False, mode="reduce-overhead", fullgraph=True
    )
    if optimizer is not None:
        optimizer.begin_step()
        optimizer.prepare_ngram_forward(tokens)
    losses = []
    for _ in range(2):
        torch.compiler.cudagraph_mark_step_begin()
        loss = compiled_model(tokens, targets)
        losses.append(loss.detach().clone())
        torch.cuda.synchronize()
    if not torch.equal(losses[0], losses[1]):
        raise AssertionError("c428 CUDAGraph replay changed forward loss")
    cudagraph_skips = int(counters["inductor"].get("cudagraph_skips", 0))
    graph_breaks = sum(int(value) for value in counters["graph_break"].values())
    if cudagraph_skips or graph_breaks:
        raise AssertionError(
            f"c428 {variant} probe skips={cudagraph_skips} breaks={graph_breaks}"
        )
    print(
        "C428_CUDAGRAPH_FORWARD_PROBE PASS "
        f"device={torch.cuda.get_device_name()} variant={variant} "
        f"batch={batch} seq={seq_len} fullgraph=pass cudagraph_skips=0 "
        f"loss={float(losses[-1].item()):.8g}"
    )


def _cuda_ms(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


@torch.no_grad()
def _run_touched_rmsprop_benchmark():
    """Standalone engineering harness; H200 numbers are diagnostic only."""
    if not torch.cuda.is_available():
        raise RuntimeError("TOUCHED_RMSPROP_BENCH requires CUDA")
    device = torch.device("cuda")
    rows = int(os.environ.get("TOUCHED_RMSPROP_BENCH_ROWS", 524288))
    cols = int(os.environ.get("TOUCHED_RMSPROP_BENCH_COLS", 384))
    iterations = int(os.environ.get("TOUCHED_RMSPROP_BENCH_ITERS", 10))
    warmup = int(os.environ.get("TOUCHED_RMSPROP_BENCH_WARMUP", 3))
    if rows <= 0 or (rows & (rows - 1)) != 0:
        raise ValueError("benchmark rows must be a positive power of two")
    torch.manual_seed(20260816)

    row_ids = torch.arange(rows, device=device, dtype=torch.int64)

    def make_mask(rate, salt):
        mixed = (row_ids * 1103515245 + salt) & 0x7FFFFFFF
        return (mixed.remainder(1000000) < int(rate * 1000000)).to(torch.int32)

    mask14 = make_mask(0.14, 12345)
    mask17 = make_mask(0.17, 67891)
    mask21 = make_mask(0.21, 24681)
    capacity = min(rows, 72 * 2048)

    def compact(mask):
        rows_used = torch.nonzero(mask, as_tuple=False).flatten().to(torch.int32)
        if rows_used.numel() > capacity:
            raise RuntimeError("synthetic active rows exceed compact-list capacity")
        active_rows = torch.empty(capacity, dtype=torch.int32, device=device)
        active_rows[: rows_used.numel()].copy_(rows_used)
        counter = torch.tensor([rows_used.numel()], dtype=torch.int32, device=device)
        return active_rows, counter

    active14, counter14 = compact(mask14)
    active17, counter17 = compact(mask17)
    active21, counter21 = compact(mask21)
    param = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
    state = torch.rand(rows, cols, device=device, dtype=torch.bfloat16)
    grad14 = torch.randn_like(param).mul_(0.01)
    grad17 = torch.randn_like(param).mul_(0.01)
    grad21 = torch.randn_like(param).mul_(0.01)
    grad14.mul_(mask14.bool().unsqueeze(1))
    grad17.mul_(mask17.bool().unsqueeze(1))
    grad21.mul_(mask21.bool().unsqueeze(1))
    step_t = torch.tensor(2048.0, dtype=torch.float32)
    lr_t = torch.tensor(0.2, dtype=torch.float32)
    beta2_t = torch.tensor(0.99945, dtype=torch.float32)
    eps_t = torch.tensor(1e-10, dtype=torch.float32)
    wd_t = torch.tensor(0.0, dtype=torch.float32)

    dense_args = (param, grad17, state, step_t, lr_t, beta2_t, eps_t, wd_t)
    touched_args = (
        param,
        grad17,
        state,
        mask17,
        step_t,
        lr_t,
        beta2_t,
        eps_t,
        wd_t,
    )
    dense_ms = _cuda_ms(lambda: rmsprop_step_fused(*dense_args), warmup, iterations)
    scan_ms = _cuda_ms(
        lambda: rmsprop_step_touched_triton(*touched_args), warmup, iterations
    )
    compact_ms = _cuda_ms(
        lambda: rmsprop_step_compact_triton(
            param,
            grad17,
            state,
            active17,
            counter17,
            step_t,
            lr_t,
            beta2_t,
            eps_t,
            wd_t,
        ),
        warmup,
        iterations,
    )

    def dense_full_step():
        for _ in range(8):
            rmsprop_step_fused(param, grad14, state, step_t, lr_t, beta2_t, eps_t, wd_t)
        for _ in range(6):
            rmsprop_step_fused(param, grad21, state, step_t, lr_t, beta2_t, eps_t, wd_t)

    def scan_full_step():
        for _ in range(8):
            rmsprop_step_touched_triton(
                param, grad14, state, mask14, step_t, lr_t, beta2_t, eps_t, wd_t
            )
        for _ in range(6):
            rmsprop_step_touched_triton(
                param, grad21, state, mask21, step_t, lr_t, beta2_t, eps_t, wd_t
            )

    def compact_full_step():
        for _ in range(8):
            rmsprop_step_compact_triton(
                param,
                grad14,
                state,
                active14,
                counter14,
                step_t,
                lr_t,
                beta2_t,
                eps_t,
                wd_t,
            )
        for _ in range(6):
            rmsprop_step_compact_triton(
                param,
                grad21,
                state,
                active21,
                counter21,
                step_t,
                lr_t,
                beta2_t,
                eps_t,
                wd_t,
            )

    full_dense_ms = _cuda_ms(dense_full_step, 1, max(2, iterations // 2))
    full_scan_ms = _cuda_ms(scan_full_step, 1, max(2, iterations // 2))
    full_compact_ms = _cuda_ms(compact_full_step, 1, max(2, iterations // 2))

    # Measure the real fixed-mask maintenance shape independently.
    masks = [torch.zeros(rows, dtype=torch.int32, device=device) for _ in range(14)]
    tokens = torch.randint(0, 8192, (72, 2048), device=device, dtype=torch.long)
    bigram_pairs = (
        ((2654435761, 2246822519), (1013904223, 6291469)),
        ((374761393, 668265263), (3266489917, 104729)),
        ((1640531527, 97531), (48271, 40503)),
        ((16777619, 2166136261), (3432918353, 461845907)),
    )
    trigram_pairs = (
        ((16777619, 2166136261, 3432918353), (461845907, 2654435769, 1540483477)),
        ((3405403843, 2654435761, 2246822519), (1013904223, 6291469, 374761393)),
        ((668265263, 3266489917, 104729), (1640531527, 97531, 48271)),
    )
    mark_grid = (triton.cdiv(tokens.numel(), 256),)

    def clear_masks():
        torch._foreach_zero_(masks)

    def mark_masks():
        mask_i = 0
        for pair in bigram_pairs:
            _mark_bigram_pair_kernel[mark_grid](
                tokens,
                masks[mask_i],
                masks[mask_i + 1],
                tokens.numel(),
                *pair[0],
                *pair[1],
                SEQ_LEN=tokens.shape[1],
                TABLE_MASK=rows - 1,
                BLOCK_SIZE=256,
                num_warps=4,
            )
            mask_i += 2
        for pair in trigram_pairs:
            _mark_trigram_pair_kernel[mark_grid](
                tokens,
                masks[mask_i],
                masks[mask_i + 1],
                tokens.numel(),
                *pair[0],
                *pair[1],
                SEQ_LEN=tokens.shape[1],
                TABLE_MASK=rows - 1,
                BLOCK_SIZE=256,
                num_warps=4,
            )
            mask_i += 2

    clear_ms = _cuda_ms(clear_masks, warmup, iterations)
    mark_ms = _cuda_ms(mark_masks, warmup, iterations)
    stamps = [torch.zeros(rows, dtype=torch.int32, device=device) for _ in range(14)]
    active_buffers = [
        torch.empty(capacity, dtype=torch.int32, device=device) for _ in range(14)
    ]
    row_to_slots = [
        torch.full((rows,), -1, dtype=torch.int32, device=device) for _ in range(14)
    ]
    counters = [torch.zeros(1, dtype=torch.int32, device=device) for _ in range(14)]
    generation = 0

    def collect_rows():
        nonlocal generation
        generation += 1
        torch._foreach_zero_(counters)
        buffer_i = 0
        for pair in bigram_pairs:
            _collect_bigram_pair_kernel[mark_grid](
                tokens,
                stamps[buffer_i],
                active_buffers[buffer_i],
                row_to_slots[buffer_i],
                counters[buffer_i],
                stamps[buffer_i + 1],
                active_buffers[buffer_i + 1],
                row_to_slots[buffer_i + 1],
                counters[buffer_i + 1],
                tokens.numel(),
                generation,
                *pair[0],
                *pair[1],
                SEQ_LEN=tokens.shape[1],
                TABLE_MASK=rows - 1,
                CAPACITY=capacity,
                BLOCK_SIZE=256,
                num_warps=4,
            )
            buffer_i += 2
        for pair in trigram_pairs:
            _collect_trigram_pair_kernel[mark_grid](
                tokens,
                stamps[buffer_i],
                active_buffers[buffer_i],
                row_to_slots[buffer_i],
                counters[buffer_i],
                stamps[buffer_i + 1],
                active_buffers[buffer_i + 1],
                row_to_slots[buffer_i + 1],
                counters[buffer_i + 1],
                tokens.numel(),
                generation,
                *pair[0],
                *pair[1],
                SEQ_LEN=tokens.shape[1],
                TABLE_MASK=rows - 1,
                CAPACITY=capacity,
                BLOCK_SIZE=256,
                num_warps=4,
            )
            buffer_i += 2

    collect_ms = _cuda_ms(collect_rows, warmup, iterations)
    max_compact_count = max(int(counter.item()) for counter in counters)
    if max_compact_count > capacity:
        raise AssertionError("compact collector overflowed its fixed capacity")
    candidate_full_ms = full_scan_ms + clear_ms + mark_ms
    compact_candidate_full_ms = full_compact_ms + collect_ms
    print(
        "TOUCHED_RMSPROP_BENCH "
        f"device={torch.cuda.get_device_name()} rows={rows} cols={cols} "
        f"single_dense_ms={dense_ms:.4f} single_scan_ms={scan_ms:.4f} "
        f"single_speedup={dense_ms / scan_ms:.3f}x "
        f"single_compact_ms={compact_ms:.4f} "
        f"single_compact_speedup={dense_ms / compact_ms:.3f}x "
        f"full14_dense_ms={full_dense_ms:.4f} full14_scan_ms={full_scan_ms:.4f} "
        f"full14_compact_ms={full_compact_ms:.4f} "
        f"clear14_ms={clear_ms:.4f} mark7_ms={mark_ms:.4f} "
        f"candidate_full_ms={candidate_full_ms:.4f} "
        f"candidate_speedup={full_dense_ms / candidate_full_ms:.3f}x "
        f"compact_collect7_ms={collect_ms:.4f} "
        f"compact_candidate_full_ms={compact_candidate_full_ms:.4f} "
        f"compact_candidate_speedup={full_dense_ms / compact_candidate_full_ms:.3f}x "
        f"compact_max_count={max_compact_count} "
        f"scan_ctas_per_table={triton.cdiv(rows, 8)} "
        f"scan_ctas_full={14 * triton.cdiv(rows, 8)} "
        f"compact_ctas_full={14 * triton.cdiv(capacity, 8)}"
    )



if os.environ.get("P1_COMPACT_BACKWARD_SELF_TEST") == "1":
    _run_p1_compact_backward_selftest()
    raise SystemExit(0)

if os.environ.get("P1_FORWARD_REUSE_SELF_TEST") == "1":
    _run_p1_forward_reuse_selftest()
    raise SystemExit(0)

if os.environ.get("P1_FP64_ENVELOPE_SELF_TEST") == "1":
    _run_p1_fp64_envelope_selftest()
    raise SystemExit(0)

if os.environ.get("P1_COMPACT_BACKWARD_TINY_MODEL_SELF_TEST") == "1":
    _run_p1_tiny_model_selftest()
    raise SystemExit(0)

if os.environ.get("P1_COMPACT_BACKWARD_REAL_SHAPE_SMOKE") == "1":
    _run_p1_real_shape_smoke()
    raise SystemExit(0)

if os.environ.get("P1_NO_SCORE_ENGINEERING") == "1":
    _run_p1_no_score_engineering()
    raise SystemExit(0)

if os.environ.get("P1_C428_FULL_STEP_ADMISSION") == "1":
    _run_p1_c428_full_step_admission()
    raise SystemExit(0)

if os.environ.get("P1_SLOT_RECYCLE_SOAK") == "1":
    _run_p1_slot_recycle_soak()
    raise SystemExit(0)

if os.environ.get("P1_COMPOSITE_H200_ADMISSION") == "1":
    _run_p1_composite_h200_admission()
    raise SystemExit(0)

if os.environ.get("P1_C428_CUDAGRAPH_FORWARD_PROBE") == "1":
    _run_c428_cudagraph_forward_probe()
    raise SystemExit(0)

if os.environ.get("TOUCHED_RMSPROP_SELF_TEST") == "1":
    _run_touched_rmsprop_selftest()
    raise SystemExit(0)

if os.environ.get("TOUCHED_RMSPROP_BENCH") == "1":
    _run_touched_rmsprop_benchmark()
    raise SystemExit(0)


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 96  # model_dim = depth * ASPECT_RATIO (d8*96=768 -> dim=768, 6 heads)
HEAD_DIM = 128  # target head dimension for attention
WINDOW_PATTERN = "TTTL"  # 3 tiny + 1 long -- sandwich norm + warmdown=0.8 variant

# Optimization
TOTAL_BATCH_SIZE = 72 * 2048  # 147456 tokens per step (grad_accum=1 with devbatch=72 on B200)
EMBEDDING_LR = 0.6  # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.035  # AutoTrust recipe-parity matrix LR
SCALAR_LR = 0.8  # x0 Muon warmdown SCALAR_LR=0.8
WEIGHT_DECAY = 0.1  # baseline WD
ADAM_BETAS = (0.8, 0.95)  # Adam beta1, beta2
DEMON_FINAL_BETA1 = 0.55  # baseline Demon
NGRAM_VE_BETAS = (0.5, 0.999)  # RMSProp only uses beta2=0.999; higher beta2 preserves gradient history for sparse tables
NGRAM_VE_LR_SCALE = 1.0  # RMSProp with full LR (no reduction)
WARMUP_RATIO = 0.0  # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.90  # AutoTrust recipe-parity warmdown
ADAM_WARMDOWN_RATIO = 0.65  # slightly longer Adam warmdown to match extended Muon warmdown
NGRAM_WARMDOWN_RATIO = 0.0  # no warmdown for bigram/trigram VE (sparse tables benefit from full-rate training)
FINAL_LR_FRAC = 0.05  # restored FLR=0.05

# Model size
DEPTH = 8  # number of transformer layers
DEVICE_BATCH_SIZE = 72  # per-device batch size -- B200

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
import argparse

_cli_parser = argparse.ArgumentParser(description="NanoChat Discovered Pretraining on B200")
_cli_parser.add_argument("--seed", type=int, default=None, help="Random seed (default: 42 or $NANOCHAT_SEED)")
_cli_parser.add_argument("--budget", type=int, default=None, help="Training budget in seconds (default: 300 or $NANOCHAT_TIME_BUDGET)")
_cli_parser.add_argument("--output", type=str, default=None, help="Save evaluation metrics to JSON file")
_cli_args, _ = _cli_parser.parse_known_args()

_SEED = _cli_args.seed if _cli_args.seed is not None else int(os.environ.get("NANOCHAT_SEED", 42))
if _cli_args.budget is not None:
    TIME_BUDGET = _cli_args.budget
elif "NANOCHAT_TIME_BUDGET" in os.environ:
    TIME_BUDGET = int(os.environ["NANOCHAT_TIME_BUDGET"])

torch.manual_seed(_SEED)
torch.cuda.manual_seed(_SEED)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
# No autocast: model is natively BF16 -- eliminates FP32->BF16 cast overhead in compile graph
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False)
B200_BF16_PEAK_FLOPS = 2.25e15

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")


def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )


config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()
# Cast entire model to BF16: enables removing autocast, simplifies compile graph
model.to(dtype=torch.bfloat16)

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts["total"]
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
assert grad_accum_steps == 1, "P1 compact backward is frozen to grad_accum=1"

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
    ngram_ve_betas=NGRAM_VE_BETAS,
    ngram_ve_lr_scale=NGRAM_VE_LR_SCALE,
    ngram_touch_capacity=_P1_COMPACT_SLOT_CAPACITY_POLICY,
)

muon_groups = []
ngram_groups = []
x0_warmdown_groups = []
adam_groups = []
adam_demon_groups = []
muon_group_lrs = []
x0_group_lrs = []
adam_group_lrs = []
for group in optimizer.param_groups:
    if group["kind"] == "muon":
        muon_groups.append(group)
        muon_group_lrs.append((group, group["initial_lr"]))
    elif group.get("is_ngram_ve", False):
        ngram_groups.append(group)
    elif group.get("is_x0_muon_warmdown", False):
        x0_warmdown_groups.append(group)
        x0_group_lrs.append((group, group["initial_lr"]))
    else:
        adam_groups.append(group)
        adam_group_lrs.append((group, group["initial_lr"]))
        if group.get("demon_beta1", False):
            adam_demon_groups.append((group, group["betas"][1]))

model = torch.compile(model, dynamic=False, mode="max-autotune", fullgraph=True)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = training_time / TIME_BUDGET)


def get_lr_multiplier(progress, warmdown_ratio=WARMDOWN_RATIO):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - warmdown_ratio:
        return 1.0
    else:
        cooldown = (1.0 - progress) / warmdown_ratio
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


MUON_PEAK_MOMENTUM = 0.94  # AutoTrust recipe-parity peak
MUON_WARMDOWN_MOMENTUM = 0.80  # AutoTrust recipe-parity warmdown
# Reverse Demon for NorMuon beta2: INCREASE beta2 during warmdown for more stable variance normalization
MUON_BETA2_PEAK = 0.95  # standard beta2 during full-LR phase
MUON_BETA2_WARMDOWN = 0.97  # target beta2 at end of warmdown
MUON_LR_BOOST = 1.0  # no LR boost
# VE RMSProp reverse-Demon: increase VE beta2 during last 30% of Muon warmdown
# Analogous to Muon's 0.95->0.97, but for ngram VE tables (0.999->0.9995)
NGRAM_VE_BETA2_WARMDOWN = 0.9999  # STRONGER delayed VE beta2 ramp (0.999->0.9999 last 30% warmdown)
def get_muon_momentum(step, progress=None):
    # Warmup: 0.85 -> 0.95 over 300 steps
    frac = min(step / 300, 1)
    base = (1 - frac) * 0.85 + frac * MUON_PEAK_MOMENTUM
    # Quadratic Demon: back-loaded shape keeps peak momentum longer
    if progress is not None:
        warmdown_start = 1.0 - WARMDOWN_RATIO
        if progress > warmdown_start:
            wd_frac = (progress - warmdown_start) / WARMDOWN_RATIO
            base = MUON_PEAK_MOMENTUM + (wd_frac ** 2) * (MUON_WARMDOWN_MOMENTUM - MUON_PEAK_MOMENTUM)
    return base


def get_muon_beta2(progress):
    """Reverse beta2: increase beta2 during warmdown for more stable variance norm."""
    warmdown_start = 1.0 - WARMDOWN_RATIO
    if progress < warmdown_start:
        return MUON_BETA2_PEAK
    else:
        wd_frac = (progress - warmdown_start) / WARMDOWN_RATIO
        return MUON_BETA2_PEAK + wd_frac * (MUON_BETA2_WARMDOWN - MUON_BETA2_PEAK)


def get_muon_lr_boost(progress):
    """Boost Muon LR during warmdown to compensate for higher beta2 reducing step size."""
    warmdown_start = 1.0 - WARMDOWN_RATIO
    if progress < warmdown_start:
        return 1.0
    else:
        wd_frac = (progress - warmdown_start) / WARMDOWN_RATIO
        return 1.0 + wd_frac * (MUON_LR_BOOST - 1.0)


def get_adam_beta1(progress, warmdown_ratio=ADAM_WARMDOWN_RATIO):
    """Forward Demon: decrease beta1 during warmdown for more responsive gradient following."""
    initial_beta1 = ADAM_BETAS[0]
    final_beta1 = DEMON_FINAL_BETA1
    warmdown_start = 1.0 - warmdown_ratio
    if progress < warmdown_start:
        return initial_beta1
    else:
        warmdown_progress = (progress - warmdown_start) / warmdown_ratio
        return initial_beta1 + (final_beta1 - initial_beta1) * warmdown_progress


# WD pulse: RECTANGULAR shape -- with 95% warmdown (starts at 5%), pulses shifted earlier
# Main pulse at 3% center, 2% total duration (1% half-width): fires at 2-4%, before warmdown onset at 5%
# Early pulse at 1.5% center, 1% total duration: fires at 1-2% progress
# Both pulses fire in the full-LR phase (0-5%), maintaining the pre-warmdown regularization timing
WD_PULSE_CENTER = 0.03   # shift main pulse to 3% (fires before warmdown at 5%)
WD_PULSE_HALF_WIDTH = 0.01  # 1% half-width: 2% total duration (tighter for earlier firing)
WD_PULSE_MAGNITUDE = 5.0  # try 5x main pulse (vs 8x) -- 5x optimal WITH Muon Demon, 8x WITHOUT; current setup HAS Demon
WD_EARLY_PULSE_CENTER = 0.015  # shift early pulse to 1.5%
WD_EARLY_PULSE_HALF_WIDTH = 0.005  # 0.5% half-width: 1% total duration
WD_EARLY_PULSE_MAGNITUDE = 3.0  # 3x early pulse (gentler, to initialize regularization)
# Mid-warmdown triangular pulse: fires at 80% total progress (= ~79% through warmdown)
# This is WITHIN the VE beta2 ramp zone (which starts at 71.5% total = 70% through warmdown)
# Hypothesis: VE beta2 stabilization provides a safety net for a mid-warmdown WD perturbation
WD_MID_PULSE_CENTER = 0.80   # 80% total progress = ~79% through warmdown
WD_MID_PULSE_HALF_WIDTH = 0.025  # 2.5% half-width: 5% total triangular duration
WD_MID_PULSE_MAGNITUDE = 4.0  # 4x magnitude (triangular shape -- less harsh than rectangular)

def get_weight_decay(progress):
    base_wd = WEIGHT_DECAY * (1 - progress)
    # Early small pulse: 3x spike at 2% progress (step ~65), 2% total duration
    early_dist = abs(progress - WD_EARLY_PULSE_CENTER)
    if early_dist < WD_EARLY_PULSE_HALF_WIDTH:
        return base_wd * WD_EARLY_PULSE_MAGNITUDE  # RECTANGULAR early pulse
    # Main pulse: 8x rectangular spike at 5% progress (step ~163), 3% total duration
    dist = abs(progress - WD_PULSE_CENTER)
    if dist < WD_PULSE_HALF_WIDTH:
        return base_wd * WD_PULSE_MAGNITUDE  # RECTANGULAR main pulse
    # Mid-warmdown triangular pulse: fires within VE beta2 stabilization zone
    mid_dist = abs(progress - WD_MID_PULSE_CENTER)
    if mid_dist < WD_MID_PULSE_HALF_WIDTH:
        # Triangular: linear ramp up then down (proven optimal shape)
        local = (progress - (WD_MID_PULSE_CENTER - WD_MID_PULSE_HALF_WIDTH)) / (2 * WD_MID_PULSE_HALF_WIDTH)
        bump = 2 * local if local < 0.5 else 2 * (1 - local)
        return base_wd * (1.0 + bump * (WD_MID_PULSE_MAGNITUDE - 1.0))
    return base_wd


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0
inv_time_budget = 1.0 / TIME_BUDGET
inv_muon_warmdown = 1.0 / WARMDOWN_RATIO
inv_adam_warmdown = 1.0 / ADAM_WARMDOWN_RATIO
muon_warmdown_start = 1.0 - WARMDOWN_RATIO
adam_warmdown_start = 1.0 - ADAM_WARMDOWN_RATIO

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    optimizer.begin_step()
    for _micro_step in range(grad_accum_steps):
        optimizer.prepare_ngram_forward(x)
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules (decoupled warmdown: Muon=0.9, Adam=0.7, Ngram VE=0.0)
    progress = min(total_training_time * inv_time_budget, 1.0)
    if progress < muon_warmdown_start:
        lrm_muon = 1.0
        muon_wd_frac = 0.0
    else:
        muon_wd_frac = (progress - muon_warmdown_start) * inv_muon_warmdown
        lrm_muon = ((1.0 - progress) * inv_muon_warmdown) * (1.0 - FINAL_LR_FRAC) + FINAL_LR_FRAC

    if progress < adam_warmdown_start:
        lrm_adam = 1.0
        adam_beta1 = ADAM_BETAS[0]
    else:
        adam_wd_frac = (progress - adam_warmdown_start) * inv_adam_warmdown
        lrm_adam = ((1.0 - progress) * inv_adam_warmdown) * (1.0 - FINAL_LR_FRAC) + FINAL_LR_FRAC
        adam_beta1 = ADAM_BETAS[0] + (DEMON_FINAL_BETA1 - ADAM_BETAS[0]) * adam_wd_frac

    frac = min(step / 300, 1)
    muon_momentum = (1 - frac) * 0.85 + frac * MUON_PEAK_MOMENTUM
    if progress > muon_warmdown_start:
        muon_momentum = MUON_PEAK_MOMENTUM + (muon_wd_frac ** 2) * (MUON_WARMDOWN_MOMENTUM - MUON_PEAK_MOMENTUM)
    muon_beta2 = MUON_BETA2_PEAK + muon_wd_frac * (MUON_BETA2_WARMDOWN - MUON_BETA2_PEAK)
    muon_lr_boost = 1.0 + muon_wd_frac * (MUON_LR_BOOST - 1.0)
    # VE RMSProp reverse-Demon: DELAYED ramp (only last 30% of Muon warmdown)
    late_frac = max(0.0, (muon_wd_frac - 0.7) / 0.3)
    ve_beta2 = NGRAM_VE_BETAS[1] + late_frac * (NGRAM_VE_BETA2_WARMDOWN - NGRAM_VE_BETAS[1])

    base_wd = WEIGHT_DECAY * (1 - progress)
    early_dist = abs(progress - WD_EARLY_PULSE_CENTER)
    if early_dist < WD_EARLY_PULSE_HALF_WIDTH:
        muon_weight_decay = base_wd * WD_EARLY_PULSE_MAGNITUDE
    else:
        dist = abs(progress - WD_PULSE_CENTER)
        if dist < WD_PULSE_HALF_WIDTH:
            muon_weight_decay = base_wd * WD_PULSE_MAGNITUDE
        else:
            mid_dist = abs(progress - WD_MID_PULSE_CENTER)
            if mid_dist < WD_MID_PULSE_HALF_WIDTH:
                local = (progress - (WD_MID_PULSE_CENTER - WD_MID_PULSE_HALF_WIDTH)) / (2 * WD_MID_PULSE_HALF_WIDTH)
                bump = 2 * local if local < 0.5 else 2 * (1 - local)
                muon_weight_decay = base_wd * (1.0 + bump * (WD_MID_PULSE_MAGNITUDE - 1.0))
            else:
                muon_weight_decay = base_wd

    muon_lr = lrm_muon * muon_lr_boost
    if progress < muon_warmdown_start:
        for group in muon_groups:
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
            group["beta2"] = muon_beta2
    else:
        for group, initial_lr in muon_group_lrs:
            group["lr"] = initial_lr * muon_lr
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
            group["beta2"] = muon_beta2
        for group, initial_lr in x0_group_lrs:
            group["lr"] = initial_lr * lrm_muon
    if progress >= adam_warmdown_start:
        for group, initial_lr in adam_group_lrs:
            group["lr"] = initial_lr * lrm_adam
        for group, beta2 in adam_demon_groups:
            group["betas"] = (adam_beta1, beta2)
    # Update ngram VE RMSProp beta2 during warmdown (delayed reverse-Demon for sparse tables)
    if progress >= muon_warmdown_start and late_frac > 0.0:
        for group in ngram_groups:
            group["beta2"] = ve_beta2
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / B200_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm_muon: {lrm_muon:.2f} lrm_adam: {lrm_adam:.2f} | dt: {dt * 1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ",
        end="",
        flush=True,
    )

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = (
    100
    * num_flops_per_token
    * TOTAL_BATCH_SIZE
    * (step - 10)
    / total_training_time
    / B200_BF16_PEAK_FLOPS
    if total_training_time > 0
    else 0
)
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")

if _cli_args.output:
    out_dict = {
        "seed": _SEED,
        "val_bpb": float(val_bpb),
        "training_seconds": float(total_training_time),
        "total_seconds": float(t_end - t_start),
        "peak_vram_gib": float(peak_vram_mb / 1024),
        "peak_vram_mb": float(peak_vram_mb),
        "mfu_percent": float(steady_state_mfu),
        "total_tokens_M": float(total_tokens / 1e6),
        "num_steps": int(step),
        "num_params_M": float(num_params / 1e6),
        "depth": int(DEPTH),
    }
    out_path = Path(_cli_args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"Saved run metrics to {out_path}")
