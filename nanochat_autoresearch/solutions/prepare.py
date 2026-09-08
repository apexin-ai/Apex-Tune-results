"""Shim forwarding to parent prepare.py."""
import sys
from pathlib import Path

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from prepare import (  # noqa: F401, E402
    BOS_TOKEN,
    CACHE_DIR,
    DATA_DIR,
    EVAL_TOKENS,
    MAX_SEQ_LEN,
    SPECIAL_TOKENS,
    TIME_BUDGET,
    TOKENIZER_DIR,
    Tokenizer,
    evaluate_bpb,
    get_token_bytes,
    make_dataloader,
)

if __name__ == "__main__":
    import runpy
    runpy.run_path(str(Path(__file__).resolve().parent.parent / "prepare.py"), run_name="__main__")
