#!/usr/bin/env python3
"""Run a script with the flex_attention_kernel argument dump armed.

    python flexdump_run.py <target.py> [args...]

Imports nothing heavy before the target: the kernel is patched lazily, the
moment the module that owns it appears in sys.modules.  The dump path comes
from $FLEX_DUMP.
"""

import os
import runpy
import sys

if len(sys.argv) < 2:
    sys.exit("usage: python flexdump_run.py <target.py> [args...]")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dump_flex_args                                            # noqa: E402

dump_flex_args.install_deferred(os.environ.get("FLEX_DUMP"),
                                limit=int(os.environ.get("FLEX_DUMP_LIMIT", "1")))

target = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(target)))
runpy.run_path(target, run_name="__main__")
