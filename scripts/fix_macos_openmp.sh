#!/usr/bin/env bash
# macOS only. torch (Intel libiomp5) and LightGBM/XGBoost (LLVM libomp) each bring their own OpenMP runtime;
# loading both in one process segfaults or hangs at exit. Point torch's copy at Homebrew's libomp so a single
# runtime is loaded. Linux/Docker builds use libgomp everywhere and don't need this. Re-run after `uv sync`.
set -euo pipefail
[[ "$(uname)" == "Darwin" ]] || exit 0
OMP="$(brew --prefix libomp)/lib/libomp.dylib"
[[ -f "$OMP" ]] || { echo "brew install libomp first"; exit 1; }
F=$(.venv/bin/python -c "import functorch, pathlib; print(pathlib.Path(functorch.__file__).parent / '.dylibs' / 'libiomp5.dylib')")
[[ -L "$F" ]] && { echo "already linked: $F -> $(readlink "$F")"; exit 0; }
mv "$F" "$F.orig" 2>/dev/null || true
ln -sf "$OMP" "$F"
echo "linked $F -> $OMP"
