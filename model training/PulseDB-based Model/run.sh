#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${BP_PYTHON:-python3}
if [ "$(uname -s)" = Darwin ]; then
  BP_TORCH_LIB=$($PYTHON_BIN -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("torch").origin).parent / "lib")')
  export DYLD_LIBRARY_PATH="$BP_TORCH_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi
exec "$PYTHON_BIN" "$PROJECT_DIR/$@"
