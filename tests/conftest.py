import atexit
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"

# Ensure `src/eigenforge` resolves regardless of checkout folder name or
# whether `pip install -e .` has been run.
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_existing_py_path = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = (
    f"{_SRC_DIR}{os.pathsep}{_existing_py_path}" if _existing_py_path else str(_SRC_DIR)
)

# Isolate test logs into a per-process temp directory and clean it up on exit
# so running pytest never creates or rotates logs/app.log in the working tree.
_LOG_FILE_PREEXISTING = "LOG_FILE" in os.environ
_TEST_TMP = Path(tempfile.mkdtemp(prefix=f"eigenforge-tests-{os.getpid()}-"))
atexit.register(shutil.rmtree, _TEST_TMP, ignore_errors=True)
os.environ.setdefault("LOG_FILE", str(_TEST_TMP / "app.log"))

_managed_log = os.environ["LOG_FILE"]
_RealPopen = subprocess.Popen
_child_seq = 0


class _IsolatingPopen(_RealPopen):
    def __init__(self, *args, **kwargs):
        global _child_seq
        env = kwargs.get("env")
        base = dict(os.environ if env is None else env)
        if not _LOG_FILE_PREEXISTING and base.get("LOG_FILE") == _managed_log:
            _child_seq += 1
            target = _TEST_TMP / f"child-{os.getpid()}-{_child_seq}" / "app.log"
            target.parent.mkdir(parents=True, exist_ok=True)
            base["LOG_FILE"] = str(target)
        py_path = base.get("PYTHONPATH", "")
        src_root = str(_SRC_DIR)
        if src_root not in py_path.split(os.pathsep):
            base["PYTHONPATH"] = f"{src_root}{os.pathsep}{py_path}" if py_path else src_root
        kwargs["env"] = base
        super().__init__(*args, **kwargs)


subprocess.Popen = _IsolatingPopen
