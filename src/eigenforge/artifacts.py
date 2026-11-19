"""模型产物追溯（Slice 6）。

每次保存模型产物时生成同旁 manifest（JSON）：产物内容 sha256、模型身份、
训练配置快照、数据集标识、随机种子、step/epoch 与 git 元数据，使任何
checkpoint 都可以回答"谁、何时、用什么配置、什么数据训练出来的"。
git 不可用（非仓库/无 git 命令）时元数据为 ``null``，不阻塞保存。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eigenforge.common.fsutil import durable_replace
from eigenforge.common.logger import logger

ALLOWED_MODEL_KINDS = {"eigengan", "eigengan_faithful", "diffusion"}
_MANIFEST_SCHEMA_VERSION = 1
# 定位本代码所在的项目根与 Git 仓库根（兼容独立仓库与子目录布局）；不依赖调用方的 CWD。
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
_REPO_ROOT = next(
    (p for p in (_PROJECT_ROOT, *_PROJECT_ROOT.parents) if (p / ".git").exists()),
    _PROJECT_ROOT,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(repo_root: Path | None = None) -> dict[str, Any] | None:
    """{'commit': sha, 'dirty': bool} 或 git 不可用时 None（并记一条 warning 说明原因）。

    ``--no-optional-locks`` 保证只读查询不会改写 ``.git/index``（并发会话/IDE 同时操作
    同一仓库时，普通 ``git status`` 会刷新 index 并争用锁）。
    """
    root = repo_root or _REPO_ROOT
    common = {
        "cwd": root, "capture_output": True, "text": True, "encoding": "utf-8",
        "errors": "replace", "timeout": 15, "check": True,
    }
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], **common).stdout.strip()
        status_cmd = ["git", "--no-optional-locks", "status", "--porcelain"]
        if repo_root is None and _PROJECT_ROOT.is_relative_to(root):
            status_cmd.extend(["--", str(_PROJECT_ROOT)])
        status = subprocess.run(status_cmd, **common).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("git metadata unavailable for {}: {}", root, exc)
        return None
    return {"commit": commit, "dirty": bool(status.strip())}


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_path: str
    artifact_sha256: str
    model_kind: str
    seed: int
    step: int
    training_config: dict[str, Any]
    dataset: dict[str, Any] | None
    metrics: dict[str, Any] | None
    git: dict[str, Any] | None
    created_utc: str
    manifest_schema_version: int

    @classmethod
    def create(
        cls,
        *,
        artifact_path: str | Path,
        model_kind: str,
        seed: int,
        step: int,
        training_config: dict[str, Any],
        dataset: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        artifact_sha256: str | None = None,
    ) -> "ArtifactManifest":
        """``artifact_sha256`` 供"先在内存里序列化、哈希，再落盘"的调用方使用（产物尚未存在）；
        省略时按原语义要求文件已存在并对文件字节取哈希。"""
        path = Path(artifact_path)
        if artifact_sha256 is None:
            if not path.is_file():
                raise FileNotFoundError(f"artifact does not exist: {path}")
        elif len(artifact_sha256) != 64 or any(c not in "0123456789abcdef" for c in artifact_sha256):
            raise ValueError("artifact_sha256 must be a 64-char lowercase hex digest")
        if model_kind not in ALLOWED_MODEL_KINDS:
            raise ValueError(
                f"model_kind must be one of {sorted(ALLOWED_MODEL_KINDS)}, got {model_kind!r}"
            )
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("step must be a non-negative integer")
        return cls(
            artifact_path=str(path),
            artifact_sha256=artifact_sha256 if artifact_sha256 is not None else _sha256_file(path),
            model_kind=model_kind,
            seed=seed,
            step=step,
            training_config=dict(training_config),
            dataset=dict(dataset) if dataset else None,
            metrics=dict(metrics) if metrics else None,
            git=git_metadata(),
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            manifest_schema_version=_MANIFEST_SCHEMA_VERSION,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_schema_version": self.manifest_schema_version,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "model_kind": self.model_kind,
            "seed": self.seed,
            "step": self.step,
            "training_config": self.training_config,
            "dataset": self.dataset,
            "metrics": self.metrics,
            "git": self.git,
            "created_utc": self.created_utc,
        }

    def to_json_text(self) -> str:
        """序列化为 JSON 文本；不触盘，调用方可在改动磁盘前先验证 manifest 可序列化。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_json(self, path: str | Path) -> Path:
        out = Path(path)
        text = self.to_json_text()
        durable_replace(out, lambda handle: handle.write(text))
        return out
