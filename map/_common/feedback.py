from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass
class StageResult:
    stage: str
    status: str
    summary: dict[str, Any] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return _jsonable({
            "stage": self.stage,
            "status": self.status,
            "summary": self.summary,
            "outputs": self.outputs,
            "elapsed_seconds": self.elapsed_seconds,
        })


class Feedback:
    """Print short progress lines and return a structured stage result."""

    def __init__(self, output_root: Path, stage: str):
        self.output_root = Path(output_root)
        self.stage = stage
        self.started = time.perf_counter()
        self.events: list[dict[str, Any]] = []

    def emit(self, message: str, **values: Any) -> None:
        record = {"message": message, **_jsonable(values)}
        self.events.append(record)
        suffix = " ".join(f"{key}={value}" for key, value in values.items())
        print(f"[{self.stage}] {message}" + (f" | {suffix}" if suffix else ""), flush=True)

    def finish(self, summary: dict[str, Any], outputs: list[Path | str], status: str = "ok") -> StageResult:
        elapsed = time.perf_counter() - self.started
        result = StageResult(self.stage, status, summary, [str(item) for item in outputs], elapsed)
        self.emit("complete", status=status, elapsed_seconds=round(elapsed, 2))
        return result
