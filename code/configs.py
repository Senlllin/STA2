"""Configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class STAConfig:
    """Dataclass reflecting ``configs/sta.yaml`` structure."""

    use_sta: bool
    use_sta_bias: bool
    k_mirror: int
    beta: float
    temperature: float

    @staticmethod
    def from_yaml(path: Path) -> "STAConfig":
        data: Dict[str, Any] = yaml.safe_load(path.read_text())
        sta = data.get("model", {}).get("sta", {})
        return STAConfig(
            use_sta=data.get("model", {}).get("use_sta", True),
            use_sta_bias=data.get("model", {}).get("use_sta_bias", True),
            k_mirror=sta.get("k_mirror", 8),
            beta=sta.get("beta", 1.5),
            temperature=sta.get("temperature", 0.07),
        )