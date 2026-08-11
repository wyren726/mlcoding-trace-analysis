from pathlib import Path
from typing import Any

from ..comparison import write_comparison
from ..shared import ExtensionContext


def run(context: ExtensionContext, output: Path) -> dict[str, Any]:
    return write_comparison(context, output, "model")


__all__ = ["run"]
