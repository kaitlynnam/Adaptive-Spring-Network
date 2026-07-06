#!/usr/bin/env python3
"""Generate SVG block diagrams for the adaptive PEJ simulation framework."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.block_diagrams import write_default_diagrams


def main() -> None:
    output_dir = Path("artifacts/block_diagrams")
    paths = write_default_diagrams(output_dir)
    print("Generated adaptive PEJ block diagrams:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
