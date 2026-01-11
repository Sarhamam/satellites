"""Integration test configuration."""

import sys
from pathlib import Path

# Ensure webapp is importable
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
