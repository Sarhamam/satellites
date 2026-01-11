"""Test configuration."""

import sys
from pathlib import Path

# Add parent directory to path so webapp package can be imported
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
