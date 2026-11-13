"""LJM Reverse Engineering 起動スクリプト.

    python main.py [ファイル ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ljm_re.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
