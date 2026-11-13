import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def jp_dir(tmp_path):
    """日本語を含む一時ディレクトリ（Windows の日本語パス対応の確認用）."""
    d = tmp_path / "テスト用フォルダ"
    d.mkdir()
    return d


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def angle_deg(a, b):
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), -1, 1))))


def has_display():
    if sys.platform == "win32" or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
