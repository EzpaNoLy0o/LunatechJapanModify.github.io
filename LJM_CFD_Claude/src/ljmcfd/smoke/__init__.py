"""Realistic Smoke Visualization Engine: 計算済みの速度場から、立体的な煙の密度分布を作る。

計算 (このパッケージ。Qt に依存しない) と描画 (``gui/smoke_view.py``) を分けている。

    速度場 (OpenFOAM の結果) → 放出源 → 流跡線 (Level 2) → 煙の密度場 (Level 3) → ボリューム描画 (Level 4)
"""
