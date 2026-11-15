"""収束判定。「計算が終わった」ことと「工学的に収束した」ことを区別する。

指定した反復数を回り切っただけでは収束とはみなさない。次の 3 点を調べる。

1. **荷重の安定**: 直近の区間で抗力・揚力の変動幅と、区間の前半と後半の平均の差が小さいこと。
2. **残差**: 最終反復の初期残差が十分小さいこと。
3. **質量収支**: 流入と流出の体積流量が釣り合っていること。
"""

from __future__ import annotations

import numpy as np

# --- しきい値 (根拠はコメントのとおり。変更する場合は README の説明も合わせること)
MIN_ITERATIONS = 100            # これ未満では区間統計が取れず判定しない
WINDOW_FRACTION = 0.2           # 判定に使う直近区間 = 全反復の 20 %
WINDOW_MIN = 50
DRAG_RANGE_LIMIT = 0.01         # 区間内の抗力の変動幅 (最大 − 最小) が平均の 1 % 以内
DRAG_DRIFT_LIMIT = 0.005        # 区間の前半・後半の平均の差が 0.5 % 以内 (まだ一方向に動いていないか)
LIFT_RANGE_LIMIT = 0.02         # 揚力の変動幅。基準は max(|揚力|, 抗力の 10 %) (揚力が 0 付近でも破綻しないように)
LIFT_DRIFT_LIMIT = 0.01
RESIDUAL_GOOD = 1.0e-3          # 全残差がこれ以下なら「十分低い」(初期値 1 から 3 桁の低下)
RESIDUAL_ACCEPTABLE = 1.0e-2    # これ以下なら「やや高い」、超えれば「高い」
MASS_IMBALANCE_LIMIT = 1.0e-3   # |流入 + 流出| / |流入| が 0.1 % 以内

LEVEL_CONVERGED = "converged"           # 工学的に十分収束
LEVEL_MOSTLY = "mostly_converged"       # 荷重は安定しているが残差がやや高い
LEVEL_NOT_CONVERGED = "not_converged"   # 未収束
LEVEL_UNDETERMINED = "undetermined"     # 反復数不足などで判定できない
LEVEL_DIVERGED = "diverged"             # 発散
LEVEL_LABELS = {
    LEVEL_CONVERGED: "工学的に十分収束",
    LEVEL_MOSTLY: "概ね収束 (荷重は安定、残差はやや高い)",
    LEVEL_NOT_CONVERGED: "未収束",
    LEVEL_UNDETERMINED: "判定不能",
    LEVEL_DIVERGED: "発散",
}


def window_size(n_iterations: int) -> int:
    return int(min(n_iterations, max(WINDOW_MIN, round(WINDOW_FRACTION * n_iterations))))


def window_statistics(values: np.ndarray) -> dict:
    """区間の平均・変動幅・前半と後半の平均の差。"""
    values = np.asarray(values, dtype=float)
    half = len(values) // 2
    return {
        "mean": float(values.mean()),
        "range": float(values.max() - values.min()),
        "drift": float(values[half:].mean() - values[:half].mean()) if half else 0.0,
        "std": float(values.std()),
    }


def assess(drag: np.ndarray, lift: np.ndarray, final_residuals: dict[str, float],
           mass_imbalance: float | None, diverged: bool = False) -> dict:
    """収束を判定する。

    drag, lift:       反復ごとの抗力・揚力 [N] (全履歴)
    final_residuals:  最終反復の初期残差 (場の名前 → 値)
    mass_imbalance:   |流入 + 流出| / |流入|。取得できなければ ``None``
    戻り値: {"level", "label", "window", "checks": [{"name", "ok", "text"}], "messages"}
    """
    drag = np.asarray(drag, dtype=float)
    lift = np.asarray(lift, dtype=float)
    n = len(drag)
    checks: list[dict] = []

    def add(name: str, ok: bool | None, text: str) -> None:
        checks.append({"name": name, "ok": ok, "text": text})

    bad_numbers = n and not (np.isfinite(drag).all() and np.isfinite(lift).all())
    bad_residuals = any(not np.isfinite(v) for v in final_residuals.values())
    if diverged or bad_numbers or bad_residuals:
        add("発散", False, "計算が発散しました (数値が無限大または非数になったか、ソルバーが異常終了しました)。")
        return _result(LEVEL_DIVERGED, 0, checks)

    if n < MIN_ITERATIONS:
        add("反復数", None, f"反復数が {n} 回と少ないため、収束を判定できません (最低 {MIN_ITERATIONS} 回)。")
        return _result(LEVEL_UNDETERMINED, 0, checks)

    window = window_size(n)
    d = window_statistics(drag[-window:])
    l = window_statistics(lift[-window:])
    drag_ref = max(abs(d["mean"]), 1e-12)
    lift_ref = max(abs(l["mean"]), 0.1 * drag_ref)

    drag_range, drag_drift = d["range"] / drag_ref, abs(d["drift"]) / drag_ref
    lift_range, lift_drift = l["range"] / lift_ref, abs(l["drift"]) / lift_ref
    drag_ok = drag_range <= DRAG_RANGE_LIMIT and drag_drift <= DRAG_DRIFT_LIMIT
    lift_ok = lift_range <= LIFT_RANGE_LIMIT and lift_drift <= LIFT_DRIFT_LIMIT
    add("抗力の安定", drag_ok,
        f"直近 {window} 反復の変動幅 {drag_range * 100:.2f} % (基準 {DRAG_RANGE_LIMIT * 100:.0f} %)、"
        f"前半・後半の差 {drag_drift * 100:.2f} % (基準 {DRAG_DRIFT_LIMIT * 100:.1f} %)")
    add("揚力の安定", lift_ok,
        f"直近 {window} 反復の変動幅 {lift_range * 100:.2f} % (基準 {LIFT_RANGE_LIMIT * 100:.0f} %)、"
        f"前半・後半の差 {lift_drift * 100:.2f} % (基準 {LIFT_DRIFT_LIMIT * 100:.0f} %)")

    residual_state = None
    if final_residuals:
        worst_name = max(final_residuals, key=lambda k: final_residuals[k])
        worst = final_residuals[worst_name]
        residual_state = "good" if worst <= RESIDUAL_GOOD else "acceptable" if worst <= RESIDUAL_ACCEPTABLE else "high"
        text = {"good": "十分低い", "acceptable": "やや高い", "high": "高い"}[residual_state]
        add("残差", {"good": True, "acceptable": None, "high": False}[residual_state],
            f"最終残差の最大は {worst_name} = {worst:.2e} — {text} (基準 {RESIDUAL_GOOD:.0e}、許容 {RESIDUAL_ACCEPTABLE:.0e})")
    else:
        add("残差", None, "残差を取得できませんでした。")

    mass_ok = None
    if mass_imbalance is None:
        add("質量収支", None, "流入・流出の流量を取得できませんでした。")
    else:
        mass_ok = mass_imbalance <= MASS_IMBALANCE_LIMIT
        add("質量収支", mass_ok,
            f"流入と流出の差 {mass_imbalance * 100:.4f} % (基準 {MASS_IMBALANCE_LIMIT * 100:.1f} %)")

    forces_ok = drag_ok and lift_ok
    if forces_ok and residual_state == "good" and mass_ok:
        level = LEVEL_CONVERGED
    elif forces_ok and residual_state in ("good", "acceptable") and mass_ok is not False:
        level = LEVEL_MOSTLY
    else:
        level = LEVEL_NOT_CONVERGED
    return _result(level, window, checks)


def _result(level: str, window: int, checks: list[dict]) -> dict:
    messages = {
        LEVEL_CONVERGED: "荷重・残差・質量収支のすべてが基準を満たしています。",
        LEVEL_MOSTLY: "荷重は安定していますが、残差が基準 (1e-3) まで下がっていません。反復数を増やすと確実になります。",
        LEVEL_NOT_CONVERGED: "収束の基準を満たしていません。結果の数値は参考値として扱い、反復数を増やすかメッシュを見直してください。",
        LEVEL_UNDETERMINED: "収束を判定できませんでした。",
        LEVEL_DIVERGED: "計算が発散したため、結果は使用できません。",
    }
    return {"level": level, "label": LEVEL_LABELS[level], "window": window, "checks": checks, "message": messages[level]}
