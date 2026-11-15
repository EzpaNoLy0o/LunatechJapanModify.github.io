"""ソルバー設定 (定常・非圧縮 RANS、k-ω SST)。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

ITERATION_CHOICES = (200, 300, 500, 700, 1000)
DEFAULT_ITERATIONS = 500


@dataclass
class SolverSettings:
    iterations: int = DEFAULT_ITERATIONS
    n_procs: int = 0                    # 並列数。0 = 自動 (物理コア数とセル数から決める)
    continue_run: bool = False          # 前回の解から続けて反復する (False なら最初から再計算)
    potential_init: bool = True         # ポテンシャル流れで速度場を初期化して立ち上がりを安定させる
    relax_p: float = 0.3                # 圧力の緩和係数
    relax_u: float = 0.7                # 運動量の緩和係数
    relax_turbulence: float = 0.7       # k・ω の緩和係数
    non_ortho_correctors: int = -1      # 非直交補正の回数。-1 = メッシュの非直交性から自動
    residual_tolerance: float = 1.0e-5  # すべての残差がこれを下回ったら反復を打ち切る

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "SolverSettings":
        data = data or {}
        settings = cls()
        for name, default in asdict(settings).items():
            if name not in data:
                continue
            value = data[name]
            if isinstance(default, bool):
                value = bool(value)
            elif isinstance(default, int):
                value = int(value)
            else:
                value = float(value)
            setattr(settings, name, value)
        return settings

    def validate(self) -> list[str]:
        errors = []
        if self.iterations < 1:
            errors.append("反復数は 1 以上にしてください。")
        if self.n_procs < 0:
            errors.append("並列数は 0 (自動) 以上にしてください。")
        for label, value in (("圧力", self.relax_p), ("運動量", self.relax_u), ("乱流", self.relax_turbulence)):
            if not 0.0 < value <= 1.0:
                errors.append(f"{label}の緩和係数は 0 より大きく 1 以下にしてください。")
        if not -1 <= self.non_ortho_correctors <= 5:
            errors.append("非直交補正は -1 (自動) から 5 の範囲にしてください。")
        if self.residual_tolerance <= 0:
            errors.append("残差の打ち切り基準は正の値にしてください。")
        return errors

    def resolved_non_ortho_correctors(self, max_non_ortho_deg: float) -> int:
        """非直交補正の回数。自動の場合はメッシュの最大非直交性から決める。

        四面体メッシュは非直交性が大きいため、補正なしでは圧力方程式の誤差が残りやすい。
        """
        if self.non_ortho_correctors >= 0:
            return self.non_ortho_correctors
        if max_non_ortho_deg < 40.0:
            return 0
        if max_non_ortho_deg < 70.0:
            return 1
        return 2


def auto_process_count(n_cells: int, physical_cores: int) -> int:
    """並列数の自動決定。1 プロセスあたり 2.5 万セル以上を目安に、物理コア数を上限とする。"""
    return int(max(1, min(physical_cores, n_cells // 25_000, 32)))
