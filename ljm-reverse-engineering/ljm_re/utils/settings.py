"""アプリ設定（JSON 保存）."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields

from .paths import config_dir

log = logging.getLogger(__name__)


@dataclass
class AppSettings:
    # 表示
    theme: str = "dark"
    length_decimals: int = 3
    angle_decimals: int = 3
    lod_max_faces: int = 4_000_000        # これを超えるメッシュは表示用LODを使う
    background_top: tuple = (0.16, 0.18, 0.22)
    background_bottom: tuple = (0.06, 0.07, 0.09)
    show_grid: bool = True
    show_world_axes: bool = True
    parallel_projection: bool = False
    highlight_artificial: bool = True
    # 選択
    smart_angle_deg: float = 6.0          # スマート選択: 隣接面の角度許容
    smart_seed_dev_deg: float = 0.0       # 0 = 制限なし（平面だけを選ぶ場合は 10° 程度）
    brush_radius_mm: float = 5.0
    rect_front_only: bool = True
    exclude_artificial_in_fit: bool = True
    # フィッティング
    plane_method: str = "ransac"
    plane_threshold_mm: float = 0.0       # 0 = 自動
    cylinder_threshold_mm: float = 0.0    # 0 = 自動
    ransac_max_iter: int = 2000
    hole_section_depth_mm: float = 0.5
    # 断面/スケッチ
    sketch_tol_mm: float = 0.1
    # 位置合わせ
    icp_max_iter: int = 60
    icp_sample: int = 30000
    # CAD
    tess_deflection_mm: float = 0.05
    deviation_tol_mm: float = 0.1
    step_schema: str = "AP214IS"
    # その他
    recent_files: list = field(default_factory=list)
    last_dir: str = ""

    @staticmethod
    def path() -> str:
        return os.path.join(config_dir(), "settings.json")

    @classmethod
    def load(cls) -> "AppSettings":
        s = cls()
        try:
            with open(cls.path(), "r", encoding="utf-8") as f:
                d = json.load(f)
            names = {f.name for f in fields(cls)}
            for k, v in d.items():
                if k in names:
                    default = getattr(s, k)
                    if isinstance(default, tuple):
                        v = tuple(v)
                    setattr(s, k, v)
        except FileNotFoundError:
            pass
        except Exception:
            log.exception("設定ファイルの読込に失敗しました。既定値を使用します。")
        return s

    def save(self):
        try:
            with open(self.path(), "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        except Exception:
            log.exception("設定ファイルの保存に失敗しました")

    def to_dict(self):
        return asdict(self)

    def add_recent(self, path: str, limit: int = 10):
        p = os.path.abspath(path)
        self.recent_files = [p] + [x for x in self.recent_files if x != p]
        self.recent_files = self.recent_files[:limit]
