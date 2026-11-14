"""前面投影面積 (流れ方向 +X から見た投影面積) の計算。

三角形を YZ 平面へ投影し、高解像度の画像へ塗りつぶして重なりを除いた面積を求める。
重なり合うパーツがあっても二重に数えない。閉塞率と Cd・Cl の基準面積に使う。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.jobs import JobContext, JobError

#: 投影画像の長辺の画素数。3000 px なら 2 m 幅の車両で 1 画素 0.7 mm。
DEFAULT_RESOLUTION = 3000


def projected_area(meshes, resolution: int = DEFAULT_RESOLUTION, clip_below_z: float | None = None) -> dict:
    """三角形メッシュ群の +X 方向投影面積を求める。

    meshes:        (points (N, 3) [m], triangles (M, 3)) のリスト (ワールド座標)
    clip_below_z:  指定すると、この高さより下 (地面の下) を面積に含めない
    戻り値:        {"area": 面積 [m²], "pixel_size": 1 画素の辺 [m], "width", "height"}
    """
    from PIL import Image, ImageDraw

    meshes = [(np.asarray(p, dtype=float), np.asarray(t, dtype=np.int64)) for p, t in meshes if len(t)]
    if not meshes:
        raise JobError("投影面積を計算する形状がありません")
    lo = np.min([p[:, 1:].min(axis=0) for p, _ in meshes], axis=0)
    hi = np.max([p[:, 1:].max(axis=0) for p, _ in meshes], axis=0)
    if clip_below_z is not None:
        lo[1] = max(lo[1], clip_below_z)
    extent = hi - lo
    if extent.min() <= 0:
        return {"area": 0.0, "pixel_size": 0.0, "width": float(max(extent[0], 0)), "height": float(max(extent[1], 0))}
    pixel = float(extent.max() / resolution)
    width = int(np.ceil(extent[0] / pixel)) + 2
    height = int(np.ceil(extent[1] / pixel)) + 2
    # 縦横 2 倍 (1 画素あたり 4 サンプル) で塗り、輪郭の画素に部分的な被覆率を持たせて誤差を減らす
    scale = 2
    image = Image.new("L", (width * scale, height * scale), 0)
    draw = ImageDraw.Draw(image)
    for points, triangles in meshes:
        yz = (points[:, 1:] - lo) / pixel * scale + scale     # 1 画素の余白
        tri = yz[triangles]                                  # (M, 3, 2)
        # 投影して面積を持たない三角形 (流れに平行な面) は描かない
        u, v = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        area2 = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
        for corners in tri[area2 > 1e-9].reshape(-1, 6).tolist():
            draw.polygon(corners, fill=255)
    coverage = np.asarray(image, dtype=np.float64) / 255.0
    if clip_below_z is not None:
        # 画像の行は Z に対応する。地面より下は余白の行 (先頭 scale 行) にしか描かれないので消す
        coverage[:scale, :] = 0.0
    area = float(coverage.sum() * (pixel / scale) ** 2)
    return {"area": area, "pixel_size": pixel, "width": float(extent[0]), "height": float(extent[1])}


def frontal_area_job(args: dict, ctx: JobContext | None = None) -> dict:
    """キャッシュ済みパーツの前面投影面積を求めるジョブ。

    args:
        parts         [{"surface": VTP の絶対パス, "matrix": 4x4 (ローカル → ワールド [m])}]
        clip_below_z  地面の高さ (地面より下を除外)。None なら除外しない
        resolution    投影画像の長辺の画素数
    """
    import pyvista as pv

    ctx = ctx or JobContext()
    parts = args.get("parts") or []
    if not parts:
        raise JobError("投影面積を計算するパーツがありません")
    meshes = []
    for index, part in enumerate(parts):
        ctx.progress(0.1 + 0.5 * index / len(parts), "前面投影面積を計算中…")
        path = Path(part["surface"])
        if not path.is_file():
            raise JobError(f"サーフェスキャッシュが見つかりません: {path}")
        surface = pv.read(path)
        matrix = np.asarray(part["matrix"], dtype=float)
        points = np.asarray(surface.points, dtype=float) @ matrix[:3, :3].T + matrix[:3, 3]
        meshes.append((points, np.asarray(surface.faces).reshape(-1, 4)[:, 1:]))
    ctx.progress(0.7, "前面投影面積を計算中…")
    result = projected_area(meshes, int(args.get("resolution", DEFAULT_RESOLUTION)), args.get("clip_below_z"))
    ctx.progress(1.0, "完了")
    return result
