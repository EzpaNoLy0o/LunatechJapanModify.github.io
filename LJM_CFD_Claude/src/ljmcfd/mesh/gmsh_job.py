"""Gmsh による風洞メッシュ生成ジョブ。

流れ:

1. キャッシュ済み B-Rep を OpenCASCADE カーネルへ読み込み、ワールド姿勢へ変換する。
2. 風洞ボックスからモデルを差し引いて流体領域を作る (地面より下・モデル内部は自動的に除かれる)。
3. 外周面を位置で分類 (inlet / outlet / sideL / sideR / ground / top)、残りをパーツへ対応付ける。
4. サイズ場 (表面からの距離・後流領域・曲率) を設定し、四面体メッシュを生成する。
   境界層プリズムは、モデルが地面や壁に接していない場合に限り生成できる (実験的)。
5. OpenFOAM の polyMesh へ変換し、品質を評価して VTK とレポートを書き出す。

出力 (``out_dir``):  ``polyMesh/``  ``volume.vtu``  ``boundary.vtp``  ``mesh_report.json``
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np

from ..core.jobs import JobContext, JobError
from .polymesh import HEX, PRISM, PYRAMID, TET, BoundaryGroup, PolyMesh, build_polymesh, write_polymesh
from .quality import compute_quality
from .settings import ALGORITHM_DELAUNAY, ALGORITHM_HXT, MeshSettings

REPORT_FILE = "mesh_report.json"
REPORT_VERSION = 1

#: Gmsh の要素種別番号 → (セル種別, 節点数)
_GMSH_CELLS = {4: (TET, 4), 5: (HEX, 8), 6: (PRISM, 6), 7: (PYRAMID, 5)}
_GMSH_FACES = {2: 3, 3: 4}

TUNNEL_PATCH_TYPES = {"inlet": "patch", "outlet": "patch", "sideL": "patch", "sideR": "patch", "top": "patch",
                      "bottom": "patch", "ground": "wall"}


def patch_name_for(name: str, index: int, used: set[str]) -> str:
    """パーツ名から OpenFOAM で安全なパッチ名 (英数字とアンダースコア) を作る。"""
    base = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_")
    if not base or not base[0].isalpha():
        base = f"part{index}_{base}".rstrip("_")
    candidate, n = base, 2
    while candidate in used or candidate in TUNNEL_PATCH_TYPES:
        candidate = f"{base}_{n}"
        n += 1
    used.add(candidate)
    return candidate


def similarity_from_matrix(matrix) -> tuple[float, np.ndarray, np.ndarray]:
    """4x4 行列を (一様スケール, 回転ベクトル [rad], 平行移動) へ分解する。相似変換でなければエラー。"""
    from scipy.spatial.transform import Rotation

    m = np.asarray(matrix, dtype=float)
    linear = m[:3, :3]
    det = float(np.linalg.det(linear))
    if det <= 0:
        raise JobError("パーツの変換行列が不正です (鏡映または縮退)")
    scale = det ** (1.0 / 3.0)
    rotation = linear / scale
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
        raise JobError("パーツの変換行列が相似変換ではありません (非一様スケールは未対応)")
    return scale, Rotation.from_matrix(rotation).as_rotvec(), m[:3, 3].copy()


def _classify_tunnel_surface(bbox, domain, tol: float) -> str | None:
    """面の境界ボックスが風洞のどの外周面に乗っているか。乗っていなければ ``None`` (= モデル表面)。"""
    names = (("inlet", 0, 0), ("outlet", 0, 1), ("sideL", 1, 2), ("sideR", 1, 3), ("__bottom__", 2, 4), ("top", 2, 5))
    for name, axis, d_index in names:
        value = domain[d_index]
        if abs(bbox[axis] - value) <= tol and abs(bbox[axis + 3] - value) <= tol:
            return name
    return None


def _connected_groups(surfaces: list[int], curves_of: dict[int, set[int]]) -> list[list[int]]:
    """辺を共有する面どうしをまとめ、連結成分 (閉じた殻) ごとのリストにする。"""
    parent = {s: s for s in surfaces}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner: dict[int, int] = {}
    for surface in surfaces:
        for curve in curves_of[surface]:
            if curve in owner:
                parent[find(surface)] = find(owner[curve])
            else:
                owner[curve] = surface
    groups: dict[int, list[int]] = {}
    for surface in surfaces:
        groups.setdefault(find(surface), []).append(surface)
    return list(groups.values())


def _attribute_surfaces(gmsh, surfaces: list[int], part_surfaces: list) -> np.ndarray:
    """モデル表面の各面がどのパーツのものかを、面上のサンプル点から各パーツ表面への距離で決める。"""
    import pyvista as pv

    if len(part_surfaces) == 1:
        return np.zeros(len(surfaces), dtype=np.int64)
    samples, owner = [], []
    fractions = (0.2, 0.5, 0.8)
    for index, tag in enumerate(surfaces):
        points = []
        # 面の輪郭上の点 (トリム後の面に確実に乗っている)
        for _, curve in gmsh.model.getBoundary([(2, tag)], combined=False, oriented=False):
            lo, hi = gmsh.model.getParametrizationBounds(1, abs(curve))
            params = [lo[0] + f * (hi[0] - lo[0]) for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
            points.append(np.asarray(gmsh.model.getValue(1, abs(curve), params)).reshape(-1, 3))
        # 面内部の点 (輪郭がすべて他パーツとの交線である場合の判別用)
        lo, hi = gmsh.model.getParametrizationBounds(2, tag)
        uv = [c for fu in fractions for fv in fractions for c in (lo[0] + fu * (hi[0] - lo[0]), lo[1] + fv * (hi[1] - lo[1]))]
        points.append(np.asarray(gmsh.model.getValue(2, tag, uv)).reshape(-1, 3))
        block = np.vstack(points)
        samples.append(block)
        owner.append(np.full(len(block), index))
    cloud = pv.PolyData(np.vstack(samples))
    owner = np.concatenate(owner)
    score = np.empty((len(part_surfaces), len(surfaces)))
    for p_index, surface in enumerate(part_surfaces):
        distance = np.abs(cloud.compute_implicit_distance(surface)["implicit_distance"])
        score[p_index] = np.bincount(owner, weights=distance, minlength=len(surfaces))
    return np.argmin(score, axis=0)


def _normals_point_into_fluid(gmsh, surfaces: list[int], fluid: int, offset: float) -> list[bool | None]:
    """各面の法線が流体側を向いているか。判定できなければ ``None``。

    面上の数点を法線方向の前後へ少しずらし、どちらが流体領域の内側かを数える。
    """
    result: list[bool | None] = []
    fractions = (0.25, 0.5, 0.75)
    for tag in surfaces:
        lo, hi = gmsh.model.getParametrizationBounds(2, tag)
        uv = [c for fu in fractions for fv in fractions for c in (lo[0] + fu * (hi[0] - lo[0]), lo[1] + fv * (hi[1] - lo[1]))]
        points = np.asarray(gmsh.model.getValue(2, tag, uv)).reshape(-1, 3)
        normals = np.asarray(gmsh.model.getNormal(tag, uv)).reshape(-1, 3)
        forward = sum(gmsh.model.isInside(3, fluid, (p + offset * n).tolist()) for p, n in zip(points, normals))
        backward = sum(gmsh.model.isInside(3, fluid, (p - offset * n).tolist()) for p, n in zip(points, normals))
        result.append(None if forward == backward else forward > backward)
    return result


def _load_part_surface(part: dict, sink: float):
    """パーツの表示メッシュをワールド座標 (沈み込み適用後) で読み込む。"""
    import pyvista as pv

    surface = pv.read(part["surface"])
    matrix = np.asarray(part["matrix"], dtype=float)
    points = np.asarray(surface.points, dtype=float) @ matrix[:3, :3].T + matrix[:3, 3]
    points[:, 2] -= sink
    return pv.PolyData(points, surface.faces)


def _write_vtk(mesh: PolyMesh, cells: dict[str, np.ndarray], out_dir: Path) -> None:
    """GUI 表示用に体積メッシュ (vtu) と境界面 (vtp、パッチ番号つき) を書く。"""
    import pyvista as pv

    cell_types = {TET: pv.CellType.TETRA, PRISM: pv.CellType.WEDGE, PYRAMID: pv.CellType.PYRAMID, HEX: pv.CellType.HEXAHEDRON}
    blocks = {}
    for kind, array in cells.items():
        if len(array) == 0:
            continue
        # VTK のくさび形は底面三角形の向きが Gmsh / OpenFOAM と逆
        blocks[cell_types[kind]] = array[:, (0, 2, 1, 3, 5, 4)] if kind == PRISM else array
    pv.UnstructuredGrid(blocks, mesh.points.astype(np.float32)).save(out_dir / "volume.vtu", binary=True)

    faces = mesh.faces[mesh.n_internal:]
    patch_ids = np.empty(len(faces), dtype=np.int32)
    for index, patch in enumerate(mesh.patches):
        patch_ids[patch["start"] - mesh.n_internal:patch["start"] - mesh.n_internal + patch["size"]] = index
    is_tri = faces[:, 3] < 0
    used, inverse = np.unique(faces[faces >= 0], return_inverse=True)
    remapped = np.full(faces.shape, -1, dtype=np.int64)
    remapped[faces >= 0] = inverse
    tri = np.column_stack([np.full(int(is_tri.sum()), 3), remapped[is_tri, :3]])
    quad = np.column_stack([np.full(int((~is_tri).sum()), 4), remapped[~is_tri]])
    surface = pv.PolyData(mesh.points[used].astype(np.float32), np.concatenate([tri.ravel(), quad.ravel()]))
    surface.cell_data["patch"] = np.concatenate([patch_ids[is_tri], patch_ids[~is_tri]])
    surface.save(out_dir / "boundary.vtp", binary=True)


def mesh_generate_job(args: dict, ctx: JobContext | None = None) -> dict:
    """風洞メッシュを生成する。失敗した場合は作りかけの出力を残さず、以前のメッシュもそのまま保つ。"""
    out_dir = Path(args["out_dir"])
    tmp_dir = out_dir.with_name(out_dir.name + ".partial")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    try:
        return _generate(args, ctx or JobContext(), out_dir, tmp_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _generate(args: dict, ctx: JobContext, out_dir: Path, tmp_dir: Path) -> dict:
    """風洞メッシュを生成する。戻り値はレポート (``mesh_report.json`` と同内容)。

    args:
        out_dir       出力フォルダ (成功時のみ置き換える)
        parts         [{"id", "name", "brep" (絶対パス/None), "surface" (絶対パス), "matrix" (4x4)}]
        domain        [xmin, xmax, ymin, ymax, zmin, zmax]  計算領域 [m]
        model_bounds  モデルのワールド境界 [m]
        has_ground    地面 (下面を壁として扱う) の有無
        sink          メッシュ生成時にモデルを下げる量 [m]
        settings      MeshSettings.to_dict()
        input_hash    入力のハッシュ (レポートへ記録)
    """
    import gmsh

    started = time.perf_counter()
    settings = MeshSettings.from_dict(args.get("settings"))
    errors = settings.validate()
    if errors:
        raise JobError(" ".join(errors))
    parts = args.get("parts") or []
    if not parts:
        raise JobError("メッシュを生成するパーツがありません")
    unsupported = [p["name"] for p in parts if not p.get("brep")]
    if unsupported:
        raise JobError(
            "次のパーツは B-Rep (STEP / IGES) ではないため、現在のメッシュ生成では扱えません: "
            + "、".join(unsupported) + "。STL からの体積メッシュ生成は未対応です。"
        )
    domain = [float(v) for v in args["domain"]]
    model = [float(v) for v in args["model_bounds"]]
    has_ground = bool(args.get("has_ground", True))
    sink = float(args.get("sink", 0.0))
    bl = settings.boundary_layer
    bl_report = {"requested": bool(bl.enabled), "generated": False, "layers": 0, "first_height": None,
                 "total_thickness": None, "n_prisms": 0, "reason": "境界層プリズムは要求されていません。"}

    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.Verbosity", 2)
        threads = settings.threads or (os.cpu_count() or 1)
        gmsh.option.setNumber("General.NumThreads", threads)
        gmsh.model.add("tunnel")
        occ = gmsh.model.occ

        # ---------------------------------------------------------------- 1. 形状の読み込みと配置
        solids: list[tuple[int, int]] = []
        for index, part in enumerate(parts):
            ctx.progress(0.02 + 0.08 * index / len(parts), f"形状を読み込み中… {part['name']}")
            if not Path(part["brep"]).is_file():
                raise JobError(f"B-Rep キャッシュが見つかりません: {part['brep']}")
            entities = occ.importShapes(str(part["brep"]), highestDimOnly=True, format="brep")
            volumes = [e for e in entities if e[0] == 3]
            if not volumes:
                # サーフェスの集合なら縫い合わせてソリッド化を試みる
                healed = occ.healShapes(entities, sewFaces=True, makeSolids=True)
                volumes = [e for e in healed if e[0] == 3]
                if not volumes:
                    raise JobError(
                        f"パーツ「{part['name']}」は閉じたソリッドではないため、流体領域から差し引けません。"
                        "形状検査で開放エッジを確認し、CAD 側で閉じた形状にしてください。"
                    )
                ctx.log(f"パーツ「{part['name']}」はサーフェスの集合だったため、縫い合わせてソリッド化しました。")
            scale, rotvec, translation = similarity_from_matrix(part["matrix"])
            # 相似変換 (拡大・回転・移動) を使う。アフィン変換は曲面を NURBS 化して精度と速度を落とす。
            occ.dilate(volumes, 0, 0, 0, scale, scale, scale)
            angle = float(np.linalg.norm(rotvec))
            if angle > 1e-12:
                axis = rotvec / angle
                occ.rotate(volumes, 0, 0, 0, axis[0], axis[1], axis[2], angle)
            occ.translate(volumes, translation[0], translation[1], translation[2] - sink)
            solids.extend(volumes)

        # ---------------------------------------------------------------- 2. 流体領域 = 風洞 − モデル
        ctx.progress(0.12, "流体領域を作成中 (ブーリアン演算)…")
        box = occ.addBox(domain[0], domain[2], domain[4], domain[1] - domain[0], domain[3] - domain[2], domain[5] - domain[4])
        try:
            result, _ = occ.cut([(3, box)], solids, removeObject=True, removeTool=True)
            occ.synchronize()
        except Exception as exc:  # noqa: BLE001 - Gmsh は種類を問わず Exception を投げる
            raise JobError(
                f"風洞からモデルを差し引くブーリアン演算に失敗しました ({exc})。"
                "形状検査で不良が無いか確認し、パーツ同士が極端に薄く重なっていないか確認してください。"
            ) from exc
        volumes = [tag for dim, tag in result if dim == 3]
        if not volumes:
            raise JobError("流体領域が残りませんでした。モデルが風洞全体を覆っています。余白を確認してください。")
        if len(volumes) > 1:
            masses = [occ.getMass(3, tag) for tag in volumes]
            keep = volumes[int(np.argmax(masses))]
            occ.remove([(3, tag) for tag in volumes if tag != keep], recursive=True)
            occ.synchronize()
            ctx.log(f"モデル内部に閉じた空洞が {len(volumes) - 1} 個あったため、外部流れの領域のみを残しました。")
            volumes = [keep]
        fluid = volumes[0]
        cad_volume = float(occ.getMass(3, fluid))

        # ---------------------------------------------------------------- 3. 境界面の分類
        tol = 1e-6 * float(np.linalg.norm([domain[1] - domain[0], domain[3] - domain[2], domain[5] - domain[4]])) + 1e-7
        bottom_name = "ground" if has_ground else "bottom"
        tunnel_surfaces: dict[str, list[int]] = {}
        vehicle_surfaces: list[int] = []
        for _, tag in gmsh.model.getBoundary([(3, fluid)], combined=False, oriented=False):
            kind = _classify_tunnel_surface(gmsh.model.getBoundingBox(2, tag), domain, tol)
            if kind is None:
                vehicle_surfaces.append(tag)
            else:
                tunnel_surfaces.setdefault(bottom_name if kind == "__bottom__" else kind, []).append(tag)
        if not vehicle_surfaces:
            raise JobError("計算領域の中にモデル表面がありません。モデルの位置と風洞の余白を確認してください。")
        for name in ("inlet", "outlet"):
            if name not in tunnel_surfaces:
                raise JobError(f"風洞の {name} 面がモデルで塞がれています。余白を大きくしてください。")

        ctx.progress(0.18, "面をパーツへ対応付け中…")
        part_surfaces = [_load_part_surface(part, sink) for part in parts]
        owner_of = _attribute_surfaces(gmsh, vehicle_surfaces, part_surfaces)
        surfaces_by_part: dict[int, list[int]] = {}
        for tag, p_index in zip(vehicle_surfaces, owner_of.tolist()):
            surfaces_by_part.setdefault(p_index, []).append(tag)
        for p_index, part in enumerate(parts):
            if p_index not in surfaces_by_part:
                ctx.log(f"注意: パーツ「{part['name']}」の表面は流体に接していません (他のパーツの内部か領域外です)。")

        # ---------------------------------------------------------------- 4. 境界層プリズム (実験的)
        curves_of = {
            tag: {abs(c) for _, c in gmsh.model.getBoundary([(2, tag)], combined=False, oriented=False)}
            for tag in vehicle_surfaces + [t for tags in tunnel_surfaces.values() for t in tags]
        }
        algorithm = settings.algorithm
        if bl.enabled:
            outer = [t for tags in tunnel_surfaces.values() for t in tags]
            outer_curves = set().union(*(curves_of[t] for t in outer))
            touching = [t for t in vehicle_surfaces if curves_of[t] & outer_curves]
            if touching:
                raise JobError(
                    "境界層プリズムは、モデルが地面や風洞壁に接している場合は生成できません (未対応)。"
                    "境界層を無効にするか、モデルを地面から離してください。"
                )
            ctx.progress(0.22, "境界層プリズムを押し出し中…")
            heights = bl.layer_heights()
            # 押し出しは面の法線方向へ行われる。法線が流体側を向いているかは形状の作られ方に
            # よるので、決め打ちせず実際に調べる。
            directions = _normals_point_into_fluid(gmsh, vehicle_surfaces, fluid, 0.5 * bl.first_height)
            known = {d for d in directions if d is not None}
            if len(known) != 1:
                raise JobError(
                    "境界層プリズムを生成できません: モデル表面の向き (法線) が面によって異なる、または判定できません。"
                    "境界層を無効にしてメッシュを生成してください。"
                )
            sign = 1.0 if known.pop() else -1.0
            gmsh.model.removeEntities([(3, fluid)])
            gmsh.option.setNumber("Geometry.ExtrudeReturnLateralEntities", 0)
            loops = [gmsh.model.geo.addSurfaceLoop(outer)]
            for group in _connected_groups(vehicle_surfaces, curves_of):
                extruded = gmsh.model.geo.extrudeBoundaryLayer(
                    [(2, t) for t in group], [1] * bl.layers, [sign * h for h in heights], True)
                tops = [extruded[i - 1][1] for i in range(1, len(extruded)) if extruded[i][0] == 3]
                gmsh.model.geo.synchronize()
                loops.append(gmsh.model.geo.addSurfaceLoop(tops))
            gmsh.model.geo.addVolume(loops)
            gmsh.model.geo.synchronize()
            algorithm = ALGORITHM_DELAUNAY     # HXT はプリズム層と併用できない

        # ---------------------------------------------------------------- 5. サイズ場
        ctx.progress(0.26, "メッシュサイズ場を設定中…")
        field = gmsh.model.mesh.field
        sampling = int(np.clip(np.sqrt(2.0e6 / max(len(vehicle_surfaces), 1)), 10, 60))
        by_factor: dict[float, list[int]] = {}
        for p_index, tags in surfaces_by_part.items():
            factor = float(settings.part_size_factors.get(parts[p_index]["id"], 1.0))
            by_factor.setdefault(round(factor, 6), []).extend(tags)
        field_ids = []
        for factor, tags in by_factor.items():
            distance = field.add("Distance")
            field.setNumbers(distance, "SurfacesList", tags)
            field.setNumber(distance, "Sampling", sampling)
            threshold = field.add("Threshold")
            size = settings.surface_size * factor
            field.setNumber(threshold, "InField", distance)
            field.setNumber(threshold, "SizeMin", size)
            field.setNumber(threshold, "SizeMax", settings.far_size)
            field.setNumber(threshold, "DistMin", 2.0 * size)
            field.setNumber(threshold, "DistMax", settings.refine_distance)
            field_ids.append(threshold)
        # モデルが地面・壁と交わる線 (タイヤの接地線など) の周りは、面同士が鋭角で接するくさび形の
        # 領域になる。ここが粗いと潰れたセルができるため、接地線の近くだけ細かくする。
        tunnel_curves = set().union(*(curves_of[t] for tags in tunnel_surfaces.values() for t in tags))
        contact_curves = sorted(set().union(*(curves_of[t] for t in vehicle_surfaces)) & tunnel_curves)
        if contact_curves:
            contact = field.add("Distance")
            field.setNumbers(contact, "CurvesList", contact_curves)
            field.setNumber(contact, "Sampling", 200)
            contact_threshold = field.add("Threshold")
            contact_size = max(settings.min_size, settings.surface_size / 4.0)
            field.setNumber(contact_threshold, "InField", contact)
            field.setNumber(contact_threshold, "SizeMin", contact_size)
            field.setNumber(contact_threshold, "SizeMax", settings.far_size)
            field.setNumber(contact_threshold, "DistMin", 2.0 * contact_size)
            field.setNumber(contact_threshold, "DistMax", settings.refine_distance)
            field_ids.append(contact_threshold)
        if settings.wake_enabled:
            length = model[1] - model[0]
            width = model[3] - model[2]
            height = model[5] - model[4]
            wake = field.add("Box")
            field.setNumber(wake, "VIn", settings.surface_size * settings.wake_size_factor)
            field.setNumber(wake, "VOut", settings.far_size)
            field.setNumber(wake, "XMin", model[0] - 0.1 * length)
            field.setNumber(wake, "XMax", model[1] + settings.wake_length)
            field.setNumber(wake, "YMin", model[2] - 0.15 * width)
            field.setNumber(wake, "YMax", model[3] + 0.15 * width)
            field.setNumber(wake, "ZMin", max(domain[4], model[4] - sink - 0.25 * height))
            field.setNumber(wake, "ZMax", model[5] + 0.25 * height)
            field.setNumber(wake, "Thickness", 0.5 * settings.wake_length)
            field_ids.append(wake)
        minimum = field.add("Min")
        field.setNumbers(minimum, "FieldsList", field_ids)
        field.setAsBackgroundMesh(minimum)
        for option, value in {
            "Mesh.MeshSizeExtendFromBoundary": 0, "Mesh.MeshSizeFromPoints": 0,
            "Mesh.MeshSizeFromCurvature": settings.curvature_elements,
            "Mesh.MeshSizeMin": settings.min_size, "Mesh.MeshSizeMax": settings.far_size,
            "Mesh.Algorithm": 6, "Mesh.Optimize": 1 if settings.optimize else 0,
            "Mesh.MaxNumThreads3D": threads,
        }.items():
            gmsh.option.setNumber(option, value)

        # ---------------------------------------------------------------- 6. メッシュ生成
        def generate(kind: str) -> None:
            gmsh.option.setNumber("Mesh.Algorithm3D", 10 if kind == ALGORITHM_HXT else 1)
            gmsh.model.mesh.generate(3)

        ctx.progress(0.32, "表面・体積メッシュを生成中…")
        t_mesh = time.perf_counter()
        try:
            generate(algorithm)
        except Exception as exc:  # noqa: BLE001
            if bl.enabled:
                raise JobError(
                    f"境界層プリズムつきのメッシュ生成に失敗しました ({exc})。"
                    "層数を減らす・第 1 層を厚くする・表面サイズを小さくする、または境界層を無効にしてください。"
                ) from exc
            if algorithm != ALGORITHM_HXT:
                raise JobError(f"メッシュ生成に失敗しました ({exc})。メッシュサイズを見直してください。") from exc
            ctx.log(f"並列メッシャー (HXT) が失敗したため、逐次 Delaunay で再試行します ({exc})")
            gmsh.model.mesh.clear()
            algorithm = ALGORITHM_DELAUNAY
            try:
                generate(algorithm)
            except Exception as exc2:  # noqa: BLE001
                raise JobError(f"メッシュ生成に失敗しました ({exc2})。メッシュサイズを見直してください。") from exc2
        mesh_seconds = time.perf_counter() - t_mesh

        # ---------------------------------------------------------------- 7. 節点・要素の取り出し
        ctx.progress(0.70, "メッシュを取り出し中…")
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        index_of = np.full(int(node_tags.max()) + 1, -1, dtype=np.int64)
        index_of[node_tags.astype(np.int64)] = np.arange(len(node_tags))
        points_all = np.asarray(coords, dtype=float).reshape(-1, 3)

        cells: dict[str, np.ndarray] = {}
        gamma = None
        types, element_tags, element_nodes = gmsh.model.mesh.getElements(3)
        for gmsh_type, tags, nodes in zip(types, element_tags, element_nodes):
            if gmsh_type not in _GMSH_CELLS:
                raise JobError(f"未対応の要素種別がメッシュに含まれています (Gmsh type {gmsh_type})")
            kind, n_nodes = _GMSH_CELLS[gmsh_type]
            cells[kind] = index_of[nodes.astype(np.int64)].reshape(-1, n_nodes)
            quality = gmsh.model.mesh.getElementQualities(tags, "minSJ")
            if float(quality.min()) <= 0:
                raise JobError(
                    f"向きが反転した要素が {int((quality <= 0).sum()):,} 個あります ({kind})。"
                    + ("境界層の厚さが局所的な形状に対して大きすぎます。" if bl.enabled else "メッシュサイズを見直してください。")
                )
            if kind == TET:
                gamma = gmsh.model.mesh.getElementQualities(tags, "gamma")
        if not cells:
            raise JobError("体積メッシュが生成されませんでした")

        # 体積要素が使う節点だけに詰め直す
        used = np.unique(np.concatenate([c.ravel() for c in cells.values()]))
        compact = np.full(len(points_all), -1, dtype=np.int64)
        compact[used] = np.arange(len(used))
        points = points_all[used]
        cells = {kind: compact[array] for kind, array in cells.items()}

        def surface_elements(tags: list[int]) -> tuple[np.ndarray, np.ndarray]:
            tris, quads = [], []
            for tag in tags:
                s_types, _, s_nodes = gmsh.model.mesh.getElements(2, tag)
                for s_type, nodes in zip(s_types, s_nodes):
                    if s_type not in _GMSH_FACES:
                        raise JobError(f"未対応の表面要素がメッシュに含まれています (Gmsh type {s_type})")
                    array = compact[index_of[nodes.astype(np.int64)]].reshape(-1, _GMSH_FACES[s_type])
                    (tris if s_type == 2 else quads).append(array)
            return (np.vstack(tris) if tris else np.empty((0, 3), dtype=np.int64),
                    np.vstack(quads) if quads else np.empty((0, 4), dtype=np.int64))

        groups: list[BoundaryGroup] = []
        for name in ("inlet", "outlet", "sideL", "sideR", bottom_name, "top"):
            if name in tunnel_surfaces:
                tris, quads = surface_elements(tunnel_surfaces[name])
                groups.append(BoundaryGroup(name, TUNNEL_PATCH_TYPES[name], tris, quads, {"kind": "tunnel"}))
        used_names: set[str] = set()
        for p_index, part in enumerate(parts):
            if p_index not in surfaces_by_part:
                continue
            tris, quads = surface_elements(surfaces_by_part[p_index])
            name = patch_name_for(part["name"], p_index, used_names)
            groups.append(BoundaryGroup(name, "wall", tris, quads, {"kind": "part", "part_id": part["id"], "part_name": part["name"]}))
    finally:
        gmsh.finalize()

    # -------------------------------------------------------------------- 8. polyMesh 変換と品質評価
    ctx.progress(0.78, "OpenFOAM 形式 (polyMesh) へ変換中…")
    mesh = build_polymesh(points, cells, groups)
    if mesh.n_unmatched_faces:
        raise JobError(f"どの境界にも属さない面が {mesh.n_unmatched_faces:,} 面あります。メッシュが領域を正しく囲んでいません。")
    ctx.progress(0.84, "メッシュ品質を評価中…")
    quality = compute_quality(mesh)
    volume_error = abs(quality["volume_total"] - cad_volume) / cad_volume

    n_prisms = int(mesh.cell_counts.get(PRISM, 0))
    if bl.enabled:
        n_wall_faces = sum(p["size"] for p in mesh.patches if p["info"].get("kind") == "part")
        if n_prisms == 0 or n_prisms != bl.layers * n_wall_faces:
            raise JobError(
                f"境界層プリズムの数が想定と一致しません (生成 {n_prisms:,} / 想定 {bl.layers * n_wall_faces:,})。"
                "境界層は生成されていません。"
            )
        # プリズム層がモデルの内側へ押し出されると、四面体と重なって体積が「層の体積の 2 倍」だけ増える。
        # 離散化による体積誤差はそれより桁違いに小さいので、層の体積の半分をしきい値にして検出する。
        prism_volume = float(quality["prism_volume"])
        if abs(quality["volume_total"] - cad_volume) > 0.5 * prism_volume:
            raise JobError("境界層がモデルの内側へ押し出されました (メッシュの体積が流体領域と一致しません)。境界層は生成されていません。")
        bl_report.update(generated=True, layers=bl.layers, first_height=bl.first_height,
                         total_thickness=bl.total_thickness(), n_prisms=n_prisms,
                         reason="モデル表面にプリズム層を生成しました (地面には生成していません)。")
    elif n_prisms:
        raise JobError("要求していないプリズム要素が含まれています")

    ctx.progress(0.90, "ファイルを書き出し中…")
    write_polymesh(mesh, tmp_dir / "polyMesh")
    _write_vtk(mesh, cells, tmp_dir)

    report = {
        "report_version": REPORT_VERSION,
        "input_hash": args.get("input_hash"),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gmsh_version": gmsh.__version__,
        "algorithm": algorithm,
        "settings": settings.to_dict(),
        "domain": domain,
        "model_bounds": model,
        "has_ground": has_ground,
        "sink": sink,
        "patches": [{"name": p["name"], "type": p["type"], "n_faces": p["size"], **p["info"]} for p in mesh.patches],
        "quality": quality,
        "tet_gamma": None if gamma is None else {
            "min": float(gamma.min()), "mean": float(gamma.mean()),
            "histogram": np.histogram(gamma, bins=np.linspace(0.0, 1.0, 11))[0].tolist(),
        },
        "boundary_layer": bl_report,
        "cad_fluid_volume": cad_volume,
        "volume_error": float(volume_error),
        "seconds_meshing": round(mesh_seconds, 2),
        "seconds_total": round(time.perf_counter() - started, 2),
        "files": {"polymesh": "polyMesh", "volume": "volume.vtu", "boundary": "boundary.vtp"},
    }
    (tmp_dir / REPORT_FILE).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp_dir.rename(out_dir)
    report["out_dir"] = str(out_dir)
    ctx.progress(1.0, "メッシュ生成完了")
    ctx.log(
        f"メッシュ生成完了: {quality['n_cells']:,} セル / {quality['n_points']:,} 節点 / "
        f"非直交性 最大 {quality['non_ortho_max']:.1f}° / {report['seconds_total']:.1f} 秒"
    )
    return report


def load_report(mesh_dir: str | Path) -> dict | None:
    """メッシュフォルダのレポートを読む。無い・壊れている・ファイルが欠けている場合は ``None``。"""
    mesh_dir = Path(mesh_dir)
    path = mesh_dir / REPORT_FILE
    if not path.is_file():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if report.get("report_version") != REPORT_VERSION:
        return None
    if not all((mesh_dir / name).exists() for name in report.get("files", {}).values()):
        return None
    report["out_dir"] = str(mesh_dir)
    return report
