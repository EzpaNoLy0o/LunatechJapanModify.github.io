"""解析結果 (流れ場) の読み込みと、断面・流線・等値面の抽出。

計算済みの結果ファイルだけを使う。ここでの操作は CFD を再実行しない。

速度を重視して VTK のフィルタを使い回す。34 万セルの実データで、断面の抽出は 1 回 約 15 ms、
流線 96 本は約 0.2 秒、等値面は約 0.1 秒 (開発機での実測)。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk

from ..solver.results import FIELDS_FILE
from ..solver.run_job import load_summary

AXES = ("x", "y", "z")


@dataclass(frozen=True)
class FieldInfo:
    key: str            # データ配列の名前
    label: str          # GUI の表示名 (日本語)
    title: str          # 3D ビューのカラーバーの見出し (VTK の既定フォントで出せる ASCII のみ)
    unit: str
    colormap: str
    symmetric: bool = False   # 0 を中心に正負へ広がる量か (圧力など)


#: 流れ場の物理量 (断面・流線の色・等値面で選べる)
VOLUME_FIELDS: dict[str, FieldInfo] = {
    "Umag": FieldInfo("Umag", "流速", "Velocity [m/s]", "m/s", "turbo"),
    "pressure": FieldInfo("pressure", "圧力 (ゲージ)", "Pressure [Pa]", "Pa", "coolwarm", True),
    "Cp": FieldInfo("Cp", "圧力係数 Cp", "Cp [-]", "", "coolwarm", True),
    "vorticity_mag": FieldInfo("vorticity_mag", "渦度", "Vorticity [1/s]", "1/s", "inferno"),
    "k": FieldInfo("k", "乱流エネルギー k", "TKE k [m2/s2]", "m²/s²", "viridis"),
    "Q": FieldInfo("Q", "Q 値 (渦の指標)", "Q-criterion [1/s2]", "1/s²", "plasma"),
}
#: 壁面の物理量
SURFACE_FIELDS: dict[str, FieldInfo] = {
    "pressure": VOLUME_FIELDS["pressure"],
    "Cp": VOLUME_FIELDS["Cp"],
    "wss_mag": FieldInfo("wss_mag", "壁面せん断応力", "Wall shear [Pa]", "Pa", "magma"),
    "yPlus": FieldInfo("yPlus", "y+", "y+ [-]", "", "viridis"),
}
COLORMAPS = ("turbo", "jet", "rainbow", "viridis", "plasma", "inferno", "magma", "coolwarm", "RdBu_r", "gray")


@dataclass
class SeedSpec:
    """流線の開始位置: X 軸に垂直な長方形の面上に、格子状に開始点を並べる。寸法はメートル。"""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.5
    width: float = 2.0        # Y 方向の幅
    height: float = 1.0       # Z 方向の高さ
    n_y: int = 12             # Y 方向の本数
    n_z: int = 8              # Z 方向の本数
    both_directions: bool = False   # 上流側へも延ばすか

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "SeedSpec":
        data = data or {}
        base = cls()
        values = {}
        for name, default in asdict(base).items():
            value = data.get(name, default)
            values[name] = bool(value) if isinstance(default, bool) else int(value) if isinstance(default, int) else float(value)
        return cls(**values)

    @property
    def count(self) -> int:
        return max(self.n_y, 1) * max(self.n_z, 1)

    def spacing(self) -> tuple[float, float]:
        """開始点の間隔 (Y, Z) [m]。1 本だけの方向は 0。"""
        return (self.width / (self.n_y - 1) if self.n_y > 1 else 0.0,
                self.height / (self.n_z - 1) if self.n_z > 1 else 0.0)

    def points(self) -> np.ndarray:
        ys = np.linspace(self.y - self.width / 2, self.y + self.width / 2, self.n_y) if self.n_y > 1 else np.array([self.y])
        zs = np.linspace(self.z - self.height / 2, self.z + self.height / 2, self.n_z) if self.n_z > 1 else np.array([self.z])
        yy, zz = np.meshgrid(ys, zs, indexing="ij")
        return np.column_stack([np.full(yy.size, self.x), yy.ravel(), zz.ravel()])

    def outline(self) -> np.ndarray:
        """開始面の四隅 (表示用)。"""
        y0, y1 = self.y - self.width / 2, self.y + self.width / 2
        z0, z1 = self.z - self.height / 2, self.z + self.height / 2
        return np.array([[self.x, y0, z0], [self.x, y1, z0], [self.x, y1, z1], [self.x, y0, z1]])


def default_seed(model_bounds, ground: bool = True) -> SeedSpec:
    """車両前面に置く既定の開始面。車幅・車高より少し広く、車両前端の少し上流。"""
    x0, x1, y0, y1, z0, z1 = model_bounds
    length, width, height = x1 - x0, y1 - y0, z1 - z0
    bottom = max(z0, 0.0) + 0.03 * height if ground else z0 - 0.1 * height
    top = z1 + 0.1 * height
    return SeedSpec(x=x0 - 0.25 * length, y=(y0 + y1) / 2, z=(bottom + top) / 2,
                    width=1.2 * width, height=top - bottom, n_y=12, n_z=8)


class ResultSet:
    """1 回の解析結果。ファイルは必要になったときに読む。"""

    def __init__(self, results_dir: str | Path, summary: dict | None = None) -> None:
        self.results_dir = Path(results_dir)
        self.summary = summary or load_summary(self.results_dir)
        if self.summary is None:
            raise FileNotFoundError(f"解析結果が見つかりません: {self.results_dir}")
        self._volume: pv.UnstructuredGrid | None = None
        self._flow: pv.UnstructuredGrid | None = None
        self._surfaces: dict[str, pv.PolyData] = {}
        self._plane = vtk.vtkPlane()
        self._cutter: vtk.vtkPlaneCutter | None = None
        self._ranges: dict[str, tuple[float, float]] = {}

    # ------------------------------------------------------------------ 読み込み
    @property
    def has_fields(self) -> bool:
        """流れ場のファイルがあるか (計算失敗の結果には無い)。"""
        return bool(self.summary.get("files", {}).get("fields")) and (self.results_dir / FIELDS_FILE).is_file()

    @property
    def volume(self) -> pv.UnstructuredGrid:
        if self._volume is None:
            if not self.has_fields:
                raise FileNotFoundError("この解析結果には流れ場のデータがありません")
            self._volume = pv.read(self.results_dir / FIELDS_FILE)
        return self._volume

    @property
    def flow(self) -> pv.UnstructuredGrid:
        """断面・流線用: 節点の速度ベクトルとスカラー量だけを持つ軽いコピー。

        抽出フィルタは入力の全配列を補間するので、使わない配列 (壁面せん断など) を外して速くする。
        """
        if self._flow is None:
            volume = self.volume
            flow = pv.UnstructuredGrid()
            flow.copy_structure(volume)
            for name in ("U", *VOLUME_FIELDS):
                if name in volume.point_data:
                    flow.point_data[name] = volume.point_data[name]
            flow.point_data.active_vectors_name = "U"
            self._flow = flow
        return self._flow

    def available_fields(self) -> list[str]:
        return [key for key in VOLUME_FIELDS if key in self.volume.point_data]

    @property
    def domain(self) -> tuple[float, ...]:
        return tuple(self.volume.bounds)

    def axis_range(self, axis: str) -> tuple[float, float]:
        index = AXES.index(axis)
        return (self.domain[2 * index], self.domain[2 * index + 1])

    @property
    def free_stream_speed(self) -> float:
        return float(self.summary["flow"]["speed"])

    # ------------------------------------------------------------------ 値の範囲
    def default_range(self, field: str, surface: bool = False) -> tuple[float, float]:
        """カラースケールの既定範囲。外れ値に引きずられないよう上下 0.5 % を除いた範囲にする。"""
        key = ("s:" if surface else "v:") + field
        if key not in self._ranges:
            if surface:
                arrays = [np.asarray(s.point_data[field]) for s in self.part_surfaces().values() if field in s.point_data]
                values = np.concatenate(arrays) if arrays else np.zeros(1)
                info = SURFACE_FIELDS[field]
            else:
                values = np.asarray(self.volume.point_data[field])
                info = VOLUME_FIELDS[field]
            values = values[np.isfinite(values)]
            lo, hi = (float(v) for v in np.percentile(values, [0.5, 99.5])) if len(values) else (0.0, 1.0)
            if field == "Umag":
                lo = 0.0
            elif field == "Cp":
                lo, hi = max(lo, -2.0), 1.0          # よどみ点が 1。負圧側は −2 までを既定にする
            elif info.symmetric:
                limit = max(abs(lo), abs(hi))
                lo, hi = -limit, limit
            if hi <= lo:
                hi = lo + 1.0
            self._ranges[key] = (lo, hi)
        return self._ranges[key]

    # ------------------------------------------------------------------ 壁面
    def _surface(self, patch: str) -> pv.PolyData:
        if patch not in self._surfaces:
            relative = self.summary["files"]["surfaces"][patch]
            surface = pv.read(self.results_dir / relative)
            if "wallShearStress" in surface.point_data:
                # OpenFOAM の wallShearStress は密度で割った値 [m²/s²]。密度を掛けて [Pa] にする
                rho = float(self.summary["flow"]["rho"])
                surface.point_data["wss_mag"] = np.linalg.norm(np.asarray(surface.point_data["wallShearStress"]), axis=1) * rho
            self._surfaces[patch] = surface
        return self._surfaces[patch]

    def part_surfaces(self) -> dict[str, pv.PolyData]:
        """車体パーツの壁面 (パッチ名 → サーフェス)。"""
        names = self.summary.get("files", {}).get("surfaces", {})
        return {name: self._surface(name) for name in names if name != "ground"}

    def ground_surface(self) -> pv.PolyData | None:
        return self._surface("ground") if "ground" in self.summary.get("files", {}).get("surfaces", {}) else None

    # ------------------------------------------------------------------ 断面
    def slice(self, axis: str, position: float) -> vtk.vtkPolyData:
        """軸に垂直な平面で流れ場を切った断面。カッターは使い回すので、位置を変えるだけなら速い。"""
        if self._cutter is None:
            cutter = vtk.vtkPlaneCutter()
            cutter.SetInputData(self.flow)
            cutter.SetPlane(self._plane)
            cutter.BuildTreeOn()
            cutter.ComputeNormalsOff()
            cutter.InterpolateAttributesOn()
            self._cutter = cutter
        normal = [0.0, 0.0, 0.0]
        normal[AXES.index(axis)] = 1.0
        origin = [0.0, 0.0, 0.0]
        origin[AXES.index(axis)] = float(position)
        self._plane.SetNormal(*normal)
        self._plane.SetOrigin(*origin)
        self._cutter.Update()
        output = self._cutter.GetOutput()
        if isinstance(output, vtk.vtkPolyData):
            return output
        # VTK の版によっては分割されたデータセットで返るので 1 つにまとめる
        append = vtk.vtkAppendPolyData()
        iterator = output.NewIterator()
        iterator.InitTraversal()
        while not iterator.IsDoneWithTraversal():
            piece = iterator.GetCurrentDataObject()
            if piece is not None and piece.GetNumberOfCells():
                append.AddInputData(piece)
            iterator.GoToNextItem()
        append.Update()
        return append.GetOutput()

    # ------------------------------------------------------------------ 流線
    def streamlines(self, seed: SeedSpec, max_length: float | None = None) -> pv.PolyData:
        """開始面から速度場に沿って積分した流線。各点に流速などの物理量と ``IntegrationTime`` を持つ。

        定常の速度場に沿った軌跡であり、時間変化する流れの記録ではない。
        """
        domain = self.domain
        if max_length is None:
            max_length = 1.5 * (domain[1] - domain[0])
        source = pv.PolyData(seed.points())
        tracer = vtk.vtkStreamTracer()
        tracer.SetInputData(self.flow)
        tracer.SetSourceData(source)
        tracer.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "U")
        tracer.SetIntegratorTypeToRungeKutta45()
        tracer.SetIntegrationStepUnit(vtk.vtkStreamTracer.CELL_LENGTH_UNIT)
        tracer.SetInitialIntegrationStep(0.2)
        tracer.SetMinimumIntegrationStep(0.01)
        tracer.SetMaximumIntegrationStep(0.5)
        tracer.SetMaximumNumberOfSteps(6000)
        tracer.SetMaximumPropagation(float(max_length))
        tracer.SetTerminalSpeed(1.0e-4 * max(self.free_stream_speed, 1e-6))
        tracer.SetComputeVorticity(False)
        tracer.SetInterpolatorTypeToCellLocator()
        if seed.both_directions:
            tracer.SetIntegrationDirectionToBoth()
        else:
            tracer.SetIntegrationDirectionToForward()
        tracer.Update()
        return pv.wrap(tracer.GetOutput())

    # ------------------------------------------------------------------ 等値面
    def isosurface(self, field: str, value: float) -> pv.PolyData:
        """物理量 ``field`` が ``value`` になる面。他の物理量も補間して持つ (色付け用)。"""
        contour = vtk.vtkContourFilter()
        contour.SetInputData(self.flow)
        contour.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, field)
        contour.SetValue(0, float(value))
        contour.ComputeNormalsOn()
        contour.Update()
        return pv.wrap(contour.GetOutput())
