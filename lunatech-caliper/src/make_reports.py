#!/usr/bin/env python3
"""make_reports.py - write reports/*.md from build/meta.json, validation.json, orientation.json."""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import geometry as G   # noqa: E402

ROOT = G.ROOT
B = os.path.join(ROOT, "build")
REP = os.path.join(ROOT, "reports")


def J(n):
    with open(os.path.join(B, n)) as f:
        return json.load(f)


def yes(b):
    return "OK" if b else "NG"


def main():
    os.makedirs(REP, exist_ok=True)
    meta, val, ori = J("meta.json"), J("validation.json"), J("orientation.json")
    p = meta["params"]
    D = meta["derived"]
    stl = {s["file"]: s for s in val["stl"]}
    body = stl["Lunatech_Caliper_310mm.stl"]
    ex = body["extent_mm"]
    asm = val["assembly"]
    rot, whl = asm["rotor"], asm["wheel"]

    # ------------------------------------------------------------ Dimensions
    pis = p["piston"]
    rows = "\n".join(
        f"| {i + 1} | Ø{d:.0f} | {D['piston_s'][i]:+.1f} | {D['piston_th'][i]:+.2f}° | "
        f"{D['R_p'] * math.sin(math.radians(D['piston_th'][i])):+.1f} | {D['housing_R'][i] * 2:.0f} |"
        for i, d in enumerate(pis["diameters"]))
    spacing = ", ".join(f"{D['piston_s'][i + 1] - D['piston_s'][i]:.1f}" for i in range(len(pis["diameters"]) - 1))
    dims = f"""# Dimensions — Lunatech Titanium Caliper Replica

すべて mm。座標系：X = キャリパー長手（ローター接線）、Y = ローター軸（+Y = アウトボード / ホイール面側）、
Z = ローター半径方向。原点 = ローター回転中心、ローター中心面 Y = 0。数値は `src/validate_model.py` が
**出力 STL を読み戻して** 計測した値です（`build/validation.json`）。

## 1. 外形寸法（キャリパー本体 + ロゴ）

| 項目 | 本モデル | 初期目標 | Bugatti 実物（公表値） | 実物比 |
|---|---|---|---|---|
| 全長 X | **{ex[0]:.1f}** | 約310 | 410 | {ex[0] / 410:.2f} |
| 全幅 Y | **{ex[1]:.1f}** | 165〜185 | 210 | {ex[1] / 210:.2f} |
| 全高 Z | **{ex[2]:.1f}** | 110〜130 | 136 | {ex[2] / 136:.2f} |
| 最大軸寸法 | **{body['max_axis_mm']:.1f}** | ≤ 320（推奨 310） | — | — |
| 体積 | {body['volume_cm3']:.0f} cm³ | — | （2.9 kg ≒ 655 cm³ Ti-6Al-4V） | — |

* 自動サイズゲート：生成時に各軸 > {p['limits']['max_axis_mm']:.0f} mm なら目標全長を縮めて再生成（`generate_caliper.py`）。
  今回の履歴：{meta['history']}
* 全幅は初期目標下限 165 mm をわずかに下回ります（{ex[1]:.1f} mm）。幅はローター厚32 + パッド + ピストン
  ストローク + ボア底 + ブリッジで決まり、ホイールスポーク側クリアランス（{whl['spoke_axial_clearance_mm']:.1f} mm）を
  優先しました。`main_bridge_y_center` / `inboard_bridge_y` / `bore_depth` を増やせば拡幅できます。

## 2. デフォルメ（非一様縮小）

| 軸 | 実物 → 模型 | 倍率 | 方針 |
|---|---|---|---|
| 長手 | 410 → {ex[0]:.0f} | {ex[0] / 410:.2f} | ピストン間隔・パッド長を圧縮、両端トラスは角度で確保 |
| 幅 | 210 → {ex[1]:.0f} | {ex[1] / 210:.2f} | ローター厚 32（実物40想定）に合わせて縮小 |
| 高さ | 136 → {ex[2]:.0f} | {ex[2] / 136:.2f} | ほぼ維持：アーチ形状・メインブリッジの存在感を優先 |

長手方向のみ強く縮め、高さ方向はほぼ維持しているため、メインブリッジやバイオニック構造が相対的に
小さくなりすぎないようにしています。

## 3. ローター（実寸参照、縮小なし）

| 項目 | 値 |
|---|---|
| 外径 / 厚さ | {p['rotor']['outer_diameter']:.0f} / {p['rotor']['thickness']:.0f}（ベンチレーテッド、フリクションリング {p['rotor']['friction_ring_thickness']:.0f} ×2、ベーン {p['rotor']['vane_count']}枚） |
| キャリパー ⇔ ローター外周（半径方向） | 設定 {p['clearance']['rotor_radial']} / 実測 **{rot['radial_clearance_over_rotor_edge_mm']:.2f}** |
| キャリパー ⇔ ローター摺動面（軸方向、パッド以外） | 設定 {p['clearance']['rotor_axial']} / 実測 **{rot['axial_clearance_to_rotor_face_mm']:.2f}** |
| パッド ⇔ ローター | {p['clearance']['pad_to_rotor']} |
| 本体とローターの干渉 | 本体頂点のローター内部侵入 **{rot['body_points_inside_rotor']} 点** |

## 4. ピストン配置（8ポット対向：外側4 + 内側4、模型用仮設定値）

ピストン軸はローター軸に平行、中心半径 R = {D['R_p']:.2f}（パッド中心）。
回転方向の入口側（-X）から小径→大径。中心間距離（円弧長）：{spacing} mm。

| # | 径 | 円弧位置 s | 角度 | X | ハウジング外径 |
|---|---|---|---|---|---|
{rows}

* パッド：円弧 {D['pad_th0']:.1f}°〜{D['pad_th1']:.1f}°、半径 {p['pad']['radial_inner']}〜{p['pad']['radial_outer']}、
  裏板 {p['pad']['backing_thickness']} + ライニング {p['pad']['lining_thickness']}、パッドピン用イヤー付き。
* ボア深さ {pis['bore_depth']}、ボア底肉厚 {pis['housing_back_wall']}、ハウジング径方向肉厚 {pis['housing_wall_radial']}。
* **注意**：ピストン径はBugatti実物の値ではありません（模型用仮設定）。

## 5. 18インチホイール適合

| 項目 | 値 | 判定 |
|---|---|---|
| バレル最小内径（**仮定値・実測ではない**） | {whl['barrel_min_inner_diameter_assumed']:.0f} | — |
| キャリパー最大半径（本体＋ボルト/ピン/ブリーダー） | {whl['caliper_max_radius_mm']:.2f} | — |
| バレルとのクリアランス | **{whl['barrel_clearance_mm']:.2f}**（目標 {whl['barrel_clearance_target'][0]}〜{whl['barrel_clearance_target'][1]}） | {yes(whl['barrel_ok'])} |
| 必要最小バレル内径（3 mm確保） | **{whl['required_min_barrel_ID_for_3mm']:.1f}** | — |
| 必要最小バレル内径（5 mm確保） | **{whl['required_min_barrel_ID_for_5mm']:.1f}** | — |
| スポーク裏面位置 Y（**仮定値**） | {whl['spoke_back_y_assumed']:.0f} | — |
| キャリパー最外 Y（r>100） | {whl['caliper_max_y_mm']:.2f} | — |
| スポーク軸方向クリアランス | **{whl['spoke_axial_clearance_mm']:.2f}**（目標 ≥ {p['clearance']['spoke_axial_target_min']}） | {yes(whl['spoke_ok'])} |
| 3 mm確保に必要なスポーク裏面位置 | Y ≥ {whl['required_spoke_back_y_for_3mm']:.1f} | — |

本体外形はホイールパラメータから自動算出される包絡半径 R_cap = バレル内径/2 − {p['clearance']['wheel_barrel_design']}
= {D['R_cap']:.1f} mm で滑らかにトリムされます。実車ホイールの実測値（バレル最小内径・スポーク裏面形状）を
`parameters.json → wheel` に入れて再生成・再検証してください。

## 6. 主要パラメータ（`src/parameters.json`）

全長 `limits.target_length_mm` / ローター `rotor.*` / ホイール `wheel.barrel_min_inner_diameter`, `wheel.spoke_back_y` /
ピストン径 `piston.diameters` / 中心間 `piston.web_between_bores` / ブリッジ厚 `body.main_bridge_half_thickness_*` /
リブ太さ `body.rib_scale` / ブレンド `body.blend_major|minor` / 開口寸法 `body.opening_scale` /
ロゴ `logo.cap_height, tracking, protrusion, center_angle_deg, radial_offset` / 造形 `fdm.*` / メッシュ `mesh.*`
"""
    open(os.path.join(REP, "Dimensions.md"), "w").write(dims)

    # ------------------------------------------------------------ Print settings
    cand = ori["candidates"]
    names = {"O1_logo_up": "1. ロゴ側を上（インボード面が下）", "O2_logo_down": "ロゴ側を下",
             "O3_arch_upright": "2. 側面を下（実装姿勢、アーチを立てる）", "O4_logo_up_tilt45": "3. 斜め45°（ロゴ上向きから傾斜）",
             "O5_roof_down": "アーチ反転（上部ブリッジが下）"}
    trows = "\n".join(
        f"| {names.get(k, k)}{' **← 採用**' if k == ori['chosen'] else ''} | {r['extent'][0]:.0f}×{r['extent'][1]:.0f}×{r['extent'][2]:.0f} | "
        f"{r['support_volume_cm3']:.0f} | {r['support_scar_area_cm2']:.0f} | {r['logo_normal_z']:+.2f} | {r['bed_contact_cm2']:.1f} | "
        f"{r['est_time_h']:.0f} | {r['score']:.0f} |" for k, r in cand.items())
    sp = ori["split"]
    so, si = sp["outboard"], sp["inboard"]
    ch = cand[ori["chosen"]]
    fdm = p["fdm"]
    ps = f"""# Print Settings — QIDI MAX3（ビルド 325×325×325 mm）

## 1. 出力ファイルと推奨用途

| ファイル | 用途 | 造形サイズ |
|---|---|---|
| `stl/Lunatech_Caliper_Split_Outboard.stl` + `_Inboard.stl` + `Lunatech_Locating_Pins.stl` | **推奨：最高品質**（ローター中心面で2分割、分割面をベッドに） | {so['extent'][0]:.0f}×{so['extent'][1]:.0f}×{so['extent'][2]:.0f} / {si['extent'][0]:.0f}×{si['extent'][1]:.0f}×{si['extent'][2]:.0f} |
| `stl/Lunatech_Caliper_FDM_Optimized.stl` | 一体造形（評価で選んだ姿勢、ベッド中央配置、ロゴ用ポケット付き） | {ch['extent'][0]:.0f}×{ch['extent'][1]:.0f}×{ch['extent'][2]:.0f} |
| `stl/Lunatech_Caliper_310mm.stl` | 一体・ロゴ一体（単色）、組立座標系のまま | {ex[0]:.0f}×{ex[1]:.0f}×{ex[2]:.0f} |
| `stl/Lunatech_Logo.stl` | 別色ロゴ（フラット、厚さ {p['logo']['insert_root_depth'] - 0.05 + p['logo']['protrusion']:.2f} mm） | 79×13×1.4 |
| `stl/Lunatech_Logo_Fitted.stl` | 曲面に沿ったロゴ（組立座標、マルチマテリアル/確認用） | — |
| `stl/Lunatech_Pistons.stl`, `Lunatech_Pads.stl`, `Lunatech_Hardware.stl` | ピストン8個、パッド2枚、パッドピン/マウントボルト/ブリーダー | ≤ 304 |
| `stl/Rotor_355x32_Quarter_1..4.stl` | ローターを印刷する場合（4分割、各 178×178×64） | ≤ 178 |

ブリム 5 mm を含めても全ファイルが 325 mm 以内（一体品は {ch['extent'][0]:.0f}+10 = {ch['extent'][0] + 10:.0f} mm）。
サポートは造形物の投影範囲内に生成されるため、XY 外形は増えません。

## 2. 造形姿勢の比較（`export_stl.py` が自動計算、`build/orientation.json`）

オーバーハング閾値 45°。サポート体積 = 下向き面とその下の面（またはベッド）までの柱体積（中実換算）。
スカー面積 = 模型表面に着地するサポートの投影面積（除去跡が残る面積）。

| 姿勢 | 外形 X×Y×Z | サポート体積 cm³ | スカー面積 cm² | ロゴ法線 Z | ベッド接地 cm² | 推定時間 h | スコア |
|---|---|---|---|---|---|---|---|
{trows}

スコア = 0.6×サポート体積 + 0.6×スカー面積 + ロゴ向き罰則 + 安定性罰則 + 2×時間（小さいほど良い）。

**結論**
* 一体造形では **{names.get(ori['chosen'], ori['chosen'])}** が最良：アーチの開口（ローター溝・パッドポケット）が上を向き、
  内部サポートが最小。ロゴは垂直面になり、0.8 mm の凸文字は積層方向に対して側面となるため輪郭がシャープに出ます。
  接地面が上部リブの稜線のみなので、**ブリム 8 mm + ツリーサポートの台座** で安定させてください。
* 外観最優先なら **2分割版** を推奨：分割面（Y=0）が完全な平面でベッドに密着、ロゴ面が真上を向き最も高品位。
  アウトボード側サポート {so['support_volume_cm3']:.0f} cm³（スカー {so['support_scar_area_cm2']:.0f} cm²）、
  インボード側 {si['support_volume_cm3']:.0f} cm³（スカー {si['support_scar_area_cm2']:.0f} cm²）。
  分割線はローター中心面＝上部ブリッジの中央稜線（スパイン）上にあり、組立後は稜線に沿うため目立ちにくい位置です。

## 3. 位置決めピン（2分割版）

* 分割面の断面を距離変換し、内接半径 ≥ ピン穴半径 + 1.6 mm の位置から最も離れた {len(sp['pin_sites_xy_in_logo_up_frame'])} 点を自動選定
  （分割面の最大内接半径 {sp['max_inradius_on_split_plane']} mm）。
* ピン Ø{fdm['pin_diameter']} × {fdm['pin_length']} mm（面取り付き）、穴 Ø{sp['pin_hole_diameter']}（片側深さ {sp['hole_depth_each_side']} mm）。
* 嵌合クリアランスは `fdm.pin_hole_clearance`（現在 {fdm['pin_hole_clearance']} mm、片側）。プリンターの実測公差に合わせて
  0.10〜0.25 で調整し再生成。ロゴ差し込みの隙間は `logo.insert_clearance`（{p['logo']['insert_clearance']} mm）。

## 4. 推奨スライサー設定（PLA / PETG）

| 項目 | 推奨値 |
|---|---|
| ノズル | {fdm['nozzle']} mm |
| 積層ピッチ | **0.14 mm**（0.12〜0.16）。ロゴ付近だけ 0.08〜0.12 の可変積層も有効 |
| 壁（ペリメータ） | 3〜4 本（1.2〜1.6 mm）— 細いリブ（Ø≥3.6 mm）が全周壁で中実になる |
| 上下面 | 5 / 4 層 |
| インフィル | ジャイロイド 15 %（展示用） |
| サポート | ツリー（オーガニック）、閾値 45°、Z 距離 0.2 mm、XY 距離 0.6 mm、インターフェース 2 層 |
| ブリム | 5〜8 mm（一体品は 8 mm 推奨） |
| 温度 | PLA 210 °C / ベッド 60 °C　PETG 240 °C / ベッド 80 °C、チャンバー：PLA は扉開放 |
| 速度 | 外壁 60 mm/s、内壁 120 mm/s、ブリッジ 30 mm/s、冷却 PLA 100 % |
| シーム | 背面（インボード側）に整列 |
| 推定時間（一体） | 約 {ch['est_time_h']:.0f} h、材料 約 {ch['est_filament_g_PLA']} g（PLA、サポート込み、概算） |

時間は 0.14 mm 積層・約 11 cm³/h の押出量 + 層替え 3 s/層 の概算モデルです。スライサーの値を優先してください。

## 5. 組立・仕上げ

1. 2分割版：ピン4本を片側に圧入 → 瞬間接着剤またはエポキシで接合。
2. ロゴ：`Lunatech_Logo.stl` を別色（シルバー/ゴールド/ブラック等）で印刷、文字ごとにポケットへ差し込み接着。
   文字は 1.35 mm 厚で曲面（最大差約 0.3 mm/文字）に馴染みます。
3. ピストン・パッド・ボルト類を別色で印刷して挿入（ピストンはボア径 +{p['piston']['bore_clearance']} mm のクリアランス）。
4. チタン調仕上げ：#400→#800 研磨、サーフェーサー、メタリックグレー + 部分的にブルー〜ブロンズの焼け色。
"""
    open(os.path.join(REP, "Print_Settings.md"), "w").write(ps)

    # ------------------------------------------------------------ Geometry validation
    def wt(s):
        w = s.get("wall_thickness_mm")
        if not w:
            return "—"
        return f"P0.1 {w['p0_1']} / P1 {w['p1']} / P5 {w['p5']} / 中央値 {w['median']}（<0.8: {w['below_0_8']}%、<1.6: {w['below_1_6']}%、<2.0: {w['below_2_0']}%）"

    srows = "\n".join(
        f"| `{s['file']}` | {s['triangles']:,} | {s['extent_mm'][0]:.1f}×{s['extent_mm'][1]:.1f}×{s['extent_mm'][2]:.1f} | "
        f"{yes(s['within_limit'])} | {yes(s['watertight'])} | {s['non_manifold_edges']} | {s['self_intersecting_faces']} | "
        f"{yes(s['normals_outward'] and s['winding_consistent'])} | {s['shells']} | {s['internal_voids']} | {s['volume_cm3']} |"
        for s in val["stl"])
    trs = "\n".join(f"| `{s['file']}` | {wt(s)} |" for s in val["stl"] if s.get("wall_thickness_mm"))
    srow = []
    for s_ in val["step"]:
        if "file_note" in s_:
            continue
        fb = s_.get("faceted_body")
        body_txt = (f"FACETED_BREP {fb['faceted_faces']:,} 面、閉 {yes(fb['closed'])}、向き {yes(fb['winding_consistent'] and fb['outward'])}、"
                    f"{fb['volume_cm3']} cm³、{fb['bbox_mm'][0]}×{fb['bbox_mm'][1]}×{fb['bbox_mm'][2]}") if fb else "—"
        occ = (f"{s_['occt_solids']} ソリッド、全有効 {yes(s_['occt_all_valid'])}、{s_['occt_volume_cm3']} cm³、"
               f"{s_['occt_bbox_mm'][0]}×{s_['occt_bbox_mm'][1]}×{s_['occt_bbox_mm'][2]}、{s_['seconds']} 秒")
        srow.append(f"| `{s_['file']}` | {s_['size_MB']} | {body_txt} | {occ} |")
    strows = "\n".join(srow)
    gv = f"""# Geometry Validation

`src/validate_model.py` が **書き出したファイルをディスクから読み戻して** 機械的に検査した結果です
（`build/validation.json`）。ハード要件の不合格：**{len(val['failures'])} 件** {val['failures'] if val['failures'] else ''}

## 1. STL 検査

| ファイル | 三角形 | 外形 mm | ≤{val['limit_mm']:.0f} | 閉鎖 | 非多様体辺 | 自己交差面 | 法線/巻き方向 | シェル数 | 内部空洞 | 体積 cm³ |
|---|---|---|---|---|---|---|---|---|---|---|
{srows}

* シェル数 > 1 のファイル（ピストン8個、ロゴ8文字、ピン4本など）は **意図的に複数部品を並べたプレート** です。
  キャリパー本体系 STL はすべて 1 シェル（浮遊ボディなし）。
* 自己交差：pymeshlab `compute_selection_by_self_intersections_per_face`（MeshLab 内部は float32 演算）。非多様体辺：辺共有数 ≠ 2 の辺。
* `FDM_Optimized` はベッド座標へ移動した際、ロゴポケット壁の「ほぼ同一平面」の 2 面が float32 精度で接触したため、
  `export_stl.py` がその 2 面を削除して穴を再三角形化し、閉鎖と自己交差 0 を再確認しています（形状差 < 0.001 mm）。
* 内部空洞：符号付き体積が負の閉シェル。法線：全体の符号付き体積 > 0 かつ巻き方向一貫。
* 等値面抽出直後の微小浮遊片（数値ノイズ）は生成時に除去：{meta['dropped_crumbs']}

## 2. 肉厚（レイキャスト：表面上の 2 万点から内向き法線方向に次の面までの距離）

| ファイル | 肉厚分布 mm |
|---|---|
{trs}

* 主要外殻・リブの肉厚は P1 ≈ 2.6 mm、中央値 ≈ 10 mm。目安（主要外殻 ≥ 2.0、リブ ≥ 1.6、細部 ≥ 0.8）に対し、
  0.8 mm 未満は全サンプルの 0.1 % 前後で、穴（パッドピン穴・ボルト穴・ブリーダー穴）の縁やロゴポケット縁など
  **鋭いエッジにレイが斜めに入る箇所** に限られます（局所的な壁ではなくエッジ効果）。
* 当初の検査で見つかったパッドピン穴まわりの 0.8〜1.3 mm 壁は、ボス追加（肉厚 ≥ 2.4 mm）で解消済み。
* 2分割版インボード側はピン穴（Ø{sp['pin_hole_diameter']}）の縁のサンプルを含みます。

## 3. 組立干渉

| 項目 | 結果 |
|---|---|
| 本体のローター内部侵入点 | **{rot['body_points_inside_rotor']}** |
| 本体 ⇔ ローター最小距離 | {rot['min_clearance_mm']} mm |
| 半径方向（ローター外周） | {rot['radial_clearance_over_rotor_edge_mm']} mm（設定 {p['clearance']['rotor_radial']}） |
| 軸方向（摺動面、パッド除く） | {rot['axial_clearance_to_rotor_face_mm']} mm（設定 {p['clearance']['rotor_axial']}） |
| ホイールバレル（ID {whl['barrel_min_inner_diameter_assumed']:.0f} 仮定） | {whl['barrel_clearance_mm']} mm → {yes(whl['barrel_ok'])}；必要最小 ID（3 mm）= {whl['required_min_barrel_ID_for_3mm']} |
| スポーク（裏面 Y={whl['spoke_back_y_assumed']:.0f} 仮定） | {whl['spoke_axial_clearance_mm']} mm → {yes(whl['spoke_ok'])} |

## 4. STEP 読み戻し（OpenCASCADE `STEPControl_Reader`）

| ファイル | MB | ファセット本体（STEP エンティティから再構築） | ファイル全体の OpenCASCADE 読込 |
|---|---|---|---|
{strows}

ソリッド数：Master 17 = 本体 1 + ピストン 8 + パッド 2 + パッドピン 2 + マウントボルト 2 + ブリーダー 2、
Assembly はローター 1 を追加、18inch はさらにホイール 1 を追加。

（注：開発途中、間引き後の本体に非多様体辺が 7 本残っていた版では、OCCT が形状修復を行うため読込に 25 分以上
かかりました。manifold 化して閉ソリッドを保証してからは 30 秒程度で読めます。）

### STEP の制限事項（重要）

* キャリパー本体は陰関数（SDF）でモデリングした有機形状のため、STEP では **ファセット B-Rep
  （`FACETED_BREP`、平面三角形 6 万面）** として格納しています。閉じた有効ソリッドで、計測・断面・干渉確認・
  ブーリアンに使えますが、NURBS 曲面ではないためフィレット編集や曲面の再パラメータ化はできません。
* 高精細形状の正は STL（約 70 万三角形、等値面 0.4 mm ボクセルから適応間引き）です。STEP 本体は
  ファイルサイズ（約 20 MB）とのバランスで 6 万面に間引いています（`generate_caliper.py` の `decimate(..., 60000)` で変更可）。
* ピストン・パッド・ピン・ボルト・ブリーダー・ローター・ホイールは CadQuery による **厳密 B-Rep**（NURBS/解析曲面）です。
* 組立 STEP は部品ごとに独立したルートプロダクトとして格納（XCAF 色・名称つき部品ファイルと結合）。
"""
    open(os.path.join(REP, "Geometry_Validation.md"), "w").write(gv)
    print("reports written")


if __name__ == "__main__":
    main()
