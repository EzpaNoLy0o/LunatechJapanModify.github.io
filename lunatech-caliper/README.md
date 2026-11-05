# Lunatech Titanium Caliper Replica

2018年の Bugatti 3D プリント製チタン8ピストンキャリパーを参考にした、FDM 展示用レプリカ
（ロゴは **Lunatech**）。355 × 32 mm ローター・18インチホイールの実寸参照モデル付き。
全長約 309 mm、各軸 320 mm 以下。QIDI MAX3（325³ mm）用。

> 展示・寸法確認用の模型です。油圧保持・制動・耐圧の機能はなく、実車には使用できません。

## フォルダ

| フォルダ | 内容 |
|---|---|
| `src/` | 再生成可能な Python CAD ソース（`parameters.json` が全寸法のマスター） |
| `cad/` | STEP：`Lunatech_Caliper_Master` / `_Assembly`（+ローター）/ `_18inch`（+ホイール）、`Rotor_355x32_Reference`、`Wheel_18inch_Reference` |
| `stl/` | 印刷用 STL（一体・FDM 最適姿勢・2分割＋ピン・ロゴ・ピストン・パッド・金物・ローター4分割） |
| `renders/` | 正面 / 背面 / 上面 / 底面 / 左右端 / 斜め前後 / ローター組立 / ホイール適合 / ロゴ詳細 |
| `reports/` | `Dimensions.md`、`Print_Settings.md`、`Geometry_Validation.md`、`Reference_Comparison.md` |
| `fonts/` | Inter Display SemiBold（SIL OFL 1.1） |

## 再生成

```bash
pip install -r requirements.txt
./run_all.sh            # 約 5〜10 分（STEP の読み戻し検査を含む）
```

| スクリプト | 役割 |
|---|---|
| `src/generate_caliper.py` | 陰関数（SDF）ボディ → 0.4 mm ボクセル等値面 → 自動サイズゲート（>320 mm なら自動短縮）→ 適応間引き → manifold 化 → ロゴのブーリアン → STEP |
| `src/export_stl.py` | STL 書き出し、造形姿勢 5 案の定量比較、Y=0 分割と位置決めピンの自動配置 |
| `src/validate_model.py` | 全ファイルを読み戻して検査：寸法、閉鎖、非多様体、自己交差、法線、肉厚、浮遊ボディ、内部空洞、ローター/ホイール干渉、STEP 有効性 |
| `src/render_views.py` | ソフトウェアレンダラ（numba 製 z バッファ）による各視点画像 |
| `src/make_reports.py` | 検査結果からレポートを生成 |
| `src/geometry.py` / `sdf_core.py` / `logo.py` / `parts.py` | 形状レイアウト / SDF カーネル / 曲面ロゴ / 厳密 B-Rep 部品 |

## モデリング方式

* **本体**：Catmull-Rom スプラインに沿った可変半径リブ、ローター円弧に沿った可変断面スイープ梁、ピストンローブ、
  穴あき膜、角丸多角形窓を多項式スムーズユニオン／サブトラクションで結合した陰関数モデル。
  接合部はすべて連続した曲面で、角が残らない。
* **ロゴ**：フォント輪郭 → 曲げたプリズム ∩ ブリッジ曲面のオフセットスラブ（厳密ブーリアン）。
  凸 0.8 mm、曲面追従、別色差し込み用ポケット＋インサートも出力。
* **部品**：ピストン、パッド、ピン、ボルト、ブリーダー、ローター、ホイールは CadQuery による厳密 B-Rep。
* **STEP 本体**：有機形状はファセット B-Rep（FACETED_BREP、有効な閉ソリッド）として格納。制限事項は
  `reports/Geometry_Validation.md` を参照。
