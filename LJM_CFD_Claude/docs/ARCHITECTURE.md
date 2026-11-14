# LJM CFD (Claude Edition) — アーキテクチャ設計

Windows 11 で動作する自動車専用 CFD ワークベンチの全体設計です。
Stage 1〜6 を通して変わらない骨格と、各ステージで追加するモジュールを定めます。

- **実装済み**: Stage 1（GUI・3D 表示・CAD 読み込み・モデル操作・形状検査）、Stage 2（仮想風洞・メッシュ生成・品質評価・polyMesh 変換）
- **未実装**: Stage 3〜6（本書の該当節は「設計」であり、動作するコードはまだありません）

---

## 1. 設計方針

| 方針 | 具体策 |
|---|---|
| 計算と可視化の分離 | 重い処理はすべて **子プロセス（ジョブ）** で実行し、結果はファイルで受け渡す。GUI は結果ファイルを読むだけ |
| GUI を止めない | CAD 変換・メッシュ生成・ソルバー起動は `QProcess`。OpenCASCADE は GIL を解放しないため、スレッドではなくプロセスを使う |
| 偽の結果を出さない | 値が無いものは「未計算」「計算失敗」と表示する。既定値・推定値で埋めない |
| エラーを握りつぶさない | 想定内のエラーは `JobError`（日本語メッセージ）、想定外はトレースバックごとログへ。未捕捉例外は `sys.excepthook` で記録 |
| Windows ⇔ WSL のパス問題を作らない | パス変換は `core/paths.py` に一元化。作業フォルダは ASCII パス（`C:\LJM_CFD_Claude\workspace`）に固定 |
| 拡張しやすさ | ジョブはレジストリ登録制、プロジェクト JSON は未知キーを保持、メッシャー／ソルバーはバックエンド差し替え可能な構成 |

## 2. プロセス構成

```
┌──────────────────────── Windows 11 ────────────────────────┐
│  GUI プロセス (PySide6 + PyVista/VTK)                       │
│   ├ Scene（データモデル）─ Viewport（3D）─ 各パネル           │
│   └ JobManager ──QProcess──┐                                │
│                            ▼                                │
│  ジョブ子プロセス  python -m ljmcfd --job <name> <args.json> │
│   ├ cad-import   STEP/IGES/STL → キャッシュ   [Stage 1 実装済] │
│   ├ cad-inspect  形状検査                     [Stage 1 実装済] │
│   ├ frontal-area   前面投影面積               [Stage 2 実装済] │
│   ├ mesh-generate  Gmsh → polyMesh            [Stage 2 実装済] │
│   └ solver-run   wsl.exe 経由で OpenFOAM を監視 [Stage 3]     │
│                            │ wsl.exe -d Ubuntu-24.04 -- bash │
└────────────────────────────┼───────────────────────────────┘
                             ▼
┌──────────────── WSL2 / Ubuntu 24.04 ───────────────────────┐
│  OpenFOAM Foundation 14（foamRun / foamToVTK）  [Stage 3]   │
└────────────────────────────────────────────────────────────┘
```

### ジョブ・プロトコル（`core/jobs.py`）

子プロセスは標準出力へ `@@LJMJOB@@ {json}` の行を出します。

| type | 内容 |
|---|---|
| `progress` | `value`（0〜1）と `message` |
| `log` | ログ 1 行 |
| `result` | 成功時の結果（最後に 1 回） |
| `error` | 失敗時のメッセージと詳細 |

それ以外の行（OpenCASCADE や OpenFOAM の生出力）は通常ログとして扱います。
ジョブ関数は `func(args: dict, ctx: JobContext) -> dict` という素の Python 関数なので、
テストからはプロセスを起こさず直接呼べます。新しいジョブは `JOB_REGISTRY` に 1 行足すだけです。

## 3. ディレクトリ構成

```
C:\LJM_CFD_Claude\                  ← 設置先（既存の LJM CFD とは別フォルダ）
├ run_ljm_cfd.bat                   起動（コンソールなし）
├ run_ljm_cfd_debug.bat             起動（コンソールあり・エラー確認用）
├ requirements.txt / pyproject.toml
├ scripts\                          セットアップ・テスト・デモモデル生成
├ docs\ARCHITECTURE.md
├ src\ljmcfd\
│  ├ __main__.py                    GUI / ジョブ の振り分け
│  ├ app.py                         GUI 起動
│  ├ core\                          Qt 非依存の共通基盤
│  │  ├ units.py                    単位換算（内部は常に SI）
│  │  ├ pose.py                     姿勢・変換行列
│  │  ├ paths.py                    Windows⇔WSL パス変換、作業フォルダ
│  │  ├ jobs.py                     ジョブ・プロトコル
│  │  ├ project.py                  プロジェクト JSON
│  │  └ logging_setup.py
│  ├ cad\                           CAD（Qt 非依存）
│  │  ├ occ_io.py                   STEP/IGES 読み込み・テッセレーション（OCP）
│  │  ├ surface.py                  表示メッシュ・STL 読み込み・凸包
│  │  ├ unit_detect.py              ファイル内の単位検出
│  │  ├ import_job.py               インポートジョブ・キャッシュ
│  │  ├ inspect.py                  形状検査
│  │  └ assembly.py                 Part / Assembly（2 段変換）
│  ├ gui\                           PySide6
│  │  ├ main_window.py  viewport.py  panels.py
│  │  ├ scene.py                    選択状態とシグナル
│  │  ├ job_runner.py               QProcess ラッパー
│  │  └ theme.py                    ダークテーマ
│  ├ tunnel\                        仮想風洞
│  │  ├ spec.py                     流れの条件・風洞寸法・検証・y+ 概算
│  │  └ frontal_area.py             前面投影面積 (画像への塗りつぶし)
│  ├ mesh\                          メッシュ
│  │  ├ settings.py                 設定・プリセット・入力ハッシュ
│  │  ├ request.py                  シーン → ジョブ引数
│  │  ├ gmsh_job.py                 Gmsh による生成ジョブ
│  │  ├ polymesh.py                 OpenFOAM polyMesh の組み立て・書き出し
│  │  └ quality.py                  非直交性・歪度などの品質評価
│  ├ solver\      [Stage 3]         ケース生成・WSL 実行・残差／荷重の解析
│  └ post\        [Stage 4-5]       結果読み込み・断面・流線・粒子アニメーション
├ tests\
├ workspace\                        実行時に生成（キャッシュ・ログ・設定）※Git 管理外
└ projects\<名前>\                  解析プロジェクト ※Git 管理外
   ├ project.ljmcfd.json            設定のみ（数十 kB）
   ├ mesh\      [Stage 2]           メッシュ本体
   ├ case\      [Stage 3]           OpenFOAM ケース
   └ results\   [Stage 3-5]         VTK・荷重履歴・アニメーション用キャッシュ
```

## 4. 座標系と規約（全ステージ共通）

| 項目 | 規約 |
|---|---|
| 風向 | 常にグローバル **+X**。相対風向は車両側の姿勢で与える |
| 上方向・地面 | **+Z** が上、地面は **Z = 0** |
| 車両の向き | ノーズを **−X**（上流）へ向ける。車両左側面は −Y |
| 内部単位 | 長さ m、速度 m/s、圧力 Pa（SI）。表示単位（mm / m）は書式だけに影響 |
| 回転 | 右手系・度。`R = Rz(yaw) · Ry(pitch) · Rx(roll)` |
| 荷重の符号（Stage 3） | Drag = +X 方向の力、Lift = +Z 方向の力、**Downforce = −Lift**。GUI では両方を別の行で表示する |

### 2 段階の変換

```
x_world = A · P_i · (s_i · x_local)
```

- `s_i` … パーツのファイル単位 → m
- `P_i` … パーツ個別の配置（回転中心 = パーツ自身の境界ボックス中心）
- `A`  … 車両全体の姿勢（回転中心 = 全体境界の XY 中心・最下点）

VTK アクターの変換行列だけを書き換えるため、移動・回転でメッシュを作り直しません。
Stage 2 のメッシュ生成では同じ行列を B-Rep / STL へ適用して風洞内に配置します。

## 5. データの流れ

```
CAD ファイル ─(cad-import ジョブ)→ workspace\cache\cad\<key>\
                                    ├ meta.json          パーツ一覧・単位・統計
                                    ├ part_NNN.vtp       表示メッシュ（面番号つき）
                                    ├ part_NNN_edges.vtp 面輪郭線
                                    ├ part_NNN_hull.npy  凸包頂点（境界計算用）
                                    └ part_NNN.brep      B-Rep（Stage 2 の入力）
                                          │
            [Stage 2] mesh-generate ジョブ ▼
                       projects\<名前>\mesh\   メッシュ + 品質レポート(JSON)
                                          │
            [Stage 3] solver-run ジョブ    ▼
                       projects\<名前>\case\   OpenFOAM ケース（WSL で実行）
                       projects\<名前>\results\ forces.csv / residuals.csv / *.vtu
                                          │
            [Stage 4-5] GUI（post\）       ▼
                       断面・流線・粒子 … 読み込んだ結果から GUI プロセス内で生成（再計算なし）
```

キャッシュのキーは「元ファイルの絶対パス・サイズ・更新時刻・品質・インポータ版」です。
同じファイルは即座に開き、ファイルが更新されれば自動で再変換されます。

### 単位の扱い

- **STEP / IGES**: OpenCASCADE が読み込み時に mm へ換算。元ファイルの単位は別途検出して表示（情報）。
- **STL**: 単位を持たないため寸法から推定し、**「推定」であることを明示**。GUI で変更可能。

## 6. ステージ別の設計

### Stage 1 — 実装済み
本書 2〜5 節のとおり。詳細は README を参照。

### Stage 2 — 実装済み

- **風洞** (`tunnel/spec.py`): モデルのワールド境界から計算領域を決める。地面は常に Z = 0。
  解析モードごとの既定値（車両: 前 3L / 後 6L / 左右 3W / 上 4H、地面あり。単体: 各方向 3D〜6D、地面なし）。
  Internal Flow は定義のみで、選択不可・検証でエラーになる。
- **前面投影面積** (`tunnel/frontal_area.py`): 三角形を YZ 平面へ投影して高解像度画像へ塗りつぶし、重なりを除いた面積を求める。
  閉塞率と Cd・Cl の基準面積に使う。未計算の間は閉塞率も基準面積も「未計算」。
- **メッシュ生成** (`mesh/gmsh_job.py`):
  1. キャッシュ B-Rep を OCC カーネルへ読み込み、相似変換（拡大・回転・移動）で配置。
     アフィン変換は曲面を NURBS 化して精度と速度を落とすため使わない。
  2. 風洞ボックスからモデルを差し引いて流体領域を作る（地面より下・パーツの重なりは自動的に処理される）。
  3. 外周面は位置で分類、残りはサンプル点から各パーツ表面への距離でパーツへ対応付け、パーツごとの壁面パッチにする。
  4. サイズ場 = 表面からの距離（パーツ別倍率）・接地線まわり・後流ボックスの最小値 + 曲率。
  5. 四面体は並列 HXT、失敗時は逐次 Delaunay へ自動で切り替え。
- **接地の扱い**: 最下点が Z ≈ 0 のモデルは `contact_sink`（既定 10 mm）だけ沈めて有限の接地面を作る。
  半径 330 mm のタイヤでは 5 mm だと接地端が約 10° のくさびになり品質警告が残る。10 mm で非直交性 70° 未満になる。
- **polyMesh 変換** (`mesh/polymesh.py`): 自前実装。面の対応付けは節点集合の辞書式ソートで行い、
  OpenFOAM の順序規約（内部面は owner → neighbour の昇順、法線は owner → neighbour、境界面はパッチごと）に従う。
  四面体・プリズム・ピラミッド・六面体に対応。
- **品質評価** (`mesh/quality.py`): OpenFOAM の checkMesh と同じ定義の非直交性・歪度、負体積、開いたセル、
  外周の閉じ具合、CAD の流体体積との差。**OpenFOAM 実機の checkMesh との突き合わせは Stage 3 で行う。**
- **境界層プリズム（実験的）**: Gmsh の `extrudeBoundaryLayer`。モデルが地面・壁に接していない場合のみ。
  面の法線が流体側を向いているかを実際に調べて押し出し方向を決め、生成後に
  「プリズム数 = 層数 × 壁面の面数」「メッシュ体積と CAD 体積の差がプリズム体積の半分未満」を検証する
  （内側へ押し出されると体積がプリズム体積の 2 倍増えるため検出できる）。
  条件を満たさなければジョブを失敗させ、境界層なしのメッシュへ黙って切り替えない。
- **メッシュの鮮度**: 形状・姿勢・風洞・設定から入力ハッシュを作り、レポートに記録。現在の状態と一致しなければ「再生成が必要」。
- **未対応**: STL パーツの体積メッシュ、半分モデル（対称面）、接地形状の境界層。
  `snappyHexMesh` バックエンドは OpenFOAM が必要なため Stage 3 以降で検討する。

### Stage 3 — OpenFOAM 連携（設計）
- `solver/wsl.py`: `wsl.exe -d <distro> -- bash -lc "..."` の起動、ディストリビューションと OpenFOAM の検出。
  ケースは WSL 側の ext4（`~/ljmcfd_runs/…`）で実行し結果だけを Windows 側へ戻す方式と、`/mnt/c` 直接実行方式を比較して決定。
- `solver/case_builder.py`: 定常・非圧縮 RANS、k-ω SST。Foundation 版のモジュラーソルバー（`foamRun` + `incompressibleFluid`）を想定。
  **辞書の正確な書式は実機の OpenFOAM 14 で確認してから確定する**（本書の時点では未検証）。
  安定化: 有界な 2 次風上、勾配リミッタ、控えめな緩和係数、初期数十反復の 1 次精度立ち上げ。
- `solver/monitor.py`: 残差・`forces` / `forceCoeffs` の履歴を逐次解析して GUI へ送る。停止要求にも対応。
- `solver/convergence.py`: 「計算完了」と「工学的に収束」を別判定。
  残差の低下量、**直近区間の荷重変動幅**、質量収支（流入出の差）で評価し、満たさなければ「未収束」と表示。
- 荷重: 圧力成分と粘性成分を分けて取得。Cd・Cl は `0.5·ρ·U²·A_ref` で無次元化し、基準面積は GUI で確認・変更可能。

### Stage 4 — 結果可視化（設計）
- `post/result_set.py`: VTK 結果の読み込みと派生量（|U|、Cp、渦度、Q 値）。
- 表示モード: Velocity / Pressure / Cp / Streamlines / Surface Pressure / Vorticity / Q-Criterion / k / ISO Surface / Smoke。
- カラースケールの最小・最大、透明度、線幅、流線本数、粒子密度を GUI から指定。

### Stage 5 — スライダー操作（設計・今回の最重要機能）
- **断面スライダー**: `vtkPlaneCutter` とセルロケータを使い、読み込み済み結果から断面を抽出。
  ドラッグ中は間引きデータで即時更新、停止後に高精細へ差し替え。CFD は再実行しない。
- **流線シード・スライダー**: シード面の X/Y/Z 位置・サイズ・本数・間隔を変更。初期位置は車両前面。
- **粒子・スモーク**: 定常速度場で流跡線を事前積分し、時刻ごとの粒子位置を配列としてキャッシュ。
  タイムラインは配列の添字を動かすだけなので前後どちらにも即座に移動できる。
  **「定常 RANS の速度場に沿った粒子軌跡であり、非定常解析ではない」旨を画面に常時表示。**
  将来の非定常対応に備え、`VelocityFieldProvider`（時刻 → 速度場）を抽象化する。
- 断面用とタイムライン用のスライダーは下部パネルに別々に配置する。

### Stage 6 — 保存・出力・配布（設計）
- プロジェクト JSON の `tunnel` / `mesh` / `solver` / `post` 節（枠は Stage 1 で確保済み）を各ステージが埋める。
- 結果の CSV / レポート出力、動画書き出し、PyInstaller による配布形態。

## 7. 精度に関する取り決め

1. 計算していない値・取得に失敗した値は表示しない（「未計算」「計算失敗」）。
2. 指定反復数の終了は「計算完了」であって「収束」ではない。収束判定は残差・荷重変動・質量収支で別に行う。
3. 境界層メッシュの有無と解像度を結果に必ず併記する。
4. 閉塞率・風洞余白・メッシュ密度など、結果の信頼性に影響する条件をレポートへ残す。
5. 検証形状として Ahmed ボディ（`scripts/make_demo_models.py`）を用意し、Stage 3 以降で文献値と比較する。

## 8. テスト方針

| 種類 | 内容 |
|---|---|
| 単体テスト | 単位・姿勢・パス変換・プロジェクト JSON・単位検出 |
| 処理テスト | OCP で生成した試験形状（直方体・アセンブリ・開いたシェル・IGES・STL）を実際に読み込み検査 |
| ジョブ経路 | GUI と同じ `python -m ljmcfd --job …` を子プロセスとして実行 |
| GUI 結合テスト | 実ウィンドウを開き、読み込み → クリック選択 → 面選択 → 計測 → 姿勢変更 → 検査 → 保存 → 再読み込み |

各ステージで「試験形状に対して実際に処理が通ること」を確認し、未確認の環境（特に Windows 実機）は未確認と報告します。
