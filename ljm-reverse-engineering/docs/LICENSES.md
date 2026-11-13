# ライセンス

## 本ソフトウェア

LJM Reverse Engineering 本体のライセンスは**未定**です（権利者が決定してください）。
本ソフトのソースコードは市販ソフト（Revo Design、Geomagic Design X 等）のコード・UI 資産を流用せず、
下記のオープンソースライブラリと独自コードで作成しています。

## 使用ライブラリ

| ライブラリ | ライセンス | 用途 | 配布時の注意 |
|---|---|---|---|
| Python 3.12 | PSF License | 実行環境 | |
| PySide6 / Shiboken6 (Qt 6) | **LGPL-3.0**（または GPL） | GUI | 動的リンクで利用。EXE は **onedir** 形式で Qt の DLL を差し替え可能な状態で同梱する。Qt のライセンス文書を同梱し、改変していない旨と入手先（https://www.qt.io/ , https://pypi.org/project/PySide6/ ）を明記する |
| VTK | BSD-3-Clause | 3D 描画 | 著作権表示を同梱 |
| PyVista / PyVistaQt | MIT | VTK ラッパー | 著作権表示を同梱 |
| NumPy / SciPy | BSD-3-Clause | 数値計算 | SciPy wheel は OpenBLAS 等（BSD）を含む |
| Matplotlib | Matplotlib License (PSF 系) | カラーマップ（PyVista 依存） | |
| Trimesh | MIT | メッシュ I/O・処理 | |
| Shapely | BSD-3-Clause | 2D 幾何（押し出し用サンプル・回転体の半断面） | 同梱の GEOS は **LGPL-2.1** |
| mapbox-earcut | ISC | 多角形三角形分割 | |
| Rtree | MIT（libspatialindex: MIT） | 空間索引（trimesh） | |
| NetworkX | BSD-3-Clause | グラフ（trimesh） | |
| ezdxf | MIT | DXF 出力 | |
| cadquery-ocp (OCP) | Apache-2.0 | OpenCASCADE バインディング | Apache-2.0 の NOTICE を同梱 |
| Open CASCADE Technology (OCCT) | **LGPL-2.1 + 追加例外** | CAD カーネル | 動的リンク（DLL）で同梱。差し替え可能な形で配布し、ライセンス文書を同梱する |
| Open3D（任意） | MIT | 点群処理（任意機能） | 既定の EXE には同梱しない |
| PyInstaller（ビルド時のみ） | GPL-2.0 + ブートローダ例外 | EXE 作成 | 生成した EXE の配布は例外により制限されない |
| pytest / pytest-qt（開発時のみ） | MIT | テスト | 配布物に含めない |

## EXE 同梱について（判断）

- 上表のライブラリはすべて商用・非商用を問わず同梱して配布可能なライセンスです。
- LGPL のライブラリ（Qt/PySide6, OCCT, GEOS）は **onedir 形式**（DLL を個別ファイルとして同梱）にすることで、
  利用者がライブラリを差し替えられる要件を満たします（onefile 形式は推奨しません）。
- 配布時は `dist/LJM_ReverseEngineering/` に各ライブラリのライセンス文書（各 `*.dist-info/LICENSE*`）を含めてください。
  PyInstaller は多くの場合 dist-info を同梱しませんので、配布前に `licenses/` フォルダへ集めることを推奨します
  （`pip-licenses --with-license-file --format=plain-vertical` 等で一括取得可能）。
- アプリ内の「設定 > ライブラリとライセンス」で、インストールされている各ライブラリのバージョンとライセンスを確認できます。
