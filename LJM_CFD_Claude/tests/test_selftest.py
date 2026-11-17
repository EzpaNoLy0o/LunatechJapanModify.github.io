"""動作確認 (``python -m ljmcfd --self-test``) と、配布版のジョブ用実行ファイルの選択。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ljmcfd import selftest
from ljmcfd.core.jobs import frozen_worker, job_command, job_environment
from ljmcfd.solver.environment import native_openfoam_available


def test_selftest_runs_the_real_pipeline_in_a_subprocess(tmp_path):
    """GUI を含めて別プロセスで実行する。OpenFOAM があれば短い解析まで通す。"""
    env = job_environment()
    env["LJMCFD_HOME"] = str(tmp_path / "home")
    has_display = sys.platform == "win32" or bool(os.environ.get("DISPLAY"))
    command = [sys.executable, "-m", "ljmcfd", "--self-test"] + ([] if has_display else ["--no-gui"])
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=900)
    output = process.stdout
    assert process.returncode == 0, output[-3000:] + process.stderr[-2000:]
    for name in ("ライブラリの読み込み", "作業フォルダへの書き込み", "STEP 読み込みジョブ (子プロセス)", "メッシュ生成ジョブ (子プロセス)"):
        assert f"[OK] {name}" in output
    assert "[失敗]" not in output
    if native_openfoam_available():
        assert "[OK] OpenFOAM 環境" in output and "[OK] CFD 解析 30 反復 (子プロセス)" in output
        assert "[OK] 結果の可視化データ" in output and "収束した値ではありません" in output
        assert "[OK] ボリューム煙の密度場" in output and "車体の内部に煙なし" in output
        if has_display:
            assert "ボリューム煙を描画" in output
        assert "結果: すべて成功" in output if has_display else "スキップ 1 件" in output
    elif sys.platform != "win32":
        # Windows では WSL に OpenFOAM があれば解析まで実行され、無ければスキップになる (どちらも正しい)
        assert "[スキップ] OpenFOAM 環境" in output and "[スキップ] CFD 解析" in output
    if has_display:
        assert "[OK] GUI と 3D 描画" in output
    saved = (tmp_path / "home" / "logs" / "selftest.txt").read_text(encoding="utf-8")
    assert "動作確認" in saved and "[OK] メッシュ生成ジョブ" in saved


def test_selftest_without_openfoam_skips_the_analysis_but_passes(tmp_path, monkeypatch, capsys):
    """OpenFOAM が無い環境では、解析の項目は「スキップ」であって失敗ではない。"""
    from ljmcfd.solver import environment

    monkeypatch.setenv("LJMCFD_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(environment, "default_environment",
                        lambda: environment.FoamEnvironment(kind=environment.KIND_NATIVE, bashrc="/nonexistent/openfoam/etc/bashrc"))
    assert selftest.run_selftest(["--no-gui"]) == 0
    output = capsys.readouterr().out
    assert "[OK] メッシュ生成ジョブ (子プロセス)" in output and "[失敗]" not in output
    assert "[スキップ] OpenFOAM 環境" in output and "OpenFOAM を使えません" in output
    assert "[スキップ] CFD 解析 30 反復 (子プロセス)" in output and "[スキップ] 結果の可視化データ" in output
    assert "[スキップ] ボリューム煙の密度場" in output
    assert "実行した項目はすべて成功 (スキップ 5 件" in output


def test_selftest_reports_failures_and_skips_dependent_steps(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LJMCFD_HOME", str(tmp_path / "home"))

    def broken(path, size_mm=None):
        raise RuntimeError("STEP を書けません (テスト用の故障)")

    monkeypatch.setattr(selftest, "write_box_step", broken)
    assert selftest.run_selftest(["--no-gui", "--no-solver"]) == 1
    output = capsys.readouterr().out
    assert "[失敗] STEP 読み込みジョブ (子プロセス)" in output and "テスト用の故障" in output
    assert "[スキップ] メッシュ生成ジョブ (子プロセス)" in output and "[スキップ] GUI と 3D 描画" in output
    assert "結果: 失敗 1 件 — STEP 読み込みジョブ (子プロセス)" in output
    assert "失敗 1 件" in (tmp_path / "home" / "logs" / "selftest.txt").read_text(encoding="utf-8")


def test_frozen_build_uses_the_console_worker_for_jobs(tmp_path, monkeypatch):
    gui = tmp_path / "LJM_CFD_Claude.exe"
    gui.write_bytes(b"")
    assert frozen_worker(str(gui)) == str(gui)                     # ワーカーが無ければ GUI の実行ファイルで代用
    worker = tmp_path / "ljmcfd-worker.exe"
    worker.write_bytes(b"")
    assert frozen_worker(str(gui)) == str(worker)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(gui))
    assert job_command("cad-import", "args.json") == [str(worker), "--job", "cad-import", "args.json"]
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    assert job_command("cad-import", "args.json")[1:] == ["-m", "ljmcfd", "--job", "cad-import", "args.json"]


def test_packaging_files_are_consistent():
    """配布定義が、実際に存在する入口スクリプトとジョブ用実行ファイル名を指している。"""
    root = Path(__file__).resolve().parents[1]
    spec = (root / "packaging" / "ljmcfd.spec").read_text(encoding="utf-8")
    assert (root / "packaging" / "launcher.py").is_file() and '"launcher.py"' in spec
    assert 'name="ljmcfd-worker"' in spec and 'name="LJM_CFD_Claude"' in spec
    build = (root / "scripts" / "build_windows.bat").read_bytes()
    assert b"packaging\\ljmcfd.spec" in build and b"ljmcfd-worker.exe" in build and b"\r\n" in build
    assert build.decode("ascii")                                    # バッチファイルは ASCII のみ (文字コードの問題を避ける)
