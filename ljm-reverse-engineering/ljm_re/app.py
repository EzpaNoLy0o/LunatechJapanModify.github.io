"""アプリケーション起動."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback


def _excepthook_factory(app_getter):
    def hook(etype, value, tb):
        text = "".join(traceback.format_exception(etype, value, tb))
        logging.getLogger("ljm_re").error("未処理の例外:\n%s", text)
        try:
            from PySide6 import QtWidgets
            if QtWidgets.QApplication.instance() is not None:
                box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Critical, "予期しないエラー",
                                            f"{etype.__name__}: {value}\n\n詳細はログを確認してください。")
                box.setDetailedText(text)
                box.exec()
        except Exception:
            pass
    return hook


def create_app(argv=None):
    from PySide6 import QtCore, QtWidgets
    os.environ.setdefault("QT_API", "pyside6")
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv or sys.argv)
    app.setApplicationName("LJM Reverse Engineering")
    app.setOrganizationName("LunaTech Japan Modify")
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description="LJM Reverse Engineering")
    parser.add_argument("files", nargs="*", help="起動時に開くファイル (.ljmre / .stl / .obj / .ply)")
    parser.add_argument("--debug", action="store_true", help="デバッグログを出力")
    parser.add_argument("--self-test", metavar="DIR", nargs="?", const="", default=None,
                        help="自己診断（サンプル生成→読込→平面/円筒フィット→STEP出力と検証）を実行して終了")
    args = parser.parse_args(argv)

    from .utils.logging_setup import setup_logging
    log_path = setup_logging(logging.DEBUG if args.debug else logging.INFO)
    sys.excepthook = _excepthook_factory(None)

    app = create_app([sys.argv[0]])
    from .gui.theme import apply_theme, japanese_font
    from .utils.settings import AppSettings
    settings = AppSettings.load()
    apply_theme(app, settings.theme)
    app.setFont(japanese_font())

    from .gui.main_window import MainWindow
    win = MainWindow(settings)
    win.show()
    if args.self_test is not None:
        from PySide6 import QtCore
        rc = {"code": 1}

        def run_test():
            try:
                rc["code"] = self_test(win, args.self_test)
            except Exception:
                logging.getLogger("ljm_re").exception("自己診断に失敗")
                rc["code"] = 1
            win.doc.undo_stack.set_clean()
            app.exit(rc["code"])
        QtCore.QTimer.singleShot(500, run_test)
        app.exec()
        print("SELF-TEST", "OK" if rc["code"] == 0 else "FAILED")
        return rc["code"]
    logging.getLogger("ljm_re").info("ログファイル: %s", log_path)
    for f in args.files:
        if f.lower().endswith(".ljmre"):
            win.open_project(f)
        else:
            win.import_files([f])
    return app.exec()


def self_test(win, out_dir: str = "") -> int:
    """配布物（EXE）の動作確認用: 主要機能を一通り実行して結果を検証する."""
    import tempfile
    import numpy as np
    from .cad import occ_backend as occ
    from .cad import reconstruct as rc
    from .core import synthetic as S
    from .core.selection import region_grow
    from .io.mesh_io import export_mesh
    log = logging.getLogger("ljm_re.selftest")
    out = out_dir or tempfile.mkdtemp(prefix="ljm_selftest_")
    os.makedirs(out, exist_ok=True)
    mesh, truth = S.make_bracket()
    p = os.path.join(out, "自己診断_ブラケット.stl")
    export_mesh(p, mesh.vertices, mesh.faces)
    win.import_files([p], unit="mm", ask_unit=False)
    assert win.runner.wait(60000), "import timeout"
    m = win.doc.objects("mesh")[0]
    win.viewer.set_selection(m.id, region_grow(m.topology, m.topology.nearest_face([0, -30, 8]), 5.0), "replace")
    win.cmd_fit_plane()
    assert win.runner.wait(60000)
    pl = win.doc.objects("plane")[0]
    assert abs(abs(pl.normal[2]) - 1) < 1e-9 and abs(pl.origin[2] - 8) < 1e-6, "plane fit"
    c = m.topology.centroids
    wall = (np.hypot(c[:, 0] - 30, c[:, 1] - 25) < 8.5) & (c[:, 2] > 0.5) & (c[:, 2] < 7.5)
    win.viewer.set_selection(m.id, wall, "replace")
    win.cmd_fit_cylinder()
    assert win.runner.wait(60000)
    cyl = win.doc.objects("cylinder")[0]
    assert abs(cyl.diameter - 16) < 0.01, f"cylinder fit {cyl.diameter}"
    ok_cad = occ.available()
    if ok_cad:
        s, _ = rc.cylinder_to_shape(cyl, "solid")
        f, _ = rc.plane_to_face(pl)
        sp = os.path.join(out, "自己診断.step")
        occ.write_step([(s, "cyl"), (f, "plane")], sp)
        rep = occ.verify_step(sp, [s, f])
        assert rep["ok"], rep["issues"]
    from .model.project_io import save_project, load_project
    pp = os.path.join(out, "自己診断.ljmre")
    save_project(win.doc, pp)
    d2, _ = load_project(pp)
    assert len(d2.objects()) == len(win.doc.objects())
    log.info("自己診断 OK: 平面/円筒フィット, STEP=%s, 出力先 %s", "OK" if ok_cad else "OCPなし", out)
    print(f"self-test output: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
