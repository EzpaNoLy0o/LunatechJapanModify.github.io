"""エントリポイント。

* ``python -m ljmcfd``                      GUI を起動
* ``python -m ljmcfd --job <name> <json>``  ジョブ (子プロセス) を実行 — GUI が内部で使用
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--job":
        if len(argv) != 3:
            sys.stderr.write("usage: python -m ljmcfd --job <name> <args.json>\n")
            return 64
        # ジョブ側では Qt を import しない (起動を軽くし、GUI 依存を持ち込まない)
        from .core.jobs import run_job_cli

        return run_job_cli(argv[1], argv[2])
    from .app import run_gui

    return run_gui(argv)


if __name__ == "__main__":
    sys.exit(main())
