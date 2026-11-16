"""配布用実行ファイルの入口。

同じスクリプトから 2 つの実行ファイルを作る (``ljmcfd.spec``):

* ``LJM_CFD_Claude``  … GUI (Windows ではコンソールなし)
* ``ljmcfd-worker``   … ジョブ用 (コンソールあり)。GUI が ``--job <名前> <引数.json>`` を付けて子プロセスとして起動し、
                        標準出力で進捗を受け取る。利用者が直接起動するものではない。
"""

import multiprocessing
import sys

from ljmcfd.__main__ import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
