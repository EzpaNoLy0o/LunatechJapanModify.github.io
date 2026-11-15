#!/bin/bash
# LJM CFD (Claude Edition) - OpenFOAM Foundation 14 を WSL2 の Ubuntu 24.04 へインストールする。
#
# 使い方 (Windows のコマンドプロンプトから):
#   wsl -d Ubuntu-24.04 -- bash /mnt/c/LJM_CFD_Claude/scripts/setup_wsl_openfoam.sh
#
# OpenFOAM Foundation の公式手順 (https://openfoam.org/download/14-ubuntu/) と同じ内容:
#   1. リポジトリの署名鍵を登録
#   2. リポジトリを追加
#   3. openfoam14 をインストール
# 途中で sudo のパスワードを求められます。
set -e

if [ "$(lsb_release -cs 2>/dev/null)" != "noble" ]; then
    echo "[注意] このスクリプトは Ubuntu 24.04 (noble) を想定しています。現在: $(lsb_release -ds 2>/dev/null)"
fi

if [ -f /opt/openfoam14/etc/bashrc ]; then
    echo "OpenFOAM 14 は既にインストールされています: /opt/openfoam14"
else
    echo "[1/3] 署名鍵を登録します"
    sudo sh -c "wget -q -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc"
    echo "[2/3] リポジトリを追加します"
    # パッケージ本体は http の配布ミラーへ転送されるため、公式手順どおり http のリポジトリを使う
    # (改ざんは上で登録した署名鍵で検出される)
    echo "deb http://dl.openfoam.org/ubuntu noble main" | sudo tee /etc/apt/sources.list.d/openfoam.list > /dev/null
    echo "[3/3] OpenFOAM 14 をインストールします (約 650 MB)"
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends openfoam14
fi

# 確認
source /opt/openfoam14/etc/bashrc
echo
echo "OpenFOAM version : $WM_PROJECT_VERSION"
echo "foamRun          : $(command -v foamRun)"
echo "mpirun           : $(command -v mpirun)"
echo "CPU cores        : $(nproc)"
echo
echo "完了しました。LJM CFD の「解析」タブで「環境を確認」を押してください。"
