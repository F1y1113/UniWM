#!/usr/bin/env bash
# download_data.sh — Download the UniWM dataset into data/.
# Usage: bash download_data.sh

set -e

mkdir -p data && cd data

for split in go_stanford recon sacson scand tartandrive; do
    echo "[DOWNLOAD] ${split}.tar"
    wget -c "https://huggingface.co/datasets/fly1113/UniWM_Dataset/resolve/main/${split}.tar"

    echo "[EXTRACT] ${split}.tar"
    tar -xf "${split}.tar"

    echo "[CLEAN] removing ${split}.tar"
    rm -f "${split}.tar"
done

echo "[DONE] Datasets are ready under $(pwd)/"
