import glob
import shutil
import os
from pathlib import Path


if __name__ == '__main__':
    """
    300W_LP matファイルを一つのフォルダに集約
    """
    # matファルダ用を作成
    mat_fold_path = './dataset/mat/'
    os.makedirs(mat_fold_path, exist_ok=True)
    # Pathオブジェクトを生成
    p = Path("../dataset/300W_LP/")
    path_ls = list(p.glob("**/*.mat"))

    for path in path_ls:
        shutil.copy(path, mat_fold_path)
