import os
import requests
import tarfile
from pathlib import Path
from GDesigner.utils.const import GDesigner_ROOT


def download():
    """
    下载 MMLU 数据集
    
    数据集将下载到: GDesigner_ROOT/datasets/MMLU/data/
    """
    # 使用 GDesigner_ROOT 确保路径一致性
    mmlu_dir = Path(GDesigner_ROOT) / "datasets" / "MMLU"
    mmlu_dir.mkdir(parents=True, exist_ok=True)
    
    tar_path = mmlu_dir / "data.tar"
    data_path = mmlu_dir / "data"
    
    # 下载 tar 文件（如果不存在）
    if not tar_path.exists():
        url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
        print(f"Downloading {url}")
        try:
            r = requests.get(url, allow_redirects=True, timeout=300)
            r.raise_for_status()
            with open(tar_path, 'wb') as f:
                f.write(r.content)
            print(f"Saved to {tar_path}")
        except requests.RequestException as e:
            print(f"Error downloading dataset: {e}")
            raise

    # 解压 tar 文件（如果数据目录不存在）
    if not data_path.exists() or not any(data_path.iterdir()):
        if not tar_path.exists():
            raise FileNotFoundError(f"Tar file not found: {tar_path}")
        print(f"Extracting {tar_path}...")
        try:
            with tarfile.open(tar_path) as tar:
                tar.extractall(mmlu_dir)
            print(f"Extracted to {data_path}")
        except tarfile.TarError as e:
            print(f"Error extracting tar file: {e}")
            raise


if __name__ == "__main__":
    download()
