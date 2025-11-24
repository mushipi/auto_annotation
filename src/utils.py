import logging
import sys
from pathlib import Path
import yaml

def setup_logging(log_level="INFO"):
    """
    ログ設定を初期化する
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path):
    """
    YAML設定ファイルを読み込む
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    """
    ディレクトリが存在することを確認し、なければ作成する
    """
    Path(path).mkdir(parents=True, exist_ok=True)
