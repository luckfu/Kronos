"""One-cell Kaggle entrypoint for V5 (120-day context, 10-day target)."""
import os
import subprocess
from pathlib import Path

repo = Path('/kaggle/working/Kronos')
if not (repo / '.git').exists():
    subprocess.run(['git', 'clone', 'https://github.com/luckfu/Kronos.git', str(repo)], check=True)
env = os.environ.copy()
env.setdefault('KRONOS_KAGGLE_ROOT', '/kaggle/working/kronos_a_share_v5_120d')
subprocess.run(['bash', 'finetune/kaggle_v5_bootstrap.sh'], cwd=repo, env=env, check=True)
subprocess.run(['bash', 'finetune/kaggle_v5_train.sh'], cwd=repo, env=env, check=True)
