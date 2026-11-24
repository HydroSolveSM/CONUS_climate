from pathlib import Path
import os
import subprocess
import sys

project_root = Path(__file__).resolve().parent.parent
script_dir = project_root / "scripts"
scripts = ["all_indices_average.py", "all_indices_trend.py", "PET_calculations.py", "ai_spei.py", "supplementary.py"]

def main():
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    for name in scripts:
        print(f"Running {name}", flush=True)
        subprocess.run([sys.executable, str(script_dir / name)], cwd=project_root, env=env, check=True)

if __name__ == "__main__":
    main()
