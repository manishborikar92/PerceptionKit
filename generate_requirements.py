import subprocess
import sys
from pathlib import Path

# --- Paths ---
project_path = Path(r"C:\Users\manis\Projects\PerceptionKit")
req_file = project_path / "requirements.txt"
pipreqs_exe = Path(r"C:\Users\manis\AppData\Roaming\Python\Python310\Scripts\pipreqs.exe")

# --- Step 1: Generate base requirements (no versions) ---
if not pipreqs_exe.exists():
    sys.exit(f"❌ pipreqs not found at {pipreqs_exe}. Install with: pip install pipreqs")

subprocess.run(
    [str(pipreqs_exe), str(project_path), "--force", "--encoding=utf-8"],
    check=True
)

# --- Step 2: Get installed packages with versions ---
freeze_output = subprocess.check_output(
    [sys.executable, "-m", "pip", "freeze"], text=True, encoding="utf-8"
)

freeze_dict = {}
for line in freeze_output.splitlines():
    if "==" in line:
        pkg, ver = line.strip().split("==", 1)
        freeze_dict[pkg.lower()] = ver

# --- Step 3: Rewrite requirements.txt with versions ---
with open(req_file, "r", encoding="utf-8") as f:
    packages = [line.strip() for line in f if line.strip()]

with open(req_file, "w", encoding="utf-8") as f:
    for pkg in packages:
        pkg_name = pkg.split("==")[0].lower()
        if pkg_name in freeze_dict:
            f.write(f"{pkg_name}=={freeze_dict[pkg_name]}\n")
        else:
            f.write(f"{pkg}\n")

print(f"✅ Final requirements.txt updated with exact versions at {req_file}")

# python generate_requirements.py