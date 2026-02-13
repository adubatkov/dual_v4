"""
Render Mermaid .mmd files to PNG images.

Requires one of:
  1. mmdc (Mermaid CLI): npm install -g @mermaid-js/mermaid-cli
  2. npx mmdc (via npx)
  3. Online: paste .mmd content into https://mermaid.live

Usage:
  python render.py              # Render all .mmd files in this directory
  python render.py file.mmd     # Render a specific file
"""

import subprocess
import sys
from pathlib import Path


def find_mmdc():
    """Find mmdc binary -- try direct, then npx."""
    for cmd in [["mmdc", "--version"], ["npx", "mmdc", "--version"]]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if r.returncode == 0:
                return cmd[:-1]  # Remove --version
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def render_mmd(mmdc_cmd: list, mmd_path: Path) -> bool:
    """Render a .mmd file to PNG using mmdc."""
    png_path = mmd_path.with_suffix(".png")
    cmd = mmdc_cmd + [
        "-i", str(mmd_path),
        "-o", str(png_path),
        "-t", "default",
        "-b", "white",
        "-w", "2400",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"[OK] {mmd_path.name} -> {png_path.name}")
            return True
        else:
            print(f"[FAIL] {mmd_path.name}: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[FAIL] {mmd_path.name}: timeout (60s)")
        return False


def main():
    script_dir = Path(__file__).parent

    if len(sys.argv) > 1:
        files = [Path(sys.argv[1])]
    else:
        files = sorted(script_dir.glob("*.mmd"))

    if not files:
        print("No .mmd files found.")
        return

    mmdc_cmd = find_mmdc()
    if not mmdc_cmd:
        print("[ERROR] mmdc not found.")
        print("  Install: npm install -g @mermaid-js/mermaid-cli")
        print("  Or paste .mmd content into https://mermaid.live")
        return

    print(f"Using: {' '.join(mmdc_cmd)}")
    success = 0
    for f in files:
        if render_mmd(mmdc_cmd, f):
            success += 1

    print(f"\nRendered {success}/{len(files)} files.")


if __name__ == "__main__":
    main()
