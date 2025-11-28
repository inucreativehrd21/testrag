#!/usr/bin/env python3
"""
Fix encoding of existing report.txt files
Converts UTF-8 to UTF-8-sig for Windows compatibility
"""
import sys
from pathlib import Path


def fix_txt_encoding(txt_file: str):
    """Convert UTF-8 file to UTF-8-sig"""
    file_path = Path(txt_file)

    if not file_path.exists():
        print(f"Error: File not found: {txt_file}")
        return False

    # Read with utf-8
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"Error: Cannot decode {txt_file} as UTF-8")
        return False

    # Write with utf-8-sig (adds BOM)
    with open(file_path, "w", encoding="utf-8-sig") as f:
        f.write(content)

    print(f"✓ Fixed encoding: {txt_file}")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_encoding.py <file.txt>")
        print("\nExample:")
        print("  python fix_encoding.py artifacts/ragas_evals/ragas_eval_20251120_060751_report.txt")
        sys.exit(1)

    txt_file = sys.argv[1]
    if fix_txt_encoding(txt_file):
        print("\n✓ Encoding fixed! Now open the file in Notepad to verify.")
    else:
        sys.exit(1)
