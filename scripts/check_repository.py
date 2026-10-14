"""Kiểm source publish: notebook, cú pháp Python, dấu merge, link và file loại trừ."""
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
GUIDES = ("README.md",)
BANNED_SUFFIXES = {".docx", ".pdf", ".ppt", ".pptx", ".tex", ".zip", ".sqlite3"}
PRIVATE_FILES = {"docs/final_report_content.md", "scripts/build_ieee_report.py", "requirements-report.txt"}


def check():
    files = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    issues, notebook_count, python_count = [], 0, 0
    for name in sorted(set(files) - {""}):
        path = ROOT / name
        if (path.suffix.lower() in BANNED_SUFFIXES or name in PRIVATE_FILES
                or name.startswith(("outputs/", "docs/"))
                or (path.suffix.lower() == ".md" and name != "README.md")):
            issues.append(f"Không publish: {name}")
        if not path.is_file() or path.suffix not in (".py", ".md", ".ipynb", ".json", ".js", ".css", ".html"):
            continue
        content = path.read_text(encoding="utf-8-sig")
        if re.search(r"^(?:<{7} .*|={7}|>{7} .*)$", content, flags=re.MULTILINE):
            issues.append(f"Dấu xung đột: {name}")
        try:
            if path.suffix == ".py":
                compile(content, name, "exec")
                python_count += 1
            elif path.suffix == ".ipynb":
                notebook = json.loads(content)
                if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
                    raise ValueError("Notebook format không hợp lệ")
                for cell in notebook["cells"]:
                    if cell["cell_type"] not in ("markdown", "code", "raw"):
                        raise ValueError("Cell type không hợp lệ")
                notebook_count += 1
        except (SyntaxError, ValueError, TypeError, KeyError) as error:
            issues.append(f"{name}: {error}")
    for name in GUIDES:
        path = ROOT / name
        if not path.is_file():
            issues.append(f"Thiếu hướng dẫn: {name}")
            continue
        for link in re.findall(r"\]\(([^\s)]+)\)", path.read_text(encoding="utf-8")):
            if link.startswith(("https://", "http://", "#", "mailto:")):
                continue
            target = unquote(link.split("#", 1)[0])
            if not (path.parent / target).exists():
                issues.append(f"Link thiếu: {name} -> {link}")
    result = {"passed": not issues, "python_files": python_count,
              "notebooks": notebook_count, "issues": issues}
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return result


if __name__ == "__main__":
    raise SystemExit(0 if check()["passed"] else 1)
