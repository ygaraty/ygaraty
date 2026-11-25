"""Stage public user docs for MkDocs; repository references stay on GitHub."""
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "_build" / "docs"


def main():
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)
    shutil.copytree(ROOT / "docs" / "assets", STAGING / "assets")
    for source in sorted((ROOT / "docs").glob("*.md")):
        text = source.read_text()
        # Links outside docs are source/research references, not website pages.
        text = re.sub(
            r"(?<=\]\()\.\./([^\s)]+)",
            r"https://github.com/ygaraty/ygaraty/blob/main/\1",
            text,
        )
        text = re.sub(r"(?<=\]\()README\.md(?=[#)])", "index.md", text)
        name = "index.md" if source.name == "README.md" else source.name
        (STAGING / name).write_text(text)
    subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict"],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
