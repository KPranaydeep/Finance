"""Use local Git only; never publish fixtures to the real GitHub repository."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_snapshot_first_publish_refresh_and_unchanged(tmp_path):
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if os.name == "nt" and git_bash.exists() else shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable")

    def git(*args, cwd=None):
        return subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        ).stdout.strip()

    remote = tmp_path / "remote.git"
    git("init", "--bare", str(remote))
    checkout = tmp_path / "writer"
    git("clone", str(remote), str(checkout))
    git("config", "user.name", "Snapshot test", cwd=checkout)
    git("config", "user.email", "snapshot@example.invalid", cwd=checkout)
    (checkout / "README.md").write_text("main branch must stay intact\n")
    git("add", "README.md", cwd=checkout)
    git("commit", "-m", "Initial", cwd=checkout)
    main = git("rev-parse", "HEAD", cwd=checkout)
    script = Path(__file__).resolve().parents[1] / "scripts/publish_public_snapshot.sh"
    snapshot = checkout / "public_record_snapshot.json"

    def publish(value):
        snapshot.write_text(value)
        subprocess.run([bash, str(script), str(snapshot)], cwd=checkout, check=True,
                       capture_output=True, text=True)
        return git("rev-parse", "refs/heads/public-snapshot", cwd=remote)

    first = publish('{"test":1}\n')
    second = publish('{"test":2}\n')
    assert first != second
    assert publish('{"test":2}\n') == second
    assert git("ls-tree", "--name-only", "public-snapshot", cwd=remote) == "public_record_snapshot.json"
    assert git("rev-parse", "HEAD", cwd=checkout) == main
    assert (checkout / "README.md").exists()
