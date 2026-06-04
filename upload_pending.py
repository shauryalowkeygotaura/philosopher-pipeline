#!/usr/bin/env python3
"""
upload_pending.py -- Download generated reels from GitHub Actions artifacts and upload to Instagram.

Usage:
    python upload_pending.py

Run this locally whenever you want to push pending reels to Instagram.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_DIR = Path(__file__).parent.resolve()

# Load .env
_env = _DIR / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from uploader import upload_reel


def _gh(*args) -> str:
    r = subprocess.run(["gh", *args], capture_output=True, text=True, check=True, cwd=str(_DIR))
    return r.stdout.strip()


def main():
    print("Fetching recent workflow runs...")
    runs_json = _gh("run", "list", "--workflow=pipeline.yml",
                    "--status=success", "--limit=10", "--json",
                    "databaseId,displayTitle,createdAt,headBranch")
    runs = json.loads(runs_json)

    if not runs:
        print("No successful runs found.")
        return

    uploaded = 0
    for run in runs:
        run_id = run["databaseId"]
        created = run["createdAt"]

        try:
            artifacts_json = _gh("api",
                f"repos/shauryalowkeygotaura/philosopher-pipeline/actions/runs/{run_id}/artifacts")
            artifacts = json.loads(artifacts_json).get("artifacts", [])
        except subprocess.CalledProcessError:
            continue

        reel_artifacts = [a for a in artifacts if a["name"].startswith("reel-")]
        if not reel_artifacts:
            continue

        for artifact in reel_artifacts:
            artifact_id = artifact["id"]
            artifact_name = artifact["name"]

            print(f"\nRun {run_id} ({created}) - artifact: {artifact_name}")

            extract_dir = _DIR / "output" / f"pending-{run_id}"
            extract_dir.mkdir(parents=True, exist_ok=True)

            # Download
            print("  Downloading...")
            subprocess.run(
                ["gh", "run", "download", str(run_id),
                 "--name", artifact_name,
                 "--dir", str(extract_dir)],
                check=True, capture_output=True, cwd=str(_DIR)
            )

            mp4_files = list(extract_dir.glob("*.mp4"))
            if not mp4_files:
                print("  No MP4 found - skipping.")
                continue

            mp4 = mp4_files[0]
            jpg = mp4.with_suffix(".jpg")

            # Re-encode to fix Linux ffmpeg codec differences
            reencoded = mp4.with_name(mp4.stem + "_win.mp4")
            enc = subprocess.run(
                ["ffmpeg", "-y", "-i", str(mp4),
                 "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                 "-c:a", "aac", "-b:a", "128k",
                 "-movflags", "+faststart", str(reencoded)],
                capture_output=True
            )
            upload_mp4 = reencoded if reencoded.exists() else mp4

            # Caption from filename
            slug = mp4.stem.rsplit("-", 1)[0]
            philosopher = slug.replace("-", " ").title()
            caption = (
                f'"{philosopher}"\n\n'
                f"#philosophy #quotes #wisdom #deepthoughts #philosophyquotes "
                f"#mindset #existentialism #stoicism #motivation"
            )

            print(f"  Uploading {upload_mp4.name}...")
            try:
                upload_reel(str(upload_mp4), caption, str(jpg) if jpg.exists() else None)
                print("  Uploaded!")
                uploaded += 1

                # Delete artifact to avoid re-uploading
                subprocess.run(
                    ["gh", "api", "--method=DELETE",
                     f"repos/shauryalowkeygotaura/philosopher-pipeline/actions/artifacts/{artifact_id}"],
                    capture_output=True, cwd=str(_DIR)
                )
                print("  Artifact deleted.")
                shutil.rmtree(str(extract_dir), ignore_errors=True)

            except Exception as e:
                print(f"  Upload failed: {e}")

    print(f"\nDone. {uploaded} reel(s) uploaded.")


if __name__ == "__main__":
    main()
