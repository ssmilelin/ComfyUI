"""Download Wan 2.2 GGUF workflow models with resume + progress log."""
from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(r"D:\project\ComfyUI\models")
LOG = ROOT / "wan22_download_progress.json"

JOBS = [
    {
        "id": "text_encoder",
        "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "fname": "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "dest": ROOT / "text_encoders" / "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "exp_bytes": int(6.73 * 1024**3),
    },
    {
        "id": "gguf_high",
        "repo": "QuantStack/Wan2.2-T2V-A14B-GGUF",
        "fname": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q3_K_S.gguf",
        "dest": ROOT / "unet" / "Wan2.2-T2V-A14B-HighNoise-Q3_K_S.gguf",
        "exp_bytes": int(6.51 * 1024**3),
    },
    {
        "id": "gguf_low",
        "repo": "QuantStack/Wan2.2-T2V-A14B-GGUF",
        "fname": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q3_K_S.gguf",
        "dest": ROOT / "unet" / "Wan2.2-T2V-A14B-LowNoise-Q3_K_S.gguf",
        "exp_bytes": int(6.51 * 1024**3),
    },
]


def write_log(data: dict) -> None:
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    LOG.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def download_one(job: dict, state: dict) -> None:
    dest: Path = job["dest"]
    if dest.exists() and dest.stat().st_size > job["exp_bytes"] * 0.95:
        state["jobs"][job["id"]] = {
            "status": "done",
            "dest": str(dest),
            "bytes": dest.stat().st_size,
            "pct": 100.0,
        }
        write_log(state)
        return

    tmpdir = dest.parent / "_dl"
    tmpdir.mkdir(parents=True, exist_ok=True)
    state["current"] = job["id"]
    state["jobs"][job["id"]] = {
        "status": "downloading",
        "dest": str(dest),
        "bytes": dest.stat().st_size if dest.exists() else 0,
        "pct": 0.0,
    }
    write_log(state)

    print(f"[{job['id']}] start {job['fname']}", flush=True)
    path = hf_hub_download(
        repo_id=job["repo"],
        filename=job["fname"],
        local_dir=str(tmpdir),
    )
    src = Path(path)
    if not src.exists():
        src = tmpdir / job["fname"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.move(str(src), str(dest))

    state["jobs"][job["id"]] = {
        "status": "done",
        "dest": str(dest),
        "bytes": dest.stat().st_size,
        "pct": 100.0,
    }
    write_log(state)
    print(f"[{job['id']}] done {dest}", flush=True)


def main() -> None:
    state = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "current": None,
        "jobs": {},
        "status": "running",
    }
    write_log(state)
    try:
        for job in JOBS:
            download_one(job, state)
        state["status"] = "completed"
        state["current"] = None
    except Exception as e:
        state["status"] = "error"
        state["error"] = str(e)
        raise
    finally:
        write_log(state)


if __name__ == "__main__":
    main()
