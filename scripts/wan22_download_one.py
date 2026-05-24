"""Download a single Wan 2.2 model file (resumable). Usage: python wan22_download_one.py <job_id>"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(r"D:\project\ComfyUI\models")

JOBS = {
    "text_encoder": {
        "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "fname": "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "dest": ROOT / "text_encoders" / "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "exp_bytes": int(6.73 * 1024**3),
    },
    "gguf_high": {
        "repo": "QuantStack/Wan2.2-T2V-A14B-GGUF",
        "fname": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q3_K_S.gguf",
        "dest": ROOT / "unet" / "Wan2.2-T2V-A14B-HighNoise-Q3_K_S.gguf",
        "exp_bytes": int(6.51 * 1024**3),
    },
    "gguf_low": {
        "repo": "QuantStack/Wan2.2-T2V-A14B-GGUF",
        "fname": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q3_K_S.gguf",
        "dest": ROOT / "unet" / "Wan2.2-T2V-A14B-LowNoise-Q3_K_S.gguf",
        "exp_bytes": int(6.51 * 1024**3),
    },
}


def log_path(job_id: str) -> Path:
    return ROOT / f"wan22_dl_{job_id}.json"


def update_job(job_id: str, **fields) -> None:
    p = log_path(job_id)
    entry = {"job_id": job_id}
    if p.exists():
        entry.update(json.loads(p.read_text(encoding="utf-8")))
    entry.update(fields)
    entry["updated_at"] = datetime.now().isoformat(timespec="seconds")
    p.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")


def download(job_id: str) -> None:
    job = JOBS[job_id]
    dest: Path = job["dest"]
    if dest.exists() and dest.stat().st_size >= job["exp_bytes"] * 0.95:
        update_job(job_id, status="done", dest=str(dest), bytes=dest.stat().st_size, pct=100.0)
        print(f"[{job_id}] already done", flush=True)
        return

    tmpdir = dest.parent / f"_dl_{job_id}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    update_job(job_id, status="downloading", dest=str(dest), bytes=dest.stat().st_size if dest.exists() else 0, pct=0.0)
    print(f"[{job_id}] start {job['fname']}", flush=True)

    path = hf_hub_download(repo_id=job["repo"], filename=job["fname"], local_dir=str(tmpdir))
    src = Path(path)
    if not src.is_file():
        src = tmpdir / job["fname"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.move(str(src), str(dest))
    update_job(job_id, status="done", dest=str(dest), bytes=dest.stat().st_size, pct=100.0)
    print(f"[{job_id}] done {dest.stat().st_size / 1e9:.2f} GB", flush=True)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in JOBS:
        print(f"usage: {sys.argv[0]} {{{','.join(JOBS)}}}", file=sys.stderr)
        sys.exit(2)
    job_id = sys.argv[1]
    try:
        download(job_id)
    except Exception as e:
        update_job(job_id, status="error", error=str(e))
        raise


if __name__ == "__main__":
    main()
