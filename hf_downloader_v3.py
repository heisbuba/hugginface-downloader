"""
HF Downloader — v3.0
Supports: Spaces · Models · Datasets
Features: parallel downloads, resume, retries, zip, Android/PC auto-detect
Python 3.10+
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# --- Configuration ---
REPO_ID     = "heisbuba/cryptovat"          # owner/name
REPO_TYPE   = "space"                       # "space" | "model" | "dataset"
MAX_THREADS = min(8, (os.cpu_count() or 4) * 4)  # Default thread cap
TIMEOUT     = 30                            # seconds per request
MAX_RETRIES = 5                             # per-file retry attempts
HF_TOKEN    = os.getenv("HF_TOKEN", "")     # Optional: export HF_TOKEN=hf_xxx

RepoType = Literal["space", "model", "dataset"]

API_ROOTS: dict[str, str] = {
    "space":   "spaces",
    "model":   "models",
    "dataset": "datasets",
}
RESOLVE_ROOTS: dict[str, str] = {
    "space":   "spaces",
    "model":   "",               # huggingface.co/{repo}/resolve/...
    "dataset": "datasets",
}

# Permanent failures where retries are skipped
_PERMANENT_ERRORS = frozenset({400, 401, 403, 404, 410})

# --- Platform Detection ---

def get_downloads_folder() -> Path:
    """Detects primary Downloads folder for Android or PC platforms."""
    seen: set[Path] = set()
    candidates = [
        Path("/storage/emulated/0/Download"),   # Android internal
        Path("/sdcard/Download"),                # Android legacy symlink
        Path.home() / "Downloads",              # Windows / macOS / Linux
    ]
    for p in candidates:
        try:
            resolved = p.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if p.exists():
            return p
    
    fallback = Path.home() / "Downloads"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback

# --- Security ---

def _safe_local_path(dest_dir: Path, file_path: str) -> Path:
    """Verifies that the file path remains within the destination directory."""
    resolved_dest  = dest_dir.resolve()
    resolved_local = (dest_dir / file_path).resolve()
    if not resolved_local.is_relative_to(resolved_dest):
        raise ValueError(
            f"⛔ Path traversal blocked: '{file_path}' would escape download dir"
        )
    return resolved_local

# --- HTTP Utilities ---

def _make_headers(token: str = "") -> dict[str, str]:
    h = {"User-Agent": "HF-Downloader/3.0"}
    t = token or HF_TOKEN
    if t:
        h["Authorization"] = f"Bearer {t}"
    return h

class _RangePreservingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Retains Range header during HTTP redirects to maintain resume state."""

    def redirect_request(
        self, req: urllib.request.Request, fp, code, msg, headers, newurl
    ) -> urllib.request.Request:
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req and "Range" in req.headers:
            new_req.add_header("Range", req.headers["Range"])
        return new_req

def _build_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_RangePreservingRedirectHandler)

def _fetch_json(url: str, token: str = "") -> list | dict:
    req = urllib.request.Request(url, headers=_make_headers(token))
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode())

# --- Repository Listing ---

def list_repo_files(repo_id: str, repo_type: RepoType, token: str = "") -> list[str]:
    """Fetches full file list from the repository tree API."""
    api_root = API_ROOTS[repo_type]
    url = (
        f"https://huggingface.co/api/{api_root}/{repo_id}"
        f"/tree/main?recursive=true"
    )
    try:
        data = _fetch_json(url, token)
    except urllib.error.HTTPError as e:
        sys.exit(
            f"❌ HTTP {e.code} listing repo — check REPO_ID / REPO_TYPE / HF_TOKEN"
        )
    except Exception as e:
        sys.exit(f"❌ Cannot reach HuggingFace API: {e}")

    return [item["path"] for item in data if item.get("type") == "file"]

# --- Core Logic ---

@dataclass
class DownloadTask:
    repo_id:   str
    repo_type: RepoType
    file_path: str
    dest_dir:  Path
    token:     str = ""

def _resolve_url(task: DownloadTask) -> str:
    """Generates the download URL based on repo type and ID."""
    resolve_root = RESOLVE_ROOTS[task.repo_type]
    if resolve_root:
        return (
            f"https://huggingface.co/{resolve_root}/{task.repo_id}"
            f"/resolve/main/{task.file_path}"
        )
    return f"https://huggingface.co/{task.repo_id}/resolve/main/{task.file_path}"

def download_file(task: DownloadTask) -> tuple[bool, str]:
    """Downloads a single file with resume and retry support."""
    url = _resolve_url(task)

    try:
        local_path = _safe_local_path(task.dest_dir, task.file_path)
    except ValueError as e:
        print(f"\n{e}", file=sys.stderr)
        return False, task.file_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    opener = _build_opener()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            headers        = _make_headers(task.token)
            existing_bytes = local_path.stat().st_size if local_path.exists() else 0

            if existing_bytes:
                headers["Range"] = f"bytes={existing_bytes}-"

            req = urllib.request.Request(url, headers=headers)

            with opener.open(req, timeout=TIMEOUT) as resp:
                mode = "ab" if existing_bytes and resp.status == 206 else "wb"
                with open(local_path, mode) as fh:
                    shutil.copyfileobj(resp, fh, length=1 << 20)   # 1 MiB chunks

            return True, task.file_path

        except urllib.error.HTTPError as e:
            if e.code == 416:  # Range not satisfiable (file complete)
                return True, task.file_path
            if e.code in _PERMANENT_ERRORS:
                return False, task.file_path
            if attempt < MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))
            else:
                return False, task.file_path

        except Exception:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))
            else:
                return False, task.file_path

    return False, task.file_path

# --- Progress Tracking ---

@dataclass
class Progress:
    total:     int
    done:      int   = field(default=0)
    succeeded: int   = field(default=0)
    failed:    int   = field(default=0)
    _start:    float = field(default_factory=time.time, repr=False)

    def update(self, success: bool) -> None:
        """Updates progress stats and renders the CLI bar."""
        self.done += 1
        if success:
            self.succeeded += 1
        else:
            self.failed += 1

        elapsed  = time.time() - self._start
        rate     = self.succeeded / elapsed if elapsed else 0
        bar_w    = 30
        filled   = int(bar_w * self.done / self.total)
        bar      = "█" * filled + "░" * (bar_w - filled)
        pct      = 100 * self.done / self.total
        print(
            f"\r[{bar}] {pct:5.1f}%  "
            f"{self.done}/{self.total}  "
            f"✅ {rate:.1f} f/s  ❌ {self.failed}",
            end="",
            flush=True,
        )

    def finish(self) -> None:
        """Prints final completion statistics."""
        elapsed = time.time() - self._start
        print(
            f"\n✅ {self.succeeded}/{self.total} OK  "
            f"| ❌ {self.failed} failed  "
            f"| ⏱ {elapsed:.1f}s"
        )

# --- Workflow Execution ---

def download_repo(
    repo_id:     str,
    repo_type:   RepoType = "space",
    max_threads: int      = MAX_THREADS,
    zip_output:  bool     = True,
    token:       str      = "",
) -> None:
    """Manages the end-to-end download, concurrency, and archiving process."""
    clean_name = repo_id.replace("/", "_")
    dest_base  = get_downloads_folder()

    temp_dir  = dest_base / f"{clean_name}_hf_tmp_{os.getpid()}"
    zip_path  = dest_base / clean_name

    print(f"\n🤗 HF Downloader v3 — {repo_type.upper()}: {repo_id}")
    print(f"📂 Destination : {dest_base}")
    print(f"⚡ Threads     : {max_threads}\n")

    print("🔍 Fetching file list …")
    files = list_repo_files(repo_id, repo_type, token)
    total = len(files)
    if not total:
        sys.exit("⚠️ No files found — check REPO_ID and REPO_TYPE.")
    print(f"📋 {total} files found\n")

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    tasks    = [DownloadTask(repo_id, repo_type, fp, temp_dir, token) for fp in files]
    progress = Progress(total=total)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as pool:
            futures = {pool.submit(download_file, t): t for t in tasks}
            for fut in concurrent.futures.as_completed(futures):
                ok, _ = fut.result()
                progress.update(ok)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted — cleaning up …")
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)

    progress.finish()

    if zip_output:
        partial_zip = Path(f"{zip_path}.zip")
        if partial_zip.exists():
            partial_zip.unlink()
        print(f"\n📦 Zipping → {zip_path}.zip …")
        try:
            shutil.make_archive(str(zip_path), "zip", temp_dir)
        except Exception as e:
            print(f"❌ Zip failed: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"✅ Saved: {zip_path}.zip")
    else:
        final_dir = dest_base / clean_name
        if final_dir.exists():
            shutil.rmtree(final_dir)
        temp_dir.rename(final_dir)
        print(f"✅ Saved: {final_dir}")

# --- Entry Point ---

def build_parser() -> argparse.ArgumentParser:
    """Configures command-line argument parsing."""
    default_threads = min(8, (os.cpu_count() or 4) * 4)
    p = argparse.ArgumentParser(
        description="Download any HuggingFace Space / Model / Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python HF_Downloader.py heisbuba/cryptovat
  python HF_Downloader.py bert-base-uncased --type model --no-zip
  python HF_Downloader.py datasets/imdb     --type dataset
  python HF_Downloader.py myrepo/private    --threads 32 --token hf_xxx
        """,
    )
    p.add_argument("repo",       nargs="?", default=REPO_ID,
                   help="owner/repo-name (default: %(default)s)")
    p.add_argument("--type",      dest="repo_type", default=REPO_TYPE,
                   choices=["space", "model", "dataset"],
                   help="repo type (default: %(default)s)")
    p.add_argument("--threads",  type=int, default=default_threads,
                   help=f"parallel threads (default: {default_threads})")
    p.add_argument("--no-zip",   action="store_true",
                   help="keep as folder instead of zipping")
    p.add_argument("--token",    default="",
                   help="HuggingFace token for private repos (overrides HF_TOKEN env)")
    return p

if __name__ == "__main__":
    args = build_parser().parse_args()
    resolved_token = args.token or HF_TOKEN

    download_repo(
        repo_id     = args.repo,
        repo_type   = args.repo_type,
        max_threads = args.threads,
        zip_output  = not args.no_zip,
        token       = resolved_token,
    )