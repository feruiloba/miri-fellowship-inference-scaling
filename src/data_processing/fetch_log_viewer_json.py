"""
Fetch the underlying inspect_ai .eval log for each row of
data/benchmarks_runs-PUBLIC VIEW.csv, and extract every JSON file inside.

Background:
- The viewer URL has the form
    https://.../viewer.html?log_file=https://logs.epoch.ai/inspect_ai_logs/<ID>.eval
- "/json" fragment routes are client-side SPA only — fragments aren't
  sent to the server, so HTTP fetches against fragment URLs return the
  viewer HTML, not JSON.
- The .eval file itself is the source of truth: a zip archive (with zstd-
  compressed entries) containing header.json, samples/*.json,
  summaries.json, reductions.json, and _journal/* files.

Two fetch modes:
- **Range mode (default for --summaries-only)**: Use HTTP Range requests to
  pull only the bytes for the entries we want (~few MB instead of multi-GB
  for huge logs). Verified: logs.epoch.ai serves Accept-Ranges: bytes.
- **Full mode**: Stream-download the entire .eval to a tempfile, then
  extract. Required when the user wants every JSON entry (per-sample files
  etc.) since 200+ range requests per zip is slower than one streaming GET.

This script:
  1. Loads EPOCH_COOKIE from .env (or environment) — needed for the WAF.
  2. Cleans up orphaned tempfiles in the output dir from prior killed runs.
  3. For each row, fetches and extracts to data/log_viewer_json/<ID>/.
  4. Skips IDs whose target directory already has files (use --overwrite
     to refetch).

Pass --keep-eval to also save the raw .eval zip alongside (forces full mode).
"""

import argparse
import gc
import io
import os
import shutil
import struct
import sys
import tempfile
import time
import zipfile
from urllib.parse import urlparse, parse_qs, unquote

import pandas as pd
import requests
from dotenv import load_dotenv

try:
    import zstandard as zstd
    HAVE_ZSTD = True
except ImportError:
    HAVE_ZSTD = False


CSV_IN = "data/benchmarks_runs-PUBLIC VIEW.csv"
DEFAULT_OUT_DIR = "data/log_viewer_json"

# Files kept in --summaries-only mode (also the default range-mode targets).
SUMMARY_FILES = ("header.json", "summaries.json")


# ----------------------------------------------------------------------------
# Zip-over-HTTP-Range reader
# ----------------------------------------------------------------------------

EOCD_SIG = b"PK\x05\x06"
ZIP64_EOCD_LOC_SIG = b"PK\x06\x07"
ZIP64_EOCD_SIG = b"PK\x06\x06"
CD_ENTRY_SIG = b"PK\x01\x02"
LOCAL_HDR_SIG = b"PK\x03\x04"


class RemoteEvalZip:
    """Read a remote zip via HTTP Range requests.
    Supports ZIP64 and zstd-compressed entries (compression method 93)."""

    EOCD_PROBE_SIZE = 1 << 16  # 64 KB — enough for EOCD + ZIP64 EOCD records

    def __init__(self, url: str, session: requests.Session,
                 total_size: int | None = None):
        self.url = url
        self.session = session
        if total_size is None:
            r = session.head(url, timeout=30)
            r.raise_for_status()
            total_size = int(r.headers["Content-Length"])
        self.total_size = total_size
        self._cd: dict | None = None

    def _range(self, start: int, end: int) -> bytes:
        """Fetch bytes [start, end] inclusive."""
        r = self.session.get(
            self.url,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=60,
        )
        if r.status_code not in (200, 206):
            raise RuntimeError(f"range fetch returned {r.status_code}")
        return r.content

    def _load_central_directory(self) -> None:
        if self._cd is not None:
            return

        probe_start = max(0, self.total_size - self.EOCD_PROBE_SIZE)
        tail = self._range(probe_start, self.total_size - 1)

        eocd_idx = tail.rfind(EOCD_SIG)
        if eocd_idx < 0:
            raise RuntimeError("EOCD signature not found in last 64 KB")

        sig, _, _, _, n_total, cd_size, cd_offset, _ = struct.unpack(
            "<4sHHHHIIH", tail[eocd_idx : eocd_idx + 22]
        )

        # ZIP64 detection
        is_zip64 = (
            cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or n_total == 0xFFFF
        )
        if is_zip64:
            loc_idx = eocd_idx - 20
            if loc_idx < 0 or tail[loc_idx : loc_idx + 4] != ZIP64_EOCD_LOC_SIG:
                raise RuntimeError("ZIP64 EOCD locator missing")
            _, _, z64_off, _ = struct.unpack(
                "<4sIQI", tail[loc_idx : loc_idx + 20]
            )
            if z64_off >= probe_start:
                z64_rec = tail[z64_off - probe_start : z64_off - probe_start + 56]
            else:
                z64_rec = self._range(z64_off, z64_off + 55)
            if z64_rec[:4] != ZIP64_EOCD_SIG:
                raise RuntimeError("ZIP64 EOCD record signature mismatch")
            (_, _, _, _, _, _, _, n_total, cd_size, cd_offset) = struct.unpack(
                "<4sQHHIIQQQQ", z64_rec[:56]
            )

        # Read central directory
        if cd_offset >= probe_start and cd_offset + cd_size <= probe_start + len(tail):
            cd_bytes = tail[cd_offset - probe_start : cd_offset - probe_start + cd_size]
        else:
            cd_bytes = self._range(cd_offset, cd_offset + cd_size - 1)

        entries: dict[str, dict] = {}
        i = 0
        for _ in range(n_total):
            if cd_bytes[i : i + 4] != CD_ENTRY_SIG:
                raise RuntimeError(f"bad CD entry signature at offset {i}")
            (
                _, _, _, _, method, _, _, _, comp_size, uncomp_size,
                name_len, extra_len, comment_len, _, _, _, local_offset,
            ) = struct.unpack("<4sHHHHHHIIIHHHHHII", cd_bytes[i : i + 46])
            i += 46
            name = cd_bytes[i : i + name_len].decode("utf-8", errors="replace")
            i += name_len
            extra = cd_bytes[i : i + extra_len]
            i += extra_len + comment_len

            # ZIP64 extra field if any size/offset is 0xFFFFFFFF
            if comp_size == 0xFFFFFFFF or uncomp_size == 0xFFFFFFFF or local_offset == 0xFFFFFFFF:
                j = 0
                while j + 4 <= len(extra):
                    hid, dsize = struct.unpack("<HH", extra[j : j + 4])
                    if hid == 0x0001:
                        z = extra[j + 4 : j + 4 + dsize]
                        zo = 0
                        if uncomp_size == 0xFFFFFFFF:
                            uncomp_size = struct.unpack("<Q", z[zo : zo + 8])[0]
                            zo += 8
                        if comp_size == 0xFFFFFFFF:
                            comp_size = struct.unpack("<Q", z[zo : zo + 8])[0]
                            zo += 8
                        if local_offset == 0xFFFFFFFF:
                            local_offset = struct.unpack("<Q", z[zo : zo + 8])[0]
                            zo += 8
                        break
                    j += 4 + dsize

            entries[name] = {
                "method": method,
                "compressed_size": comp_size,
                "uncompressed_size": uncomp_size,
                "local_offset": local_offset,
            }
        self._cd = entries

    def names(self) -> list[str]:
        self._load_central_directory()
        return list(self._cd.keys())  # type: ignore[union-attr]

    def read(self, name: str) -> bytes:
        """Read decompressed bytes for a member via range requests."""
        self._load_central_directory()
        assert self._cd is not None
        if name not in self._cd:
            raise KeyError(name)
        info = self._cd[name]

        # Fetch local header + payload in one shot, with a generous over-fetch
        # for the local header's name+extra fields (which can differ from CD).
        OVER = 4096
        start = info["local_offset"]
        end = min(start + 30 + OVER + info["compressed_size"] - 1,
                  self.total_size - 1)
        chunk = self._range(start, end)

        if chunk[:4] != LOCAL_HDR_SIG:
            raise RuntimeError("bad local header signature")
        (_, _, _, _, _, _, _, _, _, lname_len, lextra_len) = struct.unpack(
            "<4sHHHHHIIIHH", chunk[:30]
        )
        data_start = 30 + lname_len + lextra_len
        if data_start + info["compressed_size"] > len(chunk):
            # Rare: extras larger than OVER. Refetch exact data range.
            abs_start = info["local_offset"] + data_start
            abs_end = abs_start + info["compressed_size"] - 1
            payload = self._range(abs_start, abs_end)
        else:
            payload = chunk[data_start : data_start + info["compressed_size"]]

        method = info["method"]
        if method == 0:  # stored
            return payload
        if method == 8:  # deflate
            import zlib
            return zlib.decompress(payload, -zlib.MAX_WBITS)
        if method == 93:  # zstd
            if not HAVE_ZSTD:
                raise RuntimeError("zstd-compressed; install 'zstandard'")
            return zstd.ZstdDecompressor().decompress(
                payload, max_output_size=info["uncompressed_size"]
            )
        raise NotImplementedError(f"compression method {method} not supported")


# ----------------------------------------------------------------------------
# Local zip extraction (used when full-download fallback is needed)
# ----------------------------------------------------------------------------

def read_zip_member(zip_path: str, info: zipfile.ZipInfo) -> bytes:
    if info.compress_type != 93:
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(info) as f:
                return f.read()
    if not HAVE_ZSTD:
        raise RuntimeError("zstd-compressed entry; install 'zstandard'.")
    with open(zip_path, "rb") as fp:
        fp.seek(info.header_offset)
        local_hdr = fp.read(30)
        name_len = int.from_bytes(local_hdr[26:28], "little")
        extra_len = int.from_bytes(local_hdr[28:30], "little")
        fp.seek(info.header_offset + 30 + name_len + extra_len)
        compressed = fp.read(info.compress_size)
    return zstd.ZstdDecompressor().decompress(
        compressed, max_output_size=info.file_size
    )


def extract_all_json_from_file(zip_path: str, out_dir: str,
                               summaries_only: bool = False) -> int:
    keep = set(SUMMARY_FILES)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        infos = [i for i in zf.infolist()
                 if not i.is_dir() and i.filename.lower().endswith(".json")]
    for info in infos:
        if summaries_only and info.filename not in keep:
            continue
        data = read_zip_member(zip_path, info)
        rel = info.filename.lstrip("/").replace("..", "_")
        target = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(target) or out_dir, exist_ok=True)
        with open(target, "wb") as f:
            f.write(data)
        n += 1
    return n


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def extract_eval_url(log_viewer_url: str) -> str | None:
    parsed = urlparse(log_viewer_url)
    qs = parse_qs(parsed.query)
    vals = qs.get("log_file")
    if not vals:
        return None
    return unquote(vals[0])


def cleanup_orphan_tempfiles(out_dir: str) -> int:
    """Remove leftover tmp*.eval files from killed prior runs."""
    if not os.path.isdir(out_dir):
        return 0
    n = 0
    for f in os.listdir(out_dir):
        if f.startswith("tmp") and f.endswith(".eval"):
            try:
                os.unlink(os.path.join(out_dir, f))
                n += 1
            except OSError:
                pass
    return n


def fetch_via_range(url: str, session: requests.Session, run_dir: str,
                    members: tuple[str, ...]) -> tuple[int, int]:
    """Range-fetch only the given members. Returns (n_files, total_bytes)."""
    rz = RemoteEvalZip(url, session)
    n = 0
    total_bytes = 0
    os.makedirs(run_dir, exist_ok=True)
    for name in members:
        try:
            data = rz.read(name)
        except KeyError:
            continue  # member not present
        target = os.path.join(run_dir, name)
        os.makedirs(os.path.dirname(target) or run_dir, exist_ok=True)
        with open(target, "wb") as f:
            f.write(data)
        n += 1
        total_bytes += len(data)
    return n, total_bytes


def fetch_via_full_download(url: str, session: requests.Session, run_dir: str,
                             out_dir: str, summaries_only: bool,
                             keep_eval_path: str | None) -> tuple[int, int]:
    """Stream the full .eval to a tempfile and extract. Returns (n_files, bytes)."""
    tmp_path = None
    try:
        with session.get(url, timeout=60, stream=True) as r:
            if r.status_code != 200:
                raise RuntimeError(f"http status {r.status_code}")
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".eval", dir=out_dir
            ) as tf:
                tmp_path = tf.name
                total = 0
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        tf.write(chunk)
                        total += len(chunk)
        with open(tmp_path, "rb") as f:
            if f.read(2) != b"PK":
                raise RuntimeError("response is not a zip")
        os.makedirs(run_dir, exist_ok=True)
        n_files = extract_all_json_from_file(
            tmp_path, run_dir, summaries_only=summaries_only
        )
        if keep_eval_path:
            shutil.move(tmp_path, keep_eval_path)
            tmp_path = None
        return n_files, total
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def main():
    load_dotenv()

    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=CSV_IN)
    p.add_argument("--out", default=DEFAULT_OUT_DIR)
    p.add_argument("--limit", type=int, default=None,
                   help="Only fetch the first N CSV rows (for testing)")
    p.add_argument("--cookie", default=os.environ.get("EPOCH_COOKIE", ""),
                   help="Cookie header for auth (default: $EPOCH_COOKIE / .env)")
    p.add_argument("--sleep", type=float, default=0.2,
                   help="Seconds between requests")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-fetch IDs whose output directory already exists")
    p.add_argument("--keep-eval", action="store_true",
                   help="Also save the raw .eval zip per row (forces full download)")
    p.add_argument("--summaries-only", action="store_true",
                   help="Keep only header.json + summaries.json. Uses fast "
                        "range requests instead of full download.")
    p.add_argument("--full-download", action="store_true",
                   help="Force full-download mode even with --summaries-only.")
    args = p.parse_args()

    if not args.cookie:
        print("Warning: no EPOCH_COOKIE set — fetches will likely hit the WAF.",
              file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)

    n_orphans = cleanup_orphan_tempfiles(args.out)
    if n_orphans:
        print(f"Cleaned up {n_orphans} orphan tempfile(s) from prior runs")

    df = pd.read_csv(args.csv)
    if "log viewer" not in df.columns:
        sys.exit(f"CSV missing 'log viewer' column. Columns: {list(df.columns)}")

    rows = df[df["log viewer"].notna()].copy()
    if args.limit:
        rows = rows.head(args.limit)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; epoch-fetch/1.0)"})
    if args.cookie:
        session.headers.update({"Cookie": args.cookie})

    use_range = args.summaries_only and not args.keep_eval and not args.full_download

    n_ok = n_skip = n_fail = 0
    for i, row in rows.iterrows():
        viewer_url = row["log viewer"].strip()
        eval_url = extract_eval_url(viewer_url)
        if eval_url is None:
            print(f"[{i}] skip: no log_file in URL")
            n_skip += 1
            continue

        log_id = os.path.basename(eval_url).rsplit(".", 1)[0]
        run_dir = os.path.join(args.out, log_id)
        eval_path = os.path.join(args.out, f"{log_id}.eval")

        already = os.path.isdir(run_dir) and any(
            f.endswith(".json") for f in os.listdir(run_dir)
        )
        if already and not args.overwrite:
            n_skip += 1
            continue

        try:
            if use_range:
                n_files, total = fetch_via_range(
                    eval_url, session, run_dir, SUMMARY_FILES
                )
            else:
                n_files, total = fetch_via_full_download(
                    eval_url, session, run_dir, args.out,
                    summaries_only=args.summaries_only,
                    keep_eval_path=eval_path if args.keep_eval else None,
                )
            mode = "range" if use_range else "full"
            print(f"[{i}] ok  ({mode}, {total//1024} KB → {n_files} files)  → {log_id}/")
            n_ok += 1
        except Exception as e:
            print(f"[{i}] error: {e}")
            n_fail += 1
        finally:
            if (n_ok + n_fail) and (n_ok + n_fail) % 50 == 0:
                gc.collect()

        time.sleep(args.sleep)

    print(f"\nDone. ok={n_ok}  skipped={n_skip}  failed={n_fail}")


if __name__ == "__main__":
    main()
