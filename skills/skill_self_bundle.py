from pathlib import Path
import hashlib, json, os, time, zipfile, tempfile
ACT_NAME = "self_bundle"
FORCE_FLAG = Path("data/force/bundle.force")
THROTTLE_TICKS = 120

def _consume_force_flag() -> bool:
    if FORCE_FLAG.exists():
        try: FORCE_FLAG.unlink(missing_ok=True)
        except Exception: pass
        return True
    return False

def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()

def _gather_files(root: Path):
    EXCLUDE_DIRS = {"data", ".git", "__pycache__", ".pytest_cache", ".venv", "venv", ".idea", ".mypy_cache"}
    EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp"}
    for p in root.rglob("*"):
        if p.is_dir():
            if any(part in EXCLUDE_DIRS for part in p.parts): continue
            else: continue
        if any(part in EXCLUDE_DIRS for part in p.parents): continue
        if p.suffix in EXCLUDE_SUFFIXES: continue
        if "data" in p.parts and "releases" in p.parts: continue
        yield p

def act(ctx):
    s = ctx["state"]; ticks = int(s.get("ticks", 0))
    forced = _consume_force_flag()
    if not forced and (ticks % THROTTLE_TICKS != 0): return 0.0

    root = Path.cwd(); release_dir = Path("data") / "releases"
    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    fd, tmp_path = tempfile.mkstemp(prefix="fgseed-", suffix=".zip", dir=str(release_dir)); os.close(fd)
    tmp_zip = Path(tmp_path); final_zip = release_dir / f"fgseed-{ts}.zip"

    try:
        files = list(_gather_files(root))
        manifest = []
        for p in files:
            try: rel = p.relative_to(root).as_posix()
            except ValueError: continue
            manifest.append({"path": rel, "sha256": _sha256_file(p), "size": p.stat().st_size})

        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for e in manifest: zf.write(root / e["path"], arcname=e["path"])
            zf.writestr("manifest.json", json.dumps({"created_ts": ts, "file_count": len(manifest), "files": manifest}, ensure_ascii=False, indent=2))

        with zipfile.ZipFile(tmp_zip, "r") as zf:
            if zf.testzip() is not None: tmp_zip.unlink(missing_ok=True); return -0.4

        tmp_zip.replace(final_zip)
        checksum = _sha256_file(final_zip)
        (final_zip.with_suffix(final_zip.suffix + ".sha256")).write_text(checksum + "  " + final_zip.name, encoding="utf-8")
        # prune old
        zips = sorted(release_dir.glob("fgseed-*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in zips[5:]:
            try: old.unlink(missing_ok=True); (old.with_suffix(old.suffix + ".sha256")).unlink(missing_ok=True)
            except Exception: pass
        s["last_bundle"] = final_zip.as_posix(); s["last_bundle_sha256"] = checksum; s["last_bundle_ts"] = ts
        return +0.6
    except Exception:
        try: tmp_zip.unlink(missing_ok=True)
        except Exception: pass
        return -0.4
