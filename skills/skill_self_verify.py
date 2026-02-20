from pathlib import Path
import tempfile, zipfile, hashlib, json, time, shutil
ACT_NAME = "self_verify"
FORCE_FLAG = Path("data/force/verify.force")
THROTTLE_TICKS = 180

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

def _newest_bundle(release_dir: Path) -> Path | None:
    zips = sorted(release_dir.glob("fgseed-*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0] if zips else None

def act(ctx):
    s = ctx["state"]; ticks = int(s.get("ticks", 0))
    forced = _consume_force_flag()
    if not forced and (ticks % THROTTLE_TICKS != 0): return 0.0

    release_dir = Path("data") / "releases"; bundle = _newest_bundle(release_dir)
    if not bundle or not bundle.exists(): return 0.0

    tmpdir = Path(tempfile.mkdtemp(prefix="fgseed-verify-"))
    try:
        with zipfile.ZipFile(bundle, "r") as zf:
            if zf.testzip() is not None: return -0.5
            zf.extractall(tmpdir)

        mpath = tmpdir / "manifest.json"
        if not mpath.exists(): return -0.5
        data = json.loads(mpath.read_text(encoding="utf-8"))
        for e in data.get("files", []):
            rel = e.get("path"); expect = e.get("sha256")
            if not rel or not expect: return -0.5
            fp = tmpdir / rel
            if fp.is_file():
                if _sha256_file(fp) != expect: return -0.5
            else:
                if not fp.exists(): return -0.5

        s["last_verify_bundle"] = str(bundle)
        s["last_verify_ts"] = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        return +0.7
    except Exception:
        return -0.5
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
