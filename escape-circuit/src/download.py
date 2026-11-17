"""Fetch MaleCNS v1.0 bulk files. No authentication required."""
import hashlib
import urllib.request
from pathlib import Path

from . import config


def _fetch_one(name: str, force: bool) -> Path:
    dest = config.RAW_DIR / config.FILES[name]
    expected = config.EXPECTED_BYTES.get(name)
    if dest.exists() and not force:
        if expected is None or dest.stat().st_size == expected:
            return dest
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{config.BASE_URL}/{config.FILES[name]}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    if expected is not None and tmp.stat().st_size != expected:
        tmp.unlink()
        raise RuntimeError(f"{name}: size mismatch, expected {expected}")
    digest = config.EXPECTED_SHA256.get(name)
    if digest is not None and _sha256(tmp) != digest:
        tmp.unlink()
        raise RuntimeError(f"{name}: SHA-256 mismatch, expected {digest}")
    tmp.replace(dest)
    return dest


def _sha256(path: Path) -> str:
    """Hex SHA-256 of a file, read in 1 MiB blocks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_all(force: bool = False) -> dict[str, Path]:
    """Download every raw file, skipping any already present and complete."""
    return {name: _fetch_one(name, force) for name in config.FILES}
