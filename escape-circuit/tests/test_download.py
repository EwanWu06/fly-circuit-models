from src import config, download


def test_fetch_all_returns_existing_paths():
    paths = download.fetch_all()
    assert set(paths) == set(config.FILES)
    for key, path in paths.items():
        assert path.exists(), f"{key} missing"
        if key in config.EXPECTED_BYTES:
            assert path.stat().st_size == config.EXPECTED_BYTES[key]


def test_fetch_all_is_idempotent():
    first = download.fetch_all()
    mtimes = {k: p.stat().st_mtime for k, p in first.items()}
    second = download.fetch_all()
    assert {k: p.stat().st_mtime for k, p in second.items()} == mtimes


def _fake_bucket(monkeypatch, tmp_path, payload: bytes, digest: str):
    """Point download at a one-file bucket in tmp_path; never touch the network."""
    import urllib.request
    from pathlib import Path

    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    monkeypatch.setattr(config, "FILES", {"toy": "toy.feather"})
    monkeypatch.setattr(config, "EXPECTED_BYTES", {"toy": len(payload)})
    monkeypatch.setattr(config, "EXPECTED_SHA256", {"toy": digest})
    monkeypatch.setattr(urllib.request, "urlretrieve",
                        lambda url, dest: Path(dest).write_bytes(payload))


def test_fresh_download_with_wrong_digest_is_rejected(monkeypatch, tmp_path):
    import pytest
    payload = b"not the connectome"
    _fake_bucket(monkeypatch, tmp_path, payload, "0" * 64)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        download.fetch_all()
    assert list(tmp_path.iterdir()) == []      # neither the file nor a .part


def test_fresh_download_with_matching_digest_is_kept(monkeypatch, tmp_path):
    import hashlib
    payload = b"not the connectome"
    _fake_bucket(monkeypatch, tmp_path, payload,
                 hashlib.sha256(payload).hexdigest())
    paths = download.fetch_all()
    assert paths["toy"].read_bytes() == payload
    assert [p.name for p in tmp_path.iterdir()] == ["toy.feather"]
