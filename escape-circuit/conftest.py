"""Make the project root importable so bare `pytest` can `import src`.

`tests/` has no `__init__.py`, so pytest's default import mode would only
add `tests/` itself to `sys.path` when collecting a test module there —
not this directory, where `src/` actually lives. Pytest resolves that the
same way for a `conftest.py`: it walks up from this file's directory until
it finds one with no `__init__.py` and inserts that directory into
`sys.path`. Since this file sits at the repository root (which has no
`__init__.py`), just having it here is enough to put the root on
`sys.path` before test collection — no code needed, and no dependence on
`python -m pytest`'s CWD-insertion behaviour or on installing the package.
"""
