"""Make the project root importable so bare `pytest` can `import minifly`.

Same approach as escape-circuit/conftest.py: pytest inserts the directory of the first
conftest.py without an __init__.py into sys.path, which is this one.
"""
