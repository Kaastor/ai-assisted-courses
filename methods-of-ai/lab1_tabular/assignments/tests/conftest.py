import os
import pytest


def pytest_collection_modifyitems(config, items):
    # Only run these tests when explicitly enabled to avoid impacting the main lab.
    if os.environ.get("RUN_ASSIGNMENT_TESTS"):
        return
    skip_marker = pytest.mark.skip(reason="Assignment tests run only with RUN_ASSIGNMENT_TESTS=1")
    for item in items:
        item.add_marker(skip_marker)

