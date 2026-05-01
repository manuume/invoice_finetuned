import pytest

# make pytest-asyncio work without decorating every test
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
