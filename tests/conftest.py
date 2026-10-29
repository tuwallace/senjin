import pytest


@pytest.fixture(autouse=True)
def _ignore_dotenv(monkeypatch):
    """Tests must not pick up a developer's own .env."""
    monkeypatch.setenv("SENJIN_ENV_FILE", "")
