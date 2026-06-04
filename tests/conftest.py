import pytest
from asgi_lifespan import LifespanManager

from app.main import app


@pytest.fixture
async def managed_app():
    """App with ASGI lifespan (startup/shutdown) properly executed."""
    async with LifespanManager(app) as manager:
        yield manager.app
