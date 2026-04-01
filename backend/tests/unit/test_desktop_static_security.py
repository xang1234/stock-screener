from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from app.main import app


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


@pytest.mark.asyncio
async def test_desktop_static_fallback_rejects_path_traversal(client, tmp_path, monkeypatch):
    from app import main as module

    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("top-secret", encoding="utf-8")

    monkeypatch.setattr(module.settings, "desktop_mode", True)
    monkeypatch.setattr(module.settings, "frontend_dist_dir", str(dist_dir))

    assert module._safe_desktop_asset_path("../secret.txt") is None

    response = await client.get("/%2e%2e/secret.txt")

    assert response.status_code == 200
    assert response.text == "<html>ok</html>"
    assert response.text != "top-secret"
