"""Shared fixtures for the transport tests."""

import asyncio

import pytest

from helpers import LoopbackServer


@pytest.fixture
async def listen():
    """Start a loopback HTTP server per registered handler, close all after."""
    servers = []

    def _listen(handler):
        server = LoopbackServer(handler)
        server.start()
        servers.append(server)
        return server

    yield _listen
    for server in servers:
        await asyncio.to_thread(server.close)
