"""Exercise the real middleware: an AppTest alone cannot catch HTTP failures."""

import asyncio
import gzip

import pytest
from streamlit.web.server.starlette.starlette_gzip_middleware import (
    SelectiveGZipMiddleware,
)


@pytest.mark.parametrize("accept_encoding", [b"gzip", b"identity"])
@pytest.mark.parametrize("kind", ["image", "encoded", "partial", "html"])
def test_media_and_partial_responses_send_headers_before_body(accept_encoding, kind):
    body = b"portfolio evidence " * 100
    headers = [(b"content-type", b"text/html")]
    status = 200
    if kind == "image":
        headers = [(b"content-type", b"image/png")]
    elif kind == "encoded":
        body = gzip.compress(body)
        headers.append((b"content-encoding", b"gzip"))
    elif kind == "partial":
        status = 206
        headers = [(b"content-type", b"video/mp4")]
        headers.append((b"content-range", b"bytes 0-1899/3800"))
    headers.append((b"content-length", str(len(body)).encode()))
    messages = []

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        # Uvicorn rejects a body sent before the response start.
        if not messages:
            assert message["type"] == "http.response.start"
        messages.append(message)

    asyncio.run(SelectiveGZipMiddleware(app)(
        {"type": "http", "path": "/media/evidence", "headers": [
            (b"accept-encoding", accept_encoding)
        ]}, receive, send,
    ))
    assert messages[0]["status"] == status
    assert messages[-1]["type"] == "http.response.body"
    outgoing = messages[-1]["body"]
    if kind == "html" and accept_encoding == b"gzip":
        assert gzip.decompress(outgoing) == body
    else:
        assert outgoing == body
