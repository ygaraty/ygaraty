"""HTTP replies and streaming shared by inference hosts."""
import json


def reply(h, code, body=b"", mime="application/json", extra=()):
    """One complete response on the request handler `h` (this module's or the general coordinator's)."""
    if isinstance(body, (dict, list)):
        body = json.dumps(body).encode()
    elif isinstance(body, str):
        body = body.encode()
    h.send_response(code)
    h.send_header("Content-Type", mime)
    h.send_header("Content-Length", str(len(body)))
    h.send_header("Cache-Control", "no-store")
    for k, v in extra:
        h.send_header(k, v)
    h.end_headers()
    h.wfile.write(body)

def stream(h, session, events):
    """Newline-delimited JSON events, chunked as they come."""
    if callable(getattr(h, "stream", None)):
        def guarded():
            try:
                yield from events
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as error:
                session.say(f"error in chat stream: {error!r}")
                yield dict(error=repr(error))
            finally:
                if hasattr(events, 'close'):
                    events.close()
        return h.stream(guarded())
    h.send_response(200)
    h.send_header("Content-Type", "application/x-ndjson")
    h.send_header("Cache-Control", "no-store")
    h.send_header("Transfer-Encoding", "chunked")
    h.end_headers()

    def chunk(event):
        line = (json.dumps(event) + "\n").encode()
        h.wfile.write(f"{len(line):X}\r\n".encode() + line + b"\r\n")
        h.wfile.flush()
    try:
        for event in events:
            chunk(event)
    except (BrokenPipeError, ConnectionResetError):
        return
    except Exception as error:  # end the stream with the error, keep serving
        session.say(f"error in chat stream: {error!r}")
        chunk(dict(error=repr(error)))
    h.wfile.write(b"0\r\n\r\n")
    h.wfile.flush()
