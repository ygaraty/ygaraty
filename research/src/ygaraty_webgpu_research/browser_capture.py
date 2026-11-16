"""Attach sanitized CDP network capture to an existing authorized browser
session.

Does not launch a browser or start services. Tests supply a fake event client.
CDP preflight events are separate requests; JS fetch timings include them.
"""

import hashlib
from urllib.parse import urlsplit

from ygaraty_webgpu_research.host import component as _host_component

capture = _host_component("runtime").capture


def attach(client, worker):
    def request(event):
        request = event["request"]
        url = request.get("url", "")
        path = urlsplit(url).path
        route = (
            "blocks"
            if "/blocks" in path
            else "coordinator"
            if "/api/jobs/" in path
            else "other"
        )
        capture.emit(
            "browser.network_request",
            worker=worker,
            request=event["requestId"],
            browser_mono_s=event["timestamp"],
            method=request["method"],
            route=route,
            resource_hash=hashlib.sha256(url.encode()).hexdigest(),
            resource_type=event.get("type"),
            redirect=bool(event.get("redirectResponse")),
        )

    def response(event):
        r = event["response"]
        timing = {
            k: v
            for k, v in r.get("timing", {}).items()
            if k
            in (
                "requestTime",
                "proxyStart",
                "proxyEnd",
                "dnsStart",
                "dnsEnd",
                "connectStart",
                "connectEnd",
                "sslStart",
                "sslEnd",
                "sendStart",
                "sendEnd",
                "receiveHeadersStart",
                "receiveHeadersEnd",
            )
            and type(v) in (int, float)
        }
        capture.emit(
            "browser.network_response",
            worker=worker,
            request=event["requestId"],
            browser_mono_s=event["timestamp"],
            status=r["status"],
            timing=timing,
            from_disk_cache=bool(r.get("fromDiskCache")),
            from_service_worker=bool(r.get("fromServiceWorker")),
        )

    def finished(event):
        capture.emit(
            "browser.network_finished",
            worker=worker,
            request=event["requestId"],
            browser_mono_s=event["timestamp"],
            encoded_total_bytes=event["encodedDataLength"],
        )

    def failed(event):
        capture.emit(
            "browser.network_failed",
            worker=worker,
            request=event["requestId"],
            browser_mono_s=event["timestamp"],
            cancelled=bool(event.get("canceled")),
        )

    listeners = {
        "Network.requestWillBeSent": request,
        "Network.responseReceived": response,
        "Network.loadingFinished": finished,
        "Network.loadingFailed": failed,
    }
    for event, fn in listeners.items():
        client.on(event, fn)
    client.send("Network.enable")
    return listeners
