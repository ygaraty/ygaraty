"""Local-network sharing: this machine's LAN address and a self-signed cert.

Browsers enable WebGPU only in secure contexts (HTTPS, or localhost), so a
plain http://192.168.x.y link would load without a GPU on other devices.
`--lan` serves HTTPS with a certificate made here; each device accepts its
warning once, after which the page is a secure context.
"""

import ipaddress
import shutil
import socket
import subprocess
from pathlib import Path


def lan_address(probe="192.0.2.1"):
    """The IPv4 address this machine uses toward the network.

    Connecting a UDP socket only selects a route; no packet is sent.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        try:
            udp.connect((probe, 9))
            address = udp.getsockname()[0]
        except OSError as error:
            raise ValueError(
                "Could not find this machine's LAN address; "
                "pass it explicitly: --lan 192.168.x.y"
            ) from error
    return check_address(address)


def check_address(value):
    try:
        address = ipaddress.ip_address(value)
    except ValueError as error:
        raise ValueError(f"Invalid LAN address: {value}") from error
    if address.is_loopback or address.is_unspecified:
        raise ValueError(
            "--lan needs this machine's network address, not "
            f"{value}; omit --lan for links on this machine only"
        )
    return str(address)


def lan_certificate(directory, address, *, openssl=None):
    """(certificate, key) paths for `address`, created once and reused.

    The certificate also names localhost and 127.0.0.1 so this machine's
    own browser and terminal chat can use the same server.
    """
    directory = Path(directory)
    cert, key, marker = (
        directory / "cert.pem",
        directory / "key.pem",
        directory / "address",
    )
    if (
        cert.is_file()
        and key.is_file()
        and marker.is_file()
        and marker.read_text().strip() == address
    ):
        return cert, key
    tool = openssl or shutil.which("openssl")
    if not tool:
        raise ValueError(
            "--lan needs the openssl command to make a certificate"
        )
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    host = ipaddress.ip_address(address)
    names = f"IP:{host},IP:127.0.0.1,DNS:localhost"
    subprocess.run(
        [
            tool,
            "req",
            "-x509",
            "-newkey",
            "ec",
            "-pkeyopt",
            "ec_paramgen_curve:prime256v1",
            "-nodes",
            "-days",
            "397",
            "-subj",
            "/CN=Ygaraty LAN",
            "-addext",
            f"subjectAltName={names}",
            "-keyout",
            str(key),
            "-out",
            str(cert),
        ],
        check=True,
        capture_output=True,
    )
    key.chmod(0o600)
    marker.write_text(address + "\n")
    return cert, key
