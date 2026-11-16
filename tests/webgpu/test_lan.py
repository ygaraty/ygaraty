"""LAN sharing helpers: address checks and the self-signed certificate."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ygaraty.backends.webgpu.lan import check_address, lan_address, lan_certificate
from ygaraty.backends.webgpu.terminal_chat import local_base


class LanTests(unittest.TestCase):
    def test_addresses(self):
        self.assertEqual(check_address("192.168.2.120"), "192.168.2.120")
        for value in ("127.0.0.1", "0.0.0.0", "::1", "not-an-ip"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                check_address(value)
        with patch("socket.socket.connect", side_effect=OSError("offline")):
            with self.assertRaisesRegex(ValueError, "--lan 192.168"):
                lan_address()

    def test_terminal_chat_uses_https_for_its_own_lan_server(self):
        self.assertEqual(
            local_base("0.0.0.0", 8765, secure=True), "https://127.0.0.1:8765"
        )

    @unittest.skipUnless(shutil.which("openssl"), "openssl required")
    def test_certificate_names_the_address_and_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / ".lan"
            cert, key = lan_certificate(folder, "192.168.2.120")
            text = subprocess.run(
                ["openssl", "x509", "-in", str(cert), "-noout", "-text"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            self.assertIn("IP Address:192.168.2.120", text)
            self.assertIn("IP Address:127.0.0.1", text)
            self.assertEqual(key.stat().st_mode & 0o777, 0o600)
            before = cert.read_bytes()
            self.assertEqual(lan_certificate(folder, "192.168.2.120")[0], cert)
            self.assertEqual(cert.read_bytes(), before)  # reused
            lan_certificate(folder, "10.0.0.7")  # address changed: new cert
            self.assertNotEqual(cert.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
