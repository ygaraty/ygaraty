"""Native runtime selection and process-group lifecycle."""

import argparse
import signal
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ygaraty.backends.webgpu import contributors


def args(*argv):
    parser = argparse.ArgumentParser()
    contributors.add_parser(parser.add_subparsers(dest="command"))
    return parser.parse_args(["contribute", *argv])


class ContributorTests(unittest.TestCase):
    @patch("shutil.which", side_effect=lambda value: "/bin/" + value)
    def test_runtime_commands_and_validation(self, which):
        for runtime in ("deno", "wgpu", "node", "bun"):
            cmd = contributors.command(args("--runtime", runtime, "--check"))
            self.assertTrue(
                cmd[0].endswith(
                    "/deno" if runtime == "wgpu" else "/" + runtime
                )
            )
            self.assertEqual(cmd[-1], "--check")
            if runtime in {"deno", "wgpu"}:
                self.assertIn("--allow-net", cmd)
                self.assertNotIn("-A", cmd)
        for extra in (["--num-workers", "0"], [], ["file:///x#key=a"]):
            with self.assertRaises(ValueError):
                contributors.command(args(*extra))
        with self.assertRaises(ValueError):
            contributors.command(
                args(
                    "--runtime",
                    "deno",
                    "--check",
                    "--dawn-option",
                    "backend=metal",
                )
            )
        cmd = contributors.command(
            args("https://example.test/#key=a", "--num-workers", "2")
        )
        self.assertEqual(cmd[-1], "2")
        self.assertTrue(Path(cmd[1]).is_file())

    @patch("shutil.which", return_value="/bin/node")
    @patch("signal.signal")
    @patch("os.killpg")
    @patch("subprocess.Popen")
    def test_failed_worker_stops_other_workers(self, popen, kill, sig, which):
        first = Mock(pid=111)
        first.poll.return_value = 1
        second = Mock(pid=222)
        second.poll.return_value = None
        popen.side_effect = [first, second]
        result = contributors.run(
            args(
                "https://example.test/#key=a",
                "--runtime",
                "node",
                "--num-workers",
                "2",
            )
        )
        self.assertEqual(result, 1)
        self.assertEqual(popen.call_count, 2)
        kill.assert_called_once_with(222, signal.SIGTERM)
        second.wait.assert_called_once_with(timeout=5)

    @patch("shutil.which", return_value="/bin/node")
    @patch("signal.signal")
    @patch("os.killpg")
    @patch("subprocess.Popen")
    def test_partial_spawn_failure_stops_started_worker(
        self, popen, kill, sig, which
    ):
        first = Mock(pid=111)
        first.poll.return_value = None
        popen.side_effect = [first, OSError("spawn failed")]
        with self.assertRaises(OSError):
            contributors.run(
                args(
                    "https://example.test/#key=a",
                    "--runtime",
                    "node",
                    "--num-workers",
                    "2",
                )
            )
        kill.assert_called_once_with(111, signal.SIGTERM)
