"""Binary interchange across the Python coordinator and JavaScript workers."""

import json
import shutil
import struct
import subprocess
import unittest
from pathlib import Path

import numpy as np

from ygaraty.backends.webgpu.browser_assets import BROWSER_ROOT
from ygaraty.training.tensor_transport import decode, encode


class TensorTransportTests(unittest.TestCase):
    def test_bounds_checksum_and_finite_validation(self):
        frame = encode({"version": 1}, {"w": np.array([[1.0, -2.5]])})
        meta, tensors = decode(frame)
        self.assertEqual(meta, {"version": 1})
        np.testing.assert_array_equal(tensors["w"], [[1, -2.5]])
        for invalid in (
            b"",
            b"BAD!" + frame[4:],
            frame[:-1],
            frame + b"x",
            frame[:4] + struct.pack("<I", 99999999) + frame[8:],
        ):
            with self.assertRaises(ValueError):
                decode(invalid)
        with self.assertRaises(ValueError):
            encode({}, {"nan": [[np.nan]]})

    @unittest.skipUnless(
        shutil.which("node"), "Node required for cross-runtime transport"
    )
    def test_python_javascript_binary_round_trip(self):
        script = """
import {decodeFrame, encodeFrame} from MODULE;
const chunks=[]; for await (const chunk of process.stdin) chunks.push(chunk);
const input=new Uint8Array(Buffer.concat(chunks));
const frame=await decodeFrame(input);
frame.metadata.roundtrip='typescript';
process.stdout.write(await encodeFrame(frame.metadata,frame.tensors));
""".replace(
            "MODULE",
            json.dumps(Path(BROWSER_ROOT / "tensor_transport.mjs").as_uri()),
        )
        values = {
            "w": np.array([[1e-100, -3.125, 1e100]]),
            "b": np.zeros((2, 1)),
        }
        result = subprocess.run(
            ["node", "--input-type=module", "-e", script],
            input=encode({"version": 7}, values),
            capture_output=True,
            check=True,
            timeout=30,
        )
        meta, restored = decode(result.stdout)
        self.assertEqual(meta, dict(version=7, roundtrip="typescript"))
        for name in values:
            np.testing.assert_array_equal(restored[name], values[name])


if __name__ == "__main__":
    unittest.main()
