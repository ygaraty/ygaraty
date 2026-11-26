import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('summary', Path(__file__).parents[1] / 'tools/summarize_coder_backends.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)

class SummaryTests(unittest.TestCase):
    def test_failed_requests_and_quality_gate_remain_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'gpu' / 'evidence' / 'llama' / 'trials.jsonl'
            path.parent.mkdir(parents=True)
            events = [
                {'kind': 'correctness', 'passed': False},
                {'kind': 'wave', 'context': 4096, 'concurrency': 2, 'long_prompt': True,
                 'seconds': 4, 'passed': False, 'requests': [
                    {'reference_found': True, 'response': {'status': 200, 'ttft_seconds': 1,
                     'decode_tokens_per_second': 3, 'usage': {'completion_tokens': 10, 'prompt_tokens': 3500}}},
                    {'reference_found': False, 'response': {'status': 500}}]},
            ]
            path.write_text('\n'.join(map(json.dumps, events)))
            row = summary.summarize(root)[0]
            self.assertEqual(row['requests'], 2)
            self.assertEqual(row['http_success'], 1)
            self.assertEqual(row['aggregate_tokens_per_second'], 2.5)
            self.assertTrue(row['quality_gate_failed'])
            self.assertFalse(row['all_waves_passed'])
            self.assertEqual(row['prompt_tokens_max'], 3500)

if __name__ == '__main__':
    unittest.main()
