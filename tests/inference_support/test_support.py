"""Backend-neutral behavior, with no model, GPU, network or host framework."""
import io
import sys
from types import SimpleNamespace

import pytest

from ygaraty.inference_support.artifacts import local_file
from ygaraty.inference_support.chat import chat_prompt, encode_ids, validate_conversation_id
from ygaraty.inference_support.http import reply, stream
from ygaraty.inference_support.roster import ContributorRoster
from ygaraty.inference_support.throughput import Throughput


def test_prompt_and_tokenizer_adapter():
    tokenizer = SimpleNamespace(encode_batch=lambda texts: [SimpleNamespace(ids=[len(texts[0])])])
    assert encode_ids(tokenizer, 'hello') == [5]
    assert '<|im_start|>user\nhello<|im_end|>' in chat_prompt([('user', 'hello')])
    assert validate_conversation_id('c_1-A') == 'c_1-A'
    with pytest.raises(ValueError):
        validate_conversation_id('../escape')


def test_artifacts_reject_parent_and_symlink_escape(tmp_path):
    root = tmp_path / 'model'; root.mkdir()
    (root / 'config').write_text('{}')
    (tmp_path / 'secret').write_text('private')
    (root / 'link').symlink_to(tmp_path / 'secret')
    assert local_file(root, 'config') == root / 'config'
    for name in ('../secret', 'link', str(tmp_path / 'secret'), 'missing'):
        with pytest.raises(ValueError):
            local_file(root, name)


def test_throughput_counts_concurrent_wall_time_once():
    now = [0.0]
    tracker = Throughput(clock=lambda: now[0])
    tracker.start('a'); tracker.start('b')
    now[0] = 2.0
    tracker.token('a'); tracker.token('b')
    tracker.finish('a'); tracker.finish('b')
    result = tracker.snapshot()
    assert result['generated_tokens'] == 2
    assert result['active_seconds'] == 2
    assert result['average_tokens_per_s'] == 1


def test_roster_restart_retires_live_reservations(tmp_path):
    path = tmp_path / 'roster.sqlite'
    roster = ContributorRoster([{'slot': 0}], database=path)
    roster.reserve('worker', 0)
    roster.activate('worker')
    restored = ContributorRoster([{'slot': 0}], database=path)
    assert restored.records['worker']['state'] == 'former'
    with pytest.raises(ValueError):
        restored.touch('worker')


def test_http_reply_and_stream_close_iterator():
    headers = {}
    handler = SimpleNamespace(wfile=io.BytesIO(), send_response=lambda code: headers.update(status=code),
                              send_header=lambda k,v: headers.update({k:v}), end_headers=lambda: None)
    reply(handler, 200, {'ok': True})
    assert headers['Content-Length'] == str(len(handler.wfile.getvalue()))
    closed = []
    def events():
        try:
            yield {'text': 'hello'}
        finally:
            closed.append(True)
    output = []
    stream(SimpleNamespace(stream=lambda it: output.extend(it)), SimpleNamespace(say=lambda text: None), events())
    assert output == [{'text': 'hello'}]
    assert closed == [True]


def test_package_imports_with_only_standard_library():
    import subprocess
    from pathlib import Path
    source = Path(__file__).resolve().parents[2]
    code = '''import importlib, pkgutil, sys
sys.path.insert(0, sys.argv[1])
import ygaraty.inference_support as support
for module in pkgutil.walk_packages(support.__path__, support.__name__ + '.'):
    importlib.import_module(module.name)
assert not {'numpy', 'django', 'vllm', 'ygaraty.backends.webgpu', 'ygaraty_cloud'} & set(sys.modules)
'''
    subprocess.run([sys.executable, '-I', '-S', '-c', code, str(source)], check=True)
