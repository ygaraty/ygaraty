"""Summarize raw coder waves without hiding failed requests or quality gates."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def summarize(root):
    rows = []
    gates = []
    for file in sorted(root.glob('*/evidence/*/trials.jsonl')):
        host, engine = file.parts[-4], file.parent.name
        if engine not in ('llama', 'vllm', 'vllm-cudagraph'):
            continue
        events = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
        for event in events:
            if event['kind'] not in ('wave', 'warmup', 'gpu'):
                gates.append({'host': host, 'variant': engine, **event})
        keys = sorted({(e['context'], e['concurrency'], e['long_prompt'])
                       for e in events if e['kind'] == 'wave'})
        for context, concurrency, long_prompt in keys:
            waves = [e for e in events if e['kind'] == 'wave' and
                     (e['context'], e['concurrency'], e['long_prompt']) ==
                     (context, concurrency, long_prompt)]
            requests = [request for wave in waves for request in wave['requests']]
            good = [r['response'] for r in requests if r['response'].get('status') == 200]
            ttft = [r['ttft_seconds'] for r in good if r.get('ttft_seconds') is not None]
            decode = [r['decode_tokens_per_second'] for r in good
                      if r.get('decode_tokens_per_second') is not None]
            prompt_tokens = [r.get('usage', {}).get('prompt_tokens', 0) for r in good]
            rows.append(dict(
                host=host, engine=engine, context=context, concurrency=concurrency,
                workload='near_full' if long_prompt else 'short', waves=len(waves),
                requests=len(requests), http_success=len(good),
                reference_correct=sum(r['reference_found'] for r in requests),
                all_waves_passed=all(w['passed'] for w in waves),
                median_ttft_seconds=statistics.median(ttft) if ttft else None,
                max_ttft_seconds=max(ttft) if ttft else None,
                median_decode_tokens_per_second=statistics.median(decode) if decode else None,
                aggregate_tokens_per_second=sum(r.get('usage', {}).get('completion_tokens', 0)
                                                for r in good) / sum(w['seconds'] for w in waves),
                prompt_tokens_min=min(prompt_tokens, default=0),
                prompt_tokens_max=max(prompt_tokens, default=0),
                quality_gate_failed=any(e['kind'] == 'correctness' and e.get('passed') is False
                                        for e in events),
            ))
    if rows:
        with (root / 'summary.csv').open('w') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (root / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    (root / 'gates.json').write_text(json.dumps(gates, indent=2) + '\n')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2))
