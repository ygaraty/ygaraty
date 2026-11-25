// Formatting for the session page tiles and the chat page's status bar.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {barText, duration, facts, stateText, statTiles, workerCoordinator, workerLinks, workerState, workersText} from '../../../ygaraty/backends/webgpu/browser/session_stats_view.mjs';

const status = {
  model: 's5', state: 'ready', context_tokens: 4608, max_output_tokens: 256, slots: 1, route: 'relay', network: null,
  workers_required: 1, workers_ready: 1,
  stats: {total_tokens: 1234, active_requests: 0, peak_active_requests: 1, throughput_tokens_per_s: 2.5, uptime_s: 3725,
    requests: {completed: 3}, tokens_per_s: {average: 19.84}, ttft_s: {average: 0.613}},
};

test('chat status bar: tokens, speed, TTFT, context, chats, network', () => {
  assert.equal(barText(status), '1,234 tokens · avg 19.8 tok/s · TTFT 0.61 s · context 4,608 · 0/1 chats · relay');
  assert.equal(barText({stats: {}}), '– tokens · avg – · TTFT – · context – · –/– chats · –');
  assert.match(barText({...status, network: 'WebRTC: A→B connected'}), /WebRTC: A→B connected$/);
});

test('session page tiles, facts and state', () => {
  const tiles = Object.fromEntries(statTiles(status).map(([value, label]) => [label, value]));
  assert.equal(tiles['tokens so far'], '1,234');
  assert.equal(tiles['chats now / at a time'], '0 / 1');
  assert.equal(tiles.uptime, '1 h 2 min');
  assert.deepEqual(facts(status), ['1 of 1 worker ready', '4,608-token context', 'up to 256 output tokens', '1 chat at a time', 'network: relay']);
  assert.deepEqual(stateText(status), ['Ready', 'ready']);
  assert.deepEqual(stateText({state: 'unavailable', workers_required: 2, workers_ready: 1}), ['Waiting for workers (1 of 2 ready)', 'waiting']);
  assert.deepEqual(stateText(null), ['Server unavailable', '']);
  assert.equal(duration(65), '1 min 5 s');
});

test('workers: state, coordinator heartbeat and links', () => {
  const ready = {stage: 'A', state: 'ready', coordinator: 'connected', route: 'WebRTC',
    links: [{peer: 'B', state: 'connected', path: 'direct', protocol: 'udp', rtt_ms: 0.4}]};
  const loading = {stage: 'B', state: 'downloading', percent: 44.6, coordinator: 'late', heartbeat_age_s: 41.2,
    links: [{peer: 'A', state: 'connecting'}], route: 'relay (WebRTC pending)'};
  const empty = {stage: 'C', state: 'waiting'};
  assert.equal(workerState(ready), 'ready');
  assert.equal(workerState(loading), 'downloading 45%');
  assert.equal(workerState(empty), 'waiting for someone to open the link');
  assert.equal(workerCoordinator(loading), 'late (last heard 41 s ago)');
  assert.equal(workerCoordinator(empty), '–');
  assert.equal(workerLinks(ready), 'B: direct udp 0.4 ms');
  assert.equal(workerLinks(loading), 'A: connecting');
  assert.equal(workerLinks({...loading, links: []}), 'relay (WebRTC pending)');
  assert.equal(workerLinks({stage: 'A', state: 'ready', coordinator: 'connected'}), 'none (one worker)');
  assert.equal(workersText([ready, loading, empty]), 'A ready, B 45% (late), C waiting');
  assert.match(barText({...status, workers: [ready, loading]}), /workers A ready, B 45% \(late\) · relay$/);
});
