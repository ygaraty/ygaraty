// DO-25: speculative decoding with n-gram (prompt lookup) drafts, checked exactly.
//
// The first stage proposes up to k draft tokens after the last generated one; one pass through every stage
// computes the greedy token after each of the 1 + k positions; the last stage keeps the drafts that equal
// those tokens and sends the tokens back. Every kept token is the token plain greedy decoding gives.
//
// The first stage needs no word from the last stage on how many drafts were kept: tokens come back in order,
// and a group ends at the first token that differs from its draft, or after 1 + k tokens.

// The k tokens after the most recent earlier occurrence of the history's last m tokens, trying m = n, n - 1,
// ..., min; none when no such suffix recurs.
export function ngramDrafts(history, n, k, min = 1) {
  return ngramMatch(history, n, k, min).drafts;
}

// The same, with the length m of the match that proposed the drafts (0: none).
export function ngramMatch(history, n, k, min = 1) {
  if (!(k > 0)) return { drafts: [], m: 0 };
  const end = history.length;
  for (let m = Math.min(n, end - 1); m >= Math.max(1, min); m--) {
    for (let start = end - m - 1; start >= 0; start--) {
      let same = true;
      for (let j = 0; j < m; j++) {
        if (history[start + j] !== history[end - m + j]) { same = false; break; }
      }
      if (!same) continue;
      const drafts = history.slice(start + m, start + m + k);
      if (drafts.length) return { drafts, m };
    }
  }
  return { drafts: [], m: 0 };
}

// Cost-aware drafting (WS-19, revised in WS-20). A pass's time by its rows: 1; 2-8 (one group of the
// rows-per-thread kernel); 9 and more. Until measured, a bucket's time is the 1-row time times its prior ratio.
const bucket = (rows) => (rows <= 1 ? 0 : rows <= 8 ? 1 : 2);
const PRIOR_RATIO = [1, 1.6, 2.6];
// Acceptance of draft position i, given that position i - 1 was kept: decayed counts, shrunk towards a prior
// of PRIOR_STRENGTH pseudo-observations at P0 for position 1 and at position i - 1's estimate for the others.
// Pessimistic: a conversation drafts only once its own evidence says drafts hold.
export const P0 = 0.25, PRIOR_STRENGTH = 2, DECAY = 0.95;

class CostModel {
  constructor(k) {
    this.kept = new Array(k + 1).fill(0);
    this.tried = new Array(k + 1).fill(0);
    this.ms = [null, null, null];  // moving average of a pass's time per bucket
  }
  // An opportunity's draft at position i (its earlier drafts held), scored against the token that came.
  observe(i, held) {
    this.kept[i] = DECAY * this.kept[i] + (held ? 1 : 0);
    this.tried[i] = DECAY * this.tried[i] + 1;
  }
  // A completed pass's rows and time.
  record(rows, ms) {
    if (!Number.isFinite(ms) || ms < 0) return;
    const b = bucket(rows);
    this.ms[b] = this.ms[b] === null ? ms : 0.8 * this.ms[b] + 0.2 * ms;
  }
  accept(most) {
    const p = [];
    for (let i = 1, q = P0; i <= most; i++) {
      q = (this.kept[i] + PRIOR_STRENGTH * q) / (this.tried[i] + PRIOR_STRENGTH);
      p.push(q);
    }
    return p;
  }
  time(rows) {
    const b = bucket(rows);
    if (this.ms[b] !== null) return this.ms[b];
    const known = this.ms.findIndex((x) => x !== null);
    const base = known < 0 ? 1 : this.ms[known] / PRIOR_RATIO[known];
    return base * PRIOR_RATIO[b];
  }
  // The number of drafts (0..most) that maximizes expected tokens per unit of time. A pass carries `replay`
  // rows; drafts that fail make the next pass carry the kept rows again, which costs that pass a group of
  // rows instead of one.
  best(most, replay) {
    const p = this.accept(most), again = Math.max(0, this.time(2) - this.time(1));
    let bestJ = 0, bestRate = 1 / this.time(replay + 1), expected = 1, all = 1;
    for (let j = 1; j <= most; j++) {
      all *= p[j - 1];
      expected += all;
      const rate = expected / (this.time(replay + 1 + j) + (1 - all) * again);
      if (rate > bestRate) { bestRate = rate; bestJ = j; }
    }
    return bestJ;
  }
}

// Last stage: `out[i]` is the greedy token after row i of a checking pass whose rows are the last token and
// `drafts`. Returns the tokens to send: out[0], then out[i + 1] for each draft i equal to out[i], up to the
// first difference; cut after a stop token.
export function acceptDrafts(out, drafts, stopIds = []) {
  if (out.length !== drafts.length + 1) throw new Error(`${out.length} outputs for ${drafts.length} drafts`);
  const tokens = [out[0]];
  for (let i = 0; i < drafts.length && drafts[i] === out[i] && !stopIds.includes(out[i]); i++) tokens.push(out[i + 1]);
  const stop = tokens.findIndex((t) => stopIds.includes(t));
  return stop >= 0 ? tokens.slice(0, stop + 1) : tokens;
}

// After this many tokens without drafting (adaptive k fell to 0), drafting resumes with one draft; each
// resumption whose draft fails doubles the pause, up to PAUSE_MAX.
export const PAUSE = 16, PAUSE_MAX = 128;

// First stage, one conversation: what the last pass proposed and what has come back of it.
// `min`: the shortest match that may propose drafts. `adapt`: false, up to k drafts; true (WS-17), the number of
// drafts follows their acceptance (all kept: 2 more, up to k; any rejected: as many as were kept; at 0, none
// for a pause, then 1); "cost" (WS-19, WS-20), the number that maximizes expected tokens per measured
// millisecond, with acceptance learned from every opportunity's drafts, sent or not.
//
// A pass's rows are [replay | token | drafts]. After a rejection the stages restore the state they saved
// before the rejected pass, so the next pass first reruns that pass's kept rows (its own replay included):
// one run, at most k + 1 rows.
export class Drafter {
  constructor(prompt, { n = 3, k = 4, min = 1, adapt = false } = {}) {
    if (!Number.isSafeInteger(n) || n < 1 || !Number.isSafeInteger(k) || k < 0 ||
        !Number.isSafeInteger(min) || min < 1 || min > n) throw new Error("Invalid draft settings");
    this.n = n;
    this.k = k;
    this.min = min;
    this.adapt = adapt === "cost" ? "cost" : Boolean(adapt);
    this.cost = this.adapt === "cost" ? new CostModel(k) : null;
    this.shadows = [];  // cost-aware: open opportunities {at, drafts}, scored as tokens arrive
    this.sentAt = null;  // when the current pass was sent (cost-aware)
    this.rowsSent = 0;
    this.limit = k;  // the current number of drafts (adaptive)
    this.pause = 0;
    this.pauseLength = PAUSE;
    this.history = Array.from(prompt);
    this.rows = null;  // the last pass's rows: [...replay, token, ...drafts]; null after a prompt
    this.r = 0;  // the last pass's replay rows
    this.drafts = [];
    this.got = 0;
  }

  // A token from the last stage. Null while its group goes on; when the group is complete, the next pass:
  // `replay` (rows to rerun after the stages restore their saved state, when a draft was rejected), `token`
  // (the last token received) and `drafts`, at most `room` of them.
  take(token, room = Infinity, now = null) {
    this.history.push(token);
    if (this.cost) this.score(token);
    const i = this.got++;
    if (i < this.drafts.length && token === this.drafts[i]) return null;  // the draft was kept: more follow
    const kept = i;  // drafts kept in this group
    const replay = this.rows && kept < this.drafts.length ? this.rows.slice(0, this.r + 1 + kept) : [];
    room = Math.max(0, Math.min(room, this.k - replay.length));  // a pass has at most k + 1 rows
    let drafts;
    if (this.cost) drafts = this.takeCost(replay, room, now);
    else {
      if (this.adapt && this.drafts.length) {
        const resumed = this.limit === 1 && this.drafts.length === 1;
        this.limit = kept === this.drafts.length ? Math.min(this.k, this.limit + 2) : kept;
        if (kept > 0) this.pauseLength = PAUSE;
        if (this.limit === 0) {
          if (resumed) this.pauseLength = Math.min(PAUSE_MAX, 2 * this.pauseLength);
          this.pause = this.pauseLength;
        }
      } else if (this.adapt && this.limit === 0 && --this.pause <= 0) this.limit = 1;
      const want = this.adapt ? this.limit : this.k;
      drafts = ngramDrafts(this.history, this.n, Math.min(want, room), this.min);
    }
    this.rows = [...replay, token, ...drafts];
    this.r = replay.length;
    this.drafts = drafts;
    this.got = 0;
    return { replay, token, drafts, kept };
  }

  // Cost-aware: score the open opportunities' drafts against the token that just arrived.
  score(token) {
    const at = this.history.length - 1;
    this.shadows = this.shadows.filter((sh) => {
      const i = at - sh.at, held = token === sh.drafts[i];
      this.cost.observe(i + 1, held);
      return held && i + 1 < sh.drafts.length;
    });
  }

  takeCost(replay, room, now) {
    if (this.rows && this.sentAt !== null && now !== null) this.cost.record(this.rowsSent, now - this.sentAt);
    const found = ngramDrafts(this.history, this.n, this.k, this.min);
    if (found.length) this.shadows.push({ at: this.history.length, drafts: found });
    const most = Math.min(found.length, room);
    const drafts = most ? found.slice(0, this.cost.best(most, replay.length)) : [];
    this.rowsSent = replay.length + 1 + drafts.length;
    this.sentAt = now;
    return drafts;
  }
}
