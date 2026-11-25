"""Inkling 2 NumPy reference (DO-95) on a tiny synthetic model: structure checks that need no checkpoint."""

import unittest

import numpy as np

from ygaraty.backends.webgpu.checkpoints import inkling2_gguf as ink

C = ink.Inkling2Config(
    layers=6, hidden=16, heads=4, kv_heads=2, head_dim=8, vocab=40, vocab_unpadded=37, experts=4, top_k=2,
    expert_ffn=8, expert_scale=2.0, eps=1e-6, window=3, sliding=(True, False, True, True, False, True), conv=4,
    res_block=2, rel_dim=4, rel_extent=4, qk_clip=3.0, rel_clip=2.0, log_floor=2, log_alpha=0.1, situ_beta=4.0,
    situ_linear_beta=25.0, logit_cap=30.0, loop_start=2, loop_end=4, passes=3, gate_rank=3,
)


class FakeModel:
    """The `inkling2_gguf.Model` interface over random float32 tensors in the GGUF's shapes."""

    def __init__(self, seed=0, zero_gates=False):
        c, r = C, np.random.default_rng(seed)
        H, D, R, K = c.hidden, c.head_dim, c.rel_dim, c.conv
        rand = lambda *shape, s=0.3: (r.standard_normal(shape) * s).astype(np.float32)
        t = {"token_embd.weight": rand(c.vocab, H, s=1.0), "token_embd_norm.weight": 1 + rand(H, s=0.1),
             "output.weight": rand(c.vocab, H), "output_norm.weight": 1 + rand(H, s=0.1),
             "output_res_query.weight": rand(H)}
        for i in range(c.layers):
            p = f"blk.{i}."
            t |= {p + "attn_q.weight": rand(c.heads * D, H), p + "attn_k.weight": rand(c.kv_heads * D, H),
                  p + "attn_v.weight": rand(c.kv_heads * D, H), p + "attn_gate.weight": rand(c.heads * D, H),
                  p + "attn_output.weight": rand(H, c.heads * D), p + "attn_rel.weight": rand(c.heads * R, H),
                  p + "attn_rel_bias.weight": rand(c.rel_extent, R, s=1.0),
                  p + "attn_q_clip.weight": 1 + rand(c.heads, D, s=0.1),
                  p + "attn_rel_clip.weight": 1 + rand(c.heads, R, s=0.1),
                  p + "attn_sink_proj.weight": rand(c.heads, H),
                  p + "attn_k_conv.weight": rand(c.kv_heads * D, K), p + "attn_v_conv.weight": rand(c.kv_heads * D, K),
                  p + "attn_post_conv.weight": rand(H, K), p + "ffn_post_conv.weight": rand(H, K),
                  p + "ffn_res_query.weight": rand(H), p + "ffn_norm.weight": 1 + rand(H, s=0.1),
                  p + "ffn_gate_inp.weight": rand(c.experts, H), p + "exp_probs_b.bias": rand(c.experts, s=0.1)}
            if c.sliding[i]:
                t[p + "attn_norm.weight"] = 1 + rand(H, s=0.1)
            else:
                for n in "qkv":
                    t[p + f"attn_{n}_inp_norm.weight"] = 1 + rand(H, s=0.1)
                t[p + "attn_k_conv_proj.weight"] = rand(c.kv_heads * K, H)
                t[p + "attn_v_conv_proj.weight"] = rand(c.kv_heads * K, H)
            if i > 0:
                t[p + "attn_res_query.weight"] = rand(1 if c.sliding[i] else 3, H)
            if c.loop_start <= i < c.loop_end:
                t[p + "loop_msg_query.weight"] = rand(4, H)
                for w in ink.LOOP_CTRLS:
                    t[p + f"loop_{w}_ctrl_down.weight"] = rand(c.gate_rank, 2 * H)
                    for g in ("gain", "forget"):
                        t[p + f"loop_{w}_ctrl_{g}.weight"] = (
                            np.zeros((H, c.gate_rank), np.float32) if zero_gates else rand(H, c.gate_rank)
                        )
        self.t = t
        self.experts = {(i, e): (rand(2 * c.expert_ffn, H), rand(H, c.expert_ffn))
                        for i in range(c.layers) for e in range(c.experts)}
        self.config = c

    def has(self, name):
        return name in self.t

    def dense(self, name):
        return self.t[name]

    def rows(self, name, rows):
        return self.t[name][np.asarray(rows)]

    def expert(self, layer, e):
        return self.experts[(layer, e)]


class Inkling2ReferenceTest(unittest.TestCase):
    def test_visits_follow_the_loop(self):
        seen = []
        ink.run(FakeModel(), [1, 2, 3], trace=lambda v, il, p, cur: seen.append((il, p)))
        self.assertEqual(seen, C.visits())
        self.assertEqual(len(seen), 2 + 3 * 2 + 2)
        self.assertEqual([ink.slot(C, il, p) for il, p in seen], [0, 1, 2, 3, 6, 7, 8, 9, 4, 5])

    def test_prefill_is_causal(self):
        model = FakeModel()
        tokens = [5, 9, 1, 30, 2, 7, 11, 3]
        full = ink.logits(model, ink.run(model, tokens))
        self.assertEqual(full.shape, (len(tokens), C.vocab_unpadded))
        self.assertTrue(np.isfinite(full).all())
        for n in (1, 4, 6):
            part = ink.logits(model, ink.run(model, tokens[:n]))
            np.testing.assert_allclose(part, full[:n], rtol=1e-4, atol=1e-5)

    def test_zero_gates_make_every_pass_repeat_the_first(self):
        outs = {}
        ink.run(FakeModel(zero_gates=True), [4, 8, 15, 16, 23],
                trace=lambda v, il, p, cur: outs.__setitem__((il, p), cur))
        for il in range(C.loop_start, C.loop_end):
            for p in (1, 2):
                np.testing.assert_allclose(outs[(il, p)], outs[(il, 0)], rtol=1e-5, atol=1e-6)

    def test_gates_change_later_passes(self):
        outs = {}
        ink.run(FakeModel(), [4, 8, 15, 16, 23], trace=lambda v, il, p, cur: outs.__setitem__((il, p), cur))
        self.assertGreater(np.abs(outs[(C.loop_end - 1, 1)] - outs[(C.loop_end - 1, 0)]).max(), 1e-3)

    def test_convolution_matches_its_formula(self):
        r = np.random.default_rng(1)
        x = r.standard_normal((5, 4)).astype(np.float32)
        w = r.standard_normal((4, C.conv)).astype(np.float32)
        dyn = r.standard_normal((5, 2 * C.conv)).astype(np.float32)  # 2 heads of 2 channels
        got = ink.convolve(C, x, w, dyn)
        past = lambda t: x[t] if t >= 0 else np.zeros(4, np.float32)
        for t in range(5):
            out = sum(w[:, k] * past(t - (C.conv - 1) + k) for k in range(C.conv))
            heads = dyn[t].reshape(2, C.conv)
            out = out + sum(np.repeat(heads[:, lag], 2) * past(t - lag) for lag in range(C.conv))
            np.testing.assert_allclose(got[t], (x[t] + out) / np.sqrt(2 * C.conv + 1), rtol=1e-5, atol=1e-6)

    def test_clip_caps_the_norm_only(self):
        x = np.array([[3.0, 4.0], [0.3, 0.4]], np.float32)
        np.testing.assert_allclose(np.linalg.norm(ink.clip(x, 1.0), axis=1), [1.0, 0.5], rtol=1e-5)


if __name__ == "__main__":
    unittest.main()


class LoopPlacementTest(unittest.TestCase):
    """families.loop_placement: the looped core stays on one worker; memory is balanced around it."""

    def plan(self, workers, **kw):
        from ygaraty.backends.webgpu.checkpoints.families import loop_placement

        layers = [10] * 6 + [6] * 12 + [10] * 6  # Inkling-like: core layers lighter, but 72 together
        return loop_placement(layers, loop=(6, 18), res_block=6, hidden=4, workers=workers, visit_bytes=[1] * 24,
                              embedding_bytes=5, head_bytes=5, **kw)

    def test_three_workers_isolate_the_core(self):
        p = self.plan(3)
        self.assertEqual(p["splits"], [0, 6, 18, 24])
        self.assertEqual([s["loop"] for s in p["stages"]], [False, True, False])
        self.assertEqual([(t["before_layer"], t["vectors"]) for t in p["transfers"]], [(6, 2), (18, 4)])
        self.assertEqual(p["transfers"][1]["bytes_per_token"], 4 * 4 * 2)

    def test_no_boundary_inside_the_core(self):
        for workers in range(1, 14):
            splits = self.plan(workers)["splits"]
            self.assertFalse([c for c in splits if 6 < c < 18], splits)
        with self.assertRaises(ValueError):
            self.plan(14)  # 12 allowed cuts: at most 13 stages

    def test_budgets(self):
        self.assertEqual(self.plan(2, budgets=[200, 100])["splits"], [0, 18, 24])
        with self.assertRaises(ValueError):
            self.plan(3, budgets=[70, 70, 70])  # the core alone needs 72 + 12 visits of state

    def test_split_check(self):
        from ygaraty.backends.webgpu.checkpoints.families import check_loop_splits

        self.assertEqual(check_loop_splits([0, 6, 18, 24], (6, 18)), [])
        self.assertEqual(len(check_loop_splits([0, 8, 16, 24], (6, 18))), 2)


class Inkling2DistributionTest(unittest.TestCase):
    """compile_distribution for inkling2 stages: boundaries carry block streams; the core stays on one stage."""

    def stages(self, bounds):
        import dataclasses

        config = dataclasses.asdict(C) | {"family": "inkling2", "sliding": list(C.sliding)}
        return [dict(layers=[a, b], fixture=dict(wire="f16"),
                     manifest=dict(config=config, first=a, last=b, embed=a == 0, head=b == C.layers, tensors={},
                                   layers=[dict(index=i) for i in range(a, b)]))
                for a, b in zip(bounds, bounds[1:])]

    def test_three_stages(self):
        from ygaraty.backends.webgpu.distribution import compile_distribution

        stages = self.stages([0, 2, 4, 6])  # tiny model: loop 2..4, blocks of 2
        plan = compile_distribution(stages)
        self.assertEqual([(t["source"], t["destination"], t["vectors"]) for t in plan["transfers"]], [(0, 1, 2), (1, 2, 3)])
        self.assertEqual([s["input_vectors"] for s in plan["stages"]], [0, 2, 3])
        self.assertEqual(stages[2]["fixture"]["plan_route"], dict(next=0, output="token"))
        self.assertEqual(stages[1]["manifest"]["plan"]["output"], "streams_4")
        compile_distribution(stages)  # stored plans are accepted again

    def test_rejects_split_core_and_gaps(self):
        from ygaraty.backends.webgpu.distribution import compile_distribution

        with self.assertRaises(ValueError):
            compile_distribution(self.stages([0, 3, 6]))  # 3 is inside the core 2..3
        stages = self.stages([0, 2, 4, 6])
        del stages[1]
        with self.assertRaises(ValueError):
            compile_distribution(stages)

    def test_boundary_vectors(self):
        self.assertEqual([ink.boundary_vectors(C, c) for c in range(7)], [1, 2, 2, 3, 3, 4, 4])


class TaggedChatTest(unittest.TestCase):
    """inference_support.chat's tagged style (Inkling 2's message tokens)."""

    FMT = dict(style="tagged", first="<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>",
               generation="", open="<|content_text|>", close="<|end_message|>",
               assistant_end="<|content_model_end_sampling|>",
               roles=dict(system="<|message_system|>", user="<|message_user|>", assistant="<|message_model|>",
                          tool="<|message_tool|>"))

    def test_prompt_matches_the_template(self):
        from ygaraty.inference_support.chat import chat_prompt

        self.assertEqual(chat_prompt([("user", "Hi")], self.FMT),
                         self.FMT["first"] + "<|message_user|><|content_text|>Hi<|end_message|>")
        # the effort message goes after system messages, before the first other one; assistant turns are closed
        self.assertEqual(chat_prompt([("system", "S"), ("user", "a"), ("assistant", "b"), ("user", "c")], self.FMT),
                         "<|message_system|><|content_text|>S<|end_message|>" + self.FMT["first"]
                         + "<|message_user|><|content_text|>a<|end_message|>"
                         + "<|message_model|><|content_text|>b<|end_message|><|content_model_end_sampling|>"
                         + "<|message_user|><|content_text|>c<|end_message|>")

    def test_validation(self):
        from ygaraty.inference_support.chat import chat_format

        self.assertEqual(chat_format(self.FMT)["style"], "tagged")
        for broken in (dict(self.FMT, roles={"user": "x"}), dict(self.FMT, extra="1"), dict(self.FMT, close=3)):
            with self.assertRaises(ValueError):
                chat_format(broken)

    def test_stored_plan_check_without_tasks(self):
        from ygaraty.backends.webgpu.preparation import validate_distribution_plan

        plan = dict(version=3, stages=[dict(owner=0, family="inkling2")], transfers=[])
        validate_distribution_plan(plan, plan)
        with self.assertRaises(ValueError):
            validate_distribution_plan(dict(plan, transfers=[1]), plan)
