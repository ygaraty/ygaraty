"""Convert a locally supplied ChatML template to protocol pieces."""

import json


def _tagged(template, render):
    """DO-95: Inkling 2's message-token template. Its pieces are read from renderings of the template itself, so a
    changed effort default or role token shows up here (or fails the consistency check)."""
    marker = "\u2063USER\u2063"
    pieces = dict(style="tagged", open="<|content_text|>", close="<|end_message|>",
                  assistant_end="<|content_model_end_sampling|>", generation="",
                  roles=dict(system="<|message_system|>", user="<|message_user|>", assistant="<|message_model|>", tool="<|message_tool|>"))
    one = render([dict(role="user", content=marker)], True)
    user = f"{pieces['roles']['user']}{pieces['open']}{marker}{pieces['close']}"
    if not one.endswith(user):
        raise ValueError("Unexpected Inkling 2 chat template rendering")
    pieces["first"] = one[: -len(user)]
    convo = [dict(role="user", content="a"), dict(role="assistant", content="b"), dict(role="user", content="c")]
    want = pieces["first"] + "".join(
        f"{pieces['roles'][m['role']]}{pieces['open']}{m['content']}{pieces['close']}"
        + (pieces["assistant_end"] if m["role"] == "assistant" else "") for m in convo)
    if render(convo, True) != want:
        raise ValueError("Inkling 2 chat template differs from the tagged chat format")
    return pieces


def chat_format(template, thinking=False):
    """The ChatML pieces of a checkpoint's own chat template (Jinja, rendered as transformers renders it):
    `first`, what a conversation starts with before the first user message (an empty system block for some
    models), `generation`, the assistant's opening after the last one (with thinking off unless asked), and
    `sep` and `end`, what follows a message's role and its <|im_end|> (ChatML's newlines; Phi-4's <|im_sep|>
    and nothing). Raises when the template is not ChatML-shaped: such a model needs its own chat format in the
    protocol."""
    from ygaraty.backends.webgpu._legacy.qwen_protocol import CHAT_ENDS, CHAT_SEPARATORS

    import jinja2.sandbox

    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"],
    )
    env.filters["tojson"] = lambda value, **kw: json.dumps(
        value, ensure_ascii=kw.get("ensure_ascii", False)
    )
    env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(
        ValueError(message)
    )
    # Inkling 2 marks assistant turns with transformers' `{% generation %}` (assistant-mask) tag: plain Jinja
    # does not know it, and it changes nothing in the text.
    compiled = env.from_string(template.replace("{%- generation -%}", "").replace("{%- endgeneration -%}", ""))
    render = lambda messages, generation: compiled.render(  # noqa: E731
        messages=messages,
        add_generation_prompt=generation,
        enable_thinking=thinking,
        bos_token="",
        eos_token="",
    )
    if "<|message_user|>" in template and "<|content_text|>" in template:
        return _tagged(template, render)
    marker = "\u2063USER\u2063"
    plain = render([dict(role="user", content=marker)], False)
    prompt = render([dict(role="user", content=marker)], True)
    shapes = [
        (sep, end)
        for sep in CHAT_SEPARATORS
        for end in CHAT_ENDS
        if plain.endswith(f"<|im_start|>user{sep}{marker}<|im_end|>{end}")
    ]
    if not shapes or not prompt.startswith(plain):
        raise ValueError(
            "The chat template is not ChatML-shaped (<|im_start|>role ... <|im_end|>): add a chat format for it"
        )
    sep, end = shapes[0]
    block = f"<|im_start|>user{sep}{marker}<|im_end|>{end}"
    first, generation = plain[: -len(block)], prompt[len(plain) :]
    if not generation.startswith("<|im_start|>assistant"):
        raise ValueError(f"Unexpected generation prompt {generation!r}")
    out = dict(
        first=first.lstrip("\n"), generation=generation, thinking=thinking
    )
    if (sep, end) != ("\n", "\n"):
        out.update(sep=sep, end=end)
    return out
