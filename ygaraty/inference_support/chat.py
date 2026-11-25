"""Backend-independent prompt formatting and conversation identifiers."""
import re

QWEN_CHAT = dict(first="", generation="<|im_start|>assistant\n<think>\n\n</think>\n\n")


CHAT_SEPARATORS = ("\n", "<|im_sep|>")  # after the role: ChatML's newline, Phi-4's <|im_sep|>
CHAT_ENDS = ("\n", "")  # after <|im_end|>: ChatML's newline, nothing (Phi-4)


def encode_ids(tokenizer, text):
    """Token IDs of `text`, with the GIL released while the Rust tokenizer runs.

    `Tokenizer.encode` holds the GIL: one ~1.2 M-character prompt froze every other coordinator thread (all
    sessions' streams and contributor polls) for about 0.4 s, and four arriving together for 1.6 s. `encode_batch`
    returns the same IDs and releases it (4 such prompts: 0.58 s, other threads paused at most 24 ms).
    """
    return tokenizer.encode_batch([text])[0].ids


TAGGED_KEYS = ("first", "generation", "roles", "open", "close", "assistant_end")


def _tagged_format(value):
    """DO-95: messages as `{roles[role]}{open}{content}{close}` (+ `assistant_end` after an assistant turn), with
    `first` (e.g. Inkling 2's thinking-effort system message) before the first non-system message. Inkling 2:
    roles <|message_user|> ..., open <|content_text|>, close <|end_message|>, generation empty (the model opens its
    own turn)."""
    roles = value.get("roles")
    if (set(value) - {"style"} != set(TAGGED_KEYS) or not isinstance(roles, dict)
            or set(roles) != {"system", "user", "assistant", "tool"}
            or not all(isinstance(value[k], str) and len(value[k]) <= 4096 for k in TAGGED_KEYS if k != "roles")
            or not all(isinstance(v, str) and 0 < len(v) <= 64 for v in roles.values())):
        raise ValueError("Invalid chat format")
    return dict(style="tagged", **{k: (dict(roles) if k == "roles" else value[k]) for k in TAGGED_KEYS})


def chat_format(value=None):
    """A bundle's ChatML pieces (`ygaraty model inspect` derives them from the checkpoint's own template),
    checked; Qwen3.5's (thinking off) when absent. A message is `<|im_start|>{role}{sep}{content}<|im_end|>{end}`
    (`sep`, `end`: ChatML's newlines when absent)."""
    if value is None:
        return dict(QWEN_CHAT, sep="\n", end="\n")
    if isinstance(value, dict) and value.get("style") == "tagged":
        return _tagged_format(value)
    if (not isinstance(value, dict) or not isinstance(value.get("first"), str) or not isinstance(value.get("generation"), str)
            or len(value["first"]) > 4096 or not value["generation"].startswith("<|im_start|>assistant") or len(value["generation"]) > 256
            or value.get("sep", "\n") not in CHAT_SEPARATORS or value.get("end", "\n") not in CHAT_ENDS):
        raise ValueError("Invalid chat format")
    return dict(first=value["first"], generation=value["generation"], sep=value.get("sep", "\n"), end=value.get("end", "\n"))


def chat_message(role, content, fmt):
    if fmt.get("style") == "tagged":
        end = fmt["assistant_end"] if role == "assistant" else ""
        return f"{fmt['roles'][role]}{fmt['open']}{content}{fmt['close']}{end}"
    return f"<|im_start|>{role}{fmt['sep']}{content}<|im_end|>{fmt['end']}"


def chat_prompt(messages, fmt=None):
    """The prompt for a whole message list ([(role, content)]) in the chat format, ending with the assistant's
    opening; the conversation prefix only when there is no system message (the template's own default)."""
    fmt = chat_format(fmt)
    if fmt.get("style") == "tagged":  # `first` goes before the first non-system message, as Inkling 2's template
        lead = next((i for i, (role, _) in enumerate(messages) if role != "system"), len(messages))
        parts = [chat_message(role, content, fmt) for role, content in messages]
        return "".join(parts[:lead]) + fmt["first"] + "".join(parts[lead:]) + fmt["generation"]
    text = "" if messages and messages[0][0] == "system" else fmt["first"]
    return text + "".join(chat_message(role, content, fmt) for role, content in messages) + fmt["generation"]


def validate_conversation_id(value):
    """An opaque session-local context selector, not an authorization credential."""
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', value):
        raise ValueError('conversation_id must be 1–64 letters, digits, underscores or hyphens')
    return value
