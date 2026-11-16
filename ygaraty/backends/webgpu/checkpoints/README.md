# Local checkpoint loading

Audited GGUF family descriptors, packed tensor ranges and preparation helpers.
This package has no cloud/platform imports. It retains tensor-only GGUF loading,
chunked embeddings/heads, fused tensor views and existing self-test references.
Qwen3-Next remains admission-only and cannot produce a serving bundle.

Preparation computes numerical self-test references on an authorized preparation
host. Offline tests replace that computation with synthetic references; they do
not run checkpoint inference. Header access is local-only and bounded. See the
public operator guide for the `ygaraty model` commands.
