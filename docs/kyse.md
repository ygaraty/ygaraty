# Run a compressed Kyse model

Ygaraty can import Kyse's English-vocabulary Qwen3.5-0.8B files and run them
on one contributor. Install Ygaraty with the `server` and `prepare` extras,
then point launch at a directory containing `model.bin` and `tokenizer.bin`:

```sh
ygaraty launch --model kyse-english \
  --checkpoint /path/to/kyse/en-q4-mix-down \
  --num-workers 1 --open-chat
```

The first launch imports and caches the checkpoint automatically. Its pruned
token IDs and quantized weights are preserved; Kyse source code is not needed.
Launch prints the estimated contributor GPU memory requirement for your context
size and a contributor link. Open that link in a supported browser, share it,
or start a native GPU contributor on your own machine:

```sh
ygaraty contribute --runtime wgpu '<contributor link>'
```

See [command-line contributors](contributing-gpu.md#contributors-from-command-line)
for runtime setup. Once the contributor loads the weights, use terminal chat,
the web chat opened by `--open-chat`, or the [OpenAI-compatible API](api.md).

For an explicit import that can be inspected or reused:

```sh
ygaraty model import-kyse /path/to/kyse/en-q4-mix-down --output ./kyse-imported
ygaraty model inspect ./kyse-imported
```

QW35 versions 1–4 support BF16, Q8, Q4 and mixed Q8/Q4 matrices. BF16 expands
losslessly to F32; Q8 and Q4 are repacked without requantization. Low-rank
correction variants are rejected. This path currently supports the 0.8B
geometry and one worker; it does not establish support for Kyse 9B models.

The `en-q4-mix-down` checkpoint was tested on a native WebGPU contributor:
two greedy chat responses matched Kyse's Metal reference exactly. Tokenization
matched the C reference on 108 cases, including Unicode and chat delimiters.
These are compatibility checks, not a model quality benchmark.
