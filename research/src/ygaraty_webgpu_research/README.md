# Public research helpers

These helpers resolve only public Ygaraty implementations. No private application
host, checkout adapter or service registry is needed. Use `ygaraty train` for
local training and `ygaraty launch` / `ygaraty contribute` for serving.

Retained reusable code:

- `factorization_browser_training`: `factor_space_reference` and `dense_reference`
  compare losses without a hosted job or HTTP service.
- `convnet_remote_scaling`: the `report CASE_DIR ...` command compares saved
  scaling reports using public analysis.
- `qwen35_session`: session preparation, range hashing, delay simulation,
  browser-launch/memory sampling helpers and experiment option parsers.
- Other numerical, WebGPU and parity harnesses remain public. GPU execution needs
  the stated hardware and optional third-party runtimes.

The old HTTP translation benchmark and hosted serve/submit/worker orchestration
were retired. Their original sources and cloud deployment notes are preserved
in the internal historical archive. Old commands in research reports are evidence,
not supported standalone commands. Shared public kernels and references remain
available independently of those retired application integrations.

From a checkout with Ygaraty installed:

```sh
python -m pip install -e ./research
python -m ygaraty_webgpu_research.convnet_remote_scaling report CASE_DIR ...
python -m unittest discover -s research/tests
```
