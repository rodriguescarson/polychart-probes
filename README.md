# polychart-probes

Code for "Reading the Lie Factor: linear probes recover graded chart-deception
severity from vision-language model activations" (Rodrigues, 2026).

- `probekit/` — model loading (LLM + VLM, 4-bit, LoRA adapters), residual-stream
  extraction with detection masks, linear/ridge probes, steering and ablation,
  bootstrap and permutation statistics, plotting.
- `paper17/` — dataset builders (images and text specs from real OWID series with
  exact Tufte Lie Factors), extraction runners, the frozen analysis, dated
  extensions (layer sweeps, seed variance, representation-action comparison,
  holdout metrics, steering), and figure generation.

Chart generation depends on the PolyChart generator released with the dataset
paper (DOI 10.5281/zenodo.21939874; dataset DOI 10.5281/zenodo.21939803).

The analysis in `paper17/analyze_trackB.py` was committed before any data existed;
extensions are dated. See the paper's appendix for the preregistration log.

MIT license.
