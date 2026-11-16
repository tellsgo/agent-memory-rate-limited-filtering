# Agent Memory as Rate-Limited Filtering: Explicit State Overtakes Long Context at a Reader-Dependent Horizon

## Requirements

Python 3.9 or later and `pip install -r requirements.txt`. Model readings require GPUs and separately obtained model weights; CPU commands below reuse saved readings. Download `logs.tar.gz` from the release and unpack it in the repository root with `tar -xzf logs.tar.gz` before running tables. The asset contains recorded outputs, not model weights. No inference service is required for CPU checks.

## Reproduction

Run `bash run_all.sh quick` for the exhaustive theory checks (several minutes); `bash run_all.sh tables` for the nine supported table files and byte comparison (bootstrap may take several minutes); `bash run_all.sh figures` for figures 2–4. Run commands from the repository root. Figures require a locally installed Times New Roman font. GPU inference is not reproduced by the CPU workflow. The Mem0 history databases were not retained: the final table uses the recorded ledger, not a rebuilt storage ledger. Model tokenizer downloads are required to rederive token counts.

| Paper output | Command or limitation | Input level |
| --- | --- | --- |
| Main and supplement: final, cells R5-1/R5-4, R7-1, R7 cost, R8, R9, R10, reader channel | `bash run_all.sh tables` | Logged readings, frozen predictions and recorded analysis/ledger |
| Figures 2–4 | `bash run_all.sh figures` | Four recorded CSVs |
| Figure 1 | Conceptual TikZ illustration; no experiment | Not applicable |
| Theory counts, masking and replay | `bash run_all.sh quick` | Generated state-machine episodes |
| Supplement: x12, x3, x4c, x5, x6, r3, r4, policy and prompts | Not covered by the table target; see scripts in `data/` | Varies; not verified |
| Inline tables and remaining numeric claims | Not independently verified by this release | Not verified |

## Third-party models and data

The recorded readings are supplied in the asset. Model inference additionally uses model weights obtained separately under their respective licenses: Qwen2.5/QwQ (Qwen model distribution), Phi-4 (Microsoft model distribution), and GPT-OSS (model distribution). Check each model card for its applicable terms. The original GPT-OSS repository identifier contains a restricted vendor string and is therefore supplied at runtime through `GPT_OSS_MODEL_ID`; Mem0 provider configuration likewise requires `MEM0_PROVIDER` and `MEM0_BASE_URL_KEY`. These GPU paths are not part of the CPU test.

Citation: arXiv:2611.02361
