# Training and experiments

## Data pipeline

The existing 20,000-frame dataset is read in place from `data/dataset/episode_*`. `data/preprocessing/prepare.py` adapts its RGB, semantic IDs, three-class landing suitability, clean sensor CSVs and per-episode metadata. It writes calibration, episode-grouped JSONL manifests and a train-only class-statistics file. It never rerenders frames. IMF maps are produced by the existing C++ executable and cached once as versioned NPZ; cache generation resumes at complete episode boundaries and checks source/config/calibration/extractor signatures.

The fixed 13-state registry is in `common/registry.py`. Raw ultrasonic axis range and simulator vertical ground clearance have different names and validity. Hazard targets mark unsafe **and caution** source landing classes positive; suitable landing target marks only safe cells that meet the configured within-cell safe fraction. POI class masks use source semantic IDs. These target rules are in `data/preprocessing/targets.py`; three-class source masks are preserved.

Splits group entire episodes and environment/layout identifiers with fixed seed `24051991` (70/15/15 target). `data/manifests/manifest_lock.json` hashes the split files. The validator rejects episode, trajectory or environment crossing splits. Model selection uses validation; the root runner does not automatically inspect the locked test split.

## Models and shared training

The four packages each expose `model.py` and `train.py`; they delegate to `common/models/` and `common/training/engine.py`. Losses combine masked class-weighted logits losses with overlap loss for spatial binary products and categorical losses for semantic/scene outputs. Positive weights are computed from training split only. Early stopping defaults to patience 12 and at most 80 epochs. Configurations live in each model folder and `common/configs/`.

`python run.py --suite smoke` runs the tiny fixture suite. The canonical primary
entry point is `python train_research_sequence.py`; it runs every seed for
CNN-HTransformer (CoHAtNet-inspired), then CNN-ViT, CNN, and IMF-HTransformer.
`python run.py --suite full` remains a separate ablation tool. Individual developer
runs may use `python run.py --train cnn_vit --profile small`. All runs write under
`artifacts/`; interrupted runs resume from `last.pt`, matching completed runs are
skipped, and `--force` is required to replace work.

Each model folder has a thin `train.py` entry point; for example `python -m IMF_HTransformer.train --seed 7`. `python run.py` alone displays status and available commands, and never trains.

## Fairness and limits

The model families use the same grouped splits, labels, state/relation interface, shared heads, output maps and loss implementation. The four front ends necessarily differ by design. RGB baselines receive current RGB; IMF may also include temporal flow/geometry. All map-context-enabled models use the same predicted-only episode rollout, the same map rules and the same 32-token bound. The full comparison is system-level. A spatial-only 20-channel IMF ablation is the cleaner low-level feature comparison.

Before each training and validation epoch, context is rebuilt sequentially from that model snapshot. Prior frames contribute only the model's predictions plus permitted analytical depth/pose. Simulator masks never populate the inference map. Candidate verdicts receive a supervised target only when projection lands on valid, matching current depth; candidate target coverage must therefore be reported. Full-dataset context-rollout runtime and map-quality behavior remain unmeasured until the primary suite runs. See [model_contract.md](model_contract.md).

## Reproducibility

The shared engine records seed, configuration, channel registry, manifest hash, cache metadata and repository revision in atomic checkpoints. Validation histories are append-only per run. Seeds are fixed in suite YAML. Training loss and validation metrics are logged every epoch; the runner probes measured batch time before estimating total runtime. An estimate is only a throughput extrapolation, not a promised completion time.
