# RoPART

**Learning relative orientation in off-grid self-supervised vision.**

RoPART (**Ro**tation + PART) extends [PART](https://github.com/Melika-Ayoughi/PART)'s
pairwise self-supervised pretext task from relative *translation* `(Δx, Δy)` between
off-grid image patches to planar *rigid motion* SE(2), by adding a relative
*orientation* target `Δφ`. The research question is whether supervising relative
orientation yields visual representations with better performance on
orientation-sensitive downstream tasks — without degrading classification.

This is an MSc Computing individual project at Imperial College London (Department
of Computing, 70081), running June–September 2026.

## Idea

PART learns the *relative composition* of images by sampling patches off-grid and
predicting the relative translation between random patch pairs — no absolute
positions, no position embeddings. Its authors extended the target with relative
scale as a proof of concept, and list "modelling rotations" as open future work.
RoPART takes that step: predicting relative orientation between patch pairs, so the
pretext task spans translation **and** rotation. The contribution is methodological
— completing the relative pretext task towards the rigid-motion group — rather than
chasing benchmark numbers.

## Status

Proposal approved; implementation in progress. **No results yet.** The model is
built up one geometric channel-group at a time: a translation-only baseline
`(Δx, Δy)` first, then RoPART's rotation `(cos Δφ, sin Δφ)`, with relative scale
deferred as an optional final step.

## Repository layout

```
README.md          # this file
PART/              # upstream PART reference implementation (vendored)
RoPART/            # RoPART extensions, evaluation, and probes
latex/             # interim & final report (.tex, compiled .pdf) + references.bib
```

## Getting started

The upstream PART code is vendored under `PART/`, so there is nothing extra to
clone. RoPART runs on a modern stack (the upstream pins are outdated and ignored):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch torchvision wandb pyyaml scikit-learn einops scikit-image matplotlib
```

Local development and short runs use the Apple-Silicon **MPS** backend; full-scale
pretraining runs on **CUDA** via the Imperial DoC SLURM cluster. A baseline smoke
run on CIFAR-100 (data downloads on first use):

```bash
cd PART
WANDB_MODE=offline PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py \
  --model deit_small_patch4_32 --input-size 32 --data-set CIFAR --data-path ./data \
  --output_dir ./out_smoke --device mps --pretrain 1 \
  --use-pe 0 --loss WeightedMSELoss --distance_weight_mode 0 \
  --train_sampling v3 --test_sampling v3 --head_type cross_attention \
  --batch-size 64 --epochs 1 --num_workers 2 --wandb-name smoke
```

## Built on

Ayoughi M, Abnar S, Huang C, Sandino C, Lala S, Dhekane EG, et al. *How PARTs
assemble into wholes: Learning the relative composition of images.* NLDL 2026
(PMLR 307). arXiv:2506.03682.

Full bibliography in `latex/references.bib`.
