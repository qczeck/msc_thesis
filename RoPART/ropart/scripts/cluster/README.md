# ImageNet-100 pretraining on the DoC GPU cluster

The analogous pair to the CIFAR runs — **translation baseline** and **raw rotation** —
but on ImageNet-100 with **ViT-B/16 @224** (matching the source paper's ImageNet
model). SLURM, single GPU, `a40` by default.

> Scope note: this is ImageNet-**100** on **one** GPU — *not* the ImageNet-1K /
> 8-GPU / ViT-B run, which is out of scope for this project.

## 0. One-time data staging (on gpu30, NOT the cluster head nodes)

```bash
ssh gpu30viashell1
cd ~/msc_thesis/RoPART && ropart/scripts/setup_imagenet_cluster.sh
```

Allocates a `/vol/gpudata` workspace, downloads the standard ImageNet-100 subset
(~15 GB) into `$WS/imagenet100/{train,val}/<wnid>/`, and prints the data path. The
sbatch scripts resolve the workspace automatically via `ws_find ropart-in100`.

## 1. Smoke test (decides online vs offline wandb)

```bash
ssh gpucluster2viashell1
cd ~/msc_thesis/RoPART
sbatch ropart/scripts/cluster/pretrain_imagenet_baseline.sbatch \
  --epochs 1 --max-steps 50 --wandb-mode online    # try online; if init fails it falls back
squeue --me
tail -f in100-base_*.out
```

Confirms SLURM + CUDA + the NFS venv + workspace data line up, and whether the
compute node has outbound HTTPS for live wandb.

## 2. Full runs

```bash
sbatch ropart/scripts/cluster/pretrain_imagenet_baseline.sbatch
sbatch ropart/scripts/cluster/pretrain_imagenet_rotation.sbatch
```

Each is ViT-B/16 @224, 100 epochs, batch 192, `--num_pairs 256`, auto-resuming from
`$WS/out/in100_<control>/checkpoint.pth` if present (safe across requeue / the 3-day
walltime). Extra args forward to `ropart.train`.

## 3. Monitoring

- `squeue --me`, `scancel <jobid>`, `tail -f <jobname>_<jobid>.out`.
- **wandb online** (if the smoke showed egress): just watch the dashboard.
- **wandb offline**: the run dirs are on NFS home (`$REPO/wandb`, mounted on gpu30),
  so run a near-live sync loop from gpu30:
  ```bash
  ssh gpu30viashell1
  cd ~/msc_thesis/RoPART
  while :; do .venv/bin/wandb sync wandb/offline-run-* ; sleep 90; done
  ```

## Knobs

`--partition` (a40 default; a100/a30 alternatives) is set in the `#SBATCH` header.
Override per-submit env, e.g. `WANDB_MODE=online sbatch ...`, or pass train flags
after the script name (`--lr`, `--epochs`, `--batch-size`, …).
