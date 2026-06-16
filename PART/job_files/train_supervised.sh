source /coreflow/venv/bin/activate
python -m pip install wandb

python3 -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
--model deit_small_patch4_32 \
--input-size 32 \
--batch-size 256 \
--data-set CIFAR \
--data-path . \
--pretrain 0 \
--epochs 400 \
--reprob 0.25 \
--repeated-aug \
--lr 5e-4 \
--wandb-name supervised_small4

