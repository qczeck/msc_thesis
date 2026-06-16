source /coreflow/venv/bin/activate
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_224 --batch-size 256 --data-path imagenet --pretrain 0 --epochs 300 --reprob 0.25 --repeated-aug


