source /coreflow/venv/bin/activate
python -m pip install wandb

#PT_EPOCHS=$1
#FT_EPOCHS=$2
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
--model deit_base_patch16_224 \
--batch-size 256 \
--data-path imagenet \
--output_dir $ARTIFACT_DIR \
--mask-prob 0.75 \
--pretrain 1 \
--epochs 400 \
--warmup-epochs 20 \
--lr 5e-4 \
--cutmix 1 \
--mixup 0.8 \
--wandb-name pretrain_imgnet_mp3

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
--model deit_base_patch16_224 \
--batch-size 256 \
--data-path imagenet \
--pretrain 0 \
--epochs 300 \
--reprob 0.25 \
--repeated-aug \
--resume $ARTIFACT_DIR/checkpoint.pth \
--wandb-name finetune_imgnet_mp3


