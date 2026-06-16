source /coreflow/venv/bin/activate
python -m pip install wandb
#PRETRAIN_EPOCHS=$1
#FINETUNE_EPOCHS=$2

python3 main.py \
--model deit_small_patch4_32 \
--input-size 32 \
--batch-size 256 \
--data-set CIFAR \
--data-path . \
--output_dir $ARTIFACT_DIR \
--mask-prob 0.5 \
--pretrain 1 \
--epochs 1000 \
--warmup-epochs 20 \
--lr 5e-4 \
--cutmix 1 \
--mixup 0 \
--num_workers 1 \
--wandb-name mp3_small4_1000epochs_pretrain


python3 -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
--model deit_small_patch4_32 \
--input-size 32 \
--batch-size 256 \
--data-set CIFAR \
--data-path . \
--pretrain 0 \
--reprob 0.25 \
--repeated-aug \
--resume $ARTIFACT_DIR/checkpoint.pth \
--epochs 400 \
--wandb-name mp3_small4_500epochs_finetune
#--num_workers 1 \
#--lr 5e-4 \


