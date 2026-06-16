source /coreflow/venv/bin/activate
python -m pip install wandb

#python3 main.py \
#--model deit_small_patch4_32 \
#--input-size 32 \
#--batch-size 256 \
#--data-set CIFAR \
#--data-path . \
#--output_dir $ARTIFACT_DIR \
#--mask-prob 0.5 \
#--pretrain 1 \
#--epochs 500 \
#--warmup-epochs 20 \
#--lr 20e-4 \
#--cutmix 1 \
#--mixup 0 \
#--num_workers 1 \
#--loss WeightedMSELoss \
#--wandb-name port_v1

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
--wandb-name finetune_symmetric_port_v1 \
--resume $ARTIFACT_DIR/checkpoint.pth
#--num_workers 1 \

