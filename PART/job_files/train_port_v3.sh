source /miniconda/bin/activate
#python -m pip install wandb

#python3 main.py \
#--model deit_small_patch4_32 \
#--input-size 32 \
#--batch-size 256 \
#--data-set CIFAR \
#--data-path . \
#--output_dir $ARTIFACT_DIR \
#--mask-prob 0.5 \
#--pretrain 1 \
#--epochs 1000 \
#--warmup-epochs 20 \
#--lr 20e-4 \
#--cutmix 1 \
#--mixup 0 \
#--num_workers 1 \
#--loss WeightedMSELoss \
#--port_version v3_1 \
#--use_pe 0 \
#--wandb-name port_v3_1
##--symmetry False \
##--mask_diag \

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
--resume $ARTIFACT_DIR/vv9i34gegk \
--lr 5e-4 \
--wandb-name v1_v1_FT_vv9i34gegk

