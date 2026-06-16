source /coreflow/venv/bin/activate
python3 main.py \
--model deit_small_patch4_32 \
--input-size 32 \
--batch-size 256 \
--data-set CIFAR \
--data-path . \
--pretrain 0 \
--epochs 300 \
--reprob 0.25 \
--repeated-aug


