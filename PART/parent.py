
import yaml
from sklearn.model_selection import ParameterGrid

with open('yaml_files/config_mp3_cifar.yaml', 'r') as fp:
    config = yaml.load(fp)

cmd0 = 'source /coreflow/venv/bin/activate; python -m pip install wandb; git config --global --add safe.directory /mnt/task_runtime'
# cmd1 = 'python3 main.py --model deit_base_patch16_224 --batch-size 256 --data-path imagenet --output_dir $ARTIFACT_DIR --pretrain 1'
cmd1 = ('python3 main.py '
        # '--model deit_small_patch4_32 '
        '--input-size 32 '
        '--batch-size 256 '
        '--data-set CIFAR '
        '--data-path . '
        '--output_dir $ARTIFACT_DIR '
        # '--mask-prob 0.5 '
        '--pretrain 1 '
        # '--epochs 500 '
        # '--warmup-epochs 20 '
        # '--lr 2.5e-4 '
        '--cutmix 0 '
        '--mixup 0 '
        '--smoothing 0 '
        '--weight-decay 0 '
        '--num_workers 1 '
        '--use-pe 0 ')
# cmd2 = 'python3 main.py --model deit_base_patch16_224 --batch-size 256 --data-path imagenet --pretrain 0 --reprob 0.25 --repeated-aug --resume $ARTIFACT_DIR/checkpoint.pth --lr 2e-3 --layer-lr-decay 0.75'
cmd2 = ('python3 main.py '
        # '--model deit_small_patch4_32 '
        '--input-size 32 '
        '--batch-size 256 '
        '--data-set CIFAR '
        '--data-path . '
        '--pretrain 0 '
        '--epochs 400 '
        '--reprob 0.25 '
        '--repeated-aug '
        '--resume $ARTIFACT_DIR/checkpoint.pth '
        '--num_workers 1 ')

param_grid = [
    {
        # 'mask-prob': [0.25, 0.5, 0.75],
        # 'epochs': [100, 300, 600],
        # 'warmup-epochs': [20, 50],
        # 'lr': [2.5e-4, 5e-4, 10e-4, 20e-4],  #learning rate for pretraining phase
        'mask-prob': [0.5],
        'epochs': [500],
        'warmup-epochs': [20],
        'lr': [2.5e-4, 5e-4],
        'model': ['deit_base_patch4_32']#, 'deit_small_patch8_32']
    },
]

param_list = list(ParameterGrid(param_grid))
tasks = []
for param in param_list:
    name = ''
    name += ' + '.join(['{}:{}'.format(k, param[k]) for k in sorted(param)])
    config['name'] = name
    cmd1_ = cmd1
    cmd2_ = cmd2
    for key in param:
        cmd1_ += ' --%s %s' % (key, param[key])
        if key == 'model':
            cmd2_ += ' --%s %s' % (key, param[key])
    config['command'] = ';'.join([cmd0, cmd1_, cmd2_])
    tasks.append(submit(config))


