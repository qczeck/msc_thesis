
import yaml
from sklearn.model_selection import ParameterGrid

with open('yaml_files/config_port.yaml', 'r') as fp:
    config = yaml.load(fp)

cmd0 = ('source /coreflow/venv/bin/activate; '
        'python -m pip install wandb; '
        'git config --global --add safe.directory /mnt/task_runtime')
cmd1 = ('python3 main.py '
        '--model deit_small_patch4_32 '
        '--input-size 32 '
        '--batch-size 256 '
        '--data-set CIFAR '
        '--data-path . '
        '--output_dir $ARTIFACT_DIR '
        '--mask-prob 0.5 '
        '--pretrain 1 '
        '--epochs 1000 '
        '--warmup-epochs 20 '
        '--lr 20e-4 '
        '--cutmix 1 '
        '--mixup 0 '
        '--num_workers 1 '
        # '--use-pe 0 '
        '--loss WeightedMSELoss '
        # '--symmetry True '
        '--wandb-name ablation_nosym_port_v1 '
        # '--mask_diag '
        )
cmd2 = ('python3 -m torch.distributed.launch --nproc_per_node=1 --use_env main.py '
        '--model deit_small_patch4_32 '
        '--input-size 32 '
        '--batch-size 256 '
        '--data-set CIFAR '
        '--data-path . '
        '--pretrain 0 '
        '--epochs 400 '
        '--reprob 0.25 '
        '--repeated-aug '
        '--resume $ARTIFACT_DIR/checkpoint.pth '
        # '--num_workers 1 '
        '--lr 5e-4 '
        '--wandb-name ablation_nosym_finetune_port_v1 '
        )

param_grid = [
    {
        # 'mask-prob': [0.5],
        'alpha': [1],
        # 'epochs': [1000],
        # 'lr': [20e-4],
        # 'model': ['deit_small_patch4_32'],  # 'deit_base_patch4_32']
        'distance_weight_mode': [0, 2],
        # 'mask_prob_weights': [0, 0.1],
        # 'symmetry': [True, False],
        # 'use-pe': [1],
        # 'loss': ["WeightedMSELoss"]
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


