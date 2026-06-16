
import yaml
from sklearn.model_selection import ParameterGrid

with open('yaml_files/config_port.yaml', 'r') as fp:
    config = yaml.load(fp, Loader=yaml.SafeLoader)

cmd0 = ('source /miniconda/bin/activate; '
        # 'source /coreflow/venv/bin/activate; '
        'git config --global --add safe.directory /mnt/task_runtime')
cmd1 = ('python3 main.py '
        '--model deit_small_patch4_32 '
        '--input-size 32 '
        '--batch-size 256 '
        '--data-set CIFAR '
        '--data-path . '
        '--output_dir $ARTIFACT_DIR '
        '--mask-prob 0.0 '
        '--pretrain 1 '
        '--epochs 100 '
        '--warmup-epochs 20 '
        # '--lr 20e-4 '
        '--cutmix 1 '
        '--mixup 0 '
        '--num_workers 1 '
        '--use-pe 0 '
        '--loss WeightedMSELoss '
        # '--symmetry True '
        '--port_version v3 '
        '--distance_weight_mode 0 '
        '--wandb-name v3_max_mask '
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
        '--wandb-name v3_max_mask '
        '--port_version v3 '
        )

param_grid = [
    {
        # 'mask-prob': [0.0],
        # 'alpha': [1],
        # 'epochs': [1000],
        'lr': [2.5e-4, 5e-4, 10e-4, 20e-4],
        # 'model': ['deit_small_patch4_32'],  # 'deit_base_patch4_32']
        # 'distance_weight_mode': [0],
        # 'use-ce': [0],
        # 'pairwise_mlp': [1],
        # 'num_pairs': [64, 128, 256, 512, 1024, 2048, 4096],
        'maxwh': [4, 16, 32],
        # 'with_replacement': [1],  # if 1: num_pairs can be bigger than 64
        # 'ce-op': ["concat"],
        # 'column_embedding': ['scalar_norm', 'random', 'scalar'] #x
        'mask_prob_weights': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        # 'symmetry': [True],
        # 'use-pe':  [1],
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
        if key == 'model' or key == 'use-ce' or key == 'ce-op' or key == 'column_embedding':
            cmd2_ += ' --%s %s' % (key, param[key])
    config['command'] = ';'.join([cmd0, cmd1_, cmd2_])
    tasks.append(submit(config))


