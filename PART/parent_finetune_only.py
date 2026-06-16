
import yaml
from sklearn.model_selection import ParameterGrid

with open('yaml_files/config_port.yaml', 'r') as fp:
    config = yaml.load(fp, Loader=yaml.SafeLoader)


cmd0 = ('source /miniconda/bin/activate; '
        'git config --global --add safe.directory /mnt/task_runtime')
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
        # '--resume $ARTIFACT_DIR/checkpoint.pth '
        '--num_workers 20 '
        '--lr 5e-4 '
        '--wandb-name cross_attention_v3_v3_finetune '
        )

param_grid = [
    {
        # 'mask-prob': [0.0, 0.5],
        # 'alpha': [1],
        # 'epochs': [],
        'resume': ['3du3qqnv4h',
                     '9cc4ewbckv',
                     'xxyaa4ixp7',
                     't5yqvf5gp3',
                     'ifffrcq43a',
                     'tbbpzkci55',
                     'ep78d7jpp3',
                     '39vurp8dtm',
                     'wudsadnmni',
                     'udvwitdr4d',
                     '2z6egqc3dz',
                     '52kckb6w4j',
                     'vm2pwz7r8x',
                     '8tu6eqrwie',
                     'ibi8b5d3ux',
                     'uzaa6peqf7',
                     'mx2ikeetmv',
                     '6b5ii65gjt',
                     'mqjgcm3zdv',
                     'dwz95inwkv',
                     '5vkt598k28',
                     'mugnvscs9w',
                     'mehxuumzr4',
                     'innq3x8hk4',
                     'i4fisdfbj8',
                     'iee6sibbj3',
                     'gp7u99h7fa',
                     'eewedhqdz6',
                     '9jrhdyscya',
                     'wjgti5yyiy',
                     'sw637qbrcw',
                     'bbxtrum88k'],
        # 'lr': [5e-4],
        # 'batch-size': [512],
        # 'model': ['deit_small_patch4_32'],  # 'deit_base_patch4_32']
        # 'distance_weight_mode': [0],
        # 'use-ce': [0],
        # 'num_pairs': [512, 1024, 2048, 4096],
        # 'cross_attention_query_type': ['patch_cat', 'positional'],
        # 'maxwh': [4, 16, 32],
        # 'with_replacement': [1],  # if 1: num_pairs can be bigger than 64
        # 'ce-op': ["concat"],
        # 'column_embedding': ['scalar_norm', 'random', 'scalar'] #x
        # 'mask_prob_weights': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
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
    cmd2_ = cmd2
    for key in param:
        cmd2_ += ' --%s %s' % (key, param[key])
    config['command'] = ';'.join([cmd0, cmd2_])
    tasks.append(submit(config))


