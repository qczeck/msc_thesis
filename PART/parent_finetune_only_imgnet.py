
import yaml
from sklearn.model_selection import ParameterGrid

with open('yaml_files/config_big.yaml', 'r') as fp:
    config = yaml.load(fp, Loader=yaml.SafeLoader)


cmd0 = ('source /miniconda/bin/activate; '
        'git config --global --add safe.directory /mnt/task_runtime')
cmd2 = ('python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py '
        '--model deit_base_patch16_224 '
        '--input-size 224 '
        '--batch-size 256 '
        '--data-set IMNET '
        '--data-path imagenet '
        '--pretrain 0 '
        '--epochs 300 '
        '--reprob 0.25 '
        '--repeated-aug '
        # '--resume $ARTIFACT_DIR/checkpoint.pth '
        '--num_workers 20 '
        '--lr 5e-4 '
        '--wandb-name imgnet_finetune_v3_anycheckpoint_pairwise_mlp '
        )

param_grid = [
    {
        # 'mask-prob': [0.0, 0.5],
        # 'alpha': [1],
        # 'epochs': [],
        'resume': ["3sb9xdedy2", "f99u9xf3yx", "d9iyzhmq5m", "nze3e9rapm", "57q6r9cghg", "zq7jchyzxj"],
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


