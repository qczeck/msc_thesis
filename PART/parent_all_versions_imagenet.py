
import yaml
from sklearn.model_selection import ParameterGrid

with open('yaml_files/config_big.yaml', 'r') as fp:
    config = yaml.load(fp, Loader=yaml.SafeLoader)


cmd0 = ('source /miniconda/bin/activate; '
        'git config --global --add safe.directory /mnt/task_runtime '
        # 'apt update; '
        # 'apt install screen rsync'
        )
cmd1 = ('python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py '
        '--model deit_base_patch16_224 '
        '--input-size 224 '
        # '--batch-size 256 '
        '--data-set IMNET '
        '--data-path imagenet '
        '--output_dir $ARTIFACT_DIR '
        # '--mask-prob 0.5 '
        '--pretrain 1 '
        # '--epochs 400 '
        '--warmup-epochs 20 '
        # '--lr 20e-4 '
        '--cutmix 1 '
        '--mixup 0.8 '
        '--num_workers 20 '
        '--use-pe 0 '
        '--loss WeightedMSELoss '
        # '--symmetry True '
        '--distance_weight_mode 0 '
        '--train_sampling v3 '
        '--test_sampling v3 '
        '--head_type pairwise_mlp '
        # '--debug_pairwise_mlp '
        # '--debug_cross_attention '
        '--wandb-name imgnet_v3_v3_pairwise_mlp '
        # '--mask_diag '
        )
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
        '--resume $ARTIFACT_DIR/checkpoint.pth '
        '--num_workers 20 '
        '--lr 5e-4 '
        '--wandb-name imgnet_v3_v3_pairwise_mlp '
        )

param_grid = [
    {
        'mask-prob': [0.0],
        # 'alpha': [1],
        'epochs': [1600],
        'lr': [0.625e-4],
        'batch-size': [128],
        'minwh': [8],  # 16/2
        'maxwh': [32],  # 16*2
        # 'model': ['deit_small_patch4_32'],  # 'deit_base_patch4_32']
        # 'distance_weight_mode': [0],
        # 'use-ce': [0],
        'num_pairs': [4802, 5000],
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
    cmd1_ = cmd1
    cmd2_ = cmd2
    for key in param:
        cmd1_ += ' --%s %s' % (key, param[key])
        if key == 'model' or key == 'use-ce' or key == 'ce-op' or key == 'column_embedding':
            cmd2_ += ' --%s %s' % (key, param[key])
    config['command'] = ';'.join([cmd0, cmd1_, cmd2_])
    tasks.append(submit(config))


