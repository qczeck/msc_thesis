
import yaml
from sklearn.model_selection import ParameterGrid

# with open('config_imagenet64_augment.yaml', 'r') as fp:
with open('yaml_files/config_cifar.yaml', 'r') as fp:
    config = yaml.load(fp)

cmd0 = 'source /coreflow/venv/bin/activate'
cmd1 = 'python3 -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model deit_small_patch4_32 --input-size 32 --batch-size 256 --data-set CIFAR --data-path . --pretrain 1 --output_dir $ARTIFACT_DIR'
cmd2 = 'python3 -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model deit_small_patch4_32 --input-size 32 --batch-size 256 --data-set CIFAR --data-path . --pretrain 0 --reprob 0.25 --repeated-aug --resume $ARTIFACT_DIR/checkpoint.pth'

param_grid = [
    {
        'mask-prob': [0.5, 0.75], 'pretrain-epochs': [100, 200], 'warmup-epochs': [20]
    },
]

param_list = list(ParameterGrid(param_grid))
tasks = []
for param in param_list:
    name = ''
    name += '_'.join(['{}:{}'.format(k, param[k]) for k in sorted(param)])
    config['name'] = name
    cmd1_ = cmd1
    cmd2_ = cmd2
    for key in param:
        if key != 'pretrain-epochs':
            cmd1_ += ' --%s %s' % (key, param[key])
            if key == 'num-cls-token':
                cmd2_ += ' --%s %s' % (key, param[key])                
        else:
            cmd1_ += ' --%s %s' % ('epochs', param[key])
            cmd2_ += ' --%s %s' % ('epochs', 400)
    config['command'] = ';'.join([cmd0, cmd1_, cmd2_])
    tasks.append(submit(config))


