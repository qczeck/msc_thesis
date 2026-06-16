# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from torch.distributed.elastic.multiprocessing.errors import record

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from utils import WeightedMSELoss
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate, evaluate_one_img
from samplers import RASampler
import models
import utils
import wandb
from make_dataset import main2
import traceback
import random


def reproduce(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    # todo: torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True
    # Disabling this feature with torch.backends.cudnn.benchmark = False causes cuDNN to deterministically
    # select an algorithm, possibly at the cost of reduced performance


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--mask-prob', type=float, default=0., metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--pretrain', type=int, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--linear-probe', type=int, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--global-pool', type=int, default=0, metavar='PCT')
    parser.add_argument('--use-pe', type=int, default=1, metavar='PCT')
    parser.add_argument('--use-ce',
                        type=int,
                        default=0,
                        help='use column embedding, if 0: all patches are given to linear, if 1: col_embed is used')
    parser.add_argument('--ce-op',
                        type=str,
                        default="concat",
                        help="concat or add means the column_embedding will be concatenated or added to input of vit")
    parser.add_argument('--column_embedding',
                        type=str,
                        default='scalar',
                        help='learned, x, random, one-hot, scalar, scalar_norm')
    # parser.add_argument('--pairwise_mlp', type=int, default=0, help="if True, shared MLP for any 2 patch")
    parser.add_argument('--with_replacement', type=int, default=1, help="pairwise patches are not unique")
    parser.add_argument('--num_pairs', type=int, default=64, help="in pairwise_mlp, this is the number of pairs")
    parser.add_argument('--debug_pairwise_mlp', action='store_true', help="once activated, targets and indices are v1 unshuffled")
    parser.add_argument('--debug_cross_attention', action='store_true', help="once activated, targets and indices are v1 unshuffled")
    parser.add_argument('--minwh', type=int, default=2, help="the minimum width/height for a sampled patch")
    parser.add_argument('--maxwh', type=int, default=8, help="the maximum width/height for a sampled patch")
    parser.add_argument('--centered_targets', type=int, default=1, help="the targets are center of patches instead of corners")
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer-lr-decay', type=float, default=1., metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=False)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01_101/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19', 'TINY-IMNET'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default=ARTIFACT_DIR,
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--loss',
                        default="CrossEntropyLoss",
                        type=str,
                        help='The loss formula during pretraining: CrossEntropyLoss or WeightedMSELoss')
    parser.add_argument('--symmetry',
                        default=False,
                        type=bool,
                        help="If the symmetry flag is True, then the loss is regularized to be symmetrical")
    parser.add_argument('--w_x', default=1, type=float, help="this parameter weights the dx in the loss")
    parser.add_argument('--w_y', default=1, type=float, help="this parameter weights the dy in the loss")
    parser.add_argument('--w_w', default=1, type=float, help="this parameter weights the dw in the loss")
    parser.add_argument('--w_h', default=1, type=float, help="this parameter weights the dh in the loss")
    parser.add_argument('--mask_prob_weights',
                        default=0,
                        type=float,
                        help="if mask_prob_weights are none zero, then a mask with that probability per item is multiplied by loss")
    parser.add_argument('--distance_weight_mode',
                        default=0,
                        type=int,
                        help="in the WeightedMSELoss function, we can weigh the loss by distance of patches (1)")
    parser.add_argument('--alpha',
                        default=1,
                        type=float,
                        help="the loss function is criterion + alpha * symmetry")
    parser.add_argument('--mask_diag',
                        action='store_true',
                        help="mask the diagonal of the symmetry matrix")
    # parser.add_argument('--port_version',
    #                     default='v1',
    #                     type=str,
    #                     help='PORT v1, v2, v3')
    parser.add_argument('--train_sampling', default='v3', type=str, help="the sampling strategy for train during PT")
    parser.add_argument('--test_sampling', default='v1', type=str, help="the sampling strategy for test during PT")
    parser.add_argument('--shuffle_patches', action='store_true', help="if sampling==v1, shuffle_patches is used")
    parser.add_argument('--wandb-name',
                        default="default_wandb",
                        type=str,
                        help="the name of the project on wandb")
    parser.add_argument('--debug',
                        action='store_true',
                        help="if true: wandb is not initialized and more printing is performed")
    parser.add_argument('--num-samples',
                        default=1,
                        type=int,
                        help="number of examples to evaluate and visualize")
    parser.add_argument('--ref_patches', default=[35], type=list, help="reference patch for visualization")
    parser.add_argument('--head_type', default='mlp', type=str, help="mlp, pairwise_mlp, cross_attention")
    parser.add_argument('--cross_attention_query_type', default='positional', type=str, help="when head_type==cross_attention")
    return parser


@record
def main(args):
    # start multiprocessing work

    utils.init_distributed_mode(args)

    if utils.is_main_process():
        if not args.debug:
            wandb.init(project="mp3 reproduce", settings=wandb.Settings(disable_git=True, save_code=False),
                       name=f"{args.wandb_name}_pretrain{args.pretrain}}",
                       notes=f"{args}")

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    reproduce(seed)

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    dataset_train_viz, _ = build_dataset(is_train=True, args=args, viz=True)
    dataset_val_viz, _ = build_dataset(is_train=False, args=args, viz=True)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=int(1.5 * args.batch_size),
        shuffle=False, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )
    # if num_workers is 1 the default prefetch factor is 2 meaning it will load 2 batches in the future
    data_loader_one_img_train = torch.utils.data.DataLoader(
        dataset_train_viz, batch_size=1,
        shuffle=True, num_workers=0, #prefetch_factor=0,
        pin_memory=args.pin_mem, drop_last=False
    )

    data_loader_one_img_val = torch.utils.data.DataLoader(
        dataset_val_viz, batch_size=1,
        shuffle=True, num_workers=0, #prefetch_factor=0,
        pin_memory=args.pin_mem, drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        mask_prob=args.mask_prob,
        pretrain=args.pretrain,
        linear_probe=args.linear_probe,
        global_pool=args.global_pool,
        use_pe=args.use_pe,
        use_ce=args.use_ce,
        # pairwise_mlp=args.pairwise_mlp,
        with_replacement=args.with_replacement,
        num_pairs=args.num_pairs,
        ce_op=args.ce_op,
        loss=args.loss,
        head_type=args.head_type,
        column_embedding=args.column_embedding,
        num_channels=4 if args.train_sampling == "v3" or args.train_sampling == "v3_1" else 2,
        debug_pairwise_mlp=args.debug_pairwise_mlp,
        debug_cross_attention=args.debug_cross_attention,
        cross_attention_query_type=args.cross_attention_query_type
    )

    # TODO: finetuning

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    # create param groups per layer
    parameters = model_without_ddp.get_parameter_groups(base_lr=args.lr, weight_decay=args.weight_decay,
                                                        layer_lr_decay=args.layer_lr_decay)
    optimizer = create_optimizer(args, parameters=parameters)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    # criterion = LabelSmoothingCrossEntropy()

    if args.pretrain:
        if args.loss == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
        elif args.loss == "WeightedMSELoss":
            criterion = WeightedMSELoss(
                device=device,
                distance_weight_mode=args.distance_weight_mode,
                mask_prob_weights=args.mask_prob_weights,
                num_patches=model_without_ddp.num_patches,
                w_x=args.w_x, w_y=args.w_y, w_w=args.w_w, w_h=args.w_h,
            )
        elif args.loss == "L1Loss":
            criterion = torch.nn.L1Loss(reduction='mean')
        elif args.loss == "HuberLoss":
            criterion = torch.nn.HuberLoss(reduction='mean')
        elif args.loss == "SmoothL1Loss":
            criterion = torch.nn.SmoothL1Loss(reduction='mean')

    elif args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        elif args.resume.startswith('s3://'):
            checkpoint = torch.load("./mp3_checkpoint_pretrained.pth", map_location='cpu')
        else:
            # checkpoint = torch.load("./mp3_checkpoint_pretrained.pth", map_location='cpu')
            print('Trying to load from %s...' % args.resume)
            checkpoint = torch.load(args.resume, map_location='cpu')
            print('after torch.load')

        # for finetuning, the head is thrown away and a new classifier (clf) is trained from scratch
        if not args.pretrain:
            checkpoint['model'].pop('head.bias', None)
            checkpoint['model'].pop('head.weight', None)
        if 'head.input_project.weight' in checkpoint['model'] and not args.pretrain:
            checkpoint['model'].pop('head.input_project.weight', None)
            checkpoint['model'].pop('head.input_project.bias', None)
            checkpoint['model'].pop('head.multihead_attn.in_proj_weight', None)
            checkpoint['model'].pop('head.multihead_attn.in_proj_bias', None)
            checkpoint['model'].pop('head.multihead_attn.out_proj.weight', None)
            checkpoint['model'].pop('head.multihead_attn.out_proj.bias', None)
            checkpoint['model'].pop('head.output_project.weight', None)
            checkpoint['model'].pop('head.output_project.bias', None)

        # quick fix only for this experiment
        # if args.port_version.startswith('v3') and not args.pretrain and args.column_embedding != 'learned':
        # checkpoint['model']['pos_embed'] = nn.Parameter(torch.zeros(1, 65, 384))
        #     model_without_ddp.pop('col_embed', None)
        #     checkpoint['model'].pop('pos_embed', None)
        #     from timm.models.layers import trunc_normal_
        #     checkpoint['model']['col_embed'] = nn.Parameter(torch.zeros(1, 65, 384))
        #     trunc_normal_(checkpoint['model']['pos_embed'], std=.02)
        # checkpoint['model']['clf.weight'] = torch.zeros((100, 384))
        # checkpoint['model']['clf.bias'] = torch.zeros(100)
        model_without_ddp.load_state_dict(checkpoint['model'])
        # from timm.models.layers import trunc_normal_
        # trunc_normal_(model_without_ddp.clf.weight, std=.02)
        # nn.init.constant_(model_without_ddp.clf.bias, 0)

        print("Successfully loading from the resumed link...")

        # if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        #     args.start_epoch = checkpoint['epoch'] + 1
        #     if args.model_ema:
        #         utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])

    if args.eval:
        test_stats = evaluate(args, data_loader_val, model, device, args.pretrain)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        if utils.is_main_process():
            # send_metrics({'test_'+k:v for k, v in test_stats.items() if not k.startswith('output+target')})
            if not args.debug:
                try:
                    wandb.log({'test_'+k:v for k, v in test_stats.items()})
                    if args.pretrain:
                        evaluate_one_img(args, data_loader_one_img_val, model, device, args.num_samples, train_time=False)
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    print("wandb.log failed")
        wandb.finish()
        return

    print("Start training")
    start_time = time.time()
    max_accuracy = 0.0

    import multiprocessing as mp

    for epoch in range(args.start_epoch, args.epochs):

        # with mp.Pool(4) as p:
        #     p.map(f, tasks)
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            args,
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn, args.pretrain
        )

        if utils.is_main_process():
            # send_metrics({'train_'+k: v for k, v in train_stats.items() if not k.startswith('output+target')})
            if not args.debug:
                try:
                    wandb.log({'train_' + k: v for k, v in train_stats.items()})
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    print("wandb.log failed")
            if args.pretrain:
                evaluate_one_img(args, data_loader_one_img_train, model, device, args.num_samples, train_time=True)

        lr_scheduler.step(epoch)
        if args.output_dir:
            suffix = '' if args.pretrain else '_ft'
            if epoch in [100, 150, 200, 300, 400, 500, 800, 1000, 1600, 2000, 2500, 3000, 3500]:
                suffix += f"_epoch_{epoch}"
            checkpoint_paths = [output_dir / ('checkpoint%s.pth' % suffix)]
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(args, data_loader_val, model, device, args.pretrain)

        if utils.is_main_process():
            # send_metrics({'test_' + k: v for k, v in test_stats.items() if not k.startswith('output+target')})
            if not args.debug:
                try:
                    wandb.log({'test_' + k: v for k, v in test_stats.items()})
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    print("wandb.log failed")
            if args.pretrain:
                evaluate_one_img(args, data_loader_one_img_val, model, device, args.num_samples, train_time=False)

        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items() if not k.startswith('output+target')},
                     **{f'test_{k}': v for k, v in test_stats.items() if not k.startswith('output+target')},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    # if args.distributed:
    # end multiprocessing work

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if not args.debug:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
