# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch
import torchvision
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
import random
import wandb
from utils import SymmetryRegularizer
import traceback
from utils import WeightedMSELoss


def pretrain_model(sampling, model, images, targets, mlp_head, centered_targets, symmetry):
    if mlp_head:
        outputs = model(images)  # [b, num_channels, 64, 64]
    else:
        outputs, indices = model(images)  # outputs: ([256, num_channels, 61]), indices: ([2, 61])

    if sampling == 'v2' or sampling == 'v1':
        # targets.shape: torch.Size([1024, 2, 64])
        targets = targets[:, :, None, :] - targets[:, :, :, None]  # (b, 2, 64, 64): first x then y
        # todo: centered
        assert (targets + targets.permute(0, 1, 3, 2) == 0).all() == True
    elif sampling == 'v3' or sampling == "v3_1":

        b, c, num_patches = targets.shape
        w_ref = torch.repeat_interleave((targets[:, 2, :] - targets[:, 0, :]), num_patches, dim=1).reshape(b, num_patches, num_patches)
        h_ref = torch.repeat_interleave((targets[:, 3, :] - targets[:, 1, :]), num_patches, dim=1).reshape(b, num_patches, num_patches)
        w_tgt = (targets[:, 2, :] - targets[:, 0, :]).repeat(1, num_patches).reshape(b, num_patches, num_patches)
        h_tgt = (targets[:, 3, :] - targets[:, 1, :]).repeat(1, num_patches).reshape(b, num_patches, num_patches)

        if centered_targets:
            # targets = (x1,tgt - x1,ref), (y1,tgt - y1,ref), (x2,tgt - x2,ref), (y2,tgt - y2,ref)
            targets = targets[:, :, None, :] - targets[:, :, :, None]  # [256, 4, 64, 64]
            # d_xc = (targets[:, 0, :, :] + targets[:, 2, :, :]) / 2  # [256, 64, 64]
            # d_yc = (targets[:, 1, :, :] + targets[:, 3, :, :]) / 2  # [256, 64, 64]
            targets[:, :2, :, :] = torch.cat(((targets[:, 0, :, :] + targets[:, 2, :, :])[:, None, :, :],
                                              (targets[:, 1, :, :] + targets[:, 3, :, :])[:, None, :, :]), dim=1)
            targets[:, 0, :, :] = (targets[:, 0, :, :] / (2 * w_ref))  # [256, 4, 64, 64]
            targets[:, 1, :, :] = (targets[:, 1, :, :] / (2 * h_ref))  # [256, 4, 64, 64]
            # w_tgt, w_ref, h_tgt, h_ref are non-zero but targets[:, 2, :, :] is not
            # targets[:, 2, :, :] = (w_tgt / w_ref)
            # targets[:, 3, :, :] = (h_tgt / h_ref)
            targets = torch.cat((targets[:, :2, :, :], (w_tgt / w_ref)[:, None, :, :]), dim=1)  # [256, 3, 64, 64]
            # targets[:, 3, :] = h_tgt / h_ref
            targets = torch.cat((targets, (h_tgt / h_ref)[:, None, :, :]), dim=1)  # [256, 4, 64, 64]
        else:
            targets = targets[:, :2, None, :] - targets[:, :2, :, None]  # [256, 2, 64, 64]
            targets[:, 0, :] = targets[:, 0, :] / w_ref  # [256, 2, 64, 64]
            targets[:, 1, :] = targets[:, 1, :] / h_ref  # [256, 2, 64, 64]
            # targets[:, 2, :] = w_tgt / w_ref
            targets = torch.cat((targets, (w_tgt / w_ref)[:, None, :, :]), dim=1)  # [256, 3, 64, 64]
            # targets[:, 3, :] = h_tgt / h_ref
            targets = torch.cat((targets, (h_tgt / h_ref)[:, None, :, :]), dim=1)  # [256, 4, 64, 64]
        assert (targets[:, 2, :, :] != 0).all() and (targets[:, 3, :, :] != 0).all()
        assert (torch.diagonal(targets[:, 0, :, :], dim1=-2, dim2=-1) == 0).all()  # the diagonal should be 0 for dx
        assert (torch.diagonal(targets[:, 1, :, :], dim1=-2, dim2=-1) == 0).all()  # the diagonal should be 0 for dy
        assert (torch.diagonal(targets[:, 2, :, :], dim1=-2, dim2=-1) == 1).all()  # the diagonal should be 0 for dw
        assert (torch.diagonal(targets[:, 3, :, :], dim1=-2, dim2=-1) == 1).all()  # the diagonal should be 0 for dh
    if mlp_head:
        return outputs, targets
    else:
        sym_output = None
        if symmetry and sampling == 'v2':
            b, c, num_patches, num_patches = targets.shape
            #make a symmetric output:
            # 1. make a zero matrix
            # 2. in indices[0], indices[1] put output
            # 3. make the shared mask have ones at [set(indices) intersect set(indices.T)] indices
            indices_swapped = torch.cat((indices[1, :][None, :], indices[0, :][None, :]), dim=0)
            intersected_indices = set(indices_swapped.T).intersection(set(indices.T))
            if intersected_indices:
                mask = torch.zeros(num_patches, num_patches)  # bs, num_channels, num_patches, num_patches
                mask[list(indices)] = 1
                intersected_indices = torch.stack(list(intersected_indices)).T
                intersected_indices_swapped = torch.cat((intersected_indices[1, :][None, :], intersected_indices[0, :][None, :]), dim=0)
                mask[list(intersected_indices)] = 1
                mask[list(intersected_indices_swapped)] = 1
                # torch.where(mask==1)
                mask = mask.repeat(b, c, 1, 1)
                # 4. return the symmetric_output = mask(output)
                # outputs: ([256, num_channels, 61])
                # [256, num_channels, 64, 64])
                full_output = torch.zeros(b, c, num_patches, num_patches)
                full_output[:, :, indices[0], indices[1]] = outputs
                full_output = mask * full_output

        # index targets based on the pairs given to the linear layer
        targets = targets[:, :, indices[0], indices[1]]  # [b, num_channels, num_patches, num_patches] -> [b, num_channels, num_pairs]
        return outputs, targets, indices, full_output


def reconstruct_img_with_1_reference_patch(absolute_positions, output, sampling, img_patches, mlp_head, num_patch_per_w, patch_width, C, W, ref_patches, indices):
    num_patches = num_patch_per_w ** 2
    img_size = W
    H = W
    new_image_output = torch.zeros((C, W * 3, H * 3))
    if mlp_head:
        assert indices is None
        # fixing patch ref_patch and building the whole image relative to that:
        if sampling == "v1" or sampling == "v2":
            for ref_patch in ref_patches:
                x_ref_patch = int((absolute_positions[0, 0, ref_patch]).item())
                y_ref_patch = int((absolute_positions[0, 1, ref_patch]).item())
                for i in range(num_patches):
                    # reconstruct with the output values:
                    d_x = output[:, 0, ref_patch, i].item()
                    d_y = output[:, 1, ref_patch, i].item()
                    new_image_output[:,
                    (y_ref_patch * patch_width) + round(d_y * patch_width) + H: (y_ref_patch * patch_width) + round(
                        d_y * patch_width) + patch_width + H,
                    (x_ref_patch * patch_width) + round(d_x * patch_width) + W: (x_ref_patch * patch_width) + round(
                        d_x * patch_width) + patch_width + W] = img_patches[:, i, :, :].squeeze()
        elif sampling == "v3" or sampling == "v3_1":  # centered v3
            for ref_patch in ref_patches:
                xc_ref_patch = ((absolute_positions[0, 0, ref_patch] + absolute_positions[0, 2, ref_patch]) // 2).item()
                yc_ref_patch = ((absolute_positions[0, 1, ref_patch] + absolute_positions[0, 3, ref_patch]) // 2).item()
                wref = (absolute_positions[0, 2, ref_patch] - absolute_positions[0, 0, ref_patch]).item()
                href = (absolute_positions[0, 3, ref_patch] - absolute_positions[0, 1, ref_patch]).item()
                for i in range(num_patches):
                    # (d_xc/wref * wref) + xc_ref
                    xc_out = int(output[0, 0, ref_patch, i].item()) * wref + xc_ref_patch + W
                    # d_yc/href * href + yc_ref
                    yc_out = int(output[0, 1, ref_patch, i].item()) * href + yc_ref_patch + H
                    # wtgt / wref * wref
                    w_out = round(output[0, 2, ref_patch, i].item() * wref)
                    # htgt / href * href
                    h_out = round(output[0, 3, ref_patch, i].item() * href)
                    # resize patch to the correct size
                    resize = torchvision.transforms.Resize(size=(h_out, w_out))
                    if w_out <= 0 or h_out <= 0 or round(yc_out - (h_out / 2)) < 0 or round(xc_out - (w_out / 2)) < 0:
                        continue
                    if round(yc_out - (h_out / 2)) + h_out >= 3 * img_size or round(xc_out - (w_out / 2)) + w_out >= 3 * img_size:
                        continue

                    new_image_output[:,
                    round(yc_out - (h_out / 2)): round(yc_out - (h_out / 2)) + h_out,
                    round(xc_out - (w_out / 2)): round(xc_out - (w_out / 2)) + w_out] = \
                        (resize(img_patches[:, i, :, :].squeeze()))
    else:  # pairwise_mlp and cross_attention:
        assert indices is not None
        for ref_patch in ref_patches:
            # none of the chosen patches are the reference patch
            if (indices[0] == ref_patch).sum() == 0:
                continue
            # filter rows that are for that specific reference patch
            indices_copy = indices[:, indices[0] == ref_patch]  # [num_channels, num_patches_with_ref]
            output_copy = output[:, :, indices[0] == ref_patch]  # [1, num_channels, num_patches_with_ref]
            # we only have target/output values of num_pairs of patches and for different references
            for (ref_patch_, tgt_patch), (d) in zip(indices_copy.T, output_copy.squeeze().T):  # 0 <= ref and tgt <64
                assert ref_patch_.item() == ref_patch
                ref_patch, tgt_patch = ref_patch_.item(), tgt_patch.item()
                if sampling == "v1" or sampling == "v2":
                    d_x, d_y = d
                    d_x, d_y = d_x.item(), d_y.item()
                    x_ref_patch = int((absolute_positions[0, 0, ref_patch]).item())
                    y_ref_patch = int((absolute_positions[0, 1, ref_patch]).item())

                    new_image_output[:,
                    (y_ref_patch * patch_width) + round(d_y * patch_width) + H: (y_ref_patch * patch_width) + round(
                        d_y * patch_width) + patch_width + H,
                    (x_ref_patch * patch_width) + round(d_x * patch_width) + W: (x_ref_patch * patch_width) + round(
                        d_x * patch_width) + patch_width + W] = img_patches[:, tgt_patch, :, :].squeeze()
                elif sampling == "v3" or sampling == "v3_1":  # centered v3
                    d_x, d_y, d_w, d_h = d
                    d_x, d_y = d_x.item(), d_y.item()
                    xc_ref_patch = ((absolute_positions[0, 0, ref_patch] + absolute_positions[0, 2, ref_patch]) // 2).item()
                    yc_ref_patch = ((absolute_positions[0, 1, ref_patch] + absolute_positions[0, 3, ref_patch]) // 2).item()
                    wref = (absolute_positions[0, 2, ref_patch] - absolute_positions[0, 0, ref_patch]).item()
                    href = (absolute_positions[0, 3, ref_patch] - absolute_positions[0, 1, ref_patch]).item()
                    # (d_xc/wref * wref) + xc_ref
                    xc_out = int(d_x) * wref + xc_ref_patch + W
                    # d_yc/href * href + yc_ref
                    yc_out = int(d_y) * href + yc_ref_patch + H
                    # wtgt / wref * wref
                    w_out = round(d_w * wref)
                    # htgt / href * href
                    h_out = round(d_h * href)
                    # resize patch to the correct size
                    resize = torchvision.transforms.Resize(size=(h_out, w_out))
                    if w_out <= 0 or h_out <= 0 or round(yc_out - (h_out / 2)) < 0 or round(xc_out - (w_out / 2)) < 0:
                        continue
                    if round(yc_out - (h_out / 2)) + h_out >= 3 * img_size or round(xc_out - (w_out / 2)) + w_out >= 3 * img_size:
                        continue
                    new_image_output[:,
                    round(yc_out - (h_out / 2)): round(yc_out - (h_out / 2)) + h_out,
                    round(xc_out - (w_out / 2)): round(xc_out - (w_out / 2)) + w_out] = \
                        (resize(img_patches[:, tgt_patch, :, :].squeeze()))
    return new_image_output


def train_one_epoch(args, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    pretrain=False):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    mlp_head = True if model_without_ddp.head_type == "mlp" else False
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        gt_positions = targets.detach().clone()

        if mixup_fn is not None and not pretrain:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(True):
            if pretrain:
                if mlp_head:
                    outputs, targets = pretrain_model(args.train_sampling, model, samples, targets, mlp_head, args.centered_targets, args.symmetry)
                else:
                    outputs, targets, _, sym_output = pretrain_model(args.train_sampling, model, samples, targets, mlp_head, args.centered_targets, args.symmetry)

                if criterion._get_name() != "CrossEntropyLoss":
                    squared_error = (targets - outputs) ** 2
                    mse_x = torch.mean(squared_error[:, 0, :])
                    mse_y = torch.mean(squared_error[:, 1, :])
                    euclidean_error = torch.mean(torch.sqrt(squared_error[:, 0, :] + squared_error[:, 1, :]))
                    if squared_error.shape[1] > 2:  # more than 2 channels -> w and h (v3)
                        mse_w = torch.mean(squared_error[:, 2, :])
                        mse_h = torch.mean(squared_error[:, 3, :])

            else:
                outputs = model(samples)
            loss = criterion(outputs, targets)
            if (args.symmetry and mlp_head) or (args.symmetry and args.train_sampling == "v2"):
                regularizer = SymmetryRegularizer(args.mask_diag)
                reg = regularizer(outputs)
                loss += args.alpha * reg

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            if not args.debug:
                wandb.finish()
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=is_second_order)

        # torch.cuda.synchronize()
        if model_ema is not None: #todo: device is empty??
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if 'mse_x' in locals() and 'mse_y' in locals() and 'euclidean_error' in locals():
            metric_logger.update(mse_x=mse_x)
            metric_logger.update(mse_y=mse_y)
            metric_logger.update(euclidean_error=euclidean_error)
        if 'mse_w' in locals() and 'mse_h' in locals():
            metric_logger.update(mse_w=mse_w)
            metric_logger.update(mse_h=mse_h)
        if 'reg' in locals():
            metric_logger.update(regularizer_loss=reg)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    returned_metrics = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # todo: a special case that I'm debugging now:
    if args.debug_pairwise_mlp or args.debug_cross_attention:
        targets = targets.reshape(targets.shape[0], targets.shape[1], int(math.sqrt(targets.shape[2])), int(math.sqrt(targets.shape[2])))
        outputs = outputs.reshape(outputs.shape[0], outputs.shape[1], int(math.sqrt(outputs.shape[2])), int(math.sqrt(outputs.shape[2])))

    if outputs.dim() == 4:  # for WeightedMSELoss
        random_img_ind = random.randint(0, outputs.size(0)-1)
        img1 = torch.cat((outputs[random_img_ind, 0, :, :].squeeze(), targets[random_img_ind, 0, :, :].squeeze()), dim=1)
        img2 = torch.cat((outputs[random_img_ind, 1, :, :].squeeze(), targets[random_img_ind, 1, :, :].squeeze()), dim=1)
        returned_metrics["output+target_x"] = wandb.Image(img1)
        returned_metrics["output+target_y"] = wandb.Image(img2)
    if outputs.dim() == 4 and outputs.shape[1] == 4:  # dx, dy, dw, dh
        img3 = torch.cat((outputs[random_img_ind, 2, :, :].squeeze(), targets[random_img_ind, 2, :, :].squeeze()), dim=1)
        img4 = torch.cat((outputs[random_img_ind, 3, :, :].squeeze(), targets[random_img_ind, 3, :, :].squeeze()), dim=1)
        returned_metrics["output+target_w"] = wandb.Image(img3)
        returned_metrics["output+target_h"] = wandb.Image(img4)
    return returned_metrics


@torch.no_grad()
def evaluate(args, data_loader, model, device, pretrain=False):

    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module

    if model_without_ddp.loss == "CrossEntropyLoss":
        criterion = torch.nn.CrossEntropyLoss()
    elif model_without_ddp.loss == "WeightedMSELoss":
        criterion = torch.nn.MSELoss(reduction='mean')
        criterion = WeightedMSELoss(
            device=device,
            distance_weight_mode=args.distance_weight_mode,
            mask_prob_weights=args.mask_prob_weights,
            num_patches=model_without_ddp.num_patches,
            w_x=args.w_x, w_y=args.w_y, w_w=args.w_w, w_h=args.w_h,
        )
    elif model_without_ddp.loss == "L1Loss":
        criterion = torch.nn.L1Loss(reduction='mean')
    elif model_without_ddp.loss == "HuberLoss":
        criterion = torch.nn.HuberLoss(reduction='mean')
    elif model_without_ddp.loss == "SmoothL1Loss":
        criterion = torch.nn.SmoothL1Loss(reduction='mean')

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    mlp_head = True if model_without_ddp.head_type == "mlp" else False

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(True):
            if pretrain:
                if mlp_head:
                    output, target = pretrain_model(args.test_sampling, model, images, target, mlp_head, args.centered_targets)
                else:
                    output, target, _ = pretrain_model(args.test_sampling, model, images, target, mlp_head, args.centered_targets)
            else:
                output = model(images)
            loss = criterion(output, target)
            if args.symmetry and mlp_head:
                regularizer = SymmetryRegularizer(args.mask_diag)
                reg = regularizer(output)
                loss += args.alpha * reg

        if criterion._get_name() == "CrossEntropyLoss":
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # all other regression losses
        else:
            correct = output.eq(target)
            # b, c, p1, p2 = target.shape
            acc1, acc5 = (correct.sum() * 100. / (torch.prod(torch.Tensor(list(target.size()))))), torch.Tensor([0])
            squared_error = (target - output) ** 2
            mse_x = torch.mean(squared_error[:, 0, :])
            mse_y = torch.mean(squared_error[:, 1, :])
            euclidean_error = torch.mean(torch.sqrt(squared_error[:, 0, :] + squared_error[:, 1, :]))
            if squared_error.shape[1] > 2:  # more than 2 channels -> w and h (v3)
                mse_w = torch.mean(squared_error[:, 2, :])
                mse_h = torch.mean(squared_error[:, 3, :])

        batch_size = images.shape[0]
        if 'euclidean_error' in locals() and 'mse_x' in locals() and 'mse_y' in locals():
            metric_logger.meters['mse_x'].update(mse_x.item(), n=batch_size)
            metric_logger.meters['mse_y'].update(mse_y.item(), n=batch_size)
            metric_logger.meters['euclidean_error'].update(euclidean_error.item(), n=batch_size)
        if 'mse_w' in locals() and 'mse_h' in locals():
            metric_logger.meters['mse_w'].update(mse_w.item(), n=batch_size)
            metric_logger.meters['mse_h'].update(mse_h.item(), n=batch_size)

        if 'reg' in locals():
            metric_logger.update(regularizer_loss=reg)


        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_one_img(args, data_loader, model, device, num_samples, train_time):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module

    mlp_head = True if model_without_ddp.head_type == "mlp" else False
    # switch to evaluation mode
    model.eval()
    num_patch_per_w = int(math.sqrt(model_without_ddp.num_patches))
    if train_time:
        sampling = args.train_sampling
    else:
        sampling = args.test_sampling

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        absolute_positions = target.detach().clone()  # for v1, v2: [b, 2, 64] for v3: [b, 4, 64]

        B, C, W, H = images.shape
        patch_width = int(W / num_patch_per_w)

        # compute output
        with torch.cuda.amp.autocast(True):
            if mlp_head:
                indices = None
                output, target = pretrain_model(sampling, model, images, target, mlp_head, args.centered_targets)
            else:
                output, target, indices = pretrain_model(sampling, model, images, target, mlp_head, args.centered_targets)
            # img_patches: [3, 64, 4, 4]
            img_patches = images.squeeze().unfold(1, patch_width, patch_width).unfold(2, patch_width, patch_width).contiguous().view(3, -1, patch_width, patch_width)
            try:
                new_image_target = reconstruct_img_with_1_reference_patch(absolute_positions, target, sampling, img_patches, mlp_head, num_patch_per_w, patch_width, C, W, args.ref_patches, indices)
                new_image_output = reconstruct_img_with_1_reference_patch(absolute_positions, output, sampling, img_patches, mlp_head, num_patch_per_w, patch_width, C, W, args.ref_patches, indices)

                if utils.is_main_process():
                    if not args.debug:
                        wandb.log({f"ref_patch {args.ref_patches} output train time: {train_time} "
                                   f"img: {num_samples}": wandb.Image(new_image_output),
                                   f"ref_patch {args.ref_patches} target train time: {train_time} "
                                   f"img: {num_samples}": wandb.Image(new_image_target),
                                   })

            except Exception as e:
                print(e)
                traceback.print_exc()
                print(f"ERROR in image reconstruction! the indices go out of the image borders")


            visualize_uncertainty = False
            if visualize_uncertainty:
                # showing where one patch should be with respect to all reference patches:
                new_uncertainty_img = torch.zeros((C, W * 3, H * 3))
                for i in [0, 7, 35, 56, 63]:
                    for ref_patch in range(model_without_ddp.num_patches):
                        x_ref_patch = ref_patch % num_patch_per_w
                        y_ref_patch = int(ref_patch / num_patch_per_w)
                        print(f"reference patch {ref_patch} is: ({x_ref_patch}, {y_ref_patch})")
                        d_x = output[:, 0, ref_patch, i].item()
                        d_y = output[:, 1, ref_patch, i].item()
                        print(f"d_x {d_x}, d_y: {d_y}")
                        print(f"gt_x {target[:, 0, ref_patch, i].item()}, gt_y: {target[:, 1, ref_patch, i].item()}")

                        new_uncertainty_img[:,
                        (y_ref_patch * patch_width) + round(d_y * patch_width) + H: (y_ref_patch * patch_width) + round(
                            d_y * patch_width) + patch_width + H,
                        (x_ref_patch * patch_width) + round(d_x * patch_width) + W: (x_ref_patch * patch_width) + round(
                            d_x * patch_width) + patch_width + W] = img_patches[:, i, :, :].squeeze()
                    # print(f"patch: {ref_patch}, error x: {torch.mean(d_x - target[:, 0, ref_patch, j])} and error y: {torch.mean(d_y - target[:, 1, ref_patch, j])}")
                    if utils.is_main_process():
                        if not args.debug:
                            try:
                                wandb.log({f"uncertainty of patch {i} with respect to everything else": wandb.Image(new_uncertainty_img)})
                            except Exception as e:
                                print(e)
                                traceback.print_exc()
                                print("wandb.log failed")

                    new_uncertainty_img = torch.zeros((C, W * 3, H * 3))

            if utils.is_main_process():
                if not args.debug:
                    try:
                        wandb.log({f'original image {num_samples}': wandb.Image(images.squeeze()),})
                    except Exception as e:
                        print(e)
                        traceback.print_exc()
                        print("wandb.log failed")

        if num_samples <= 1:
            break
        else:
            num_samples -= 1
    return