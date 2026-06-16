
from torchvision import datasets, transforms
from torchvision.datasets.cifar import CIFAR10
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from samplers import RASampler
import utils
from torchvision.datasets.folder import DatasetFolder, default_loader, IMG_EXTENSIONS
from typing import Any, Callable, Optional, Tuple
import math
import time
import datetime
from PIL import Image
import wandb
import skimage
import numpy as np
import os
from viz_utils import reconstruct_image_unfold_v4, viz_img_bbs
import traceback


def patchify_and_resize_np(start_x, start_y, end_x, end_y, image, resize):
    # image.shape: [32, 32, 3]
    image_patched = image[start_y:end_y, start_x:end_x, :]
    if resize is not None:
        image_patched = skimage.transform.resize(image_patched, resize)

    return image_patched


def patchify_and_resize(start_x, start_y, end_x, end_y, image, resize):
    # print(image[:, start_y:end_y, start_x:end_x].shape)
    image_patched = image[:, start_y:end_y, start_x:end_x]
    if resize is not None:
        image_patched = skimage.transform.resize(image_patched, resize)

    return image_patched


def patchify_and_resize_batch_np(start_x, start_y, end_x, end_y, image, resize):
    # image.shape: [batch, 32, 32, 3]
    image_patched = image[:, start_y:end_y, start_x:end_x, :]
    if resize is not None:
        image_patched = skimage.transform.resize(image_patched, resize)

    return image_patched


class VectorizedCIFAR(CIFAR10):
    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

    def __init__(
            self,
            sampling: str,
            shuffle_patches: bool,
            img_size: int,
            patch_size: int,
            min_wh: int,
            max_wh: int,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            visualize: bool = False,
    ) -> None:

        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        # # load targets
        self.img_size = img_size
        self.patch_size = patch_size
        self.sampling = sampling
        self.shuffle_patches = shuffle_patches
        self.visualize = visualize

        if sampling == 'v1':
            # v1 unshuffled
            self.targets = torch.zeros(2, self.num_patches, dtype=int)  # [2, 64]
            x = torch.arange(0, img_size, patch_size, dtype=int)
            y = torch.arange(0, img_size, patch_size, dtype=int)
            target_x, target_y = torch.meshgrid(x, y, indexing='xy')
            self.targets[0], self.targets[1] = target_x.flatten(), target_y.flatten()

        elif sampling == 'v2':
            x_start = torch.arange(img_size - patch_size + 1)
            y_start = torch.arange(img_size - patch_size + 1)
            # save x_start and y_start in a file for future loading
            t = torch.cartesian_prod(y_start, x_start).T  # [y_start, x_start] torch.Size([2, 841])
            self.targets = torch.cat((t[None, 1, :], t[None, 0, :]), dim=0)  # [2, 841] switch y and x to x and y

        elif sampling == 'v3':
            x_starts, y_starts, x_ends, y_ends = (torch.randint(0, img_size, (100000,)),
                                                  torch.randint(0, img_size, (100000,)),
                                                  torch.randint(0, img_size, (100000,)),
                                                  torch.randint(0, img_size, (100000,)))

            for x1, x2, y1, y2 in zip(x_starts, x_ends, y_starts, y_ends):
                if abs(x1 - x2) < min_wh or abs(x1 - x2) > max_wh:
                    continue
                if abs(y1 - y2) < min_wh or abs(y1 - y2) > max_wh:
                    continue

                x_s = min(x1, x2)
                y_s = min(y1, y2)
                x_e = max(x1, x2)
                y_e = max(y1, y2)
                # print(f"adding a patch with the following coordinates: {x_s}, {y_s}, {x_e}, {y_e}")
                if 'target' in locals():
                    target = torch.cat((target, torch.tensor([x_s, y_s, x_e, y_e])[:, None]), dim=1)
                else:
                    target = torch.tensor([x_s, y_s, x_e, y_e])[:, None]
            print(f"{target.shape[1]} patches have been extracted with min {min_wh} and max {max_wh} ")
            self.targets = target  # [4, ~10k]
        elif sampling == 'v3_1':  # todo: this is for debugging visualization of v3 and the targets in v3
            # v1 unshuffled
            self.targets = torch.zeros(2, self.num_patches, dtype=int)  # [2, 64]
            x = torch.arange(0, img_size, patch_size, dtype=int)
            y = torch.arange(0, img_size, patch_size, dtype=int)
            target_x, target_y = torch.meshgrid(x, y, indexing='xy')
            self.targets[0], self.targets[1] = target_x.flatten(), target_y.flatten()
            self.targets = torch.cat((self.targets, target_x.flatten()[None, :] + patch_size), dim=0)
            self.targets = torch.cat((self.targets, target_y.flatten()[None, :] + patch_size), dim=0)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # self.targets: list 50000, self.data (50000, 32, 32, 3)
        # randomly choose the number of patches and then reshape
        if self.sampling == 'v1' or self.sampling == 'v3_1':
            # shuffled v1
            if self.shuffle_patches:
                r = torch.randperm(self.num_patches)
                self.targets = self.targets[:, r]
            target = self.targets

        elif self.sampling == 'v2' or self.sampling == 'v3':
            rand_ind = torch.randint(0, self.targets.shape[1], (self.num_patches,))  # torch.Size([64])
            # rand_ind = torch.arange(128*5, 128*5+64)
            target = self.targets[:, rand_ind]  # [num_channels, 64]: chosen x1, y1, x2, y2

        img = self.data[index]

        if utils.is_main_process():
            if self.visualize:
                try:
                    wandb.log({'original image': wandb.Image(img)})
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    print("wandb.log failed")

        visualize = False
        if visualize:
            # img[32,32,3]
            patch_func = np.vectorize(patchify_and_resize_np,
                                      excluded=['image', 'resize'],
                                      # otypes=['uint8'],
                                      signature='(),(),(),()->(l,m,k)'
                                      )
            if self.sampling == 'v1' or self.sampling == 'v2':
                # patches: [64, 4, 4, 3]
                patches = patch_func(target[0], target[1], target[0] + self.patch_size, target[1] + self.patch_size,
                                     image=img,
                                     resize=None
                                     # resize=(3, self.patch_size, self.patch_size)
                                     )
                bbs = torch.cat((target, torch.cat(
                    (target[0].unsqueeze(dim=0) + self.patch_size, target[1].unsqueeze(dim=0) + self.patch_size),
                    dim=0)), dim=0)
            elif self.sampling == 'v3':
                # patches: [64, 4, 4, 3]
                patches = patch_func(target[0], target[1], target[2], target[3],
                                     image=img,
                                     resize=(self.patch_size, self.patch_size, 3)
                                     )
                bbs = target

            for i in [8, 16, 32, 64]:  # i is the number of patches visualized
                # img_patches: [64, 4, 4, 3]: ndarray
                viz_img_bbs(img, bbs[:, :i])
                # reconstruct_image_unfold_v4(patches, target)
                reconstruct_image_unfold_v4(patches[:i, :, :, :], bbs[:, :i])  # img_patches: [64, 4, 4, 3], [4, 64]
                # reconstruct_image_unfold(self.data[index], self.targets)

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)  # [3, 32, 32]

        patchify = np.vectorize(patchify_and_resize,
                                excluded=['image', 'resize'],
                                # otypes=['uint8'],
                                signature='(),(),(),()->(l,m,k)'
                                )

        if self.sampling == 'v1' or self.sampling == 'v2':
            # img_patches: [64, 3, 4, 4]
            img_patches = patchify(target[0], target[1], target[0]+self.patch_size, target[1]+self.patch_size,
                                   image=np.asarray(img),
                                   resize=None
                                   # resize=(3, self.patch_size, self.patch_size)
                                   )
        elif self.sampling == 'v3' or self.sampling == "v3_1":
            # img_patches: [64, 3, 4, 4]
            img_patches = patchify(target[0], target[1], target[2], target[3],
                                   image=np.asarray(img),
                                   resize=(3, self.patch_size, self.patch_size)
                                   )
        patches_per_width = int(math.sqrt(self.num_patches))
        img_patches = torch.from_numpy(img_patches)
        # [3, 64, 4, 4] -> [3, 8, 8, 4, 4] -> [3, 8, 4, 8, 4]
        img_patches = img_patches.permute(1, 0, 2, 3)
        img_patches = img_patches.reshape(3, patches_per_width, patches_per_width, self.patch_size, self.patch_size)
        img_patches = img_patches.permute(0, 1, 3, 2, 4).reshape(3, self.img_size, self.img_size)

        if self.target_transform is not None:
            target = self.target_transform(target)
        # print('reshaping and targets time {}'.format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
        if self.sampling == 'v1' or self.sampling == 'v2':
            # todo: I should do the normalization somewhere else
            target = (target / self.patch_size)

        return img_patches, target  # [4, 64]


class VectorizedIMAGENET(DatasetFolder):
    def __init__(
            self,
            root: str,
            sampling: str,
            img_size: int,
            patch_size: int,
            shuffle_patches: bool,
            min_wh: int,
            max_wh: int,
            visualize: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        # # load targets
        self.img_size = img_size
        self.patch_size = patch_size
        self.sampling = sampling
        self.shuffle_patches = shuffle_patches
        self.visualize = visualize

        if sampling == 'v1':
            # v1 unshuffled
            self.targets = torch.zeros(2, self.num_patches, dtype=int)  # [2, 64]
            x = torch.arange(0, img_size, patch_size, dtype=int)
            y = torch.arange(0, img_size, patch_size, dtype=int)
            target_x, target_y = torch.meshgrid(x, y, indexing='xy')
            self.targets[0], self.targets[1] = target_x.flatten(), target_y.flatten()

        elif sampling == 'v2':
            x_start = torch.arange(img_size - patch_size + 1)
            y_start = torch.arange(img_size - patch_size + 1)
            # save x_start and y_start in a file for future loading
            t = torch.cartesian_prod(y_start, x_start).T  # [y_start, x_start] torch.Size([2, 841])
            self.targets = torch.cat((t[None, 1, :], t[None, 0, :]), dim=0)  # [2, 841] switch y and x to x and y

        elif sampling == 'v3':
            x_starts, y_starts, x_ends, y_ends = (torch.randint(0, img_size, (100000000,)),
                                                  torch.randint(0, img_size, (100000000,)),
                                                  torch.randint(0, img_size, (100000000,)),
                                                  torch.randint(0, img_size, (100000000,)))

            for x1, x2, y1, y2 in zip(x_starts, x_ends, y_starts, y_ends):
                if abs(x1 - x2) < min_wh or abs(x1 - x2) > max_wh:
                    continue
                if abs(y1 - y2) < min_wh or abs(y1 - y2) > max_wh:
                    continue

                x_s = min(x1, x2)
                y_s = min(y1, y2)
                x_e = max(x1, x2)
                y_e = max(y1, y2)
                # print(f"adding a patch with the following coordinates: {x_s}, {y_s}, {x_e}, {y_e}")
                if 'target' in locals():
                    target = torch.cat((target, torch.tensor([x_s, y_s, x_e, y_e])[:, None]), dim=1)
                else:
                    target = torch.tensor([x_s, y_s, x_e, y_e])[:, None]
            print(f"{target.shape[1]} patches have been extracted with min {min_wh} and max {max_wh} ")
            self.targets = target  # [4, ~10k]
        # this is for debugging visualization of v3 and the targets in v3
        elif sampling == 'v3_1':
            # v1 unshuffled
            self.targets = torch.zeros(2, self.num_patches, dtype=int)  # [2, 64]
            x = torch.arange(0, img_size, patch_size, dtype=int)
            y = torch.arange(0, img_size, patch_size, dtype=int)
            target_x, target_y = torch.meshgrid(x, y, indexing='xy')
            self.targets[0], self.targets[1] = target_x.flatten(), target_y.flatten()
            self.targets = torch.cat((self.targets, target_x.flatten()[None, :] + patch_size), dim=0)
            self.targets = torch.cat((self.targets, target_y.flatten()[None, :] + patch_size), dim=0)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.sampling == 'v1' or self.sampling == 'v3_1':
            # shuffled v1
            if self.shuffle_patches:
                r = torch.randperm(self.num_patches)
                self.targets = self.targets[:, r]
            target = self.targets

        elif self.sampling == 'v2' or self.sampling == 'v3':
            rand_ind = torch.randint(0, self.targets.shape[1], (self.num_patches,))  # torch.Size([64])
            # rand_ind = torch.arange(128*5, 128*5+64)
            target = self.targets[:, rand_ind]  # [num_channels, 64]: chosen x1, y1, x2, y2

        path, _ = self.samples[index]  # the target is not returned
        img = self.loader(path)

        if utils.is_main_process():
            if self.visualize:
                img_tensor = transforms.ToTensor()(transforms.Resize((self.img_size, self.img_size))(img))  # [3, 224, 224]
                try:
                    wandb.log({'original image': wandb.Image(img_tensor)})
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    print("wandb.log failed")

        visualize = False
        #todo: needs to be checked for imagenet, it's correct for cifar
        if visualize:
            # img[3, 32, 32]
            patch_func = np.vectorize(patchify_and_resize,
                                      excluded=['image', 'resize'],
                                      # otypes=['uint8'],
                                      signature='(),(),(),()->(l,m,k)'
                                      )
            if self.sampling == 'v1' or self.sampling == 'v2':
                # patches: [64, 4, 4, 3]
                patches = patch_func(target[0], target[1], target[0] + self.patch_size, target[1] + self.patch_size,
                                     image=img_tensor,
                                     resize=None
                                     # resize=(3, self.patch_size, self.patch_size)
                                     )
                bbs = torch.cat((target, torch.cat((target[0].unsqueeze(dim=0) + self.patch_size, target[1].unsqueeze(dim=0) + self.patch_size), dim=0)), dim=0)
            elif self.sampling == 'v3':
                # patches: [64, 3, 4, 4]
                patches = patch_func(target[0], target[1], target[2], target[3],
                                     image=img_tensor,
                                     resize=(3, self.patch_size, self.patch_size)
                                     )
                patches = np.asarray(patches.permute(0, 2, 3, 1))
                bbs = target

            img_tensor = np.asarray(img_tensor.permute(1, 2, 0))
            for i in [8, 16, 32, 64]:  # i is the number of patches visualized
                # img_patches: [64, 4, 4, 3]: ndarray
                viz_img_bbs(img_tensor, bbs[:, :i])
                reconstruct_image_unfold_v4(patches[:i, :, :, :], bbs[:, :i])  # img_patches: [64, 4, 4, 3], [4, 64]

        if self.transform is not None:
            img = self.transform(img)  # [3, 32, 32]

        patchify = np.vectorize(patchify_and_resize,
                                excluded=['image', 'resize'],
                                # otypes=['uint8'],
                                signature='(),(),(),()->(l,m,k)'
                                )

        if self.sampling == 'v1' or self.sampling == 'v2':
            # img_patches: [64, 3, 4, 4]
            img_patches = patchify(target[0], target[1], target[0]+self.patch_size, target[1]+self.patch_size,
                                   image=np.asarray(img),
                                   resize=None
                                   # resize=(3, self.patch_size, self.patch_size)
                                   )
        elif self.sampling == 'v3' or self.sampling == "v3_1":
            # img_patches: [64, 3, 4, 4]
            img_patches = patchify(target[0], target[1], target[2], target[3],
                                   image=np.asarray(img),
                                   resize=(3, self.patch_size, self.patch_size)
                                   )
        patches_per_width = int(math.sqrt(self.num_patches))
        img_patches = torch.from_numpy(img_patches)
        # [3, 64, 4, 4] -> [3, 8, 8, 4, 4] -> [3, 8, 4, 8, 4]
        img_patches = img_patches.permute(1, 0, 2, 3)
        img_patches = img_patches.reshape(3, patches_per_width, patches_per_width, self.patch_size, self.patch_size)
        img_patches = img_patches.permute(0, 1, 3, 2, 4).reshape(3, self.img_size, self.img_size)

        if self.target_transform is not None:
            target = self.target_transform(target)
        # print('reshaping and targets time {}'.format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
        if self.sampling == 'v1' or self.sampling == 'v2':
            # todo: I should do the normalization somewhere else
            target = (target / self.patch_size)

        return img_patches, target


class PatchCIFARPerItem(CIFAR10):

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

    def __init__(
            self,
            img_size: int,
            patch_size: int,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        # # load targets
        self.img_size = img_size
        self.patch_size = patch_size
        x_start = torch.arange(img_size - patch_size + 1)
        y_start = torch.arange(img_size - patch_size + 1)
        # save x_start and y_start in a file for future loading
        t = torch.cartesian_prod(y_start, x_start).T  # [y_start, x_start] torch.Size([2, 841])
        self.targets = torch.cat((t[None, 1, :], t[None, 0, :]), dim=0)  # switch y and x to x and y

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # self.targets: list 50000, self.data (50000, 32, 32, 3)
        # randomly choose the number of patches and then reshape
        rand_ind = torch.randint(0, self.targets.shape[1], (self.num_patches,))  # torch.Size([64])

        img = self.data[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)  # [3, 32, 32]
        target = self.targets[:, rand_ind]  # [2, 64]: chosen x and ys

        for x, y in target.T:
            if 'img_patches' not in locals():
                img_patches = img[:, None, y:y+self.patch_size, x:x+self.patch_size]  # [3, 1, 4, 4]
            else:
                img_patches = torch.cat((img_patches, img[:, None, y:y+self.patch_size, x:x+self.patch_size]), dim=1)
        # img_patches: [3, 64, 4, 4]
        patches_per_width = int(math.sqrt(self.num_patches))
        # [3, 64, 4, 4] -> [3, 8, 8, 4, 4] -> [3, 8, 4, 8, 4]
        img_patches = img_patches.reshape(3, patches_per_width, patches_per_width, self.patch_size, self.patch_size)
        img_patches = img_patches.permute(0, 1, 3, 2, 4).reshape(3, self.img_size, self.img_size)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # print('reshaping and targets time {}'.format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
        return img_patches, target  # [2, 64]


class CompactPatchCIFAR100(CIFAR10):

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

    def __init__(
            self,
            patch_size: int,
            root: str,
            img_size: int = 32,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super().__init__('./', train=train, transform=transform, target_transform=target_transform, download=download)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.data, self.targets = generate_dataset(patch_size=patch_size, data_path=root, is_train=train)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # randomly choose the number of patches and then reshape
        rand_ind = torch.randint(0, self.targets.shape[1], (self.num_patches,))  # torch.Size([64])

        img_patches = self.data[index]  # self.data: [50000, 841, 4, 4, 3] -> img_patches: [841, 4, 4, 3]
        img_patches = img_patches[rand_ind, :, :, :]  # img_patches: [64, 4, 4, 3]
        target = self.targets[:, rand_ind]  # [2, 64]: chosen x and ys

        # reconstruct_image_unfold_v2(img_patches, target)  # img_patches: [64, 4, 4, 3], [2, 64]
        # self.data[index]: [841, 4, 4, 3], [2, 841]
        # reconstruct_image_unfold_v2(self.data[index], self.targets)

        # img_patches: [64, 4, 4, 3]
        patches_per_width = int(math.sqrt(self.num_patches))
        # [3, 64, 4, 4] -> [3, 8, 8, 4, 4] -> [3, 8, 4, 8, 4]
        # [64, 4, 4, 3] -> [8, 8, 4, 4, 3] -> [8, 4, 8, 4, 3]
        img_patches = img_patches.reshape(patches_per_width, patches_per_width, self.patch_size, self.patch_size, 3)
        img_patches = img_patches.transpose(0, 2, 1, 3, 4).reshape(self.img_size, self.img_size, 3)  # [32, 32, 3]

        img_patches = Image.fromarray(img_patches)
        if self.transform is not None:
            img_patches = self.transform(img_patches)  # [3, 32, 32]

        if self.target_transform is not None:
            target = self.target_transform(target)

        # visualize the reconstruction of img_patches of that image and self.data[index] as well
        # plt.imshow(img_patches.permute(1, 2, 0))
        # plt.show()

        return img_patches, (target/self.patch_size)  # [2, 64]


class ImagePatchFolder(DatasetFolder):
    def __init__(
        self,
        num_patches,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

        self.num_patches = num_patches
        # merge all pathes of patches of the same image
        imgs = {}
        for sample in self.samples:
            path = sample[0]
            img_id = int(path.split('/')[-2])
            if img_id in imgs:
                imgs[img_id].append(sample)
            else:
                imgs[img_id] = [sample]
        # check if all images have 841 patches
        for k, v in imgs.items():
            if len(v) != 841:
                print(k)
        imgs = [self.samples[i: i + 841] for i in range(0, len(self.samples), 841)]
        self.samples = imgs
        # load targets
        self.targets = torch.load(f"{root}/x_ys.pt")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # start_time = time.time()
        rand_ind = torch.randint(0, len(self.targets), (self.num_patches,))  # torch.Size([64])
        paths_plus_target = self.samples[index]
        # print('self.samples[index] time {}'.format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
        # start_time = time.time()
        for ind in rand_ind:
            img = self.loader(paths_plus_target[ind.item()][0])
            if self.transform is not None:
                if 'img_patches' not in locals():
                    img_patches = self.transform(img)[:, None, :, :]  # torch.Size([3, 1, 4, 4])
                else:
                    img_patches = torch.cat((img_patches, self.transform(img)[:, None, :, :]), dim=1)
        # print('for loop time {}'.format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
        # start_time = time.time()
        # randomly choose the number of patches and then reshape
        # img_patches = img_patches[:, rand_ind, :, :]  # [3, 64, 4, 4]
        patches_per_width = int(math.sqrt(self.num_patches))
        # [3, 64, 4, 4] -> [3, 8, 8, 4, 4] -> [3, 8, 4, 8, 4]
        img_patches = img_patches.reshape(3, patches_per_width, patches_per_width, 4, 4)
        img_patches = img_patches.permute(0, 1, 3, 2, 4).reshape(3, 32, 32)
        # randomly choose the same targets and then reshape
        target = self.targets[rand_ind, :]
        if self.target_transform is not None:
            target = self.target_transform(target)
        # print('reshaping and targets time {}'.format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
        return img_patches, target


def test_dataloader_dataset(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # dataset = ImagePatchFolder(64, f"./{args.data_set}_patched/train/", transform=transforms.ToTensor())
    dataset_2 = datasets.CIFAR100(args.data_path, train=True, transform=transforms.ToTensor())
    dataset_3 = PatchCIFARPerItem(32, 4, f".", train=True, transform=transforms.ToTensor(), download=True)
    dataset_4 = VectorizedCIFAR('v2', 32, 4, f".", train=True, transform=transforms.ToTensor(), download=True)
    dataset_5 = CompactPatchCIFAR100(4, "./CIFAR100_compact_patch", train=True, transform=transforms.ToTensor(), download=True)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    # if args.repeated_aug:
    #     sampler_train = RASampler(
    #         dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    #     )
    # else:
    #     sampler_train = torch.utils.data.DistributedSampler(
    #         dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    #     )

    if args.repeated_aug:
        sampler_train_2 = RASampler(
            dataset_2, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train_2 = torch.utils.data.DistributedSampler(
            dataset_2, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if args.repeated_aug:
        sampler_train_3 = RASampler(
            dataset_3, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train_3 = torch.utils.data.DistributedSampler(
            dataset_3, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if args.repeated_aug:
        sampler_train_4 = RASampler(
            dataset_4, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train_4 = torch.utils.data.DistributedSampler(
            dataset_4, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if args.repeated_aug:
        sampler_train_5 = RASampler(
            dataset_5, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train_5 = torch.utils.data.DistributedSampler(
            dataset_5, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    # data_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     sampler=sampler_train,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     drop_last=True,
    # )

    data_loader_2 = torch.utils.data.DataLoader(
        dataset_2,
        sampler=sampler_train_2,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )

    data_loader_3 = torch.utils.data.DataLoader(
        dataset_3,
        sampler=sampler_train_3,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )

    data_loader_4 = torch.utils.data.DataLoader(
        dataset_4,
        sampler=sampler_train_4,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )

    data_loader_5 = torch.utils.data.DataLoader(
        dataset_5,
        sampler=sampler_train_5,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )

    # start_time = time.time()
    # for samples, targets in data_loader:
    #     samples = samples.to(device, non_blocking=True)
    #     print(f" samples.shape: {samples.shape}")
    #     targets = targets.to(device, non_blocking=True)
    #     print("1 batch loaded with patched cifar100")
    #     break
    # print('patched cifar100 time {}'.format(str(datetime.timedelta(seconds=time.time() - start_time))))

    start_time = time.time()
    for samples, targets in data_loader_2:
        samples = samples.to(device, non_blocking=True)
        print(f" samples.shape: {samples.shape}")
        targets = targets.to(device, non_blocking=True)
        print("1 batch loaded with normal cifar100")
        break
    print('normal cifar100 time {}'.format(str(datetime.timedelta(seconds=time.time() - start_time))))

    start_time = time.time()
    for samples, targets in data_loader_3:
        samples = samples.to(device, non_blocking=True)
        print(f" samples.shape: {samples.shape}")
        targets = targets.to(device, non_blocking=True)
        print("1 batch loaded with patched cifar100 on the fly")
        break
    print('on-the-fly cifar100 time {}'.format(str(datetime.timedelta(seconds=time.time() - start_time))))

    start_time = time.time()
    for samples, targets in data_loader_4:
        samples = samples.to(device, non_blocking=True)
        print(f" samples.shape: {samples.shape}")
        targets = targets.to(device, non_blocking=True)
        print("1 batch loaded with patched cifar100 vectorized, no for loop ")
        break
    print('vectorized cifar100 time {}'.format(str(datetime.timedelta(seconds=time.time() - start_time))))

    start_time = time.time()
    for samples, targets in data_loader_5:
        samples = samples.to(device, non_blocking=True)
        print(f" samples.shape: {samples.shape}")
        targets = targets.to(device, non_blocking=True)
        print("1 batch loaded with compact cifar100")
        break
    print('compact cifar100 time {}'.format(str(datetime.timedelta(seconds=time.time() - start_time))))


    # old:
    # 1 batch loaded with patched cifar100
    # patched cifar100 time 0:05:40
    # 1 batch loaded with normal cifar100
    # normal cifar100 time 0:00:00

    # new:
    # vectorized cifar100 time 0:00:01.900800
    # normal cifar100 time 0:00:01.327485
    # on-the-fly cifar100 time 0:00:03.585607


def generate_dataset(patch_size, data_path, is_train):

    if is_train:
        folder = 'train'
    else:
        folder = 'val'

    if os.path.exists(f"{data_path}/{folder}/data.pt"):
        return torch.load(f"{data_path}/{folder}/data.pt"), torch.load(f"{data_path}/{folder}/x_ys.pt")

    Path(f"{data_path}/{folder}/").mkdir(parents=True, exist_ok=True)

    img_size = 32

    dataset = datasets.CIFAR100("./", train=is_train, download=True)

    # make a list of x, ys that we want to sample
    x_start = torch.arange(img_size - patch_size + 1)
    y_start = torch.arange(img_size - patch_size + 1)
    # save x_start and y_start in a file for future loading
    t = torch.cartesian_prod(y_start, x_start).T  # [y_start, x_start] torch.Size([2, 841])
    patch_x_ys = torch.cat((t[None, 1, :], t[None, 0, :]), dim=0)  # switch y and x to x and y

    # for v3 we need x2_start and y2_start as well
    start_x = patch_x_ys[0]
    start_y = patch_x_ys[1]
    end_x = patch_x_ys[0] + patch_size
    end_y = patch_x_ys[1] + patch_size

    torch.save(patch_x_ys, f"{data_path}/{folder}/x_ys.pt")

    patchify_batch = np.vectorize(patchify_and_resize_batch_np,
                                  excluded=['image', 'resize'],
                                  otypes=['uint8'],
                                  signature='(),(),(),()->(h,l,m,k)'
                                  )

    # img_patches.shape: (841, 50000, 4, 4, 3)
    img_patches = patchify_batch(start_x, start_y, end_x, end_y,
                                 image=dataset.data,  # [50000, 32, 32, 3]
                                 resize=None
                                 # resize=(samples.shape[0], self.patch_size, self.patch_size, 3)
                                 )

    # torch.Size([50000, 841, 4, 4, 3])
    img_patches = img_patches.transpose(1, 0, 2, 3, 4)

    torch.save(img_patches, f'{data_path}/{folder}/data.pt', pickle_protocol=4)
    print("dataset is generated successfully!")
    return img_patches, patch_x_ys


def test_compact_patch_cifar100(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    train_data = CompactPatchCIFAR100(4, "./CIFAR100_compact_patch", train=True, transform=transforms.ToTensor(), download=True)
    val_data = CompactPatchCIFAR100(4, "./CIFAR100_compact_patch", train=False, transform=transforms.ToTensor(), download=True)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    if args.repeated_aug:
        sampler_train = RASampler(
            train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    data_loader_train = torch.utils.data.DataLoader(
        train_data, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        val_data, batch_size=int(1.5 * args.batch_size),
        shuffle=False, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )

    for samples, targets in data_loader_train:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

    for samples, targets in data_loader_val:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)


def main2(args):
    # start_time = time.time()
    # print('patched cifar100 time {}'.format(str(datetime.timedelta(seconds=time.time() - start_time))))
    test_dataloader_dataset(args)
    # test_compact_patch_cifar100(args)

