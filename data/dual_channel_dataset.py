"""Dual-channel dataset for CycleGAN virtual H&E staining.

Loads two strictly aligned input channels (brightfield + autofluorescence)
and stacks them into a 2-channel tensor for domain A.
Domain B remains standard RGB H&E images.
"""

import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch


class DualChannelDataset(BaseDataset):
    """Dataset for dual-channel (BF + AF) to H&E translation.

    Directory structure:
        dataroot/trainA_BF/   -- brightfield images
        dataroot/trainA_AF/   -- autofluorescence images (same filenames)
        dataroot/trainB/      -- H&E RGB images (for unaligned training)
        dataroot/testA_BF/    -- test BF images
        dataroot/testA_AF/    -- test AF images
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.dir_A_bf = os.path.join(opt.dataroot, opt.phase + "A_BF")
        self.dir_A_af = os.path.join(opt.dataroot, opt.phase + "A_AF")

        bf_paths = sorted(make_dataset(self.dir_A_bf, opt.max_dataset_size))
        af_paths = sorted(make_dataset(self.dir_A_af, opt.max_dataset_size))

        # Match BF and AF by filename (not just sorted index)
        af_dict = {os.path.basename(p): p for p in af_paths}
        self.A_pairs = []
        for bf_p in bf_paths:
            fname = os.path.basename(bf_p)
            if fname in af_dict:
                self.A_pairs.append((bf_p, af_dict[fname]))
        assert len(self.A_pairs) > 0, (
            f"No matching BF/AF filename pairs found. "
            f"BF dir: {self.dir_A_bf} ({len(bf_paths)} files), "
            f"AF dir: {self.dir_A_af} ({len(af_paths)} files). "
            f"All BF and AF images must have identical filenames."
        )
        if len(self.A_pairs) < len(bf_paths):
            print(f"Warning: {len(bf_paths) - len(self.A_pairs)} BF images have no AF match and will be skipped.")

        self.A_size = len(self.A_pairs)

        # Domain B: H&E images (unaligned for CycleGAN)
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.B_size = len(self.B_paths)
        btoA = self.opt.direction == "BtoA"
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        # Domain A: load BF and AF, apply same spatial transform
        A_bf_path, A_af_path = self.A_pairs[index]

        img_bf = Image.open(A_bf_path).convert("L")
        img_af = Image.open(A_af_path).convert("L")

        # Generate identical spatial transform params for alignment
        transform_params = get_params(self.opt, img_bf.size)
        A_bf = get_transform(self.opt, transform_params, grayscale=True)(img_bf)
        A_af = get_transform(self.opt, transform_params, grayscale=True)(img_af)

        # Stack as 2-channel tensor
        A = torch.cat([A_bf, A_af], dim=0)  # (2, H, W)

        # Domain B: random index for unaligned training
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        img_b = Image.open(B_path).convert("RGB")
        B = self.transform_B(img_b)

        return {"A": A, "B": B, "A_paths": A_bf_path, "B_paths": B_path}

    def __len__(self):
        return max(self.A_size, self.B_size) if hasattr(self, 'B_size') else self.A_size
