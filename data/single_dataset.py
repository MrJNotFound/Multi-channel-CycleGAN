import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch


class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    Supports dual-channel input when --input_nc 2 by loading from {dataroot}_BF and {dataroot}_AF directories.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        input_nc = self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc

        if input_nc == 2:
            # Dual-channel mode: load BF + AF from separate dirs, matched by filename
            self.dir_bf = opt.dataroot + "_BF"
            self.dir_af = opt.dataroot + "_AF"
            bf_paths = sorted(make_dataset(self.dir_bf, opt.max_dataset_size))
            af_paths = sorted(make_dataset(self.dir_af, opt.max_dataset_size))
            af_dict = {os.path.basename(p): p for p in af_paths}
            self.bf_af_pairs = []
            for bf_p in bf_paths:
                fname = os.path.basename(bf_p)
                if fname in af_dict:
                    self.bf_af_pairs.append((bf_p, af_dict[fname]))
            assert len(self.bf_af_pairs) > 0, (
                f"No matching BF/AF filename pairs found."
            )
            self.A_paths = [p[0] for p in self.bf_af_pairs]  # for __len__
            self.transform = get_transform(self.opt, grayscale=True)
            self.dual_channel = True
        elif input_nc == 1:
            self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
            self.transform = get_transform(opt, grayscale=True)
            self.dual_channel = False
        else:
            self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
            self.transform = get_transform(opt, grayscale=False)
            self.dual_channel = False

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        if self.dual_channel:
            bf_path, af_path = self.bf_af_pairs[index]
            img_bf = Image.open(bf_path).convert("L")
            img_af = Image.open(af_path).convert("L")
            params = get_params(self.opt, img_bf.size)
            A_bf = get_transform(self.opt, params, grayscale=True)(img_bf)
            A_af = get_transform(self.opt, params, grayscale=True)(img_af)
            A = torch.cat([A_bf, A_af], dim=0)
            return {"A": A, "A_paths": bf_path}
        else:
            A_path = self.A_paths[index]
            if self.opt.input_nc == 1:
                A_img = Image.open(A_path).convert("L")
            else:
                A_img = Image.open(A_path).convert("RGB")
            A = self.transform(A_img)
            return {"A": A, "A_paths": A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
