"""CUT-Stack — CUT contrastive loss with stacked dual-channel input.

Unlike single-channel CUT, this variant takes stacked BF+AF (2-channel)
directly into a standard ResNet generator — no dual-branch heads, no fusion.

Serves as the dual-channel ablation baseline for CUT: measures the value
of a second input modality without architectural changes.

Usage: --model cut_stack  (requires --input_nc 2, --dataset_mode dual_channel)
"""

from models.cut.standard import CUTModel


class CUTStackModel(CUTModel):
    """CUT with stacked 2-channel input — standard generator, no dual heads.

    G: standard ResnetGenerator (2 -> 3)   stacked BF+AF -> RGB H&E
    D: standard PatchGAN (3)               RGB H&E discriminator
    F: mlp_sample (2 input channels)       feature projection

    Keeps CUT's full PatchNCE contrastive loss.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = CUTModel.modify_commandline_options(parser, is_train)
        # Stack variant uses dual-channel dataset
        parser.set_defaults(dataset_mode="dual_channel")
        # Identity NCE needs input_nc == output_nc (2->3 here), so disable it
        parser.set_defaults(nce_idt=False)
        return parser
