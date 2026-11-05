from gui.launcher import build_option_parser
from models.cut_model import CUTModel
from models.networks import Downsample, Upsample
import torch
import pytest


@pytest.mark.parametrize('gpu_ids',[[],[0]])
@pytest.mark.parametrize('antialias',[True,False])
def test_cut_sampling_does_not_depend_on_gpu_list(gpu_ids,antialias):
    parser,_=build_option_parser('train','cut_single_none_base','unaligned')
    opt=parser.parse_args(['--dataroot','unused','--input_nc','1',
                          '--nce_idt','False','--ngf','4','--ndf','4'])
    opt.isTrain=True; opt.device=torch.device('cpu'); opt.gpu_ids=gpu_ids
    opt.no_antialias=opt.no_antialias_up=not antialias
    model=CUTModel(opt)
    expected=2 if antialias else 0
    assert sum(isinstance(x,Downsample) for x in model.netG.modules()) == expected
    assert sum(isinstance(x,Upsample) for x in model.netG.modules()) == expected
