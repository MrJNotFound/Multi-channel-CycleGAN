import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")
from tool.code_virtual_staining.reference_cartilage_color import reference_profile_normalize_bgr


def fixture(h,w,left,right,reference):
    yy,xx=np.mgrid[:h,:w]
    depth=np.clip((xx-left)/(right-left),0,1)
    mask=(xx>=left)&(xx<=right)&(yy>=20)&(yy<h-20)
    lab=np.zeros((h,w,3),np.float32)
    lab[:]=[89,0,0]
    tissue=np.stack([80-10*depth,10+8*depth,-4-12*depth],axis=-1) if reference else np.full((h,w,3),[86,1,-1],np.float32)
    tissue[...,0]+=np.sin(yy*.16)*.7
    lab[mask]=tissue[mask]
    bone=(xx>=right+9)&(xx<w-8)&(yy>=20)&(yy<h-20)
    lab[bone]=[45,30,-28]
    rgb=np.rint(cv2.cvtColor(lab.astype(np.float32),cv2.COLOR_LAB2BGR)*255).clip(0,255).astype(np.uint8)
    return rgb,mask,bone


def test_matches_pink_outer_and_purple_inner_without_white_band_or_bone_changes():
    reference,rm,_=fixture(280,300,60,210,True)
    target,tm,bone=fixture(240,240,60,155,False)
    original=target.copy()
    output,alpha,report=reference_profile_normalize_bgr(reference,target,rm,tm)
    lab=cv2.cvtColor(output.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
    outer=lab[40:200,70:80].mean(axis=(0,1))
    inner=lab[40:200,140:150].mean(axis=(0,1))
    assert outer[1]>9 and outer[2]>-8  # Pink surface, not a pale/white rim.
    assert inner[1]>15 and inner[2]<-12  # Stronger purple by adjacent bone.
    assert report['mean_profile_delta_e_after'] < .2*report['mean_profile_delta_e_before']
    assert np.isfinite(alpha).all() and alpha.min()>=0 and alpha.max()<=1
    np.testing.assert_array_equal(target,original)
    np.testing.assert_array_equal(output[bone],target[bone])
    assert report['neighboring_bone_changed_pixels']==0
    assert report['distant_background_changed_pixels']==0
    np.testing.assert_array_equal(output[:10],target[:10])


def test_narrow_interface_closes_annotation_gap_without_wide_feather():
    reference,rm,_=fixture(280,300,60,210,True)
    target,tm,bone=fixture(240,240,60,155,False)
    output,alpha,report=reference_profile_normalize_bgr(reference,target,rm,tm)
    assert report['inner_edge_shift_quantiles_px'][1]==8
    # Full stain reaches the labeled cartilage edge instead of fading to white there.
    assert alpha[100,155] == 1
    assert alpha[100,140] == 1
    assert alpha[100,164] == 0  # Deeply stained bone is protected.
    assert report['maximum_allowed_outside_band_px'] <= 37
    assert np.all(output[:,210:]==target[:,210:])


def test_auto_orientation_is_mirror_consistent():
    reference,rm,_=fixture(280,300,60,210,True)
    target,tm,_=fixture(240,240,60,155,False)
    first,alpha,report=reference_profile_normalize_bgr(reference,target,rm,tm)
    mirrored,ma,mr=reference_profile_normalize_bgr(reference[:,::-1],target[:,::-1],rm[:,::-1],tm[:,::-1])
    np.testing.assert_array_equal(mirrored[:,::-1],first)
    np.testing.assert_array_equal(ma[:,::-1],alpha)
    assert report['target_outer_side']=='left' and mr['target_outer_side']=='right'


def test_unit_gain_preserves_longitudinal_texture_contrast():
    reference,rm,_=fixture(280,300,60,210,True)
    target,tm,_=fixture(240,240,60,155,False)
    output,_,report=reference_profile_normalize_bgr(reference,target,rm,tm)
    a=cv2.cvtColor(target.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
    b=cv2.cvtColor(output.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
    np.testing.assert_allclose(b[50:190,105]-b[70,105],a[50:190,105]-a[70,105],atol=.6)
    assert report['contrast_gain']==1


def test_zero_strength_and_invalid_input():
    reference,rm,_=fixture(280,300,60,210,True)
    target,tm,_=fixture(240,240,60,155,False)
    output,alpha,_=reference_profile_normalize_bgr(reference,target,rm,tm,strength=0)
    np.testing.assert_array_equal(output,target)
    assert not alpha.any()
    with pytest.raises(ValueError):
        reference_profile_normalize_bgr(reference,target,rm,np.zeros_like(tm))
    with pytest.raises(ValueError):
        reference_profile_normalize_bgr(reference,target,rm,tm,max_edge_shift=100)
