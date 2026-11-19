import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from tool.code_virtual_staining.color_normalization import reinhard_normalize_masked_bgr


def images():
    rng = np.random.default_rng(9)
    base = rng.normal(0,1,(120,140,3)).astype(np.float32)
    reference_lab = base*[3,2,2]+[70,12,-14]
    target_lab = base*[2,1,1]+[86,2,-2]
    reference = np.rint(cv2.cvtColor(reference_lab.astype(np.float32),cv2.COLOR_LAB2BGR)*255).clip(0,255).astype(np.uint8)
    target = np.rint(cv2.cvtColor(target_lab.astype(np.float32),cv2.COLOR_LAB2BGR)*255).clip(0,255).astype(np.uint8)
    mask = np.zeros((120,140),np.uint8)
    mask[10:-10,10:-10] = 255
    return reference,target,mask


def test_mask_statistics_match_without_modifying_background_or_input():
    reference,target,mask = images()
    original = target.copy()
    output,alpha,report = reinhard_normalize_masked_bgr(reference,target,mask,mask,feather_width=0)
    assert report["mean_color_delta_e_after"] < .3
    assert report["mean_color_delta_e_after"] < .05*report["mean_color_delta_e_before"]
    np.testing.assert_allclose(report["target_after_lab_std"],report["reference_lab_std"],atol=.25)
    np.testing.assert_array_equal(output[mask==0],target[mask==0])
    np.testing.assert_array_equal(target,original)
    assert np.all(alpha[mask>0]==1)
    assert report["outside_mask_changed_pixels"]==0


def test_inside_feather_is_smooth_and_zero_at_mask_boundary():
    reference,target,mask = images()
    output,alpha,report = reinhard_normalize_masked_bgr(reference,target,mask,mask,feather_width=12)
    assert np.all(alpha[mask==0]==0)
    assert alpha[60,10]==0
    assert alpha[60,22]==1
    assert np.all(np.diff(alpha[60,10:23])>0)
    # Smoothstep has small increments at both ends of the transition.
    assert alpha[60,11] < .03
    assert alpha[60,22]-alpha[60,21] < .03
    np.testing.assert_array_equal(output[alpha==0],target[alpha==0])
    assert report["mean_color_delta_e_after"] < .25*report["mean_color_delta_e_before"]


def test_cropped_mask_touching_all_edges_has_correct_distances():
    reference,target,_ = images()
    mask = np.ones(target.shape[:2],np.uint8)
    output,alpha,_ = reinhard_normalize_masked_bgr(reference,target,mask,mask,feather_width=5)
    assert not np.any(alpha[[0,-1]])
    assert not np.any(alpha[:,[0,-1]])
    assert alpha[60,70]==1
    np.testing.assert_array_equal(output[alpha==0],target[alpha==0])


def test_different_reference_size_constant_channels_and_zero_strength():
    reference=np.full((40,50,3),[185,160,180],np.uint8)
    target=np.full((60,90,3),[220,218,221],np.uint8)
    refmask=np.ones(reference.shape[:2],bool)
    mask=np.ones(target.shape[:2],bool)
    output,_,report=reinhard_normalize_masked_bgr(reference,target,refmask,mask,feather_width=0)
    np.testing.assert_allclose(output[30,45],reference[10,10],atol=1)
    assert np.isfinite(report['channel_gain']).all()
    output,alpha,_=reinhard_normalize_masked_bgr(reference,target,refmask,mask,strength=0)
    np.testing.assert_array_equal(output,target)
    assert not np.any(alpha)


def test_invalid_masks_and_parameters_are_rejected():
    reference,target,mask=images()
    with pytest.raises(ValueError,match="为空"):
        reinhard_normalize_masked_bgr(reference,target,mask,np.zeros_like(mask))
    with pytest.raises(ValueError,match="尺寸"):
        reinhard_normalize_masked_bgr(reference,target,mask,mask[:10])
    with pytest.raises(ValueError):
        reinhard_normalize_masked_bgr(reference,target,mask,mask,feather_width=float('nan'))
