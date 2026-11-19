import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from tool.code_virtual_staining import patch_stitching as stitching
from tool.code_virtual_staining import slide_windows as windows


@pytest.mark.parametrize("length,window,stride,expected", [
    (21,8,5,[0,5,10,13]),
    (18,7,5,[0,5,10,11]),
    (18,8,5,[0,5,10]),
    (13,8,8,[0,5]),
    (9,8,8,[0,1]),
    (8,8,5,[0]),
    (4,8,5,[0]),
])
def test_positions_anchor_last_window_inside_image(length, window, stride, expected):
    starts = windows._positions_1d(length,window,stride)
    assert starts == expected
    assert starts == sorted(set(starts))
    assert starts[-1]+min(window,length) == length
    covered = np.zeros(length,bool)
    for start in starts:
        covered[start:start+window] = True
    assert covered.all()


@pytest.mark.parametrize("shape,overlap", [
    ((18,21),(3,2)),  # Both axes end at an irregular position.
    ((9,13),(0,0)),   # Tail overlaps even if the requested overlap is zero.
    ((17,18),(3,2)),  # Regularly aligned: do not duplicate the last tile.
    ((5,13),(3,2)),   # Height smaller than a window.
    ((11,4),(3,2)),   # Width smaller than a window.
    ((7,8),(3,2)),    # Exactly one window.
    ((1,1),(3,2)),    # No space exists to move inward: use the actual size.
])
@pytest.mark.parametrize("weight_mode", ["hann","gaussian"])
def test_roundtrip_recovers_original_dimensions_and_pixels(tmp_path, shape, overlap, weight_mode):
    image = np.random.default_rng(123).integers(0,256,(*shape,3),dtype=np.uint8)
    image[0,0] = 255
    source_before = image.copy()
    patches,xs,ys,stride = windows.sliding_window_patches(image,8,7,*overlap)
    assert len(patches) == len(xs)*len(ys)
    expected_shape = (min(7,shape[0]),min(8,shape[1]),3)
    items=[]
    for x,y,patch in patches:
        assert patch.shape == expected_shape
        assert 0 <= x <= max(0,shape[1]-8)
        assert 0 <= y <= max(0,shape[0]-7)
        np.testing.assert_array_equal(patch,image[y:y+patch.shape[0],x:x+patch.shape[1]])
        path = tmp_path/f"patch_y{y:06d}_x{x:06d}.png"
        assert cv2.imwrite(str(path),patch)
        parsed = stitching._parse_xy_from_name(path.name)
        assert parsed == (x,y)
        items.append((*parsed,path))
    # Exercise the actual stitching function with automatic dimensions and the
    # stored filename coordinates, rather than calculating a canvas in the test.
    result,*_,width,height = stitching.stitch_group(items,None,None,weight_mode,False,(255,255,255),3)
    assert (width,height) == (shape[1],shape[0])
    np.testing.assert_array_equal(result,image)
    np.testing.assert_array_equal(image,source_before)
    assert stride == (8-overlap[0],7-overlap[1])


def test_grayscale_tiles_do_not_gain_padding():
    image = np.arange(18*21,dtype=np.uint8).reshape(18,21)
    patches,*_ = windows.sliding_window_patches(image,8,7,3,2)
    for x,y,patch in patches:
        assert patch.shape == (7,8,3)
        for channel in range(3):
            np.testing.assert_array_equal(patch[...,channel],image[y:y+7,x:x+8])


def test_new_split_directory_preserves_and_excludes_old_padded_tiles(tmp_path):
    old = tmp_path/"sample"
    old.mkdir()
    leftover = old/"patch_y000015_x000015.png"
    leftover.write_bytes(b"old padded tile")
    new = windows.create_patch_directory(tmp_path,"sample")
    assert new == tmp_path/"sample_2"
    assert list(new.iterdir()) == []
    assert leftover.read_bytes() == b"old padded tile"
    # An existing empty directory is still safe to use.
    assert windows.create_patch_directory(tmp_path,"sample") == new
    (new/"new.png").touch()
    assert windows.create_patch_directory(tmp_path,"sample") == tmp_path/"sample_3"


def test_irregular_last_step_does_not_replace_normal_stride_in_summary():
    assert stitching._most_common_positive_diff([0,8,9]) == 8


@pytest.mark.parametrize("length,window,stride", [(0,8,4),(10,0,4),(10,8,0),(10,8,9)])
def test_invalid_position_parameters(length, window, stride):
    with pytest.raises(ValueError):
        windows._positions_1d(length,window,stride)
