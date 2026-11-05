"""Small deterministic algorithm fixtures; no demo dataset or GUI is run."""
from unittest.mock import patch

import numpy as np
import pytest
import tifffile

from tool.code_3D_analysis import func_cell_tracking_v2 as tracking


def frame(*objects, shape=(96, 96)):
    image = np.zeros(shape, dtype=np.int64)
    for label, x, y, width, height in objects:
        image[y:y + height, x:x + width] = label
    return image


def square(label=1, x=20, y=20, width=20):
    return frame((label, x, y, width, width))


def test_birth_has_one_observation_and_no_missing_count():
    confirmed, candidates, events = tracking.track_glomeruli([square()])
    assert not confirmed and len(candidates) == 1
    track = candidates[0]
    assert [(h[0], h[4]) for h in track.history] == [(0, 'matched')]
    assert track.missing_count == 0
    assert not any(e['type'] in ('predicted', 'interpolated') for e in events)


def test_label_changes_do_not_change_identity_or_inputs():
    images = [square(label) for label in (7, 99, 2, 400)]
    copies = [a.copy() for a in images]
    confirmed, candidates, _ = tracking.track_glomeruli(iter(images))
    assert len(confirmed) == 1 and not candidates
    track = confirmed[0]
    assert track.matched_labels == {0: 7, 1: 99, 2: 2, 3: 400}
    assert track.match_count == 4
    assert len(track.history) == 4
    for actual, original in zip(images, copies):
        np.testing.assert_array_equal(actual, original)


def test_bounded_gap_uses_both_observations():
    images = [square()] * 3 + [None, None, square(9, x=23)]
    confirmed, _, events = tracking.track_glomeruli(images)
    assert len(confirmed) == 1
    history = confirmed[0].history
    assert [h[0] for h in history] == list(range(6))
    assert [h[4] for h in history] == ['matched'] * 3 + ['interpolated'] * 2 + ['matched']
    assert history[3][1] == pytest.approx(30.5)
    assert history[4][1] == pytest.approx(31.5)
    assert 3 not in confirmed[0].matched_labels
    assert 4 not in tracking.build_detection_to_track(confirmed)
    assert len([e for e in events if e['type'] == 'interpolated']) == 2


@pytest.mark.parametrize('missing', [1, 2, 3])
def test_gap_at_or_below_limit_preserves_identity(missing):
    confirmed, _, _ = tracking.track_glomeruli([square()] * 3 + [None] * missing + [square(9)])
    assert len(confirmed) == 1
    assert confirmed[0].matched_labels[3 + missing] == 9


def test_gap_above_limit_creates_new_identity_without_false_tail():
    confirmed, _, _ = tracking.track_glomeruli([square()] * 3 + [None] * 4 + [square(9)] * 3)
    assert len(confirmed) == 2
    assert [h[0] for h in confirmed[0].history] == [0, 1, 2]
    assert [h[0] for h in confirmed[1].history] == [7, 8, 9]


def test_terminal_predictions_do_not_enter_history():
    confirmed, _, events = tracking.track_glomeruli([square()] * 3 + [None] * 2)
    assert [h[0] for h in confirmed[0].history] == [0, 1, 2]
    assert not confirmed[0].active
    assert not any(e['type'] == 'interpolated' for e in events)
    assert events[-1]['discarded_predictions'] == 2


def test_isolated_hits_cannot_accumulate_confirmation():
    confirmed, candidates, _ = tracking.track_glomeruli([square(), None, square(), None, square()])
    assert not confirmed
    assert len(candidates) == 1
    assert candidates[0].match_count == 3
    assert candidates[0].hit_streak == 1


def test_candidate_can_recover_then_confirm_consecutive_hits():
    confirmed, candidates, _ = tracking.track_glomeruli([square(), None] + [square()] * 3)
    assert len(confirmed) == 1 and not candidates
    assert confirmed[0].hit_streak == 3


def test_confirm_threshold_one_applies_at_birth():
    confirmed, candidates, _ = tracking.track_glomeruli([square()], confirm_thresh=1)
    assert len(confirmed) == 1 and not candidates


def test_two_nearby_objects_keep_identity_when_labels_swap():
    images = []
    for z in range(6):
        labels = (1, 2) if z % 2 == 0 else (2, 1)
        images.append(frame((labels[0], 15 + z, 20, 12, 20),
                            (labels[1], 35 + z, 20, 12, 20)))
    confirmed, _, _ = tracking.track_glomeruli(images)
    assert len(confirmed) == 2
    left = min(confirmed, key=lambda t: t.history[0][1])
    assert [left.matched_labels[z] for z in range(6)] == [1, 2, 1, 2, 1, 2]
    assert all(h[4] == 'matched' for t in confirmed for h in t.history)


def test_pre_assignment_gate_preserves_valid_alternative():
    # The old cost preferred label 10 (same area, but outside position gate)
    # and rejected it AFTER assignment, never considering feasible label 11.
    observations = [
        [{'label': 1, 'cx': 30., 'cy': 30., 'area': 1000.}],
        [{'label': 10, 'cx': 39., 'cy': 30., 'area': 1000.},
         {'label': 11, 'cx': 30., 'cy': 30., 'area': 600.}],
    ]
    with patch.object(tracking, 'extract_features', side_effect=observations):
        confirmed, candidates, _ = tracking.track_glomeruli([square()] * 2, confirm_thresh=1)
    original = next(t for t in confirmed if t.id == 0)
    assert original.matched_labels == {0: 1, 1: 11}
    assert all(t.matched_labels.get(1) != 11 for t in confirmed if t.id != 0)
    assert not candidates


def test_assignment_can_leave_infeasible_rows_unmatched():
    cost = np.array([[np.inf, 2.], [np.inf, np.inf], [.5, 4.]])
    assert sorted(tracking._assign(cost, 10.)) == [(0, 1), (2, 0)]


def test_merge_is_withheld_and_original_identities_recover():
    normal = frame((1, 15, 20, 20, 20), (2, 45, 20, 20, 20))
    merged = frame((90, 15, 20, 50, 20))
    confirmed, candidates, events = tracking.track_glomeruli([normal] * 3 + [merged] * 2 + [normal])
    assert len(confirmed) == 2 and not candidates
    assert sum(e['type'] == 'merge_conflict' for e in events) == 2
    for track in confirmed:
        assert track.matched_labels[2] == track.matched_labels[5]
        assert 3 not in track.matched_labels and 4 not in track.matched_labels
        assert [h[4] for h in track.history[3:5]] == ['interpolated'] * 2


def test_short_split_does_not_replace_confirmed_identity():
    normal = square(1, width=30)
    split = frame((10, 20, 20, 30, 10), (11, 20, 30, 30, 10), (12, 20, 40, 30, 10))
    confirmed, candidates, events = tracking.track_glomeruli([normal] * 3 + [split, normal])
    assert len(confirmed) == 1
    assert confirmed[0].matched_labels == {0: 1, 1: 1, 2: 1, 4: 1}
    assert confirmed[0].history[3][4] == 'interpolated'
    assert len(candidates) == 3
    assert any(e['type'] == 'match_rejected' and e['reason'] == 'split' for e in events)


def test_exact_mapping_excludes_nearby_unconfirmed_detection():
    normal = square()
    nearby_fp = frame((1, 20, 20, 20, 20), (77, 44, 20, 5, 5))
    confirmed, candidates, _ = tracking.track_glomeruli([normal] * 3 + [nearby_fp, normal])
    mapping = tracking.build_detection_to_track(confirmed)
    assert set(mapping[3]) == {1}
    assert len(candidates) == 1


def test_mask_overlap_is_translation_aware_without_wrapping():
    previous = tracking.extract_features(square(), include_masks=True)[0]
    moved = tracking.extract_features(square(x=23), include_masks=True)[0]
    track = tracking.KalmanTrack(0, 0, previous['cx'], previous['cy'], previous['area'], feature=previous)
    track.X[0] += 3
    assert tracking._overlap(track, moved) == pytest.approx((1., 1.))
    track.X[0] += 200
    assert tracking._overlap(track, moved) == (0., 0.)


def test_sparse_large_ids_and_foreground_only_image():
    image = np.full((5, 5), 2**40, dtype=np.int64)
    assert tracking.extract_features(image)[0]['label'] == 2**40
    sparse = np.pad(image, 2)
    assert tracking.extract_features(sparse)[0]['area'] == 25


@pytest.mark.parametrize('bad', [
    np.zeros((8, 8, 3), dtype=np.uint8),
    np.zeros((8, 8), dtype=float),
    np.full((8, 8), -1, dtype=int),
    np.zeros((0, 8), dtype=int),
])
def test_invalid_label_images_are_rejected(bad):
    with pytest.raises(ValueError):
        tracking.track_glomeruli([bad])


def test_inconsistent_shapes_are_rejected():
    with pytest.raises(ValueError, match='shape'):
        tracking.track_glomeruli([square(), np.zeros((100, 100), dtype=int)])


def test_explicit_slice_indices_count_omitted_sections():
    confirmed, _, events = tracking.track_glomeruli(
        [square()] * 4, slice_indices=[10, 11, 12, 15])
    assert len(confirmed) == 1
    assert [h[0] for h in confirmed[0].history] == list(range(10, 16))
    assert [e['z'] for e in events if e['type'] == 'missing_slice'] == [13, 14]


@pytest.mark.parametrize('indices', [[0], [0, 1, 2], [1, 1], [2, 1], [0, 1.5], [-1, 1]])
def test_invalid_slice_indices_are_rejected(indices):
    with pytest.raises(ValueError):
        tracking.track_glomeruli([square(), square()], slice_indices=indices)


@pytest.mark.parametrize('options', [
    {'max_missing': -1}, {'confirm_thresh': 0}, {'candidate_max_missing': 1.5},
    {'pos_gate': 0}, {'cost_gate': np.nan}, {'split_area_ratio': 1.},
    {'merge_area_ratio': 1.}, {'iou_weight': -1}, {'read_error': 'ignore'},
])
def test_invalid_options_are_rejected(options):
    with pytest.raises(ValueError):
        tracking.track_glomeruli([], **options)


def test_file_read_failure_is_explicit_or_counts_as_missing(tmp_path):
    absent = tmp_path / 'missing.tif'
    with pytest.raises(OSError, match='slice 0'):
        tracking.track_glomeruli([absent])
    confirmed, _, events = tracking.track_glomeruli(
        [square()] * 3 + [absent, square()], read_error='missing')
    assert len(confirmed) == 1
    assert confirmed[0].history[3][4] == 'interpolated'
    assert any(e.get('reason') == 'read_error' for e in events)


def test_tiff_paths_preserve_uint16_labels(tmp_path):
    paths = []
    for z, label in enumerate([256, 400, 600]):
        path = tmp_path / f'{z}.tif'
        tifffile.imwrite(path, square(label).astype(np.uint16))
        paths.append(path)
    confirmed, candidates, _ = tracking.track_glomeruli(paths)
    assert len(confirmed) == 1 and not candidates
    assert confirmed[0].matched_labels == {0: 256, 1: 400, 2: 600}


def test_empty_sequence_and_background_only():
    assert tracking.track_glomeruli([]) == ([], [], [])
    assert tracking.track_glomeruli([np.zeros((5, 5), dtype=int)]) == ([], [], [])


def test_covariance_remains_finite_symmetric_and_positive():
    track = tracking.KalmanTrack(0, 0, 20., 20., 1000.)
    for z in range(1, 60):
        track.predict()
        track.update(z, 20. + .5 * z, 20., 1000. + 100. * np.sin(z / 5.))
        np.testing.assert_allclose(track.P, track.P.T, atol=1e-10)
        assert np.isfinite(track.P).all() and np.isfinite(track.X).all()
        assert np.linalg.eigvalsh(track.P).min() >= -1e-8
        assert track.X[2] > 0.
