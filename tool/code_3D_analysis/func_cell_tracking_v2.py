"""Associate instances in registered, equally spaced serial sections.

Accept integer label arrays or paths in increasing Z order. Return the original
three objects and history tuple format; only bounded gaps enter the histories.
"""
from itertools import zip_longest
from numbers import Integral
from os import PathLike

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops


class KalmanTrack:
    """Constant-rate state [cx, cy, area, vx, vy, darea], in pixel units."""

    def __init__(self, track_id, z, cx, cy, area, *, feature=None,
                 position_noise=2.0, area_noise_ratio=0.15):
        self.id = track_id
        self.last_seen = z
        self.X = np.array([cx, cy, area, 0., 0., 0.], dtype=float)
        self.position_noise = position_noise
        self.area_noise_ratio = area_noise_ratio
        self.H = np.eye(3, 6)
        self.F = np.eye(6)
        self.F[:3, 3:] = np.eye(3)
        self.R = self.measurement_noise(area)
        self.P = np.diag(np.r_[np.diag(self.R), np.diag(self.R)])
        self.Q = np.zeros((6, 6))
        self.history = [(z, float(cx), float(cy), float(area), 'matched')]
        self.matched_labels = {}  # z -> original per-slice instance label
        self.last_feature = feature
        if feature is not None:
            self.matched_labels[z] = feature['label']
        self._pending = []
        self.missing_count = 0
        self.match_count = 1
        self.hit_streak = 1
        self.active = True
        self.is_candidate = True

    def measurement_noise(self, area):
        radius = np.sqrt(max(area, 1.) / np.pi)
        pos_sigma = max(self.position_noise, 0.1 * radius)
        area_sigma = max(1., self.area_noise_ratio * max(area, 1.))
        return np.diag([pos_sigma**2, pos_sigma**2, area_sigma**2])

    def predict(self):
        # One equally spaced Z step; random acceleration scales with size.
        sigmas = np.sqrt(np.diag(self.measurement_noise(self.X[2]))) * 0.5
        self.Q.fill(0.)
        for axis, sigma in enumerate(sigmas):
            indices = [axis, axis + 3]
            self.Q[np.ix_(indices, indices)] = sigma**2 * np.array([[.25, .5], [.5, 1.]])
        self.X = self.F @ self.X
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.X[2] = max(self.X[2], 1.)
        return tuple(self.X[:3])

    def update(self, z, cx, cy, area, *, feature=None):
        """Update from an observation and return newly bounded gap indices."""
        observation = np.array([cx, cy, area], dtype=float)
        self.R = self.measurement_noise(area)
        innovation = observation - self.H @ self.X
        covariance = self.H @ self.P @ self.H.T + self.R
        gain = np.linalg.solve(covariance, self.H @ self.P).T
        self.X += gain @ innovation
        residual = np.eye(6) - gain @ self.H
        # Joseph form and explicit symmetrization for numerical stability.
        self.P = residual @ self.P @ residual.T + gain @ self.R @ gain.T
        self.P = (self.P + self.P.T) * .5
        self.X[2] = max(self.X[2], 1.)

        left = self.history[-1]
        filled = []
        for gap_z in self._pending:
            weight = (gap_z - left[0]) / (z - left[0])
            values = (1. - weight) * np.array(left[1:4]) + weight * observation
            self.history.append((gap_z, *map(float, values), 'interpolated'))
            filled.append(gap_z)
        self._pending.clear()
        self.history.append((z, float(cx), float(cy), float(area), 'matched'))
        self.last_seen = z
        self.missing_count = 0
        self.match_count += 1
        self.hit_streak += 1
        if feature is not None:
            self.matched_labels[z] = feature['label']
            self.last_feature = feature
        return filled

    def interpolate(self, z):
        """Keep a provisional gap, without adding a final history point."""
        self._pending.append(z)
        self.missing_count += 1
        self.hit_streak = 0

    def finish(self):
        self.active = False
        self._pending.clear()
        self.last_feature = None  # release the last cropped mask


def extract_features(label_image, min_area=15.0, *, include_masks=False):
    """One region per positive label; disconnected equal labels stay grouped.

    Internally compact sparse IDs before regionprops. Optional cropped masks
    keep legacy feature-only callers lightweight and preserve original IDs.
    """
    img = np.asarray(label_image)
    if img.ndim != 2 or not np.issubdtype(img.dtype, np.integer):
        raise ValueError('Expected a 2D integer instance-label image.')
    if img.size == 0 or np.any(img < 0):
        raise ValueError('Label image must be nonempty and nonnegative.')
    if not np.isfinite(min_area) or min_area <= 0:
        raise ValueError('min_area must be positive and finite.')
    values, inverse = np.unique(img, return_inverse=True)
    offset = int(values[0] != 0)  # reserve background even if none is present
    compact = inverse.reshape(img.shape).astype(np.int32) + offset
    feats = []
    for region in regionprops(compact):
        if region.area < min_area:
            continue
        feature = {
            'label': int(values[region.label - offset]),
            'cx': float(region.centroid[1]), 'cy': float(region.centroid[0]),
            'area': float(region.area), 'bbox': tuple(region.bbox),
        }
        if include_masks:
            feature['mask'] = region.image.copy()
        feats.append(feature)
    return feats


def _overlap(track, detection):
    """IoU and intersection/old area for the translated last observed mask."""
    previous = track.last_feature
    if previous is None or 'mask' not in previous or 'mask' not in detection:
        return 0., 0.
    dy, dx = np.rint(track.X[1::-1] - [previous['cy'], previous['cx']]).astype(int)
    py0, px0, py1, px1 = previous['bbox']
    py0, py1, px0, px1 = py0 + dy, py1 + dy, px0 + dx, px1 + dx
    dy0, dx0, dy1, dx1 = detection['bbox']
    y0, x0 = max(py0, dy0), max(px0, dx0)
    y1, x1 = min(py1, dy1), min(px1, dx1)
    if y1 <= y0 or x1 <= x0:
        return 0., 0.
    old_crop = previous['mask'][y0-py0:y1-py0, x0-px0:x1-px0]
    new_crop = detection['mask'][y0-dy0:y1-dy0, x0-dx0:x1-dx0]
    intersection = float(np.count_nonzero(old_crop & new_crop))
    union = previous['area'] + detection['area'] - intersection
    return intersection / max(union, 1.), intersection / max(previous['area'], 1.)


def _assign(cost, unmatched_cost):
    """Give each track its own dummy column, allowing it to stay unmatched."""
    n_tracks, n_detections = cost.shape
    if not n_tracks or not n_detections:
        return []
    augmented = np.full((n_tracks, n_detections + n_tracks), np.inf)
    augmented[:, :n_detections] = cost
    augmented[np.arange(n_tracks), n_detections + np.arange(n_tracks)] = unmatched_cost
    rows, columns = linear_sum_assignment(augmented)
    return [(int(i), int(j)) for i, j in zip(rows, columns)
            if j < n_detections and np.isfinite(cost[i, j])]


def _load_label(source):
    if isinstance(source, np.ndarray):
        return source
    if not isinstance(source, (str, PathLike)):
        raise TypeError('Each input must be a label array, path, or None for a missing slice.')
    # Lazy IO: array tracking has no OpenCV or image decoder dependency.
    from skimage.io import imread
    return imread(source)


def build_detection_to_track(tracks):
    """Return {z: {original_label: track}} from accepted observations only."""
    mapping = {}
    for track in tracks:
        for z, label in track.matched_labels.items():
            by_label = mapping.setdefault(z, {})
            if label in by_label:
                raise ValueError(f'Duplicate assignment for slice {z}, label {label}.')
            by_label[label] = track
    return mapping


def track_glomeruli(label_image_paths, max_missing=3, confirm_thresh=3, cost_gate=20.0,
                    pos_gate=0.45, split_area_ratio=0.5, split_min_area=400.0,
                    min_interp_area=25.0, *, merge_area_ratio=2.0,
                    mahalanobis_gate=9.21, iou_weight=1.0,
                    candidate_max_missing=1, min_area=15.0,
                    position_noise=2.0, area_noise_ratio=0.15,
                    slice_indices=None, read_error='raise'):
    """Associate serial-section instances without modifying the input masks.

    Accept paths, arrays, or streams of either; None means an unavailable slice.
    Optional slice_indices are strictly increasing nonnegative integer indices;
    skipped indices count as missing sections. XY units are pixels and pixels
    squared. Z is an equally spaced slice index, not a physical coordinate.

    Gate BEFORE assignment; confirmed tracks have priority over candidates.
    Confirmation requires consecutive observed hits, including birth, while
    match_count records all hits. Candidate gaps use candidate_max_missing,
    capped by max_missing. Multi-track merge observations cannot match or spawn.

    Histories contain raw matched measurements and linear interpolation ONLY
    inside reconnected gaps. Terminal predictions are discarded. matched_labels
    records the exact detection correspondence; no mask interpolation is done.

    Returns (confirmed_tracks, unconfirmed_tracks, events). Unconfirmed does NOT
    establish false positivity. 'split'/'merge' are geometric hypotheses.
    """
    for name, value, minimum in (
            ('max_missing', max_missing, 0), ('confirm_thresh', confirm_thresh, 1),
            ('candidate_max_missing', candidate_max_missing, 0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}.')
    for name, value in (
            ('cost_gate', cost_gate), ('pos_gate', pos_gate),
            ('split_min_area', split_min_area), ('min_area', min_area),
            ('mahalanobis_gate', mahalanobis_gate), ('position_noise', position_noise),
            ('area_noise_ratio', area_noise_ratio)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive and finite.')
    if not 0 < split_area_ratio < 1 or not np.isfinite(merge_area_ratio) or merge_area_ratio <= 1:
        raise ValueError('Require 0 < split_area_ratio < 1 < merge_area_ratio.')
    if not np.isfinite(iou_weight) or iou_weight < 0:
        raise ValueError('iou_weight must be finite and nonnegative.')
    if not np.isfinite(min_interp_area) or min_interp_area < 0:
        raise ValueError('min_interp_area must be finite and nonnegative.')
    if read_error not in ('raise', 'missing'):
        raise ValueError("read_error must be 'raise' or 'missing'.")

    active, all_tracks, events = [], [], []
    image_shape = None

    def finish(track, z, reason):
        events.append({'type': 'terminated', 'z': z, 'track_id': track.id,
                       'reason': reason, 'discarded_predictions': len(track._pending)})
        track.finish()

    def process(z, detections):
        nonlocal active
        for track in active:
            track.predict()

        # One region covering two established objects is not a one-to-one
        # observation for either. Coast without assigning an invented split.
        blocked = set()
        overlaps = {(i, j): _overlap(track, det)
                    for i, track in enumerate(active) for j, det in enumerate(detections)}
        for j, det in enumerate(detections):
            supported = [track for i, track in enumerate(active)
                         if not track.is_candidate and overlaps[i, j][1] >= .5]
            if len(supported) >= 2 and det['area'] >= 1.3 * max(t.X[2] for t in supported):
                blocked.add(j)
                events.append({'type': 'merge_conflict', 'z': z, 'label': det['label'],
                               'track_ids': [t.id for t in supported]})

        cost = np.full((len(active), len(detections)), np.inf)
        rejected = {}
        for i, track in enumerate(active):
            px, py, pa = track.X[:3]
            radius = np.sqrt(max(pa, 1.) / np.pi)
            covariance = track.P[:2, :2] + track.measurement_noise(pa)[:2, :2]
            position_limit = pos_gate * min(2., 1. + .25 * track.missing_count)
            for j, det in enumerate(detections):
                if j in blocked:
                    continue
                delta = np.array([det['cx'] - px, det['cy'] - py])
                distance = float(np.linalg.norm(delta) / radius)
                ratio = det['area'] / max(pa, 1.)
                mahalanobis = float(delta @ np.linalg.solve(covariance, delta))
                reason = None
                if distance > position_limit:
                    reason = 'position'
                elif mahalanobis > mahalanobis_gate:
                    reason = 'mahalanobis'
                elif pa > split_min_area and ratio < split_area_ratio:
                    reason = 'split'
                elif pa > split_min_area and ratio > merge_area_ratio:
                    reason = 'merge'
                iou = overlaps[i, j][0]
                value = (2. * distance + 3. * abs(np.log(ratio))
                         + .5 * np.sqrt(max(mahalanobis, 0.) / mahalanobis_gate)
                         + iou_weight * (1. - iou))
                if reason is None and value > cost_gate:
                    reason = 'cost'
                if reason is None:
                    cost[i, j] = value
                elif distance <= max(1., position_limit):
                    # Log one nearby explanation, not every distant pair.
                    if i not in rejected or distance < rejected[i][0]:
                        rejected[i] = (distance, det['label'], reason)

        matched_t, matched_d = set(), set()
        for candidate_pass in (False, True):
            rows = [i for i, t in enumerate(active) if t.is_candidate == candidate_pass]
            columns = [j for j in range(len(detections)) if j not in matched_d and j not in blocked]
            local_cost = cost[np.ix_(rows, columns)]
            for ri, cj in _assign(local_cost, cost_gate + 1.e-6):
                i, j = rows[ri], columns[cj]
                track, det = active[i], detections[j]
                left_z = track.last_seen
                filled = track.update(z, det['cx'], det['cy'], det['area'], feature=det)
                matched_t.add(track.id)
                matched_d.add(j)
                events.append({'type': 'matched', 'z': z, 'track_id': track.id,
                               'label': det['label'], 'cost': float(cost[i, j])})
                for gap_z in filled:
                    events.append({'type': 'interpolated', 'z': gap_z, 'track_id': track.id,
                                   'left_z': left_z, 'right_z': z, 'method': 'linear'})

        # Only pre-existing tracks can miss this slice. Add births afterwards.
        for i, track in enumerate(active):
            if track.id in matched_t:
                continue
            if i in rejected:
                _, label, reason = rejected[i]
                events.append({'type': 'match_rejected', 'z': z, 'track_id': track.id,
                               'label': label, 'reason': reason})
            limit = min(max_missing, candidate_max_missing) if track.is_candidate else max_missing
            if track.missing_count >= limit:
                finish(track, z, 'max_missing')
            elif track.X[2] <= min_interp_area:
                finish(track, z, 'pole_end')
            else:
                track.interpolate(z)
                events.append({'type': 'predicted', 'z': z, 'track_id': track.id})

        for j, det in enumerate(detections):
            if j in matched_d or j in blocked:
                continue
            track = KalmanTrack(len(all_tracks), z, det['cx'], det['cy'], det['area'],
                                feature=det, position_noise=position_noise,
                                area_noise_ratio=area_noise_ratio)
            active.append(track)
            all_tracks.append(track)
            events.append({'type': 'new_candidate', 'z': z, 'track_id': track.id,
                           'label': det['label']})

        for track in active:
            if track.active and track.is_candidate and track.hit_streak >= confirm_thresh:
                track.is_candidate = False
                events.append({'type': 'confirmed', 'z': z, 'track_id': track.id})
        active = [track for track in active if track.active]

    sentinel = object()
    frames = (enumerate(label_image_paths) if slice_indices is None else
              zip_longest(slice_indices, label_image_paths, fillvalue=sentinel))
    previous_z = None
    for z, source in frames:
        if z is sentinel or source is sentinel:
            raise ValueError('slice_indices and inputs must have the same length.')
        if isinstance(z, bool) or not isinstance(z, Integral) or z < 0 or (
                previous_z is not None and z <= previous_z):
            raise ValueError('slice_indices must be strictly increasing nonnegative integers.')
        z = int(z)
        if previous_z is not None:
            for missing_z in range(previous_z + 1, z):
                events.append({'type': 'missing_slice', 'z': missing_z, 'reason': 'index_gap'})
                process(missing_z, [])
        if source is None:
            image = None
            events.append({'type': 'missing_slice', 'z': z, 'reason': 'explicit'})
        else:
            try:
                image = _load_label(source)
            except (OSError, ValueError) as exc:
                if read_error == 'raise':
                    raise OSError(f'Cannot read label image at slice {z}: {source}') from exc
                image = None
                events.append({'type': 'missing_slice', 'z': z, 'reason': 'read_error',
                               'path': str(source)})
        if image is None:
            detections = []
        else:
            detections = extract_features(image, min_area, include_masks=True)
            if image_shape is None:
                image_shape = image.shape
            elif image.shape != image_shape:
                raise ValueError(f'Inconsistent image shape at slice {z}: {image.shape} != {image_shape}.')
        process(z, detections)
        previous_z = z

    for track in active:
        finish(track, previous_z, 'end_of_input')
    confirmed = [track for track in all_tracks if not track.is_candidate]
    unconfirmed = [track for track in all_tracks if track.is_candidate]
    return confirmed, unconfirmed, events
