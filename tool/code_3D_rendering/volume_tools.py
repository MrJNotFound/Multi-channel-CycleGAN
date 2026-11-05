import numpy as np


def remove_white_background(img: np.ndarray, white_threshold: float) -> np.ndarray:
    if img.ndim == 2:
        mask = img >= white_threshold
    else:
        mask = np.all(img >= white_threshold, axis=-1)
    out = img.copy()
    out[mask] = 0
    return out


def clip_volume_to_planes(volume: np.ndarray, z: int, y: int, x: int) -> np.ndarray:
    if volume.ndim not in (3, 4):
        raise ValueError("volume must be 3D or 4D (z,y,x[,c])")
    clipped = np.zeros_like(volume)
    if volume.ndim == 3:
        clipped[: z + 1, : y + 1, : x + 1] = volume[: z + 1, : y + 1, : x + 1]
    else:
        clipped[: z + 1, : y + 1, : x + 1, :] = volume[: z + 1, : y + 1, : x + 1, :]
    return clipped


def extract_orth_slices(volume: np.ndarray, z: int, y: int, x: int):
    if volume.ndim == 3:
        xy = volume[z, :, :]
        xz = volume[:, y, :]
        yz = volume[:, :, x]
    elif volume.ndim == 4:
        xy = volume[z, :, :, :]
        xz = volume[:, y, :, :]
        yz = volume[:, :, x, :]
    else:
        raise ValueError("volume must be 3D or 4D (z,y,x[,c])")
    return xy, xz, yz


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    data = img.astype(np.float32, copy=False)
    minv = float(np.min(data))
    maxv = float(np.max(data))
    if maxv <= minv:
        return np.zeros_like(data, dtype=np.uint8)
    scaled = (data - minv) / (maxv - minv) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def upsample_along_z(arr_2d: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 1.0 or arr_2d.shape[0] < 2:
        return arr_2d
    nz = arr_2d.shape[0]
    new_nz = max(nz, int(np.round(nz * factor)))
    xp = np.arange(nz, dtype=np.float64)
    x = np.linspace(0.0, float(nz - 1), new_nz, dtype=np.float64)
    flat = arr_2d.reshape(nz, -1).astype(np.float32)
    out = np.empty((new_nz, flat.shape[1]), dtype=np.float32)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(x, xp, flat[:, c])
    return out.reshape((new_nz,) + arr_2d.shape[1:]).astype(arr_2d.dtype)


def clamp_index(idx: int, size: int) -> int:
    if size <= 0:
        return 0
    return int(np.clip(idx, 0, size - 1))

