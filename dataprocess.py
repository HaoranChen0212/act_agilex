import argparse
import math
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import h5py
import numpy as np


DEFAULT_FOV = 90.0
DEFAULT_RANGE_ARG = "yaw=-180:180:6,pitch=-80:10:1"
DEFAULT_PATCH_SIZE = 512
DEFAULT_JPEG_QUALITY = 95

CAMERA_NAMES = (
    "pano_yaw_m180",
    "pano_yaw_m120",
    "pano_yaw_m060",
    "pano_yaw_p000",
    "pano_yaw_p060",
    "pano_yaw_p120",
)

PRESERVED_CAMERA_NAMES = (
    "image_wide",
    "cam_left_wrist",
    "cam_right_wrist",
)

COPY_DATASET_PATHS = (
    "action",
    "base_action",
    "timestamp",
    "observations/qpos",
    "observations/qvel",
    "observations/basePos",
    "observations/base_orientation",
    "observations/effort",
)


@dataclass(frozen=True)
class AxisRangeSpec:
    start: float
    end: float
    count: int


@dataclass(frozen=True)
class RangeSpec:
    yaw: AxisRangeSpec
    pitch: AxisRangeSpec


def parse_axis_assignment(token, axis_name):
    token = token.strip()
    prefix = f"{axis_name}="
    if not token.startswith(prefix):
        raise ValueError(
            f"Expected '{axis_name}=start:end:count' inside --range, but got '{token}'."
        )

    parts = token[len(prefix):].split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Expected '{axis_name}=start:end:count', but got '{token}'."
        )

    start_text, end_text, count_text = parts
    try:
        start = float(start_text)
        end = float(end_text)
        count = int(count_text)
    except ValueError as exc:
        raise ValueError(
            f"Could not parse numeric values from '{token}'."
        ) from exc

    if count <= 0:
        raise ValueError(f"{axis_name} count must be positive, but got {count}.")

    return AxisRangeSpec(start=start, end=end, count=count)


def parse_range_spec(range_arg):
    tokens = [token.strip() for token in range_arg.split(",") if token.strip()]
    if len(tokens) != 2:
        raise ValueError(
            "--range must look like "
            "'yaw=-180:180:8,pitch=-60:60:5'."
        )

    spec_by_name = {}
    for axis_name in ("yaw", "pitch"):
        matching = [token for token in tokens if token.startswith(f"{axis_name}=")]
        if len(matching) != 1:
            raise ValueError(
                "--range must contain exactly one yaw spec and one pitch spec."
            )
        spec_by_name[axis_name] = parse_axis_assignment(matching[0], axis_name)

    return RangeSpec(yaw=spec_by_name["yaw"], pitch=spec_by_name["pitch"])


def _round_angle(value):
    rounded = round(float(value), 6)
    if abs(rounded) < 1e-6:
        return 0.0
    return rounded


def sample_axis(start, end, count, wrap_full_circle=False):
    if count <= 0:
        raise ValueError(f"count must be positive, but got {count}.")

    if count == 1:
        return [_round_angle((start + end) * 0.5)]

    span = end - start
    is_full_circle = False
    if wrap_full_circle:
        span_abs = abs(span)
        is_full_circle = (
            span_abs >= 360.0 - 1e-6
            and math.isclose(math.fmod(span_abs, 360.0), 0.0, abs_tol=1e-6)
        )

    if is_full_circle:
        values = start + span * np.arange(count, dtype=np.float64) / count
    else:
        values = np.linspace(start, end, count, dtype=np.float64)

    return [_round_angle(value) for value in values.tolist()]


def validate_pitch_values(values):
    clipped = [float(np.clip(value, -89.0, 89.0)) for value in values]
    return [_round_angle(value) for value in clipped]


def validate_fov(fov):
    if not (0.0 < fov < 180.0):
        raise ValueError(f"fov must be inside (0, 180), but got {fov}.")


def validate_patch_size(patch_size):
    if patch_size <= 0:
        raise ValueError(f"patch-size must be positive, but got {patch_size}.")


def build_local_rays(patch_size, fov_deg):
    half_extent = math.tan(math.radians(fov_deg) * 0.5)
    coords = (np.arange(patch_size, dtype=np.float32) + 0.5) / patch_size
    coords = (coords * 2.0 - 1.0) * half_extent
    xx, yy = np.meshgrid(coords, coords)

    rays = np.stack([xx, -yy, np.ones_like(xx)], axis=-1)
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays / norms


def rotation_matrix(yaw_deg, pitch_deg):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)

    yaw_matrix = np.array(
        [
            [cos_yaw, 0.0, sin_yaw],
            [0.0, 1.0, 0.0],
            [-sin_yaw, 0.0, cos_yaw],
        ],
        dtype=np.float32,
    )
    pitch_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_pitch, sin_pitch],
            [0.0, -sin_pitch, cos_pitch],
        ],
        dtype=np.float32,
    )

    return yaw_matrix @ pitch_matrix


@lru_cache(maxsize=32)
def build_remap(image_height, image_width, patch_size, fov_deg, yaw_deg, pitch_deg):
    local_rays = build_local_rays(patch_size, fov_deg)
    world_rays = local_rays @ rotation_matrix(yaw_deg, pitch_deg).T

    x = world_rays[..., 0]
    y = np.clip(world_rays[..., 1], -1.0, 1.0)
    z = world_rays[..., 2]

    lon = np.arctan2(x, z)
    lat = np.arcsin(y)

    map_x = (lon / (2.0 * math.pi) + 0.5) * (image_width - 1)
    map_y = (0.5 - lat / math.pi) * (image_height - 1)
    map_y = np.clip(map_y, 0.0, image_height - 1)

    return map_x.astype(np.float32), map_y.astype(np.float32)


def extract_view(panorama, fov_deg, yaw_deg, pitch_deg, patch_size):
    height, width = panorama.shape[:2]
    map_x, map_y = build_remap(
        image_height=height,
        image_width=width,
        patch_size=patch_size,
        fov_deg=float(fov_deg),
        yaw_deg=float(yaw_deg),
        pitch_deg=float(pitch_deg),
    )
    return cv2.remap(
        panorama,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


def build_view_specs(range_arg=DEFAULT_RANGE_ARG):
    range_spec = parse_range_spec(range_arg)
    yaw_values = sample_axis(
        range_spec.yaw.start,
        range_spec.yaw.end,
        range_spec.yaw.count,
        wrap_full_circle=True,
    )
    pitch_values = validate_pitch_values(
        sample_axis(
            range_spec.pitch.start,
            range_spec.pitch.end,
            range_spec.pitch.count,
            wrap_full_circle=False,
        )
    )

    view_angles = [(yaw, pitch) for pitch in pitch_values for yaw in yaw_values]
    if len(view_angles) != len(CAMERA_NAMES):
        raise ValueError(
            f"Expected {len(CAMERA_NAMES)} view angles, but derived {len(view_angles)} "
            f"from range '{range_arg}'."
        )
    return list(zip(CAMERA_NAMES, view_angles))


def decode_compressed_frame(frame):
    if isinstance(frame, np.ndarray):
        encoded = frame.tobytes()
    elif isinstance(frame, (bytes, bytearray, memoryview, np.bytes_)):
        encoded = bytes(frame)
    else:
        raise TypeError(f"Unsupported frame type: {type(frame)!r}")

    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is not None:
        return image

    trimmed = encoded.rstrip(b"\x00")
    if not trimmed:
        raise ValueError("Compressed frame is empty after trimming trailing zeros.")

    image = cv2.imdecode(np.frombuffer(trimmed, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode compressed pano frame.")
    return image


def normalize_direct_frame(frame):
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            if np.nanmax(array) <= 1.0 and np.nanmin(array) >= 0.0:
                array = array * 255.0
            array = np.clip(array, 0, 255)
        elif np.issubdtype(array.dtype, np.integer):
            array = np.clip(array, 0, 255)
        else:
            raise TypeError(f"Unsupported direct image dtype: {array.dtype}")
        array = array.astype(np.uint8)

    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
        if array.shape[-1] == 1:
            return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
        if array.shape[-1] == 4:
            return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
        return array
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        transposed = np.transpose(array, (1, 2, 0))
        return normalize_direct_frame(transposed)

    raise ValueError(f"Could not infer frame layout from shape {array.shape}.")


def decode_frame(frame):
    if isinstance(frame, np.ndarray) and frame.ndim == 1 and frame.dtype == np.uint8:
        return decode_compressed_frame(frame)
    if isinstance(frame, (bytes, bytearray, memoryview, np.bytes_)):
        return decode_compressed_frame(frame)
    return normalize_direct_frame(frame)


def encode_jpeg(image, quality=DEFAULT_JPEG_QUALITY):
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError("Failed to encode tangent image as JPEG.")
    return np.frombuffer(encoded.tobytes(), dtype=np.uint8)


def pad_encoded_frames(encoded_frames):
    max_len = max(frame.shape[0] for frame in encoded_frames)
    padded = np.zeros((len(encoded_frames), max_len), dtype=np.uint8)
    lengths = np.zeros(len(encoded_frames), dtype=np.int32)

    for index, encoded in enumerate(encoded_frames):
        padded[index, : encoded.shape[0]] = encoded
        lengths[index] = encoded.shape[0]

    return padded, lengths


def ensure_parent_group(handle, dataset_path):
    parent = Path(dataset_path).parent
    if str(parent) == ".":
        return handle

    group = handle
    for part in parent.parts:
        group = group.require_group(part)
    return group


def copy_dataset(source_file, dest_file, dataset_path):
    if dataset_path not in source_file:
        return

    parent_group = ensure_parent_group(dest_file, dataset_path)
    dataset_name = Path(dataset_path).name
    data = source_file[dataset_path][()]
    parent_group.create_dataset(dataset_name, data=data)


def build_encoded_view_datasets(
    pano_dataset,
    view_specs,
    fov=DEFAULT_FOV,
    patch_size=DEFAULT_PATCH_SIZE,
    jpeg_quality=DEFAULT_JPEG_QUALITY,
):
    encoded_by_camera = {camera_name: [] for camera_name, _ in view_specs}
    for frame_index in range(pano_dataset.shape[0]):
        panorama = decode_frame(pano_dataset[frame_index])
        for camera_name, (yaw, pitch) in view_specs:
            tangent = extract_view(
                panorama=panorama,
                fov_deg=fov,
                yaw_deg=yaw,
                pitch_deg=pitch,
                patch_size=patch_size,
            )
            encoded_by_camera[camera_name].append(encode_jpeg(tangent, quality=jpeg_quality))
    return encoded_by_camera


def build_encoded_preserved_image_datasets(
    source_file,
    camera_names=PRESERVED_CAMERA_NAMES,
    jpeg_quality=DEFAULT_JPEG_QUALITY,
):
    encoded_by_camera = {}
    for camera_name in camera_names:
        dataset_path = f"observations/images/{camera_name}"
        if dataset_path not in source_file:
            raise KeyError(
                f"Episode is missing required RGB camera dataset '{dataset_path}'."
            )

        source_dataset = source_file[dataset_path]
        encoded_frames = []
        for frame_index in range(source_dataset.shape[0]):
            image = decode_frame(source_dataset[frame_index])
            encoded_frames.append(encode_jpeg(image, quality=jpeg_quality))
        encoded_by_camera[camera_name] = encoded_frames
    return encoded_by_camera


def write_encoded_image_group(dest_file, encoded_by_camera):
    images_group = dest_file.require_group("observations").create_group("images")
    for camera_name, encoded_frames in encoded_by_camera.items():
        padded_frames, _ = pad_encoded_frames(encoded_frames)
        images_group.create_dataset(camera_name, data=padded_frames)


def convert_episode(
    source_path,
    dest_path,
    overwrite=False,
    fov=DEFAULT_FOV,
    range_arg=DEFAULT_RANGE_ARG,
    patch_size=DEFAULT_PATCH_SIZE,
    jpeg_quality=DEFAULT_JPEG_QUALITY,
):
    validate_fov(fov)
    validate_patch_size(patch_size)

    source_path = Path(source_path)
    dest_path = Path(dest_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source episode does not exist: {source_path}")
    if dest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Destination episode already exists. Use --overwrite to replace it: {dest_path}"
        )

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    view_specs = build_view_specs(range_arg)

    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f"{dest_path.stem}.",
        suffix=".tmp.hdf5",
        dir=str(dest_path.parent),
    )
    os.close(temp_fd)
    temp_path = Path(temp_name)

    try:
        with h5py.File(source_path, "r") as source_file, h5py.File(temp_path, "w") as dest_file:
            pano_key = "observations/panos/equirectangular"
            if pano_key not in source_file:
                raise KeyError(
                    f"Episode {source_path} is missing required dataset '{pano_key}'."
                )

            for attr_name, attr_value in source_file.attrs.items():
                dest_file.attrs[attr_name] = attr_value
            dest_file.attrs["compress"] = True

            for dataset_path in COPY_DATASET_PATHS:
                copy_dataset(source_file, dest_file, dataset_path)

            encoded_by_camera = build_encoded_preserved_image_datasets(
                source_file,
                jpeg_quality=jpeg_quality,
            )
            encoded_by_camera.update(build_encoded_view_datasets(
                source_file[pano_key],
                view_specs=view_specs,
                fov=fov,
                patch_size=patch_size,
                jpeg_quality=jpeg_quality,
            ))
            write_encoded_image_group(dest_file, encoded_by_camera)

        os.replace(temp_path, dest_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    return dest_path


def find_episode_paths(source_dir):
    source_dir = Path(source_dir)
    return sorted(source_dir.glob("episode_*.hdf5"))


def convert_dataset_directory(
    source_dir,
    dest_dir,
    overwrite=False,
    fov=DEFAULT_FOV,
    range_arg=DEFAULT_RANGE_ARG,
    patch_size=DEFAULT_PATCH_SIZE,
    jpeg_quality=DEFAULT_JPEG_QUALITY,
):
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory does not exist: {source_dir}")
    if source_dir.resolve() == dest_dir.resolve():
        raise ValueError("Source and destination directories must be different.")

    episode_paths = find_episode_paths(source_dir)
    if not episode_paths:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {source_dir}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    converted_paths = []
    total = len(episode_paths)
    for index, source_path in enumerate(episode_paths, start=1):
        dest_path = dest_dir / source_path.name
        print(f"[{index}/{total}] converting {source_path.name}")
        converted_paths.append(
            convert_episode(
                source_path=source_path,
                dest_path=dest_path,
                overwrite=overwrite,
                fov=fov,
                range_arg=range_arg,
                patch_size=patch_size,
                jpeg_quality=jpeg_quality,
            )
        )

    print(f"Converted {len(converted_paths)} episodes from {source_dir} to {dest_dir}")
    return converted_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert pano HDF5 episodes into ACT datasets with wide, wrists, and six tangent pano views."
    )
    parser.add_argument("source_dir", type=Path, help="Directory containing source episode_*.hdf5 files.")
    parser.add_argument("dest_dir", type=Path, help="Directory for converted episode_*.hdf5 files.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination episode files if they already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    convert_dataset_directory(
        source_dir=args.source_dir,
        dest_dir=args.dest_dir,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
