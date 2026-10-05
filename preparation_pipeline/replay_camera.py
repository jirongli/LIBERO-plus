#!/usr/bin/env python3
"""Replay LIBERO-10 demonstrations from two RoboTwin-style static cameras.

Each source demonstration is rendered from multiple fixed camera-pose groups.
Every recorded MuJoCo state is restored directly before rendering, so the new
camera videos remain exactly aligned with the successful source trajectory.
Actions are preserved as training targets but are never executed open loop.
Azimuth is stratified independently over the two RoboTwin-style hemispheres:

* camera_1: scene-dependent radius, azimuth [0, 180] deg
* camera_2: scene-dependent radius, azimuth [180, 360] deg
* both: elevation [30, 65] deg

The robot-base clearance constraints can make part of a nominal hemisphere
unreachable, so each task's feasible azimuth interval is computed before it is
split into equal-width strata.  For every source demonstration, each camera
independently shuffles the ``N`` strata.  Thus both cameras cover every stratum
once while their pairings remain random.

The cameras point at the center of the task-relevant BDDL init regions.  Their
sampled poses stay fixed for the whole trajectory.  Output files retain the source
LIBERO HDF5 structure and add ``obs/camera_1_rgb`` and
``obs/camera_2_rgb`` datasets.  Image resolution and camera optics follow the
LIBERO setup used by lingbot-va (128x128 and MuJoCo/LIBERO's native FOV).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CAMERA_NAMES = ("camera_1", "camera_2")
ROBOTWIN_RANGES = {
    "camera_1": {"azimuth": (0.0, 180.0)},
    "camera_2": {"azimuth": (180.0, 360.0)},
}
DEFAULT_RADIUS_RANGE = (0.85, 0.90)
SCENE_RADIUS_RANGES = {
    "kitchen_table": (0.85, 0.90),
    "living_room_table": (0.95, 1.05),
    "study_table": (0.90, 0.98),
}
ELEVATION_RANGE = (30.0, 65.0)
CAMERA_TARGET_Z_OFFSET = 0.06
ROBOT_REAR_CLEARANCE = 0.10
MIN_CAMERA_ROBOT_XY_DISTANCE = 0.25
MAX_CAMERA_SAMPLE_ATTEMPTS = 10_000
FEASIBLE_AZIMUTH_GRID_SIZE = 36_001
FEASIBLE_AZIMUTH_MARGIN_DEGREES = 0.5


def _as_text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _format_vector(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def _matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized wxyz quaternion."""
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quat /= np.linalg.norm(quat)
    return quat


def look_at_quaternion(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return MuJoCo camera quaternion (local -Z looks at target, +Y is up)."""
    backward = position - target
    backward /= np.linalg.norm(backward)
    right = np.cross(np.array([0.0, 0.0, 1.0]), backward)
    if np.linalg.norm(right) < 1e-8:
        right = np.cross(np.array([0.0, 1.0, 0.0]), backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    up /= np.linalg.norm(up)
    return _matrix_to_wxyz(np.column_stack((right, up, backward)))


def sample_camera_pose(
    rng: np.random.Generator,
    camera_name: str,
    target: np.ndarray,
    min_camera_x: float,
    radius_range: tuple[float, float],
    robot_base_position: np.ndarray | None = None,
    min_camera_robot_xy_distance: float = MIN_CAMERA_ROBOT_XY_DISTANCE,
    azimuth_range: tuple[float, float] | None = None,
    azimuth_bin: int | None = None,
    num_azimuth_bins: int | None = None,
) -> dict[str, Any]:
    """Sample a camera pose without placing it behind the scene's robot base."""
    config = ROBOTWIN_RANGES[camera_name]
    if azimuth_range is None:
        azimuth_range = config["azimuth"]
    for attempt in range(1, MAX_CAMERA_SAMPLE_ATTEMPTS + 1):
        radius = rng.uniform(*radius_range)
        azimuth_deg = rng.uniform(*azimuth_range)
        elevation_deg = rng.uniform(*ELEVATION_RANGE)
        azimuth = np.deg2rad(azimuth_deg)
        elevation = np.deg2rad(elevation_deg)
        position = target + radius * np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ]
        )
        camera_robot_xy_distance = (
            float(np.linalg.norm(position[:2] - robot_base_position[:2]))
            if robot_base_position is not None
            else None
        )
        far_enough_from_robot = (
            camera_robot_xy_distance is None
            or camera_robot_xy_distance >= min_camera_robot_xy_distance
        )
        if float(position[0]) >= min_camera_x and far_enough_from_robot:
            quaternion = look_at_quaternion(position, target)
            return {
                "position": position,
                "quaternion": quaternion,
                "radius": float(radius),
                "radius_range": radius_range,
                "azimuth": float(azimuth_deg),
                "azimuth_range": azimuth_range,
                "azimuth_bin": azimuth_bin,
                "num_azimuth_bins": num_azimuth_bins,
                "elevation": float(elevation_deg),
                "sample_attempts": attempt,
                "target": target.copy(),
                "robot_base_x": min_camera_x - ROBOT_REAR_CLEARANCE,
                "robot_base_position": (
                    robot_base_position.copy()
                    if robot_base_position is not None
                    else None
                ),
                "min_camera_x": min_camera_x,
                "camera_robot_xy_distance": camera_robot_xy_distance,
                "min_camera_robot_xy_distance": min_camera_robot_xy_distance,
            }
    raise RuntimeError(
        f"Failed to sample a valid pose for {camera_name} in "
        f"{MAX_CAMERA_SAMPLE_ATTEMPTS} attempts "
        f"within azimuth range {azimuth_range} and "
        f"min_camera_x={min_camera_x:.3f}, "
        f"min_camera_robot_xy_distance={min_camera_robot_xy_distance:.3f}"
    )


def azimuth_stratum(
    camera_name: str,
    bin_index: int,
    num_bins: int,
    sampling_range: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Return one equal-width azimuth stratum for a static camera."""
    if num_bins < 1:
        raise ValueError("num_bins must be positive")
    if not 0 <= bin_index < num_bins:
        raise ValueError(
            f"bin_index must satisfy 0 <= bin_index < {num_bins}, got {bin_index}"
        )
    low, high = (
        ROBOTWIN_RANGES[camera_name]["azimuth"]
        if sampling_range is None
        else sampling_range
    )
    width = (high - low) / num_bins
    bin_low = low + bin_index * width
    return bin_low, bin_low + width


def feasible_azimuth_range(
    camera_name: str,
    target: np.ndarray,
    min_camera_x: float,
    radius_range: tuple[float, float],
    robot_base_position: np.ndarray | None = None,
    min_camera_robot_xy_distance: float = MIN_CAMERA_ROBOT_XY_DISTANCE,
) -> tuple[float, float]:
    """Return the contiguous azimuth interval containing feasible poses.

    Feasibility accounts for both the rear x boundary and the minimum XY
    distance from the robot base.  For a fixed azimuth, horizontal camera
    radius spans a continuous interval.  Squared XY distance is convex in
    that radius, so testing the valid interval endpoints is sufficient to
    determine whether at least one pose exists.  A small inward margin avoids
    strata whose boundary is only attainable at a measure-zero combination
    of radius and elevation.
    """
    nominal_low, nominal_high = ROBOTWIN_RANGES[camera_name]["azimuth"]
    if robot_base_position is None:
        robot_base_position = np.array(
            [min_camera_x - ROBOT_REAR_CLEARANCE, 0.0, 0.0], dtype=float
        )
    azimuth_degrees = np.linspace(
        nominal_low,
        nominal_high,
        FEASIBLE_AZIMUTH_GRID_SIZE,
        dtype=np.float64,
    )
    azimuth = np.deg2rad(azimuth_degrees)
    cosine = np.cos(azimuth)
    sine = np.sin(azimuth)
    horizontal_low = radius_range[0] * np.cos(
        np.deg2rad(ELEVATION_RANGE[1])
    )
    horizontal_high = radius_range[1] * np.cos(
        np.deg2rad(ELEVATION_RANGE[0])
    )
    lower = np.full_like(azimuth, horizontal_low)
    upper = np.full_like(azimuth, horizontal_high)
    x_delta = float(target[0]) - min_camera_x
    negative_cosine = cosine < -1e-12
    upper[negative_cosine] = np.minimum(
        upper[negative_cosine],
        x_delta / -cosine[negative_cosine],
    )
    positive_cosine = cosine > 1e-12
    lower[positive_cosine] = np.maximum(
        lower[positive_cosine],
        -x_delta / cosine[positive_cosine],
    )
    nonempty = upper >= lower

    base_delta = np.asarray(target[:2], dtype=float) - np.asarray(
        robot_base_position[:2], dtype=float
    )

    def distance_squared(horizontal_radius: np.ndarray) -> np.ndarray:
        dx = base_delta[0] + horizontal_radius * cosine
        dy = base_delta[1] + horizontal_radius * sine
        return dx * dx + dy * dy

    max_distance_squared = np.maximum(
        distance_squared(lower), distance_squared(upper)
    )
    feasible_mask = nonempty & (
        max_distance_squared >= min_camera_robot_xy_distance**2
    )
    if camera_name == "camera_1":
        invalid = np.flatnonzero(~feasible_mask)
        boundary_index = int(invalid[0] - 1) if invalid.size else len(azimuth) - 1
        if boundary_index < 0:
            raise RuntimeError(f"No feasible azimuth for {camera_name}")
        feasible = (
            nominal_low,
            float(azimuth_degrees[boundary_index])
            - FEASIBLE_AZIMUTH_MARGIN_DEGREES,
        )
    else:
        invalid = np.flatnonzero(~feasible_mask[::-1])
        boundary_index = (
            len(azimuth) - int(invalid[0])
            if invalid.size
            else 0
        )
        if boundary_index >= len(azimuth):
            raise RuntimeError(f"No feasible azimuth for {camera_name}")
        feasible = (
            float(azimuth_degrees[boundary_index])
            + FEASIBLE_AZIMUTH_MARGIN_DEGREES,
            nominal_high,
        )
    if feasible[1] - feasible[0] <= 1e-6:
        raise RuntimeError(
            f"Empty feasible azimuth range for {camera_name}: {feasible}"
        )
    return feasible


def get_task_camera_target(base_env: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute a task-level target from task-relevant table init regions.

    BDDL region x/y ranges are relative to ``workspace_offset``.  Only direct
    ``on(object, region)`` placements whose region is defined on the task's
    workspace are usable here; regions attached to a fixture or another object
    are in that target's local frame.  ``obj_of_interest`` selects the objects
    that are relevant to the task rather than unrelated scene distractors.
    """
    parsed_problem = base_env.parsed_problem
    regions = parsed_problem["regions"]
    objects_of_interest = set(parsed_problem.get("obj_of_interest", ()))

    region_rectangles: list[np.ndarray] = []
    used_objects: list[str] = []
    used_regions: list[str] = []
    for state in parsed_problem["initial_state"]:
        if len(state) < 3 or state[0] != "on":
            continue
        object_name, region_name = state[1], state[2]
        region = regions.get(region_name)
        if region is None or region["target"] != base_env.workspace_name:
            continue
        if objects_of_interest and object_name not in objects_of_interest:
            continue
        rectangles = np.asarray(region["ranges"], dtype=float)
        if rectangles.size == 0:
            continue
        region_rectangles.extend(rectangles.reshape(-1, 4))
        used_objects.append(object_name)
        used_regions.append(region_name)

    workspace_offset = np.asarray(base_env.workspace_offset, dtype=float)
    if region_rectangles:
        rectangles = np.asarray(region_rectangles, dtype=float)
        relative_bbox = np.array(
            [
                np.min(rectangles[:, 0]),
                np.min(rectangles[:, 1]),
                np.max(rectangles[:, 2]),
                np.max(rectangles[:, 3]),
            ],
            dtype=float,
        )
        relative_xy = np.array(
            [
                (relative_bbox[0] + relative_bbox[2]) / 2.0,
                (relative_bbox[1] + relative_bbox[3]) / 2.0,
            ]
        )
        source = "bddl_task_init_region_bbox"
    else:
        relative_bbox = None
        relative_xy = np.zeros(2, dtype=float)
        source = "workspace_center_fallback"

    target = workspace_offset + np.array(
        [relative_xy[0], relative_xy[1], CAMERA_TARGET_Z_OFFSET], dtype=float
    )
    metadata = {
        "target_source": source,
        "target_relative_xy": relative_xy,
        "target_relative_bbox": relative_bbox,
        "target_objects": sorted(set(used_objects)),
        "target_regions": sorted(set(used_regions)),
    }
    return target, metadata


def set_cameras_in_xml(xml_string: str, poses: dict[str, dict[str, Any]]) -> str:
    root = ET.fromstring(xml_string)
    # Public LIBERO demonstrations contain absolute paths from the machine that
    # originally generated them (often under ``chiliocosm/assets``).  Repair
    # those paths before asking MuJoCo to load the recorded model.
    import robosuite

    robosuite_root = Path(robosuite.__file__).resolve().parent
    libero_assets = REPO_ROOT / "libero/libero/assets"
    for asset in root.iter():
        if asset.tag not in {"mesh", "texture"} or not asset.get("file"):
            continue
        old_path = Path(asset.get("file"))
        if old_path.exists():
            continue
        parts = old_path.parts
        if "robosuite" in parts:
            index = max(i for i, part in enumerate(parts) if part == "robosuite")
            asset.set("file", str(robosuite_root.joinpath(*parts[index + 1 :])))
        elif "assets" in parts:
            index = max(i for i, part in enumerate(parts) if part == "assets")
            asset.set("file", str(libero_assets.joinpath(*parts[index + 1 :])))

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MuJoCo model XML does not contain <worldbody>")
    cameras = {camera.get("name"): camera for camera in root.iter("camera")}
    for name, pose in poses.items():
        camera = cameras.get(name)
        if camera is None:
            camera = ET.SubElement(worldbody, "camera", name=name)
        camera.set("mode", "fixed")
        camera.set("pos", _format_vector(pose["position"]))
        camera.set("quat", _format_vector(pose["quaternion"]))
        # Do not set fovy here. LIBERO cameras do not override it and inherit
        # MuJoCo's native/default camera optics.
        camera.attrib.pop("fovy", None)
    return ET.tostring(root, encoding="unicode")


def copy_hdf5_tree(source: h5py.File, destination: h5py.File) -> None:
    for key, value in source.attrs.items():
        destination.attrs[key] = value
    for key in source:
        source.copy(key, destination)


def resolve_bddl_path(raw_path: str) -> Path:
    path = Path(raw_path)
    candidates = [path, REPO_ROOT / path]
    if "bddl_files" in path.parts:
        suffix = Path(*path.parts[path.parts.index("bddl_files") + 1 :])
        candidates.append(REPO_ROOT / "libero/libero/bddl_files" / suffix)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve BDDL path {raw_path!r}")


def make_environment(source: h5py.File, height: int, width: int):
    # Import after fixing sys.path so the script works from any current directory.
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv

    data = source["data"]
    env_args = json.loads(_as_text(data.attrs["env_args"]))
    env_kwargs = dict(env_args.get("env_kwargs", {}))
    # Processed LIBERO datasets store the fully expanded robosuite controller
    # config. ControlEnv accepts a controller name and expands it again, so
    # forwarding the stored dict through **kwargs would pass
    # controller_configs twice.
    stored_controller = env_kwargs.pop("controller_configs", None)
    if isinstance(stored_controller, dict) and stored_controller.get("type"):
        env_kwargs["controller"] = stored_controller["type"]
    bddl_path = resolve_bddl_path(_as_text(data.attrs["bddl_file_name"]))
    env_kwargs.update(
        {
            "bddl_file_name": str(bddl_path),
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "ignore_done": True,
            "use_camera_obs": True,
            "camera_depths": False,
            "camera_names": list(CAMERA_NAMES),
            "camera_heights": height,
            "camera_widths": width,
            "camera_segmentations": None,
        }
    )
    return OffScreenRenderEnv(**env_kwargs)


def replay_file(
    source_path: Path,
    destination_path: Path,
    seed: int,
    source_file_index: int,
    height: int,
    width: int,
    compression: str | None,
    limit_demos_per_file: int | None,
    camera_groups: int,
) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.", suffix=".tmp", dir=destination_path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    env = None
    try:
        with h5py.File(source_path, "r") as source, h5py.File(temporary_path, "w") as output:
            copy_hdf5_tree(source, output)
            env = make_environment(source, height, width)
            output_data = output["data"]
            output_env_args = json.loads(_as_text(output_data.attrs["env_args"]))
            output_env_kwargs = output_env_args.setdefault("env_kwargs", {})
            output_env_kwargs.update(
                {
                    "camera_names": list(CAMERA_NAMES),
                    "camera_heights": height,
                    "camera_widths": width,
                    "camera_depths": False,
                    "camera_segmentations": None,
                }
            )
            output_data.attrs["env_args"] = json.dumps(output_env_args)
            output_data.attrs["replay_camera_config"] = json.dumps(
                {
                    "source": "RoboTwin/assets/embodiments/aloha-agilex/config.yml",
                    "default_radius_range": DEFAULT_RADIUS_RANGE,
                    "scene_radius_ranges": SCENE_RADIUS_RANGES,
                    "elevation_range": ELEVATION_RANGE,
                    "camera_1_azimuth_range": ROBOTWIN_RANGES["camera_1"]["azimuth"],
                    "camera_2_azimuth_range": ROBOTWIN_RANGES["camera_2"]["azimuth"],
                    "azimuth_sampling": "independent_permuted_strata",
                    "azimuth_stratification_domain": "task_feasible_range",
                    "camera_groups_per_source_demo": camera_groups,
                    "bin_pairing": "independent_random_permutation_per_camera",
                    "trajectory_replay": "recorded_mujoco_states",
                    "target_source": "bddl_task_init_region_bbox",
                    "target_z_offset_from_workspace": CAMERA_TARGET_Z_OFFSET,
                    "robot_rear_clearance": ROBOT_REAR_CLEARANCE,
                    "min_camera_robot_xy_distance": (
                        MIN_CAMERA_ROBOT_XY_DISTANCE
                    ),
                    "fovy": "libero_default",
                    "height": height,
                    "width": width,
                }
            )

            demo_names = sorted(
                source["data"].keys(),
                key=lambda name: int(name.rsplit("_", 1)[-1]),
            )
            if limit_demos_per_file is not None:
                demo_names = demo_names[:limit_demos_per_file]
            for output_demo_name in list(output_data.keys()):
                del output_data[output_demo_name]

            total_output_demos = len(demo_names) * camera_groups
            if "num_demos" in output_data.attrs:
                output_data.attrs["num_demos"] = total_output_demos

            for source_demo_index, demo_name in enumerate(demo_names):
                source_demo = source["data"][demo_name]
                states = source_demo["states"][()]
                actions = source_demo["actions"][()]
                if not len(states) or len(states) != len(actions):
                    raise ValueError(
                        f"{source_path.name}/{demo_name}: states and actions must be non-empty "
                        f"and equal length, got {len(states)} and {len(actions)}"
                    )

                env.reset()
                target, target_metadata = get_task_camera_target(env.env)
                radius_range = SCENE_RADIUS_RANGES.get(
                    env.env.workspace_name, DEFAULT_RADIUS_RANGE
                )
                robot_root_body = env.env.robots[0].robot_model.root_body
                robot_body_id = env.env.sim.model.body_name2id(robot_root_body)
                robot_base_position = np.asarray(
                    env.env.sim.data.body_xpos[robot_body_id], dtype=float
                ).copy()
                robot_base_x = float(robot_base_position[0])
                min_camera_x = robot_base_x + ROBOT_REAR_CLEARANCE
                feasible_azimuth_ranges = {
                    name: feasible_azimuth_range(
                        name,
                        target,
                        min_camera_x,
                        radius_range,
                        robot_base_position=robot_base_position,
                    )
                    for name in CAMERA_NAMES
                }
                permutation_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [seed, source_file_index, source_demo_index, 0]
                    )
                )
                camera_bin_permutations = {
                    name: permutation_rng.permutation(camera_groups).tolist()
                    for name in CAMERA_NAMES
                }

                for group_index in range(camera_groups):
                    camera_bins = {
                        name: camera_bin_permutations[name][group_index]
                        for name in CAMERA_NAMES
                    }
                    rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [
                                seed,
                                source_file_index,
                                source_demo_index,
                                group_index,
                                1,
                            ]
                        )
                    )
                    poses = {}
                    for name in CAMERA_NAMES:
                        bin_index = camera_bins[name]
                        poses[name] = sample_camera_pose(
                            rng,
                            name,
                            target,
                            min_camera_x,
                            radius_range,
                            robot_base_position=robot_base_position,
                            azimuth_range=azimuth_stratum(
                                name,
                                bin_index,
                                camera_groups,
                                feasible_azimuth_ranges[name],
                            ),
                            azimuth_bin=bin_index,
                            num_azimuth_bins=camera_groups,
                        )
                        poses[name].update(target_metadata)

                    model_xml = set_cameras_in_xml(
                        _as_text(source_demo.attrs["model_file"]), poses
                    )
                    env.reset_from_xml_string(model_xml)
                    env.sim.reset()

                    images = {name: [] for name in CAMERA_NAMES}
                    for state in states:
                        observation = env.regenerate_obs_from_state(state)
                        for name in CAMERA_NAMES:
                            images[name].append(observation[f"{name}_image"])
                    recorded_final_state_success = bool(env.check_success())
                    # LIBERO records states before actions, while rewards describe
                    # the transitions after those actions. The final recorded state
                    # can therefore precede the successful final transition.
                    source_final_success = (
                        bool(source_demo["rewards"][-1] > 0)
                        if "rewards" in source_demo
                        else None
                    )

                    output_demo_index = (
                        source_demo_index * camera_groups + group_index
                    )
                    output_demo_name = f"demo_{output_demo_index}"
                    source.copy(source_demo, output_data, name=output_demo_name)
                    output_demo = output_data[output_demo_name]
                    obs_group = output_demo["obs"]
                    for name in CAMERA_NAMES:
                        dataset_name = f"{name}_rgb"
                        if dataset_name in obs_group:
                            del obs_group[dataset_name]
                        obs_group.create_dataset(
                            dataset_name,
                            data=np.stack(images[name]),
                            compression=compression,
                        )
                    output_demo.attrs["model_file"] = model_xml
                    output_demo.attrs["source_demo"] = demo_name
                    output_demo.attrs["camera_group_index"] = group_index
                    for name in CAMERA_NAMES:
                        output_demo.attrs[f"{name}_bin_permutation"] = json.dumps(
                            camera_bin_permutations[name]
                        )
                    output_demo.attrs["trajectory_replay"] = (
                        "recorded_mujoco_states"
                    )
                    output_demo.attrs["recorded_final_state_success"] = (
                        recorded_final_state_success
                    )
                    if source_final_success is not None:
                        output_demo.attrs["source_final_reward_success"] = (
                            source_final_success
                        )
                    output_demo.attrs["camera_pose_info"] = json.dumps(
                        {
                            name: {
                                key: (
                                    value.tolist()
                                    if isinstance(value, np.ndarray)
                                    else value
                                )
                                for key, value in pose.items()
                            }
                            for name, pose in poses.items()
                        }
                    )
                    completed = output_demo_index + 1
                    print(
                        f"[{source_path.name}] {completed}/{total_output_demos} "
                        f"{demo_name} group={group_index} "
                        f"bins=({camera_bins['camera_1']},"
                        f"{camera_bins['camera_2']}): rendered "
                        f"{len(states)} recorded states, "
                        f"success={recorded_final_state_success}",
                        flush=True,
                    )
        env.close()
        env = None
        os.replace(temporary_path, destination_path)
    finally:
        if env is not None:
            env.close()
        if temporary_path.exists():
            temporary_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "datasets/libero_10")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/libero_10_replay")
    parser.add_argument("--height", type=int, default=128, help="Output image height (lingbot-va LIBERO: 128)")
    parser.add_argument("--width", type=int, default=128, help="Output image width (lingbot-va LIBERO: 128)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--camera-groups",
        type=int,
        default=6,
        help="Number of stratified azimuth camera groups per source demonstration",
    )
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="gzip")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-files", type=int, default=None, help="Useful for a small test run")
    parser.add_argument(
        "--limit-demos-per-file",
        type=int,
        default=None,
        help="Replay only the first N demonstrations from each task file",
    )
    parser.add_argument("--num-shards", type=int, default=1, help="Split input files into this many disjoint shards")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard handled by this process")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if input_dir == output_dir:
        raise ValueError("Input and output directories must differ")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if args.limit_demos_per_file is not None and args.limit_demos_per_file < 1:
        raise ValueError("--limit-demos-per-file must be at least 1")
    if args.camera_groups < 1:
        raise ValueError("--camera-groups must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < num_shards")
    source_files = sorted(input_dir.glob("*.hdf5"))
    if args.limit_files is not None:
        source_files = source_files[: args.limit_files]
    indexed_source_files = list(enumerate(source_files))
    indexed_source_files = indexed_source_files[
        args.shard_index :: args.num_shards
    ]
    if not indexed_source_files:
        print(
            f"Shard {args.shard_index}/{args.num_shards} has no input files in {input_dir}",
            flush=True,
        )
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    compression = None if args.compression == "none" else args.compression

    for file_progress, (source_file_index, source_path) in enumerate(
        indexed_source_files, start=1
    ):
        destination_path = output_dir / source_path.name
        if destination_path.exists() and not args.overwrite:
            print(
                f"[{file_progress}/{len(indexed_source_files)}] "
                f"skip existing {destination_path}"
            )
            continue
        print(
            f"[{file_progress}/{len(indexed_source_files)}] "
            f"replay {source_path} -> {destination_path}"
        )
        replay_file(
            source_path,
            destination_path,
            args.seed,
            source_file_index,
            args.height,
            args.width,
            compression,
            args.limit_demos_per_file,
            args.camera_groups,
        )


if __name__ == "__main__":
    main()
