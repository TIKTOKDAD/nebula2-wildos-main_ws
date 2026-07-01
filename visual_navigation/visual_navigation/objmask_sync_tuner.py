#!/usr/bin/env python3
"""Recommend synchronization settings for ObjectMask/LiDAR triangulation."""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

from object_search_msgs.msg import ObjectMaskWithTf
from sensor_msgs.msg import PointCloud2
from tf2_msgs.msg import TFMessage
import rclpy
from rclpy.qos import ReliabilityPolicy
from rclpy.utilities import remove_ros_args

from visual_navigation.wildos_sync_tuner import load_yaml
from visual_navigation.wildos_sync_tuner import nearest_sample
from visual_navigation.wildos_sync_tuner import percentile
from visual_navigation.wildos_sync_tuner import print_measurements
from visual_navigation.wildos_sync_tuner import round_up
from visual_navigation.wildos_sync_tuner import Sample
from visual_navigation.wildos_sync_tuner import samples_by_stamp
from visual_navigation.wildos_sync_tuner import TopicSpec
from visual_navigation.wildos_sync_tuner import valid_delays
from visual_navigation.wildos_sync_tuner import WildosSyncTuner


@dataclass(frozen=True)
class MaskLidarPair:
    """Nearest timestamp pair selected from mask and LiDAR streams."""

    mask: Sample
    lidar: Sample


def find_default_config() -> Path:
    """Find the source config, falling back to the installed package."""
    source_path = Path(
        "visual_navigation/configs/triangulation3d_objsearch_conf.yaml"
    )
    if source_path.is_file():
        return source_path.resolve()

    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("visual_navigation")).joinpath(
        "configs", "triangulation3d_objsearch_conf.yaml"
    )


def build_topic_specs(config: Dict[str, Any]) -> List[TopicSpec]:
    """Build the mask, LiDAR, TF, and processing-output topic list."""
    return [
        TopicSpec(
            key="object_mask",
            label="object mask",
            topic=str(config["object_mask_topic"]),
            message_type=ObjectMaskWithTf,
        ),
        TopicSpec(
            key="lidar",
            label="LiDAR point cloud",
            topic=str(config["lidar_topic"]),
            message_type=PointCloud2,
        ),
        TopicSpec(
            key="tf",
            label="dynamic TF",
            topic="/tf",
            message_type=TFMessage,
        ),
        TopicSpec(
            key="tf_static",
            label="static TF",
            topic="/tf_static",
            message_type=TFMessage,
        ),
        TopicSpec(
            key="processing_output",
            label="particle output",
            topic=str(
                config.get("particle_viz_topic", "/object_hypotheses")
            ),
            message_type=PointCloud2,
        ),
    ]


def match_mask_lidar_pairs(
    mask_samples: Sequence[Sample],
    lidar_samples: Sequence[Sample],
) -> List[MaskLidarPair]:
    """Match every in-range mask to its closest LiDAR header timestamp."""
    ordered_masks = samples_by_stamp(mask_samples)
    ordered_lidar = samples_by_stamp(lidar_samples)
    if not ordered_masks or not ordered_lidar:
        return []

    lidar_stamps = [sample.stamp for sample in ordered_lidar]
    first_lidar_stamp = lidar_stamps[0]
    last_lidar_stamp = lidar_stamps[-1]
    pairs = []
    for mask in ordered_masks:
        if mask.stamp < first_lidar_stamp or mask.stamp > last_lidar_stamp:
            continue
        lidar = nearest_sample(ordered_lidar, lidar_stamps, mask.stamp)
        pairs.append(MaskLidarPair(mask=mask, lidar=lidar))
    return pairs


def pair_queue_requirements(
    samples: Dict[str, List[Sample]],
    pairs: Sequence[MaskLidarPair],
) -> Dict[str, List[int]]:
    """Estimate each ATS queue depth when both messages became available."""
    requirements = {"object_mask": [], "lidar": []}
    for pair in pairs:
        available_at = max(
            pair.mask.receive_monotonic,
            pair.lidar.receive_monotonic,
        )
        for key, selected in (
            ("object_mask", pair.mask),
            ("lidar", pair.lidar),
        ):
            newer_messages = sum(
                1
                for sample in samples[key]
                if sample.stamp is not None and
                sample.receive_monotonic <= available_at and
                sample.stamp > selected.stamp + 1e-9
            )
            requirements[key].append(newer_messages + 1)
    return requirements


def mask_to_output_latencies(
    mask_samples: Sequence[Sample],
    output_samples: Sequence[Sample],
) -> List[float]:
    """Return conservative arrival-to-output latency estimates."""
    masks = sorted(mask_samples, key=lambda sample: sample.receive_monotonic)
    outputs = sorted(
        output_samples, key=lambda sample: sample.receive_monotonic
    )
    latencies = []
    mask_index = 0
    for output in outputs:
        if mask_index >= len(masks):
            break
        mask = masks[mask_index]
        if mask.receive_monotonic > output.receive_monotonic:
            continue
        latencies.append(
            output.receive_monotonic - mask.receive_monotonic
        )
        mask_index += 1
    return latencies


def best_effort_warning(
    node: WildosSyncTuner,
    key: str,
    topic: str,
) -> Optional[str]:
    """Warn when the current integer QoS cannot match a publisher."""
    endpoints = node.endpoint_info.get(key, ())
    if any(
        endpoint.qos_profile.reliability ==
        ReliabilityPolicy.BEST_EFFORT
        for endpoint in endpoints
    ):
        return (
            f"{topic} offers BEST_EFFORT, while ObjectMaskTriangulator's "
            "integer QoS requests RELIABLE. A SensorData/BEST_EFFORT "
            "subscription must be configured in code."
        )
    return None


def print_recommendations(
    node: WildosSyncTuner,
    config: Dict[str, Any],
    rates: Dict[str, Optional[float]],
    percentile_fraction: float,
    processing_seconds: Optional[float],
) -> None:
    """Analyze samples and print a recommended triangulation YAML fragment."""
    pairs = match_mask_lidar_pairs(
        node.samples["object_mask"], node.samples["lidar"]
    )
    offsets = [abs(pair.mask.stamp - pair.lidar.stamp) for pair in pairs]
    measured_slop = percentile(offsets, percentile_fraction)
    if measured_slop is None:
        recommended_slop = float(config.get("syncsub_slop", 0.2))
    else:
        recommended_slop = round_up(
            max(0.01, measured_slop * 1.25 + 0.005), 0.01
        )

    requirements = pair_queue_requirements(node.samples, pairs)
    queue_percentiles = {
        key: percentile(values, percentile_fraction)
        for key, values in requirements.items()
    }
    queue_values = [
        value for value in queue_percentiles.values() if value is not None
    ]
    if queue_values:
        recommended_queue = max(
            5, math.ceil(max(queue_values) * 1.20) + 2
        )
    else:
        recommended_queue = int(config.get("syncsub_queue_size", 30))

    observed_process_latencies = mask_to_output_latencies(
        node.samples["object_mask"], node.samples["processing_output"]
    )
    if processing_seconds is not None:
        block_time = processing_seconds
        block_source = "--processing-seconds"
    elif len(observed_process_latencies) >= 3:
        block_time = percentile(
            observed_process_latencies, percentile_fraction
        )
        block_source = "mask-to-particle-output latency (upper bound)"
    else:
        block_time = None
        block_source = ""

    lidar_rate = rates.get("lidar")
    mask_rate = rates.get("object_mask")
    input_rates = [
        rate for rate in (lidar_rate, mask_rate) if rate is not None
    ]
    if block_time is not None and input_rates:
        recommended_qos_depth = max(
            10, math.ceil(max(input_rates) * block_time * 1.20)
        )
    else:
        recommended_qos_depth = int(config.get("qos_history_depth", 30))

    if block_time is not None and mask_rate is not None:
        recommended_buffer = max(
            1, math.ceil(mask_rate * block_time * 1.20)
        )
    else:
        recommended_buffer = int(config.get("buffer_size", 1))

    if mask_rate is not None and mask_rate > 0.0:
        recommended_timer = round_up(
            min(0.20, max(0.05, 0.5 / mask_rate)), 0.01
        )
    else:
        recommended_timer = float(config.get("timer_duration", 0.05))

    warnings = []
    if len(pairs) < 5:
        warnings.append(
            f"Only {len(pairs)} complete Mask/LiDAR pairs were observed. "
            "Keep the object visible and collect for longer."
        )
    if block_time is None:
        warnings.append(
            "Processing time could not be estimated because fewer than three "
            "particle outputs were received. qos_history_depth and "
            "buffer_size were left unchanged; use --processing-seconds for "
            "a firm estimate."
        )
    for key, config_key in (
        ("object_mask", "object_mask_topic"),
        ("lidar", "lidar_topic"),
    ):
        warning = best_effort_warning(node, key, str(config[config_key]))
        if warning:
            warnings.append(warning)

    mask_delays = valid_delays(node.samples["object_mask"])
    mask_delay = percentile(mask_delays, percentile_fraction)
    if node.samples["object_mask"] and mask_delay is None:
        warnings.append(
            "ObjectMask header delay was invalid. In simulation, run with "
            "'--ros-args -p use_sim_time:=true'."
        )

    print("\n=== Analysis / 分析 ===")
    if measured_slop is not None:
        print(
            f"Matched pairs={len(pairs)}, nearest timestamp offset "
            f"P{percentile_fraction * 100:.0f}={measured_slop:.4f}s, "
            f"max={max(offsets):.4f}s"
        )
    for key, label in (
        ("object_mask", "ObjectMask queue"),
        ("lidar", "LiDAR queue"),
    ):
        value = queue_percentiles.get(key)
        if value is not None:
            print(
                f"{label:22s} P{percentile_fraction * 100:.0f}="
                f"{value:.1f}, max={max(requirements[key])}"
            )
    if mask_delay is not None:
        print(
            f"ObjectMask header age P{percentile_fraction * 100:.0f}="
            f"{mask_delay:.3f}s"
        )
    if block_time is not None:
        print(
            f"Processing/blocking estimate={block_time:.3f}s "
            f"from {block_source}"
        )

    print("\n=== Recommended YAML / 建议配置 ===")
    print(f"qos_history_depth: {recommended_qos_depth}")
    print(f"syncsub_queue_size: {recommended_queue}")
    print(f"syncsub_slop: {recommended_slop:.2f}")
    print(f"\nbuffer_size: {recommended_buffer}")
    print(f"timer_duration: {recommended_timer:.2f}")

    if warnings:
        print("\n=== Warnings / 注意 ===")
        for warning in dict.fromkeys(warnings):
            print(f"- {warning}")


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    """Parse non-ROS arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Observe ObjectMaskWithTf and PointCloud2 and recommend "
            "triangulation synchronization settings."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="triangulation3d_objsearch_conf.yaml path",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="wall-clock collection duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="percentile used for recommendations (default: 95)",
    )
    parser.add_argument(
        "--processing-seconds",
        type=float,
        default=None,
        help="measured worst triangulation processing duration",
    )
    arguments = parser.parse_args(argv)
    if arguments.duration <= 0.0:
        parser.error("--duration must be positive")
    if not 50.0 <= arguments.percentile <= 100.0:
        parser.error("--percentile must be between 50 and 100")
    if arguments.processing_seconds is not None and (
        arguments.processing_seconds <= 0.0
    ):
        parser.error("--processing-seconds must be positive")
    return arguments


def main() -> None:
    """Run the ObjectMask/LiDAR synchronization tuner."""
    args = parse_arguments(remove_ros_args(args=sys.argv)[1:])
    config_path = (args.config or find_default_config()).resolve()
    config = load_yaml(config_path)
    specs = build_topic_specs(config)
    percentile_fraction = args.percentile / 100.0

    print(f"Triangulation config: {config_path}")
    print(f"Collecting for {args.duration:.1f}s; keep the object visible.")

    rclpy.init(args=sys.argv)
    node = WildosSyncTuner(specs, node_name="objmask_sync_tuner")
    try:
        node.discover_publishers()
        node.start_subscriptions()
        node.collect(args.duration)
        node.refresh_publishers()
        rates = print_measurements(node, percentile_fraction)
        print_recommendations(
            node=node,
            config=config,
            rates=rates,
            percentile_fraction=percentile_fraction,
            processing_seconds=args.processing_seconds,
        )
    except KeyboardInterrupt:
        print("\nCollection interrupted; analyzing available samples.")
        node.refresh_publishers()
        rates = print_measurements(node, percentile_fraction)
        print_recommendations(
            node=node,
            config=config,
            rates=rates,
            percentile_fraction=percentile_fraction,
            processing_seconds=args.processing_seconds,
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
