#!/usr/bin/env python3
"""Measure the WildOS ROS inputs and recommend synchronization settings."""

import argparse
from bisect import bisect_left
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

from graphnav_msgs.msg import NavigationGraph
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage
import yaml


CAMERA_NAMES = ("front", "left", "right")


@dataclass(frozen=True)
class TopicSpec:
    """Description of one monitored ROS topic."""

    key: str
    label: str
    topic: str
    message_type: Any
    synchronized: bool = False


@dataclass(frozen=True)
class Sample:
    """Receive time and message timestamp for one ROS message."""

    receive_monotonic: float
    receive_ros: float
    stamp: Optional[float]


def message_stamp(message: Any) -> Optional[float]:
    """Return a message timestamp in seconds when one is available."""
    if hasattr(message, "header"):
        stamp = message.header.stamp
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    transforms = getattr(message, "transforms", None)
    if transforms:
        stamps = []
        for transform in transforms:
            stamp = transform.header.stamp
            stamps.append(float(stamp.sec) + float(stamp.nanosec) * 1e-9)
        return max(stamps)
    return None


def percentile(values: Iterable[float], fraction: float) -> Optional[float]:
    """Calculate a linearly interpolated percentile without NumPy."""
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def arrival_rate(samples: Sequence[Sample]) -> Optional[float]:
    """Calculate the average arrival rate of a sample sequence."""
    if len(samples) < 2:
        return None
    elapsed = samples[-1].receive_monotonic - samples[0].receive_monotonic
    if elapsed <= 0.0:
        return None
    return (len(samples) - 1) / elapsed


def arrival_intervals(samples: Sequence[Sample]) -> List[float]:
    """Return positive intervals between message arrivals."""
    return [
        current.receive_monotonic - previous.receive_monotonic
        for previous, current in zip(samples, samples[1:])
        if current.receive_monotonic > previous.receive_monotonic
    ]


def valid_delays(samples: Sequence[Sample]) -> List[float]:
    """Return plausible ROS receive-time minus header-time delays."""
    delays = []
    for sample in samples:
        if sample.stamp is None:
            continue
        delay = sample.receive_ros - sample.stamp
        if -0.05 <= delay <= 300.0:
            delays.append(max(0.0, delay))
    return delays


def round_up(value: float, step: float) -> float:
    """Round a positive value up to a configured step."""
    return math.ceil(value / step) * step


def nearest_sample(
    ordered_samples: Sequence[Sample],
    ordered_stamps: Sequence[float],
    stamp: float,
) -> Sample:
    """Return the sample whose header timestamp is closest to stamp."""
    index = bisect_left(ordered_stamps, stamp)
    candidates = []
    if index < len(ordered_samples):
        candidates.append(ordered_samples[index])
    if index > 0:
        candidates.append(ordered_samples[index - 1])
    return min(candidates, key=lambda sample: abs(sample.stamp - stamp))


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML mapping from disk."""
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def format_camera_topic(template: str, index: int) -> str:
    """Apply the same positional camera-name formatting used by WildOS."""
    try:
        return template.format(CAMERA_NAMES[index])
    except (IndexError, KeyError, ValueError) as error:
        raise ValueError(
            f"Cannot format camera topic '{template}' for camera {index}"
        ) from error


def find_default_config() -> Path:
    """Find the source-tree config, falling back to the installed package."""
    source_path = Path("visual_navigation/configs/wildos_nav_conf.yaml")
    if source_path.is_file():
        return source_path.resolve()

    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("visual_navigation")).joinpath(
        "configs", "wildos_nav_conf.yaml"
    )


def find_default_graph_config(wildos_config: Path) -> Optional[Path]:
    """Find graphnav_builder.yaml near a source-tree WildOS config."""
    for parent in wildos_config.parents:
        candidate = parent / "graphnav_builder/config/graphnav_builder.yaml"
        if candidate.is_file():
            return candidate.resolve()
    return None


def graph_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a ROS parameter mapping from graphnav_builder YAML."""
    node_config = config.get("graphnav_builder", {})
    if isinstance(node_config, dict):
        parameters = node_config.get("ros__parameters", {})
        if isinstance(parameters, dict):
            return parameters
    return {}


def build_topic_specs(
    config: Dict[str, Any],
    map_topic: Optional[str],
) -> List[TopicSpec]:
    """Build the set of WildOS input and diagnostic topics to monitor."""
    camera_count = int(config.get("num_cameras", 1))
    if camera_count not in (1, 3):
        raise ValueError("WildOS supports one or three cameras")

    image_template = str(config["camera_img_topic"])
    info_template = str(config["camera_info_topic"])
    specs: List[TopicSpec] = []
    for index in range(camera_count):
        image_topic = format_camera_topic(image_template, index)
        info_topic = format_camera_topic(info_template, index)
        image_type = CompressedImage if "compressed" in image_topic else Image
        specs.extend(
            [
                TopicSpec(
                    key=f"camera_{index}_image",
                    label=f"camera[{index}] image",
                    topic=image_topic,
                    message_type=image_type,
                    synchronized=True,
                ),
                TopicSpec(
                    key=f"camera_{index}_info",
                    label=f"camera[{index}] info",
                    topic=info_topic,
                    message_type=CameraInfo,
                    synchronized=True,
                ),
            ]
        )

    specs.extend(
        [
            TopicSpec(
                key="odom",
                label="odometry",
                topic=str(config["odometry_topic"]),
                message_type=Odometry,
                synchronized=True,
            ),
            TopicSpec(
                key="nav_graph",
                label="navigation graph",
                topic=str(config["navigation_graph_topic"]),
                message_type=NavigationGraph,
                synchronized=True,
            ),
        ]
    )
    if map_topic:
        specs.append(
            TopicSpec(
                key="upstream_map",
                label="upstream GridMap",
                topic=map_topic,
                message_type=GridMap,
            )
        )

    specs.extend(
        [
            TopicSpec("tf", "dynamic TF", "/tf", TFMessage),
            TopicSpec("tf_static", "static TF", "/tf_static", TFMessage),
            TopicSpec(
                key="wildos_output",
                label="WildOS scored graph",
                topic=str(
                    config.get("scored_navgraph_topic", "/scored_nav_graph")
                ),
                message_type=NavigationGraph,
            ),
        ]
    )
    return specs


class WildosSyncTuner(Node):
    """Record timing and QoS data without changing the ROS system."""

    def __init__(
        self,
        specs: Sequence[TopicSpec],
        node_name: str = "wildos_sync_tuner",
    ) -> None:
        """Create an idle monitor for the supplied topic specifications."""
        super().__init__(node_name)
        self.specs = list(specs)
        self.samples: Dict[str, List[Sample]] = {
            spec.key: [] for spec in self.specs
        }
        self.endpoint_info: Dict[str, Sequence[Any]] = {}
        self._monitor_subscriptions = []

    def discover_publishers(self, timeout: float = 3.0) -> None:
        """Wait briefly for endpoints and save their offered QoS profiles."""
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            found_all = self.refresh_publishers()
            if found_all:
                break
            rclpy.spin_once(self, timeout_sec=0.1)

    def refresh_publishers(self) -> bool:
        """Refresh publisher endpoint information and report completeness."""
        found_all = True
        for spec in self.specs:
            endpoints = self.get_publishers_info_by_topic(spec.topic)
            self.endpoint_info[spec.key] = endpoints
            if not endpoints:
                found_all = False
        return found_all

    def start_subscriptions(self) -> None:
        """Create monitoring subscriptions compatible with offered QoS."""
        for spec in self.specs:
            endpoints = self.endpoint_info.get(spec.key, ())
            transient_local = bool(endpoints) and all(
                endpoint.qos_profile.durability ==
                DurabilityPolicy.TRANSIENT_LOCAL
                for endpoint in endpoints
            )
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=100,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=(
                    DurabilityPolicy.TRANSIENT_LOCAL
                    if transient_local
                    else DurabilityPolicy.VOLATILE
                ),
            )
            subscription = self.create_subscription(
                spec.message_type,
                spec.topic,
                lambda message, key=spec.key: self.record(key, message),
                qos,
            )
            self._monitor_subscriptions.append(subscription)

    def record(self, key: str, message: Any) -> None:
        """Record one incoming message."""
        self.samples[key].append(
            Sample(
                receive_monotonic=time.monotonic(),
                receive_ros=self.get_clock().now().nanoseconds * 1e-9,
                stamp=message_stamp(message),
            )
        )

    def collect(self, duration: float) -> None:
        """Spin this monitor for a fixed wall-clock duration."""
        deadline = time.monotonic() + duration
        next_status = time.monotonic() + 5.0
        while rclpy.ok() and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            rclpy.spin_once(self, timeout_sec=min(0.2, max(0.0, remaining)))
            if time.monotonic() >= next_status:
                counts = ", ".join(
                    f"{spec.key}={len(self.samples[spec.key])}"
                    for spec in self.specs
                    if self.samples[spec.key]
                )
                self.get_logger().info(f"Collected: {counts or 'no messages'}")
                next_status += 5.0


def samples_by_stamp(samples: Sequence[Sample]) -> List[Sample]:
    """Return only stamped samples, ordered by header timestamp."""
    return sorted(
        (sample for sample in samples if sample.stamp is not None),
        key=lambda sample: sample.stamp,
    )


def synchronized_groups(
    samples: Dict[str, List[Sample]],
    sync_keys: Sequence[str],
) -> List[Dict[str, Sample]]:
    """Match each NavGraph to the nearest timestamp from every input."""
    ordered = {key: samples_by_stamp(samples[key]) for key in sync_keys}
    if any(not values for values in ordered.values()):
        return []

    lower_bound = max(values[0].stamp for values in ordered.values())
    upper_bound = min(values[-1].stamp for values in ordered.values())
    stamp_arrays = {
        key: [sample.stamp for sample in values]
        for key, values in ordered.items()
    }

    groups = []
    for reference in ordered["nav_graph"]:
        if reference.stamp < lower_bound or reference.stamp > upper_bound:
            continue
        group = {"nav_graph": reference}
        for key in sync_keys:
            if key == "nav_graph":
                continue
            group[key] = nearest_sample(
                ordered[key], stamp_arrays[key], reference.stamp
            )
        groups.append(group)
    return groups


def queue_requirements(
    samples: Dict[str, List[Sample]],
    groups: Sequence[Dict[str, Sample]],
    sync_keys: Sequence[str],
) -> Dict[str, List[int]]:
    """Estimate how deep each ATS queue was when a match became available."""
    requirements = {key: [] for key in sync_keys}
    for group in groups:
        available_at = max(
            sample.receive_monotonic for sample in group.values()
        )
        for key in sync_keys:
            selected = group[key]
            newer_messages = sum(
                1
                for sample in samples[key]
                if sample.stamp is not None and
                sample.receive_monotonic <= available_at and
                sample.stamp > selected.stamp + 1e-9
            )
            requirements[key].append(newer_messages + 1)
    return requirements


def matched_graph_lags(
    nav_samples: Sequence[Sample],
    map_samples: Sequence[Sample],
) -> List[float]:
    """Match graph and map stamps and return arrival-time differences."""
    ordered_maps = samples_by_stamp(map_samples)
    if not ordered_maps:
        return []
    map_stamps = [sample.stamp for sample in ordered_maps]
    lags = []
    for nav_sample in samples_by_stamp(nav_samples):
        map_sample = nearest_sample(ordered_maps, map_stamps, nav_sample.stamp)
        if abs(map_sample.stamp - nav_sample.stamp) <= 0.05:
            lag = nav_sample.receive_monotonic - map_sample.receive_monotonic
            if lag >= 0.0:
                lags.append(lag)
    return lags


def enum_name(value: Any) -> str:
    """Format a ROS QoS enum for terminal output."""
    return getattr(value, "name", str(value))


def print_measurements(
    node: WildosSyncTuner,
    percentile_fraction: float,
) -> Dict[str, Optional[float]]:
    """Print rates, delays, and QoS; return rates indexed by logical key."""
    rates: Dict[str, Optional[float]] = {}
    print("\n=== Measured topics / 话题测量 ===")
    for spec in node.specs:
        samples = node.samples[spec.key]
        rate = arrival_rate(samples)
        rates[spec.key] = rate
        delays = valid_delays(samples)
        delay_value = percentile(delays, percentile_fraction)
        rate_text = f"{rate:.2f} Hz" if rate is not None else "n/a"
        delay_text = (
            f"{delay_value:.3f} s"
            if delay_value is not None
            else "n/a"
        )
        print(
            f"{spec.label:22s} {rate_text:>10s}  "
            f"P{percentile_fraction * 100:.0f} delay={delay_text:>9s}  "
            f"n={len(samples):5d}  {spec.topic}"
        )

    print("\n=== Offered publisher QoS / 发布端 QoS ===")
    for spec in node.specs:
        endpoints = node.endpoint_info.get(spec.key, ())
        if not endpoints:
            print(f"{spec.label:22s} no publisher discovered")
            continue
        profiles = []
        for endpoint in endpoints:
            qos = endpoint.qos_profile
            profiles.append(
                f"{enum_name(qos.reliability)}/"
                f"{enum_name(qos.durability)}/depth={qos.depth}"
            )
        print(f"{spec.label:22s} {', '.join(sorted(set(profiles)))}")
    return rates


def print_recommendations(
    node: WildosSyncTuner,
    config: Dict[str, Any],
    rates: Dict[str, Optional[float]],
    percentile_fraction: float,
    processing_seconds: Optional[float],
) -> None:
    """Analyze collected samples and print a recommended YAML fragment."""
    sync_specs = [spec for spec in node.specs if spec.synchronized]
    sync_keys = [spec.key for spec in sync_specs]
    groups = synchronized_groups(node.samples, sync_keys)
    warnings = []

    spans = []
    for group in groups:
        group_stamps = [sample.stamp for sample in group.values()]
        spans.append(max(group_stamps) - min(group_stamps))
    measured_slop = percentile(spans, percentile_fraction)
    recommended_slop = None
    if measured_slop is not None:
        recommended_slop = round_up(
            max(0.01, measured_slop * 1.25 + 0.005), 0.01
        )

    requirements = queue_requirements(node.samples, groups, sync_keys)
    queue_percentiles = {
        key: percentile(values, percentile_fraction)
        for key, values in requirements.items()
    }
    valid_queue_values = [
        value for value in queue_percentiles.values() if value is not None
    ]
    recommended_queue = None
    if valid_queue_values:
        recommended_queue = max(
            5, math.ceil(max(valid_queue_values) * 1.20) + 2
        )

    output_samples = node.samples.get("wildos_output", [])
    block_source = ""
    if processing_seconds is not None:
        block_time = processing_seconds
        block_source = "--processing-seconds"
    elif len(output_samples) >= 5:
        block_time = percentile(
            arrival_intervals(output_samples), percentile_fraction
        )
        block_source = "WildOS output interval (conservative upper bound)"
    else:
        block_time = None
        warnings.append(
            "Not enough /scored_nav_graph samples to estimate callback "
            "blocking. Keep qos_history_depth unchanged or pass "
            "--processing-seconds with the measured worst inference time."
        )

    synchronized_rates = [
        rates[key] for key in sync_keys if rates.get(key) is not None
    ]
    recommended_qos_depth = None
    if block_time is not None and synchronized_rates:
        recommended_qos_depth = max(
            10, math.ceil(max(synchronized_rates) * block_time * 1.20)
        )

    tf_rate = rates.get("tf")
    recommended_tf_depth = None
    if block_time is not None and tf_rate is not None:
        recommended_tf_depth = max(
            10, math.ceil(tf_rate * block_time * 1.20)
        )

    nav_delays = valid_delays(node.samples.get("nav_graph", []))
    nav_delay = percentile(nav_delays, percentile_fraction)
    if nav_delay is not None:
        recommended_cache_time = max(5, math.ceil(nav_delay + 2.0))
    else:
        recommended_cache_time = int(
            config.get("tf_lookup_config", {}).get("cache_time", 10)
        )
        warnings.append(
            "Header-to-ROS-clock delay was invalid. In simulation, rerun "
            "with '--ros-args -p use_sim_time:=true'."
        )

    map_samples = node.samples.get("upstream_map", [])
    graph_lags = matched_graph_lags(
        node.samples.get("nav_graph", []), map_samples
    )
    graph_lag = percentile(graph_lags, percentile_fraction)

    for spec in sync_specs:
        samples = node.samples[spec.key]
        if len(samples) < 2:
            if spec.key.endswith("_info"):
                warnings.append(
                    f"{spec.topic} produced fewer than two messages. Cache "
                    "this CameraInfo separately instead of synchronizing it."
                )
            else:
                warnings.append(
                    f"{spec.topic} produced fewer than two messages; no "
                    "reliable synchronization recommendation is possible."
                )
        endpoints = node.endpoint_info.get(spec.key, ())
        if any(
            endpoint.qos_profile.reliability ==
            ReliabilityPolicy.BEST_EFFORT
            for endpoint in endpoints
        ):
            warnings.append(
                f"{spec.topic} offers BEST_EFFORT, while the integer QoS used "
                "by WildOS requests RELIABLE. Configure a per-topic "
                "BEST_EFFORT/SensorData QoS profile in code."
            )

    if len(groups) < 5:
        warnings.append(
            f"Only {len(groups)} complete timestamp groups were observed; "
            "collect for longer before applying the values."
        )

    current_tf = config.get("tf_lookup_config", {})
    qos_depth = (
        recommended_qos_depth
        if recommended_qos_depth is not None
        else int(config.get("qos_history_depth", 100))
    )
    queue_size = (
        recommended_queue
        if recommended_queue is not None
        else int(config.get("syncsub_queue_size", 150))
    )
    slop = (
        recommended_slop
        if recommended_slop is not None
        else float(config.get("syncsub_slop", 0.2))
    )
    tf_depth = (
        recommended_tf_depth
        if recommended_tf_depth is not None
        else int(current_tf.get("qos_history_depth", 100))
    )

    print("\n=== Analysis / 分析 ===")
    if measured_slop is not None:
        print(
            f"Matched groups: {len(groups)}, timestamp-span "
            f"P{percentile_fraction * 100:.0f}={measured_slop:.4f}s, "
            f"max={max(spans):.4f}s"
        )
    if valid_queue_values:
        for spec in sync_specs:
            value = queue_percentiles.get(spec.key)
            if value is not None:
                maximum = max(requirements[spec.key])
                print(
                    f"ATS retained depth {spec.label:22s}: "
                    f"P{percentile_fraction * 100:.0f}={value:.1f}, "
                    f"max={maximum}"
                )
    if graph_lag is not None:
        print(
            f"GridMap -> NavGraph extra arrival lag "
            f"P{percentile_fraction * 100:.0f}={graph_lag:.3f}s"
        )
    if block_time is not None:
        print(
            f"QoS blocking-time estimate={block_time:.3f}s "
            f"from {block_source}"
        )

    print("\n=== Recommended YAML / 建议配置 ===")
    print(f"qos_history_depth: {qos_depth}")
    print(f"syncsub_queue_size: {queue_size}")
    print(f"syncsub_slop: {slop:.2f}")
    print("\ntf_lookup_config:")
    print(f"    buffer_size: {int(current_tf.get('buffer_size', 1))}")
    print(f"    cache_time: {recommended_cache_time}")
    print(
        f"    timer_duration: "
        f"{float(current_tf.get('timer_duration', 0.05)):.2f}"
    )
    print(
        f"    lookup_timeout: "
        f"{float(current_tf.get('lookup_timeout', 0.0)):g}"
    )
    print(f"    qos_history_depth: {tf_depth}")
    print(
        f"    wait_for_oldest: "
        f"{str(bool(current_tf.get('wait_for_oldest', False))).lower()}"
    )
    print(
        f"    clear_buffer_on_process: "
        f"{str(bool(current_tf.get('clear_buffer_on_process', True))).lower()}"
    )
    print(
        f"    spin_thread: "
        f"{str(bool(current_tf.get('spin_thread', False))).lower()}"
    )

    if warnings:
        print("\n=== Warnings / 注意 ===")
        for warning in dict.fromkeys(warnings):
            print(f"- {warning}")


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    """Parse non-ROS command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Observe WildOS topics and recommend QoS, "
            "ApproximateTimeSynchronizer, and TF lookup settings."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="wildos_nav_conf.yaml path",
    )
    parser.add_argument(
        "--graph-config",
        type=Path,
        default=None,
        help=(
            "graphnav_builder.yaml path, used to discover the upstream "
            "GridMap"
        ),
    )
    parser.add_argument(
        "--map-topic",
        default=None,
        help="override the upstream GridMap topic",
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
        help="measured worst WildOS inference time for QoS depth estimation",
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
    """Run the WildOS synchronization tuner."""
    cli_args = remove_ros_args(args=sys.argv)[1:]
    args = parse_arguments(cli_args)
    config_path = (args.config or find_default_config()).resolve()
    wildos_config = load_yaml(config_path)

    graph_config_path = args.graph_config
    if graph_config_path is None:
        graph_config_path = find_default_graph_config(config_path)
    map_topic = args.map_topic
    if map_topic is None and graph_config_path is not None:
        parameters = graph_parameters(load_yaml(graph_config_path.resolve()))
        configured_topic = parameters.get("traversability_map_topic")
        if configured_topic:
            map_topic = str(configured_topic)

    specs = build_topic_specs(wildos_config, map_topic)
    percentile_fraction = args.percentile / 100.0

    print(f"WildOS config: {config_path}")
    if graph_config_path is not None:
        print(f"GraphNav config: {graph_config_path.resolve()}")
    print(f"Collecting for {args.duration:.1f}s; do not pause the simulation.")

    rclpy.init(args=sys.argv)
    node = WildosSyncTuner(specs)
    try:
        node.discover_publishers()
        node.start_subscriptions()
        node.collect(args.duration)
        node.refresh_publishers()
        rates = print_measurements(node, percentile_fraction)
        print_recommendations(
            node=node,
            config=wildos_config,
            rates=rates,
            percentile_fraction=percentile_fraction,
            processing_seconds=args.processing_seconds,
        )
    except KeyboardInterrupt:
        print("\nCollection interrupted; analyzing the available samples.")
        node.refresh_publishers()
        rates = print_measurements(node, percentile_fraction)
        print_recommendations(
            node=node,
            config=wildos_config,
            rates=rates,
            percentile_fraction=percentile_fraction,
            processing_seconds=args.processing_seconds,
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
