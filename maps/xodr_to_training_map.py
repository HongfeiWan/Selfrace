import argparse
import hashlib
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class CubicProfile:
    s: float
    a: float
    b: float
    c: float
    d: float

    @classmethod
    def from_element(cls, elem: ET.Element, s_attr: str = "s") -> "CubicProfile":
        return cls(
            s=float(elem.attrib.get(s_attr, 0.0)),
            a=float(elem.attrib.get("a", 0.0)),
            b=float(elem.attrib.get("b", 0.0)),
            c=float(elem.attrib.get("c", 0.0)),
            d=float(elem.attrib.get("d", 0.0)),
        )

    def eval(self, s: float) -> float:
        u = s - self.s
        return self.a + self.b * u + self.c * u * u + self.d * u * u * u


@dataclass(frozen=True)
class GeometrySegment:
    s: float
    x: float
    y: float
    hdg: float
    length: float
    kind: str
    curvature: float = 0.0


@dataclass
class LaneInfo:
    lane_id: int
    lane_type: str
    widths: List[CubicProfile]
    lane_change: str

    def width_at(self, ds_section: float) -> float:
        if not self.widths:
            return 0.0
        profile = pick_profile(self.widths, ds_section)
        return max(0.0, profile.eval(ds_section))


@dataclass
class LaneSection:
    s: float
    left: Dict[int, LaneInfo]
    right: Dict[int, LaneInfo]


@dataclass
class RoadModel:
    road_id: int
    length: float
    is_junction: bool
    geometry_segments: List[GeometrySegment]
    elevation_profiles: List[CubicProfile]
    lane_offset_profiles: List[CubicProfile]
    sections: List[LaneSection]


def pick_profile(profiles: List[CubicProfile], s: float) -> CubicProfile:
    selected = profiles[0]
    for profile in profiles:
        if profile.s <= s + 1e-9:
            selected = profile
        else:
            break
    return selected


def parse_geometry(road: ET.Element) -> List[GeometrySegment]:
    segments = []
    for geom in road.findall("./planView/geometry"):
        children = list(geom)
        if len(children) != 1:
            raise ValueError(f"road {road.attrib.get('id')} geometry must have exactly one child")
        child = children[0]
        if child.tag not in {"line", "arc"}:
            raise ValueError(
                f"unsupported OpenDRIVE geometry '{child.tag}' in road {road.attrib.get('id')}; "
                "only line and arc are supported"
            )
        segments.append(
            GeometrySegment(
                s=float(geom.attrib["s"]),
                x=float(geom.attrib["x"]),
                y=float(geom.attrib["y"]),
                hdg=float(geom.attrib["hdg"]),
                length=float(geom.attrib["length"]),
                kind=child.tag,
                curvature=float(child.attrib.get("curvature", 0.0)),
            )
        )
    segments.sort(key=lambda g: g.s)
    return segments


def pose_at_s(segments: List[GeometrySegment], road_s: float) -> Tuple[float, float, float]:
    if not segments:
        raise ValueError("road has no planView geometry")
    seg = segments[0]
    for candidate in segments:
        if candidate.s <= road_s + 1e-9:
            seg = candidate
        else:
            break
    ds = max(0.0, min(road_s - seg.s, seg.length))
    if seg.kind == "line" or abs(seg.curvature) < 1e-12:
        return seg.x + ds * math.cos(seg.hdg), seg.y + ds * math.sin(seg.hdg), seg.hdg
    k = seg.curvature
    hdg2 = seg.hdg + k * ds
    x = seg.x + (math.sin(hdg2) - math.sin(seg.hdg)) / k
    y = seg.y - (math.cos(hdg2) - math.cos(seg.hdg)) / k
    return x, y, hdg2


def parse_profiles(parent: Optional[ET.Element], tag: str, s_attr: str = "s") -> List[CubicProfile]:
    if parent is None:
        return [CubicProfile(0.0, 0.0, 0.0, 0.0, 0.0)]
    profiles = [CubicProfile.from_element(elem, s_attr=s_attr) for elem in parent.findall(tag)]
    if not profiles:
        profiles = [CubicProfile(0.0, 0.0, 0.0, 0.0, 0.0)]
    profiles.sort(key=lambda p: p.s)
    return profiles


def parse_lane_section(elem: ET.Element) -> LaneSection:
    def parse_side(side_name: str) -> Dict[int, LaneInfo]:
        result = {}
        side = elem.find(side_name)
        if side is None:
            return result
        for lane in side.findall("lane"):
            lane_id = int(lane.attrib["id"])
            road_mark = lane.find("roadMark")
            result[lane_id] = LaneInfo(
                lane_id=lane_id,
                lane_type=lane.attrib.get("type", "none"),
                widths=parse_profiles(lane, "width", s_attr="sOffset"),
                lane_change=(road_mark.attrib.get("laneChange", "none") if road_mark is not None else "none").upper(),
            )
        return result

    return LaneSection(
        s=float(elem.attrib.get("s", 0.0)),
        left=parse_side("left"),
        right=parse_side("right"),
    )


def parse_lane_sections(road: ET.Element) -> List[LaneSection]:
    sections = [parse_lane_section(elem) for elem in road.findall("./lanes/laneSection")]
    sections.sort(key=lambda sec: sec.s)
    return sections


def parse_road_model(road: ET.Element) -> RoadModel:
    return RoadModel(
        road_id=int(road.attrib["id"]),
        length=float(road.attrib["length"]),
        is_junction=road.attrib.get("junction", "-1") != "-1",
        geometry_segments=parse_geometry(road),
        elevation_profiles=parse_profiles(road.find("elevationProfile"), "elevation"),
        lane_offset_profiles=parse_profiles(road.find("lanes"), "laneOffset"),
        sections=parse_lane_sections(road),
    )


def parse_road_models(xodr_path: Path) -> Dict[int, RoadModel]:
    root = ET.parse(xodr_path).getroot()
    return {int(road.attrib["id"]): parse_road_model(road) for road in root.findall("road")}


def profile_value_at(profiles: List[CubicProfile], s: float) -> float:
    return pick_profile(profiles, s).eval(s)


def lateral_point(segments: List[GeometrySegment], elevation_profiles: List[CubicProfile], road_s: float, t: float) -> Dict[str, float]:
    x, y, hdg = pose_at_s(segments, road_s)
    return {
        "x": x - math.sin(hdg) * t,
        "y": y + math.cos(hdg) * t,
        "z": profile_value_at(elevation_profiles, road_s),
    }


def lane_offsets_for_section(section: LaneSection, lane_id: int, ds_section: float, lane_offset: float) -> Tuple[float, float, float]:
    if lane_id > 0:
        lanes = section.left
        inner = lane_offset
        for inner_lane_id in sorted([lid for lid in lanes if 0 < lid < lane_id]):
            inner += lanes[inner_lane_id].width_at(ds_section)
        width = lanes[lane_id].width_at(ds_section)
        return inner, inner + width, width

    lanes = section.right
    inner = lane_offset
    for inner_lane_id in sorted([lid for lid in lanes if lane_id < lid < 0], reverse=True):
        inner -= lanes[inner_lane_id].width_at(ds_section)
    width = lanes[lane_id].width_at(ds_section)
    return inner - width, inner, width


def lane_center_pose(
    segments: List[GeometrySegment],
    elevation_profiles: List[CubicProfile],
    lane_offset_profiles: List[CubicProfile],
    section: LaneSection,
    lane_id: int,
    road_s: float,
) -> Tuple[Dict[str, float], float, float]:
    ds_section = road_s - section.s
    lane_offset = profile_value_at(lane_offset_profiles, road_s)
    t0, t1, width = lane_offsets_for_section(section, lane_id, ds_section, lane_offset)
    point = lateral_point(segments, elevation_profiles, road_s, (t0 + t1) * 0.5)
    _, _, ref_hdg = pose_at_s(segments, road_s)
    travel_hdg = ref_hdg if lane_id < 0 else ref_hdg + math.pi
    return point, travel_hdg, width


def ordered_cross_section_points(
    segments: List[GeometrySegment],
    elevation_profiles: List[CubicProfile],
    lane_offset_profiles: List[CubicProfile],
    section: LaneSection,
    lane_id: int,
    road_s: float,
    travel_hdg: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    ds_section = road_s - section.s
    lane_offset = profile_value_at(lane_offset_profiles, road_s)
    t0, t1, _ = lane_offsets_for_section(section, lane_id, ds_section, lane_offset)
    p_a = lateral_point(segments, elevation_profiles, road_s, t0)
    p_b = lateral_point(segments, elevation_profiles, road_s, t1)
    center = ((p_a["x"] + p_b["x"]) * 0.5, (p_a["y"] + p_b["y"]) * 0.5)
    left_normal = (-math.sin(travel_hdg), math.cos(travel_hdg))
    score_a = (p_a["x"] - center[0]) * left_normal[0] + (p_a["y"] - center[1]) * left_normal[1]
    return (p_a, p_b) if score_a > 0 else (p_b, p_a)


def sample_s_values(start_s: float, end_s: float, step: float) -> List[float]:
    if end_s <= start_s:
        return []
    values = [start_s]
    current = start_s + step
    while current < end_s - 1e-9:
        values.append(current)
        current += step
    if end_s - values[-1] > 1e-6:
        values.append(end_s)
    return values


def make_waypoint_id(road_id: int, lane_id: int, road_s: float) -> int:
    raw = f"{road_id}:{lane_id}:{road_s:.3f}"
    digest = hashlib.sha1(raw.encode("ascii")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 63) - 1)


def convert_xodr_to_stage1(xodr_path: Path, map_name: str, quad_step: float, waypoint_step: float) -> Dict:
    road_models = parse_road_models(xodr_path)
    quads = []
    driving_waypoints = []
    poly_id = 0

    for road in road_models.values():
        if not road.sections:
            continue

        for section_idx, section in enumerate(road.sections):
            section_end = road.sections[section_idx + 1].s if section_idx + 1 < len(road.sections) else road.length
            section_start = max(0.0, section.s)
            section_end = min(road.length, section_end)
            driving_lanes: Iterable[LaneInfo] = list(section.left.values()) + list(section.right.values())
            for lane in driving_lanes:
                if lane.lane_type.lower() != "driving" or lane.lane_id == 0:
                    continue

                s_values = sample_s_values(section_start, section_end, quad_step)
                for s0, s1 in zip(s_values, s_values[1:]):
                    smid = (s0 + s1) * 0.5
                    _, travel_hdg, width_mid = lane_center_pose(
                        road.geometry_segments,
                        road.elevation_profiles,
                        road.lane_offset_profiles,
                        section,
                        lane.lane_id,
                        smid,
                    )
                    if width_mid <= 1e-3:
                        continue

                    if lane.lane_id < 0:
                        back_s, front_s = s0, s1
                    else:
                        back_s, front_s = s1, s0
                    left_back, right_back = ordered_cross_section_points(
                        road.geometry_segments,
                        road.elevation_profiles,
                        road.lane_offset_profiles,
                        section,
                        lane.lane_id,
                        back_s,
                        travel_hdg,
                    )
                    left_front, right_front = ordered_cross_section_points(
                        road.geometry_segments,
                        road.elevation_profiles,
                        road.lane_offset_profiles,
                        section,
                        lane.lane_id,
                        front_s,
                        travel_hdg,
                    )

                    quads.append(
                        {
                            "polyId": poly_id,
                            "vertices": [right_front, left_front, left_back, right_back],
                            "road_id": road.road_id,
                            "lane_id": lane.lane_id,
                            "q": smid,
                        }
                    )
                    poly_id += 1

                for wp_s in sample_s_values(section_start, section_end, waypoint_step):
                    point, travel_hdg, width = lane_center_pose(
                        road.geometry_segments,
                        road.elevation_profiles,
                        road.lane_offset_profiles,
                        section,
                        lane.lane_id,
                        wp_s,
                    )
                    if width <= 1e-3:
                        continue
                    driving_waypoints.append(
                        {
                            "id": make_waypoint_id(road.road_id, lane.lane_id, wp_s),
                            "road_id": road.road_id,
                            "section_id": section_idx,
                            "lane_id": lane.lane_id,
                            "s": wp_s,
                            "is_junction": road.is_junction,
                            "lane_width": width,
                            "lane_change": lane.lane_change,
                            "lane_type": "Driving",
                            "transform": {
                                "location": point,
                                "rotation": {
                                    "pitch": 0.0,
                                    "yaw": math.degrees(travel_hdg),
                                    "roll": 0.0,
                                },
                            },
                        }
                    )

    return {
        "map_name": map_name,
        "quads": quads,
        "driving_waypoints": driving_waypoints,
        "traffic_controls": [],
    }


def save_json(path: Path, data: Dict, indent: Optional[int] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _max_vertex_delta(left_quads: List[Dict], right_quads: List[Dict]) -> float:
    max_delta = 0.0
    for left, right in zip(left_quads, right_quads):
        for left_vertex, right_vertex in zip(left["vertices"], right["vertices"]):
            for axis in ("x", "y", "z"):
                max_delta = max(max_delta, abs(float(left_vertex[axis]) - float(right_vertex[axis])))
    return max_delta


def _max_waypoint_delta(left_wps: List[Dict], right_wps: List[Dict]) -> Tuple[float, float]:
    max_location_delta = 0.0
    max_yaw_delta = 0.0
    for left, right in zip(left_wps, right_wps):
        left_loc = left["transform"]["location"]
        right_loc = right["transform"]["location"]
        for axis in ("x", "y", "z"):
            max_location_delta = max(max_location_delta, abs(float(left_loc[axis]) - float(right_loc[axis])))
        left_yaw = float(left["transform"]["rotation"]["yaw"])
        right_yaw = float(right["transform"]["rotation"]["yaw"])
        yaw_delta = abs((left_yaw - right_yaw + 180.0) % 360.0 - 180.0)
        max_yaw_delta = max(max_yaw_delta, yaw_delta)
    return max_location_delta, max_yaw_delta


def _validate_poly_ids(quads: List[Dict]) -> List[str]:
    errors = []
    poly_ids = [q.get("polyId") for q in quads]
    expected = list(range(len(quads)))
    if poly_ids != expected:
        errors.append("quad polyId values are not consecutive from 0")
    return errors


def _validate_quad_geometry(quads: List[Dict]) -> List[str]:
    errors = []
    for quad in quads:
        verts = quad.get("vertices", [])
        if len(verts) != 4:
            errors.append(f"quad {quad.get('polyId')} does not have 4 vertices")
            continue
        xy = [(float(v["x"]), float(v["y"])) for v in verts]
        area2 = 0.0
        for i in range(4):
            x1, y1 = xy[i]
            x2, y2 = xy[(i + 1) % 4]
            area2 += x1 * y2 - x2 * y1
        if abs(area2) < 1e-6:
            errors.append(f"quad {quad.get('polyId')} has near-zero area")
    return errors


def validate_stage1_against_xodr(
    stage1_data: Dict,
    xodr_path: Path,
    map_name: str,
    quad_step: float,
    waypoint_step: float,
    tolerance: float = 1e-7,
) -> Dict:
    expected = convert_xodr_to_stage1(xodr_path, map_name, quad_step, waypoint_step)
    errors: List[str] = []

    for key in ("map_name",):
        if stage1_data.get(key) != expected.get(key):
            errors.append(f"{key} differs: {stage1_data.get(key)!r} != {expected.get(key)!r}")

    for key in ("quads", "driving_waypoints", "traffic_controls"):
        actual_len = len(stage1_data.get(key, []))
        expected_len = len(expected.get(key, []))
        if actual_len != expected_len:
            errors.append(f"{key} count differs: {actual_len} != {expected_len}")

    errors.extend(_validate_poly_ids(stage1_data.get("quads", [])))
    errors.extend(_validate_quad_geometry(stage1_data.get("quads", [])))

    max_vertex_delta = float("inf")
    if len(stage1_data.get("quads", [])) == len(expected.get("quads", [])):
        max_vertex_delta = _max_vertex_delta(stage1_data["quads"], expected["quads"])
        if max_vertex_delta > tolerance:
            errors.append(f"quad vertex max delta {max_vertex_delta:.3g} exceeds tolerance {tolerance:.3g}")

    max_wp_location_delta = float("inf")
    max_wp_yaw_delta = float("inf")
    if len(stage1_data.get("driving_waypoints", [])) == len(expected.get("driving_waypoints", [])):
        max_wp_location_delta, max_wp_yaw_delta = _max_waypoint_delta(
            stage1_data["driving_waypoints"], expected["driving_waypoints"]
        )
        if max_wp_location_delta > tolerance:
            errors.append(
                f"waypoint location max delta {max_wp_location_delta:.3g} exceeds tolerance {tolerance:.3g}"
            )
        if max_wp_yaw_delta > 1e-6:
            errors.append(f"waypoint yaw max delta {max_wp_yaw_delta:.3g} exceeds tolerance 1e-6")

    report = {
        "ok": not errors,
        "errors": errors,
        "quads": len(stage1_data.get("quads", [])),
        "driving_waypoints": len(stage1_data.get("driving_waypoints", [])),
        "traffic_controls": len(stage1_data.get("traffic_controls", [])),
        "max_vertex_delta": max_vertex_delta,
        "max_waypoint_location_delta": max_wp_location_delta,
        "max_waypoint_yaw_delta": max_wp_yaw_delta,
    }
    return report


def print_validation_report(map_name: str, report: Dict) -> None:
    status = "OK" if report["ok"] else "FAILED"
    print(
        f"Validation {status} for {map_name}: "
        f"quads={report['quads']}, driving_waypoints={report['driving_waypoints']}, "
        f"traffic_controls={report['traffic_controls']}, "
        f"max_vertex_delta={report['max_vertex_delta']:.3g}, "
        f"max_waypoint_location_delta={report['max_waypoint_location_delta']:.3g}, "
        f"max_waypoint_yaw_delta={report['max_waypoint_yaw_delta']:.3g}"
    )
    for error in report["errors"]:
        print(f"  - {error}")


def run_preprocessor(stage1_path: Path, processed_path: Path) -> None:
    from preprocessor import preprocess_map

    preprocess_map(str(stage1_path), str(processed_path))


def run_cross_data_generation(processed_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualize_all_dijkstra_and_save_cross_info import visualize_lane_paths

    original_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        visualize_lane_paths(str(processed_path))
    finally:
        plt.show = original_show
        plt.close("all")

    cross_path = processed_path.with_name(f"cross_data_{processed_path.name}")
    if not cross_path.exists():
        raise FileNotFoundError(f"cross data was not generated: {cross_path}")
    return cross_path


def infer_map_name(xodr_path: Path, explicit_name: Optional[str]) -> str:
    if explicit_name:
        return explicit_name
    return xodr_path.stem


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an OpenDRIVE .xodr map into Selfrace training map JSON files."
    )
    parser.add_argument("--xodr", type=Path, help="Input .xodr file path, for example maps/xodr/Town01.xodr")
    parser.add_argument("--all", action="store_true", help="Convert every Town*.xodr in maps/xodr. Implies stage1 generation per town.")
    parser.add_argument("--map-name", default=None, help="Map name used in output filenames. Defaults to the xodr stem.")
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR, help="Directory for generated JSON files. Defaults to maps/.")
    parser.add_argument("--quad-step", type=float, default=1.0, help="Longitudinal quad sampling step in meters.")
    parser.add_argument("--waypoint-step", type=float, default=2.0, help="Raw driving waypoint sampling step in meters.")
    parser.add_argument("--stage1-only", action="store_true", help="Only write carla_map_data_*_stitched.json and skip preprocessing.")
    parser.add_argument("--skip-cross-data", action="store_true", help="Skip cross_data_processed_map_* generation.")
    parser.add_argument("--validate", action="store_true", default=True, help="Validate generated stage1 JSON against the source xodr. Enabled by default.")
    parser.add_argument("--no-validate", action="store_false", dest="validate", help="Skip stage1 validation.")
    parser.add_argument("--indent", type=int, default=None, help="Optional JSON indentation for the stage1 file.")
    return parser


def convert_one(args: argparse.Namespace, xodr_path: Path, map_name: str) -> Path:
    output_dir = args.output_dir.resolve()
    stage1_path = output_dir / f"carla_map_data_{map_name}_stitched.json"
    processed_path = output_dir / f"processed_map_{map_name}_stitched.json"

    print(f"Converting OpenDRIVE: {xodr_path}")
    stage1 = convert_xodr_to_stage1(xodr_path, map_name, args.quad_step, args.waypoint_step)
    if not stage1["quads"]:
        raise RuntimeError(f"no driving lane quads were generated for {map_name}")
    if not stage1["driving_waypoints"]:
        raise RuntimeError(f"no driving waypoints were generated for {map_name}")
    save_json(stage1_path, stage1, indent=args.indent)
    print(f"Wrote stage1 map: {stage1_path}")
    print(f"  quads={len(stage1['quads'])}, driving_waypoints={len(stage1['driving_waypoints'])}, traffic_controls=0")

    if args.validate:
        saved_stage1 = load_json(stage1_path)
        report = validate_stage1_against_xodr(saved_stage1, xodr_path, map_name, args.quad_step, args.waypoint_step)
        print_validation_report(map_name, report)
        if not report["ok"]:
            raise RuntimeError(f"stage1 validation failed for {map_name}")

    if args.stage1_only:
        return stage1_path

    run_preprocessor(stage1_path, processed_path)
    if not args.skip_cross_data:
        cross_path = run_cross_data_generation(processed_path)
        print(f"Wrote cross data: {cross_path}")

    print(f"Done. Training map: {processed_path}")
    return stage1_path


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.all:
        xodr_paths = sorted((THIS_DIR / "xodr").glob("Town*.xodr"))
        if args.map_name:
            raise ValueError("--map-name cannot be used with --all")
        if not xodr_paths:
            raise FileNotFoundError(f"no Town*.xodr files found in {THIS_DIR / 'xodr'}")
    else:
        if args.xodr is None:
            raise ValueError("--xodr is required unless --all is used")
        xodr_path = args.xodr.resolve()
        if not xodr_path.exists():
            raise FileNotFoundError(f"input xodr not found: {xodr_path}")
        xodr_paths = [xodr_path]
    if args.quad_step <= 0 or args.waypoint_step <= 0:
        raise ValueError("--quad-step and --waypoint-step must be positive")

    generated = []
    for xodr_path in xodr_paths:
        map_name = infer_map_name(xodr_path, args.map_name)
        generated.append(convert_one(args, xodr_path.resolve(), map_name))
    if len(generated) > 1:
        print("Generated stage1 files:")
        for path in generated:
            print(f"  {path}")


if __name__ == "__main__":
    main()
