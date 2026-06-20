"""
train.py — AI Training Pipeline for TORCS Corkscrew Track
==========================================================
AI controls: steer, accel, brake (full control)
Code controls: gears only

Usage:
  python train.py collect     # Phase 1: collect data
  python train.py bc          # Phase 2: Behavioral Cloning
  python train.py play        # Run trained model
"""

import numpy as np
import os
import json
from datetime import datetime

import sys
COMMAND_ARGS = sys.argv[1:]
COMMAND = COMMAND_ARGS[0] if COMMAND_ARGS else None
COMMAND_FLAGS = set(COMMAND_ARGS[1:])
sys.argv = [sys.argv[0]]


def get_int_flag(name, default):
    args = COMMAND_ARGS[1:]
    prefix = name + "="
    for i, arg in enumerate(args):
        if arg.startswith(prefix):
            try:
                return int(arg[len(prefix):])
            except ValueError:
                return default
        if arg == name and i + 1 < len(args):
            try:
                return int(args[i + 1])
            except ValueError:
                return default
    return default

import snakeoil3_gym as snakeoil3

PI = 3.14159265359

# ============================================================
# CONFIG
# ============================================================
STATE_DIM     = 17
LEGACY_STATE_DIM = 20
ACTION_DIM    = 3       # steer, accel, brake
LAP_THRESHOLD = 3600
DATA_FILE     = "driving_data.json"
GUARDIAN_FILE = "guardian_data.json"
HUMAN_ENVELOPE_FILE = "human_racing_envelope.json"
CORRECTION_FILE = "correction_data.json"
CORRECTION_PENDING_FILE = "correction_data_toCombine.json"
CORRECTION_REPEAT = 5
GUARDIAN_REPEAT = 3
PLAY_LOG_DIR  = "play_logs"
PORT          = 3001


# First-corner safety profile for online testing.
# Values are intentionally conservative: the model can still steer, but it is
# not allowed to arrive at Corkscrew's first hard corner far faster than the
# human dataset distribution.
FIRST_CORNER_SPEED_PROFILE = [
    (300.0, 232.0),
    (330.0, 225.0),
    (350.0, 210.0),
    (370.0, 180.0),
    (390.0, 145.0),
    (410.0, 115.0),
    (440.0, 105.0),
    (500.0, 115.0),
]

HYBRID_LAUNCH_END = 230.0
HYBRID_FIRST_CORNER_START = 260.0
HYBRID_FIRST_CORNER_END = 535.0
HYBRID_FIRST_CORNER_SPEED_PROFILE = [
    (220.0, 232.0),
    (260.0, 226.0),
    (300.0, 214.0),
    (330.0, 198.0),
    (360.0, 168.0),
    (390.0, 138.0),
    (410.0, 116.0),
    (430.0, 104.0),
    (455.0, 98.0),
    (480.0, 108.0),
    (510.0, 125.0),
    (535.0, 142.0),
]
HYBRID_FIRST_CORNER_LANE_PROFILE = [
    (180.0, 0.55),
    (220.0, 0.55),
    (240.0, 0.36),
    (260.0, 0.05),
    (285.0, -0.34),
    (320.0, -0.55),
    (360.0, -0.56),
    (390.0, -0.45),
    (415.0, 0.00),
    (440.0, 0.32),
    (460.0, 0.42),
    (500.0, 0.38),
    (535.0, 0.16),
]

SECOND_SECTOR_SPEED_PROFILE = [
    (680.0, 185.0),
    (700.0, 175.0),
    (725.0, 160.0),
    (750.0, 145.0),
    (780.0, 128.0),
    (820.0, 135.0),
    (850.0, 155.0),
]

SECOND_SECTOR_LANE_PROFILE = [
    (700.0, 0.35),
    (725.0, 0.20),
    (750.0, -0.20),
    (780.0, -0.45),
    (805.0, -0.20),
    (835.0, 0.20),
    (850.0, 0.35),
]

FIRST_CORNER_LANE_PROFILE = [
    (220.0, 0.55),
    (240.0, 0.35),
    (260.0, -0.05),
    (280.0, -0.45),
    (320.0, -0.56),
    (360.0, -0.56),
    (390.0, -0.50),
    (420.0, 0.10),
    (440.0, 0.35),
    (460.0, 0.38),
    (480.0, 0.45),
    (500.0, 0.42),
    (520.0, 0.20),
]

LAUNCH_LANE_CORRIDOR = [
    # Dist, lower trackPos, upper trackPos. Bounds are based on the cleaned
    # human launch data; the helper only acts outside this corridor.
    (0.0, 0.28, 0.38),
    (20.0, 0.00, 0.34),
    (40.0, -0.32, 0.26),
    (60.0, -0.48, 0.16),
    (80.0, -0.54, 0.11),
    (100.0, -0.58, 0.07),
    (120.0, -0.60, 0.15),
    (140.0, -0.54, 0.25),
    (160.0, -0.34, 0.46),
    (180.0, -0.08, 0.68),
    (200.0, 0.30, 0.78),
    (220.0, 0.05, 0.68),
    (240.0, -0.08, 0.55),
]

THIRD_SECTOR_SPEED_PROFILE = [
    (1420.0, 215.0),
    (1450.0, 190.0),
    (1480.0, 165.0),
    (1510.0, 155.0),
    (1540.0, 148.0),
    (1580.0, 142.0),
    (1620.0, 152.0),
    (1660.0, 168.0),
]

THIRD_SECTOR_LANE_PROFILE = [
    (1450.0, -0.60),
    (1480.0, -0.17),
    (1510.0, 0.38),
    (1540.0, 0.34),
    (1570.0, 0.12),
    (1600.0, -0.18),
    (1630.0, -0.50),
    (1660.0, -0.50),
]

CORKSCREW_SPEED_PROFILE = [
    (2250.0, 270.0),
    (2300.0, 250.0),
    (2330.0, 225.0),
    (2360.0, 195.0),
    (2390.0, 160.0),
    (2420.0, 130.0),
    (2450.0, 108.0),
    (2500.0, 100.0),
    (2550.0, 120.0),
]

GUARDIAN_FIRST_CORNER_SPEED_PROFILE = [
    (300.0, 242.0),
    (330.0, 232.0),
    (350.0, 218.0),
    (370.0, 196.0),
    (390.0, 166.0),
    (410.0, 138.0),
    (435.0, 118.0),
    (470.0, 118.0),
    (500.0, 132.0),
]

BASE_STEER_RATE = 0.035
LAUNCH_STEER_RATE = 0.075
GLOBAL_ASSIST_STEER_RATE = 0.085
SAFETY_STEER_RATE = 0.060
RECOVERY_STEER_RATE = 0.120
FIRST_CORNER_LANE_GAIN = 0.42
FIRST_CORNER_ANGLE_GAIN = 0.12
FIRST_CORNER_LANE_LIMIT = 0.55
FIRST_CORNER_LANE_MARGIN = 0.28
SECOND_SECTOR_LANE_GAIN = 0.50
SECOND_SECTOR_ANGLE_GAIN = 0.15
SECOND_SECTOR_LANE_LIMIT = 0.60
SECOND_SECTOR_LEFT_MARGIN = 0.20
SECOND_SECTOR_RIGHT_MARGIN = 0.38
SECOND_SECTOR_HEADING_EDGE = 0.72
THIRD_SECTOR_LANE_GAIN = 0.46
THIRD_SECTOR_ANGLE_GAIN = 0.16
THIRD_SECTOR_LANE_LIMIT = 0.55
THIRD_SECTOR_LANE_MARGIN = 0.36
GLOBAL_ASSIST_CENTER_START = 0.75
GLOBAL_ASSIST_EDGE_START = 0.90
GLOBAL_ASSIST_OFFTRACK = 1.0
GUARDIAN_FORCE_FIELD_TRACK = 0.82
GUARDIAN_WARN_TRACK = 0.88
GUARDIAN_EDGE_TRACK = 0.94
GUARDIAN_ABORT_TRACK = 1.02
GUARDIAN_HEADING_ANGLE = 0.50
GUARDIAN_LAUNCH_SUPPRESS_DIST = 180.0
GUARDIAN_SPEED_SUPPRESS_DIST = 180.0
GUARDIAN_LIFT_OVERSPEED = 18.0
GUARDIAN_STRONG_LIFT_OVERSPEED = 35.0
GUARDIAN_RACING_LINE_TRACK = 0.72
GUARDIAN_RACING_LINE_ANGLE = 0.24
GUARDIAN_RACING_LINE_SPEED_Y = 9.0
GUARDIAN_RACING_LINE_TARGET_BONUS = 28.0
GUARDIAN_STEER_RATE = 0.115
GUARDIAN_HARD_STEER_RATE = 0.180
RACING_LINE_GUARDIAN_EMERGENCY_STEER_RATE = 0.75
HYBRID_STEER_RATE = 0.24
HYBRID_HARD_STEER_RATE = 0.62
SIMPLE_TRAJECTORY_MIN_DIST = 900.0
RACING_LINE_GUARDIAN_EMERGENCY_REASONS = (
    "rlg_offtrack_abort",
    "rlg_simple_edge",
    "rlg_simple_projected_edge",
    "rlg_simple_yaw",
    "rlg_simple_slip",
    "rlg_simple_speed",
    "rlg_simple_brake",
    "rlg_simple_low_speed_recovery",
    "rlg_simple_human_trajectory",
    "rlg_edge_veto",
    "rlg_edge_shield",
    "rlg_edge_shield_cap",
    "rlg_edge_recovery_cap",
    "rlg_edge_emergency_brake",
    "rlg_first_corner_entry_guard",
    "rlg_first_corner_entry_brake",
    "rlg_post_apex_cross_damper",
    "rlg_post_apex_cross_brake",
    "rlg_post_apex_left_guard",
    "rlg_post_apex_stability_brake",
    "rlg_slip_veto",
)
GUARDIAN_BRAKE_DECEL_MS2 = 13.0
GUARDIAN_FIRST_CORNER_TRAJECTORY_START = 350.0
GUARDIAN_FIRST_CORNER_TRAJECTORY_END = 455.0
GUARDIAN_FIRST_CORNER_TRAJECTORY_MARGIN = 0.16
GUARDIAN_FIRST_CORNER_TRAJECTORY_STRONG_MARGIN = 0.30
GUARDIAN_FIRST_CORNER_TRAJECTORY_STEER_GAIN = 0.74
GUARDIAN_FIRST_CORNER_TRAJECTORY_SPEED_Y_GAIN = 0.035
GUARDIAN_FIRST_CORNER_TRAJECTORY_ANGLE_GAIN = 0.16
GUARDIAN_FIRST_CORNER_TRAJECTORY_STEER_LIMIT = 0.78
GUARDIAN_FIRST_CORNER_LATE_STEER_START = 420.0
GUARDIAN_FIRST_CORNER_LATE_STEER_CAP = 0.58
GUARDIAN_FIRST_CORNER_LATE_SLIP_STEER_CAP = 0.48
HUMAN_ENVELOPE_VERSION = 3
HUMAN_ENVELOPE_BIN_M = 10.0
HUMAN_ENVELOPE_MIN_SAMPLES = 40
HUMAN_ENVELOPE_FIRST_CORNER_START = 300.0
HUMAN_ENVELOPE_FIRST_CORNER_END = 500.0
HUMAN_ENVELOPE_ASSIST_START_DIST = 180.0
HUMAN_ENVELOPE_TRACK_MARGIN = 0.12
HUMAN_ENVELOPE_ANGLE_MARGIN = 0.08
HUMAN_ENVELOPE_SPEED_Y_MARGIN = 3.0
HUMAN_ENVELOPE_SPEED_SOFT_MARGIN = 5.0
HUMAN_ENVELOPE_SPEED_HARD_MARGIN = 15.0
HUMAN_ENVELOPE_STEER_GAIN = 0.52
HUMAN_ENVELOPE_SPEED_Y_GAIN = 0.030
HUMAN_ENVELOPE_ANGLE_GAIN = 0.10
HUMAN_ENVELOPE_SPEED_TRUST_TRACK = 0.94
HUMAN_ENVELOPE_MAJOR_ANGLE = 0.35
HUMAN_ENVELOPE_MAJOR_SPEED_Y = 12.0
HUMAN_ACTION_STEER_MARGIN = 0.10
HUMAN_ACTION_BRAKE_MARGIN = 0.08
HUMAN_ACTION_ACCEL_MARGIN = 0.12
LAUNCH_TRACTION_DIST = 125.0
LAUNCH_TRACTION_SPEED = 145.0
LAUNCH_TRACTION_ACCEL_PROFILE = [
    (0.0, 0.42),
    (20.0, 0.55),
    (45.0, 0.70),
    (80.0, 0.84),
    (125.0, 0.96),
]


def _interp_profile(points, x):
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            t = (x - x0) / max(1e-6, x1 - x0)
            return y0 + t * (y1 - y0)
    return points[-1][1]


def _interp_corridor(points, x):
    if x <= points[0][0]:
        return points[0][1], points[0][2]
    for p0, p1 in zip(points, points[1:]):
        x0, lo0, hi0 = p0
        x1, lo1, hi1 = p1
        if x <= x1:
            t = (x - x0) / max(1e-6, x1 - x0)
            return lo0 + t * (lo1 - lo0), hi0 + t * (hi1 - hi0)
    return points[-1][1], points[-1][2]


def _apply_speed_profile(info, profile_name, profile, lap_dist, speed_x, accel_cmd, brake_cmd):
    if lap_dist < profile[0][0] or lap_dist > profile[-1][0]:
        return accel_cmd, brake_cmd

    target_speed = _interp_profile(profile, lap_dist)
    overspeed = speed_x - target_speed
    info["speed_profile"] = profile_name
    info["target_speed"] = target_speed
    info["overspeed"] = overspeed

    if overspeed > -3.0:
        accel_cmd = min(accel_cmd, 0.05)
        info["active"] = True
        info["reasons"].append(profile_name + "_coast")
    if overspeed > 0.0:
        brake_needed = min(1.0, 0.18 + overspeed / 55.0)
        brake_cmd = max(brake_cmd, brake_needed)
        accel_cmd = 0.0
        info["active"] = True
        info["reasons"].append(profile_name + "_brake")

    return accel_cmd, brake_cmd


def _track_value(track, idx, default=100.0):
    if not track or idx < 0 or idx >= len(track):
        return default
    value = float(track[idx])
    if not np.isfinite(value) or value <= 0.0:
        return default
    return value


def _track_min(track, indexes, default=100.0):
    values = [_track_value(track, idx, default) for idx in indexes]
    return min(values) if values else default


def _target_speed_from_track(track):
    if not track:
        return None, None

    front = _track_value(track, 9, 100.0)
    center_clear = _track_min(track, range(7, 12), front)
    wide_clear = _track_min(track, range(5, 14), front)
    lookahead_clear = min(front, center_clear * 1.15, wide_clear * 1.35)

    if lookahead_clear < 30.0:
        target_speed = 105.0
    elif lookahead_clear < 42.0:
        target_speed = 130.0
    elif lookahead_clear < 58.0:
        target_speed = 155.0
    elif lookahead_clear < 78.0:
        target_speed = 185.0
    elif lookahead_clear < 105.0:
        target_speed = 220.0
    elif lookahead_clear < 140.0:
        target_speed = 245.0
    else:
        target_speed = 260.0

    return target_speed, lookahead_clear


def _guardian_target_speed_from_track(track):
    if not track:
        return None, None

    front = _track_value(track, 9, 100.0)
    center_clear = _track_min(track, range(7, 12), front)
    wide_clear = _track_min(track, range(5, 14), front)
    lookahead_clear = min(front, center_clear * 1.10, wide_clear * 1.25)

    if lookahead_clear > 180.0:
        target_speed = 285.0
    elif lookahead_clear > 150.0:
        target_speed = 265.0
    elif lookahead_clear > 120.0:
        target_speed = 240.0
    elif lookahead_clear > 100.0:
        target_speed = 215.0
    elif lookahead_clear > 80.0:
        target_speed = 185.0
    elif lookahead_clear > 60.0:
        target_speed = 155.0
    elif lookahead_clear > 50.0:
        target_speed = 138.0
    elif lookahead_clear > 40.0:
        target_speed = 120.0
    elif lookahead_clear > 30.0:
        target_speed = 102.0
    elif lookahead_clear > 22.0:
        target_speed = 88.0
    elif lookahead_clear > 16.0:
        target_speed = 72.0
    else:
        target_speed = 58.0

    return target_speed, lookahead_clear


def _simple_supervisor_target_speed(clear):
    if clear is None:
        return 155.0
    if clear > 150.0:
        return 255.0
    if clear > 115.0:
        return 225.0
    if clear > 90.0:
        return 195.0
    if clear > 70.0:
        return 165.0
    if clear > 55.0:
        return 135.0
    if clear > 42.0:
        return 112.0
    if clear > 32.0:
        return 88.0
    if clear > 24.0:
        return 68.0
    return 48.0


def apply_rule_based_track_supervisor(
        dist_ep, speed_x, speed_y, track_pos, angle, track,
        raw_steer, accel_cmd, brake_cmd, model_action=None,
        human_envelope=None):
    lap_dist = dist_ep % LAP_THRESHOLD if dist_ep >= 0.0 else 0.0
    abs_track_pos = abs(track_pos)
    abs_angle = abs(angle)
    abs_speed_y = abs(speed_y)

    front = _track_value(track, 9, 100.0)
    center_clear = _track_min(track, range(7, 12), front)
    wide_clear = _track_min(track, range(5, 14), front)
    clear = min(front, center_clear * 1.08, wide_clear * 1.28)

    speed_factor = float(np.clip(speed_x / 100.0, 0.45, 1.35))
    yaw_factor = float(np.clip(speed_x / 170.0, 0.25, 1.05))
    predicted_track_pos = (
        track_pos +
        (speed_y / 35.0) * speed_factor +
        angle * yaw_factor * 0.26)
    outward_speed = max(0.0, track_pos * speed_y)

    human_env = _human_envelope_at(human_envelope, lap_dist)
    future_horizon_m = float(np.clip((max(0.0, speed_x) / 3.6) * 0.70, 18.0, 42.0))
    future_human_env = _human_envelope_at(human_envelope, lap_dist + future_horizon_m)
    human_line_ok = False
    human_near_line = False
    human_future_near_line = False
    human_future_error = 0.0
    human_future_violation = 0.0
    human_trajectory_risk = 0.0
    human_trajectory_steer = raw_steer
    if human_env is not None:
        track_low = human_env["track_low"]
        track_high = human_env["track_high"]
        human_track_ok = track_low - 0.08 <= track_pos <= track_high + 0.08
        human_predicted_ok = track_low - 0.14 <= predicted_track_pos <= track_high + 0.14
        human_angle_ok = human_env["angle_low"] - 0.11 <= angle <= human_env["angle_high"] + 0.11
        human_speed_y_ok = human_env["speed_y_low"] - 3.2 <= speed_y <= human_env["speed_y_high"] + 3.2
        human_near_line = human_track_ok and human_predicted_ok
        human_line_ok = human_near_line and human_angle_ok and human_speed_y_ok
    if future_human_env is not None:
        future_low = future_human_env["track_low"]
        future_high = future_human_env["track_high"]
        future_target = future_human_env["track_target"]
        human_future_error = future_target - predicted_track_pos
        if predicted_track_pos < future_low - 0.10:
            human_future_violation = predicted_track_pos - (future_low - 0.10)
        elif predicted_track_pos > future_high + 0.10:
            human_future_violation = predicted_track_pos - (future_high + 0.10)
        human_future_near_line = abs(human_future_violation) < 0.10
        transition_delta = 0.0
        if human_env is not None:
            transition_delta = future_human_env["track_target"] - human_env["track_target"]
        human_trajectory_risk = max(
            0.0,
            abs(human_future_violation) / 0.36,
            max(0.0, abs(human_future_error) - 0.28) / 0.42 if abs(transition_delta) > 0.18 else 0.0)
        if human_trajectory_risk > 0.0:
            human_trajectory_steer = float(np.clip(
                future_human_env["steer_p50"] +
                human_future_error * 0.48 -
                speed_y * 0.012 +
                angle * 0.28,
                -0.92, 0.92))

    sensor_target_speed = _simple_supervisor_target_speed(clear)
    target_speed = sensor_target_speed
    if human_env is not None:
        profile_speed_p50 = human_env["speed_p50"]
        profile_speed_hard = human_env["speed_hard"]
        if future_human_env is not None:
            profile_speed_p50 = min(profile_speed_p50, future_human_env["speed_p50"])
            profile_speed_hard = min(profile_speed_hard, future_human_env["speed_hard"])
        if human_line_ok:
            target_speed = max(target_speed, max(55.0, profile_speed_p50 - 12.0))
        elif (human_near_line or human_future_near_line) and abs_angle < 0.55 and abs_speed_y < 8.0:
            target_speed = max(target_speed, max(50.0, profile_speed_p50 - 34.0))
        elif human_trajectory_risk > 0.0 and abs_angle < 0.62 and abs_speed_y < 9.5:
            target_speed = max(target_speed, max(50.0, profile_speed_p50 - 46.0))
        target_speed = min(target_speed, profile_speed_hard + 10.0)

    edge_speed_cap_needed = (
        not human_line_ok or
        abs_track_pos > 0.94 or
        abs(predicted_track_pos) > 0.98 or
        outward_speed > 2.8)
    if abs_track_pos > 0.66 and edge_speed_cap_needed:
        edge_cap = float(np.interp(abs_track_pos, [0.58, 0.74, 0.88, 0.98], [150.0, 112.0, 72.0, 42.0]))
        target_speed = min(target_speed, edge_cap)
    projected_speed_cap_needed = (
        not human_near_line or
        abs(predicted_track_pos) > 0.96 or
        outward_speed > 2.2)
    if abs(predicted_track_pos) > 0.82 and projected_speed_cap_needed:
        projected_cap = float(np.interp(min(abs(predicted_track_pos), 1.05), [0.78, 0.90, 1.05], [105.0, 68.0, 38.0]))
        target_speed = min(target_speed, projected_cap)
    yaw_cap_start = 0.48 if human_line_ok else 0.38
    if abs_angle > yaw_cap_start:
        yaw_cap = float(np.interp(min(abs_angle, 0.82), [yaw_cap_start, 0.58, 0.82], [132.0, 88.0, 52.0]))
        target_speed = min(target_speed, yaw_cap)
    slip_cap_start = 7.5 if human_line_ok else 5.6
    if abs_speed_y > slip_cap_start:
        slip_cap = float(np.interp(min(abs_speed_y, 14.0), [slip_cap_start, 9.5, 14.0], [132.0, 92.0, 52.0]))
        target_speed = min(target_speed, slip_cap)

    overspeed = speed_x - target_speed
    edge_risk_start = 0.86 if human_line_ok else 0.70
    projected_risk_start = 0.92 if human_near_line else 0.78
    yaw_risk_start = 0.46 if human_line_ok else 0.36
    slip_risk_start = 7.2 if human_line_ok else 5.6
    edge_risk = max(0.0, (abs_track_pos - edge_risk_start) / max(1e-6, 0.98 - edge_risk_start))
    projected_edge_risk = max(0.0, (abs(predicted_track_pos) - projected_risk_start) / max(1e-6, 1.02 - projected_risk_start))
    yaw_risk = max(0.0, (abs_angle - yaw_risk_start) / 0.26)
    slip_risk = max(0.0, (abs_speed_y - slip_risk_start) / 7.0)
    outward_risk = max(0.0, (outward_speed - 0.6) / 5.0)
    speed_risk = max(0.0, overspeed / 38.0)
    risk_score = max(
        edge_risk * 1.35,
        projected_edge_risk * 1.45,
        yaw_risk * 1.10,
        slip_risk,
        outward_risk * 1.20,
        speed_risk)

    edge_active = (
        abs_track_pos > (0.90 if human_line_ok else 0.78) or
        abs(predicted_track_pos) > (0.96 if human_near_line else 0.86) or
        (abs_track_pos > 0.50 and outward_speed > 1.2))
    human_trajectory_active = (
        speed_x > 55.0 and
        lap_dist >= SIMPLE_TRAJECTORY_MIN_DIST and
        future_human_env is not None and
        human_trajectory_risk > 0.22 and
        not edge_active)
    stability_active = (
        speed_x > 62.0 and
        (abs_angle > yaw_risk_start or abs_speed_y > slip_risk_start) and
        (abs_track_pos > 0.16 or clear < 75.0))
    speed_active = (
        speed_x > target_speed + (16.0 if human_line_ok else 10.0) and
        (clear < 80.0 or abs_track_pos > 0.54 or abs_angle > 0.32 or abs_speed_y > 5.0))
    low_speed_recovery = speed_x < 7.0 and 0.78 < abs_track_pos < 1.0
    abort = abs_track_pos >= 1.0

    info = {
        "active": False,
        "abort": abort,
        "mode": "PASSIVE",
        "previous_mode": "PASSIVE",
        "reasons": [],
        "lap_dist": lap_dist,
        "risk_score": risk_score,
        "risk": {
            "line_error": -track_pos,
            "predicted_track_pos": predicted_track_pos,
            "edge_risk": edge_risk,
            "projected_edge_risk": projected_edge_risk,
            "yaw_risk": yaw_risk,
            "slip_risk": slip_risk,
            "outward_risk": outward_risk,
            "speed_risk": speed_risk,
            "human_future_error": human_future_error,
            "human_future_violation": human_future_violation,
            "human_trajectory_risk": human_trajectory_risk,
            "outward_speed": outward_speed,
            "overspeed": overspeed,
        },
        "target": {
            "track": 0.0,
            "target_speed": target_speed,
            "sensor_target_speed": sensor_target_speed,
            "front_clear": front,
            "center_clear": center_clear,
            "wide_clear": wide_clear,
            "clear": clear,
            "human_speed_p50": human_env["speed_p50"] if human_env is not None else None,
            "human_speed_soft": human_env["speed_soft"] if human_env is not None else None,
            "human_speed_hard": human_env["speed_hard"] if human_env is not None else None,
            "human_track_low": human_env["track_low"] if human_env is not None else None,
            "human_track_high": human_env["track_high"] if human_env is not None else None,
            "human_line_ok": human_line_ok,
            "human_near_line": human_near_line,
            "human_future_near_line": human_future_near_line,
            "human_future_track_low": future_human_env["track_low"] if future_human_env is not None else None,
            "human_future_track_target": future_human_env["track_target"] if future_human_env is not None else None,
            "human_future_track_high": future_human_env["track_high"] if future_human_env is not None else None,
            "future_horizon_m": future_horizon_m,
        },
        "lookahead": {
            "predicted_track_pos": predicted_track_pos,
            "front": front,
            "center_clear": center_clear,
            "wide_clear": wide_clear,
        },
        "command_before": {
            "steer": raw_steer,
            "accel": accel_cmd,
            "brake": brake_cmd,
        },
        "model_action": model_action,
    }

    if abort:
        accel_cmd = 0.0
        if speed_x > 18.0:
            brake_cmd = max(brake_cmd, 0.35)
        _add_reason(info, "rlg_offtrack_abort")
        info["active"] = True
        info["mode"] = "ABORT"
        info["command_after"] = {"steer": raw_steer, "accel": accel_cmd, "brake": brake_cmd}
        return raw_steer, accel_cmd, brake_cmd, info

    if not (edge_active or human_trajectory_active or stability_active or speed_active or low_speed_recovery):
        info["command_after"] = {"steer": raw_steer, "accel": accel_cmd, "brake": brake_cmd}
        return raw_steer, accel_cmd, brake_cmd, info

    info["active"] = True
    if low_speed_recovery:
        info["mode"] = "SIMPLE_RECOVERY"
    elif edge_active:
        info["mode"] = "SIMPLE_EDGE"
    elif human_trajectory_active:
        info["mode"] = "SIMPLE_TRAJECTORY"
    elif stability_active:
        info["mode"] = "SIMPLE_STABILITY"
    else:
        info["mode"] = "SIMPLE_SPEED"

    safe_steer = float(np.clip(
        -track_pos * 0.75 -
        speed_y * 0.025 +
        angle * 1.10,
        -0.95, 0.95))
    if human_trajectory_active:
        safe_steer = human_trajectory_steer
        _add_reason(info, "rlg_simple_human_trajectory")

    if abs_track_pos > 0.72 or abs(predicted_track_pos) > 0.80:
        side = np.sign(track_pos if abs_track_pos > 0.38 else predicted_track_pos)
        if side == 0.0:
            side = 1.0
        edge_strength = float(np.clip(max(abs_track_pos, abs(predicted_track_pos)) - 0.70, 0.0, 0.30) / 0.30)
        edge_steer = float(np.clip(
            -side * (0.52 + 0.40 * edge_strength) -
            speed_y * 0.030 +
            angle * 0.42,
            -0.98, 0.98))
        if side > 0.0:
            safe_steer = min(safe_steer, edge_steer)
        else:
            safe_steer = max(safe_steer, edge_steer)
        _add_reason(info, "rlg_simple_edge" if abs_track_pos > 0.72 else "rlg_simple_projected_edge")

    if stability_active:
        _add_reason(info, "rlg_simple_yaw" if abs_angle > 0.36 else "rlg_simple_slip")

    if low_speed_recovery:
        safe_steer = float(np.clip(-np.sign(track_pos) * 0.82, -0.82, 0.82))
        raw_steer = safe_steer
        accel_cmd = max(accel_cmd, 0.28)
        brake_cmd = min(brake_cmd, 0.02)
        steer_blend = 1.0
        _add_reason(info, "rlg_simple_low_speed_recovery")
    else:
        steer_blend = 0.0
        if edge_active:
            steer_blend = float(np.clip(0.56 + risk_score * 0.18, 0.56, 0.88))
        elif human_trajectory_active:
            steer_blend = float(np.clip(0.24 + human_trajectory_risk * 0.12, 0.24, 0.48))
        elif stability_active:
            steer_blend = float(np.clip(0.18 + risk_score * 0.12, 0.18, 0.42))
        if steer_blend > 0.0:
            raw_steer = float(np.clip(raw_steer * (1.0 - steer_blend) + safe_steer * steer_blend, -0.95, 0.95))

        if speed_active:
            _add_reason(info, "rlg_simple_speed")
            accel_cmd = min(accel_cmd, 0.35 if human_line_ok and overspeed < 24.0 else (0.12 if overspeed < 18.0 else 0.0))

        brake_needed = 0.0
        if overspeed > (18.0 if human_line_ok else 10.0):
            brake_limit = 0.38 if human_line_ok and not (edge_active or stability_active) else 0.70
            brake_needed = max(brake_needed, min(brake_limit, 0.10 + overspeed / 110.0))
        if stability_active and speed_x > 70.0:
            brake_needed = max(brake_needed, min(0.76, 0.18 + yaw_risk * 0.22 + slip_risk * 0.18))
            accel_cmd = min(accel_cmd, 0.04)
        if human_trajectory_active and speed_x > target_speed + 12.0:
            brake_needed = max(brake_needed, min(0.42, 0.10 + max(0.0, overspeed) / 130.0))
            accel_cmd = min(accel_cmd, 0.16)
        if edge_active and speed_x > 38.0:
            brake_needed = max(brake_needed, min(0.92, 0.26 + edge_risk * 0.24 + projected_edge_risk * 0.28 + outward_risk * 0.20))
            accel_cmd = 0.0 if abs_track_pos > 0.82 or abs(predicted_track_pos) > 0.90 else min(accel_cmd, 0.08)
        if brake_needed > 0.0:
            brake_cmd = max(brake_cmd, brake_needed)
            _add_reason(info, "rlg_simple_brake")

    info["safe_steer"] = safe_steer
    info["human_trajectory_steer"] = human_trajectory_steer
    info["steer_blend"] = steer_blend
    info["speed_only"] = speed_active and not edge_active and not human_trajectory_active and not stability_active and not low_speed_recovery
    info["overspeed"] = overspeed
    info["target_speed"] = target_speed
    info["command_after"] = {"steer": raw_steer, "accel": accel_cmd, "brake": brake_cmd}
    return raw_steer, accel_cmd, brake_cmd, info


def apply_hybrid_assist(dist_ep, speed_x, speed_y, track_pos, angle,
                        raw_steer, accel_cmd, brake_cmd):
    lap_dist = dist_ep % LAP_THRESHOLD if dist_ep >= 0.0 else 0.0
    abs_track_pos = abs(track_pos)
    abs_angle = abs(angle)
    abs_speed_y = abs(speed_y)
    info = {
        "active": False,
        "abort": abs_track_pos >= 1.0,
        "mode": "PASSIVE",
        "sector": None,
        "reasons": [],
        "lap_dist": lap_dist,
        "target_speed": None,
        "overspeed": None,
        "target_track_pos": None,
        "lane_error": None,
        "lane_steer": None,
        "steer_blend": 0.0,
        "accel_before": accel_cmd,
        "brake_before": brake_cmd,
        "steer_before": raw_steer,
    }

    if info["abort"]:
        accel_cmd = 0.0
        if speed_x > 15.0:
            brake_cmd = max(brake_cmd, 0.35)
        _add_reason(info, "hybrid_offtrack_abort")
        info["active"] = True
        info["mode"] = "ABORT"
        info["accel_after"] = accel_cmd
        info["brake_after"] = brake_cmd
        info["steer_after"] = raw_steer
        return raw_steer, accel_cmd, brake_cmd, info

    if 0.0 <= lap_dist <= HYBRID_LAUNCH_END:
        lower, upper = _interp_corridor(LAUNCH_LANE_CORRIDOR, min(lap_dist, LAUNCH_LANE_CORRIDOR[-1][0]))
        center = (lower + upper) * 0.5
        lane_error = track_pos - center
        edge_error = max(0.0, abs_track_pos - 0.58)
        lane_steer = float(np.clip(
            -lane_error * 0.42 + angle * 0.18 - speed_y * 0.010,
            -0.55, 0.55))
        info["sector"] = "launch"
        info["target_track_pos"] = center
        info["lane_error"] = lane_error
        info["lane_steer"] = lane_steer
        info["lower_track_pos"] = lower
        info["upper_track_pos"] = upper

        traction_limit = _interp_profile(
            LAUNCH_TRACTION_ACCEL_PROFILE,
            max(0.0, min(lap_dist, LAUNCH_TRACTION_DIST)))
        if speed_x < LAUNCH_TRACTION_SPEED:
            accel_cmd = min(accel_cmd, traction_limit)
            info["active"] = True
            _add_reason(info, "hybrid_launch_traction")

        # The launch is not a racing-line follower. The raw BC model already
        # survives this part better than a narrow human corridor chase; only
        # intervene when the car is actually drifting toward a barrier.
        launch_risk = (
            abs_track_pos > 0.62 or
            (abs_track_pos > 0.50 and abs_angle > 0.20) or
            abs_speed_y > 8.0
        )
        if launch_risk:
            blend = float(np.clip(0.22 + edge_error * 1.2 + max(0.0, abs_angle - 0.18) * 0.6, 0.22, 0.72))
            raw_steer = float(np.clip(raw_steer * (1.0 - blend) + lane_steer * blend, -0.95, 0.95))
            accel_cmd = min(accel_cmd, 0.52 if abs_track_pos < 0.72 else 0.28)
            info["active"] = True
            info["mode"] = "HYBRID_LAUNCH"
            info["steer_blend"] = blend
            _add_reason(info, "hybrid_launch_lane")

        if abs_track_pos > 0.78:
            edge_steer = float(np.clip(
                -track_pos * 0.72 + angle * 0.16 - speed_y * 0.010,
                -0.88, 0.88))
            raw_steer = edge_steer
            accel_cmd = min(accel_cmd, 0.12)
            if speed_x > 55.0:
                brake_cmd = max(brake_cmd, 0.24)
            if abs_track_pos > 0.90:
                accel_cmd = 0.0
                brake_cmd = max(brake_cmd, 0.48)
            info["active"] = True
            info["mode"] = "HYBRID_LAUNCH_EDGE"
            info["edge_steer"] = edge_steer
            info["steer_blend"] = 1.0
            _add_reason(info, "hybrid_launch_edge")

        if abs_track_pos > 0.70 and speed_x > 65.0:
            accel_cmd = min(accel_cmd, 0.18)
            brake_cmd = max(brake_cmd, 0.18)
            info["active"] = True
            _add_reason(info, "hybrid_launch_barrier_brake")

    if HYBRID_FIRST_CORNER_START <= lap_dist <= HYBRID_FIRST_CORNER_END:
        target_speed = _interp_profile(HYBRID_FIRST_CORNER_SPEED_PROFILE, lap_dist)
        target_track = _interp_profile(HYBRID_FIRST_CORNER_LANE_PROFILE, lap_dist)
        lane_error = track_pos - target_track
        lane_steer = float(np.clip(
            lane_error * 0.74 - angle * 0.24 - speed_y * 0.020,
            -0.92, 0.92))
        overspeed = speed_x - target_speed
        line_risk = abs(lane_error) > 0.20 or abs_track_pos > 0.70 or abs_speed_y > 7.0 or abs_angle > 0.28

        info["sector"] = "first_corner"
        info["target_speed"] = target_speed
        info["target_track_pos"] = target_track
        info["lane_error"] = lane_error
        info["lane_steer"] = lane_steer
        info["overspeed"] = overspeed

        if overspeed > -4.0:
            accel_cmd = min(accel_cmd, 0.08)
            info["active"] = True
            info["mode"] = "HYBRID_FIRST_CORNER"
            _add_reason(info, "hybrid_first_corner_lift")
        if overspeed > 4.0:
            brake = min(0.95, 0.16 + overspeed / 72.0)
            if abs_speed_y > 9.0 or abs_track_pos > 0.80:
                brake = max(brake, 0.68)
            brake_cmd = max(brake_cmd, brake)
            accel_cmd = 0.0
            info["active"] = True
            info["mode"] = "HYBRID_FIRST_CORNER"
            _add_reason(info, "hybrid_first_corner_brake")

        if line_risk:
            blend = float(np.clip(0.36 + abs(lane_error) * 0.22 + abs_speed_y * 0.018, 0.36, 0.76))
            if abs_track_pos > 0.82:
                blend = max(blend, 0.82)
            raw_steer = float(np.clip(raw_steer * (1.0 - blend) + lane_steer * blend, -0.95, 0.95))
            accel_cmd = min(accel_cmd, 0.12)
            info["active"] = True
            info["mode"] = "HYBRID_FIRST_CORNER"
            info["steer_blend"] = blend
            _add_reason(info, "hybrid_first_corner_lane")

        if abs_track_pos > 0.86:
            edge_steer = float(np.clip(
                track_pos * 0.92 - angle * 0.18 - speed_y * 0.020,
                -0.98, 0.98))
            raw_steer = edge_steer
            accel_cmd = 0.0
            if speed_x > 70.0:
                brake_cmd = max(brake_cmd, 0.70)
            info["active"] = True
            info["mode"] = "HYBRID_FIRST_CORNER_EDGE"
            info["edge_steer"] = edge_steer
            info["steer_blend"] = 1.0
            _add_reason(info, "hybrid_first_corner_edge")

    info["accel_after"] = accel_cmd
    info["brake_after"] = brake_cmd
    info["steer_after"] = raw_steer
    return raw_steer, accel_cmd, brake_cmd, info


def _quantile(values, q):
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float32), q))


def _human_envelope_cache_is_fresh(envelope, data_file):
    if not isinstance(envelope, dict) or envelope.get("version") != HUMAN_ENVELOPE_VERSION:
        return False
    if not os.path.exists(data_file):
        return False
    try:
        return (
            envelope.get("source_file") == data_file and
            envelope.get("source_size") == os.path.getsize(data_file) and
            abs(float(envelope.get("source_mtime", -1.0)) - os.path.getmtime(data_file)) < 0.001
        )
    except (OSError, TypeError, ValueError):
        return False


def build_human_racing_envelope(data_file=DATA_FILE):
    if not os.path.exists(data_file):
        return None

    try:
        with open(data_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  Human envelope skipped: cannot read {data_file}: {e}")
        return None

    if not isinstance(raw_data, list) or not raw_data:
        return None

    bins = {}
    start = 0.0
    end = float(LAP_THRESHOLD)
    bin_m = HUMAN_ENVELOPE_BIN_M
    for sample in raw_data:
        if not isinstance(sample, dict):
            continue
        state = sample.get("state", [])
        action = sample.get("action", [])
        if len(state) < STATE_DIM or len(action) < ACTION_DIM:
            continue
        try:
            lap_pos = float(state[5]) % 1.0
            lap_dist = lap_pos * LAP_THRESHOLD
            if lap_dist < start - bin_m * 0.5 or lap_dist > end + bin_m * 0.5:
                continue
            bin_center = round(lap_dist / bin_m) * bin_m
            if bin_center >= LAP_THRESHOLD:
                bin_center = float(LAP_THRESHOLD)
            if bin_center < start or bin_center > end:
                continue
            entry = bins.setdefault(bin_center, {
                "speed_x": [],
                "track_pos": [],
                "angle": [],
                "speed_y": [],
                "steer": [],
                "accel": [],
                "brake": [],
            })
            entry["speed_x"].append(float(state[2]) * 300.0)
            entry["track_pos"].append(float(state[1]))
            entry["angle"].append(float(state[0]) * PI)
            entry["speed_y"].append(float(state[3]) * 300.0)
            entry["steer"].append(float(action[0]))
            entry["accel"].append(float(action[1]))
            entry["brake"].append(float(action[2]))
        except (TypeError, ValueError, IndexError):
            continue

    points = []
    for dist in sorted(bins):
        entry = bins[dist]
        n = len(entry["speed_x"])
        if n < HUMAN_ENVELOPE_MIN_SAMPLES:
            continue
        track_p10 = _quantile(entry["track_pos"], 0.10)
        track_p50 = _quantile(entry["track_pos"], 0.50)
        track_p90 = _quantile(entry["track_pos"], 0.90)
        angle_p10 = _quantile(entry["angle"], 0.10)
        angle_p50 = _quantile(entry["angle"], 0.50)
        angle_p90 = _quantile(entry["angle"], 0.90)
        speed_y_p10 = _quantile(entry["speed_y"], 0.10)
        speed_y_p50 = _quantile(entry["speed_y"], 0.50)
        speed_y_p90 = _quantile(entry["speed_y"], 0.90)
        speed_p50 = _quantile(entry["speed_x"], 0.50)
        speed_p90 = _quantile(entry["speed_x"], 0.90)
        speed_p95 = _quantile(entry["speed_x"], 0.95)
        steer_p05 = _quantile(entry["steer"], 0.05)
        steer_p10 = _quantile(entry["steer"], 0.10)
        steer_p50 = _quantile(entry["steer"], 0.50)
        steer_p90 = _quantile(entry["steer"], 0.90)
        steer_p95 = _quantile(entry["steer"], 0.95)
        accel_p10 = _quantile(entry["accel"], 0.10)
        accel_p50 = _quantile(entry["accel"], 0.50)
        accel_p90 = _quantile(entry["accel"], 0.90)
        brake_p50 = _quantile(entry["brake"], 0.50)
        brake_p90 = _quantile(entry["brake"], 0.90)
        brake_p95 = _quantile(entry["brake"], 0.95)
        brake_p99 = _quantile(entry["brake"], 0.99)

        points.append({
            "dist": float(dist),
            "samples": int(n),
            "track_target": track_p50,
            "track_low": max(-0.98, track_p10 - HUMAN_ENVELOPE_TRACK_MARGIN),
            "track_high": min(0.98, track_p90 + HUMAN_ENVELOPE_TRACK_MARGIN),
            "angle_target": angle_p50,
            "angle_low": angle_p10 - HUMAN_ENVELOPE_ANGLE_MARGIN,
            "angle_high": angle_p90 + HUMAN_ENVELOPE_ANGLE_MARGIN,
            "speed_y_target": speed_y_p50,
            "speed_y_low": speed_y_p10 - HUMAN_ENVELOPE_SPEED_Y_MARGIN,
            "speed_y_high": speed_y_p90 + HUMAN_ENVELOPE_SPEED_Y_MARGIN,
            "speed_p50": speed_p50,
            "speed_soft": speed_p90 + HUMAN_ENVELOPE_SPEED_SOFT_MARGIN,
            "speed_hard": speed_p95 + HUMAN_ENVELOPE_SPEED_HARD_MARGIN,
            "steer_low": max(-1.0, steer_p05 - HUMAN_ACTION_STEER_MARGIN),
            "steer_high": min(1.0, steer_p95 + HUMAN_ACTION_STEER_MARGIN),
            "steer_p10": steer_p10,
            "steer_p50": steer_p50,
            "steer_p90": steer_p90,
            "accel_low": max(0.0, accel_p10 - HUMAN_ACTION_ACCEL_MARGIN),
            "accel_p50": accel_p50,
            "accel_high": min(1.0, accel_p90 + HUMAN_ACTION_ACCEL_MARGIN),
            "brake_p50": brake_p50,
            "brake_p90": brake_p90,
            "brake_p95": brake_p95,
            "brake_cap": min(1.0, brake_p99 + HUMAN_ACTION_BRAKE_MARGIN),
        })

    if not points:
        return None

    return {
        "version": HUMAN_ENVELOPE_VERSION,
        "source_file": data_file,
        "source_size": os.path.getsize(data_file),
        "source_mtime": os.path.getmtime(data_file),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "bin_m": HUMAN_ENVELOPE_BIN_M,
        "sectors": {
            "whole_track": {
                "start": 0.0,
                "end": float(LAP_THRESHOLD),
                "points": points,
            },
            "first_corner": {
                "start": HUMAN_ENVELOPE_FIRST_CORNER_START,
                "end": HUMAN_ENVELOPE_FIRST_CORNER_END,
                "points": [
                    p for p in points
                    if HUMAN_ENVELOPE_FIRST_CORNER_START <= p["dist"] <= HUMAN_ENVELOPE_FIRST_CORNER_END
                ],
            }
        },
    }


def load_human_racing_envelope(data_file=DATA_FILE, envelope_file=HUMAN_ENVELOPE_FILE):
    if os.path.exists(envelope_file):
        try:
            with open(envelope_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if _human_envelope_cache_is_fresh(cached, data_file):
                return cached
        except (OSError, json.JSONDecodeError):
            pass

    envelope = build_human_racing_envelope(data_file)
    if envelope is None:
        return None

    try:
        with open(envelope_file, "w", encoding="utf-8") as f:
            json.dump(envelope, f, indent=2)
        print(f"  Human racing envelope saved: {os.path.abspath(envelope_file)}")
    except OSError as e:
        print(f"  Human envelope cache not saved: {e}")
    return envelope


def _interpolate_human_envelope(points, lap_dist):
    if not points:
        return None
    if lap_dist < points[0]["dist"] or lap_dist > points[-1]["dist"]:
        return None
    if lap_dist <= points[0]["dist"]:
        return dict(points[0])
    for p0, p1 in zip(points, points[1:]):
        if lap_dist <= p1["dist"]:
            t = (lap_dist - p0["dist"]) / max(1e-6, p1["dist"] - p0["dist"])
            out = {"dist": float(lap_dist)}
            for key in p0:
                if key in ("dist", "samples"):
                    continue
                out[key] = float(p0[key] + (p1[key] - p0[key]) * t)
            out["samples"] = int(min(p0.get("samples", 0), p1.get("samples", 0)))
            return out
    return dict(points[-1])


def _human_envelope_at(human_envelope, lap_dist):
    if not isinstance(human_envelope, dict):
        return None
    if lap_dist < HUMAN_ENVELOPE_ASSIST_START_DIST:
        return None
    sector = human_envelope.get("sectors", {}).get("whole_track")
    if not isinstance(sector, dict):
        sector = human_envelope.get("sectors", {}).get("first_corner")
    if not isinstance(sector, dict):
        return None
    return _interpolate_human_envelope(sector.get("points", []), lap_dist)


def summarize_human_racing_envelope(human_envelope):
    if not isinstance(human_envelope, dict):
        return None
    sectors = {}
    for name, sector in human_envelope.get("sectors", {}).items():
        if isinstance(sector, dict):
            sectors[name] = {
                "start": sector.get("start"),
                "end": sector.get("end"),
                "points": len(sector.get("points", [])),
            }
    return {
        "version": human_envelope.get("version"),
        "source_file": human_envelope.get("source_file"),
        "source_size": human_envelope.get("source_size"),
        "source_mtime": human_envelope.get("source_mtime"),
        "generated_at": human_envelope.get("generated_at"),
        "bin_m": human_envelope.get("bin_m"),
        "sectors": sectors,
    }


def _add_reason(info, reason):
    if reason not in info["reasons"]:
        info["reasons"].append(reason)


def apply_global_assist(speed_x, speed_y, track_pos, angle, track, raw_steer, accel_cmd, brake_cmd):
    info = {
        "active": False,
        "reasons": [],
        "target_speed": None,
        "lookahead_clear": None,
        "center_steer": None,
        "stability_steer": None,
        "accel_before": accel_cmd,
        "brake_before": brake_cmd,
        "steer_before": raw_steer,
    }

    abs_track_pos = abs(track_pos)
    abs_angle = abs(angle)
    stable_center = abs_track_pos < 0.65 and abs_angle < 0.28 and abs(speed_y) < 8.0
    elevated_risk = abs_track_pos > 0.75 or abs_angle > 0.45 or abs(speed_y) > 12.0
    high_risk = abs_track_pos > 0.90 or abs_angle > 0.65 or abs(speed_y) > 18.0

    target_speed, lookahead_clear = _target_speed_from_track(track)
    info["target_speed"] = target_speed
    info["lookahead_clear"] = lookahead_clear
    if target_speed is not None:
        overspeed = speed_x - target_speed
        info["overspeed"] = overspeed
        if overspeed > 15.0 and not stable_center:
            accel_limit = 0.75 if not elevated_risk else 0.45
            accel_cmd = min(accel_cmd, accel_limit)
            info["active"] = True
            info["reasons"].append("global_lookahead_lift")
        if overspeed > 30.0 and (elevated_risk or lookahead_clear < 45.0):
            brake_base = 0.06 if not high_risk else 0.12
            brake_cmd = max(brake_cmd, min(0.45, brake_base + (overspeed - 30.0) / 135.0))
            accel_cmd = min(accel_cmd, 0.20 if high_risk else 0.40)
            info["active"] = True
            info["reasons"].append("global_lookahead_brake")

    if abs_track_pos > GLOBAL_ASSIST_CENTER_START:
        center_steer = float(np.clip(-track_pos * 0.34 + angle * 0.08, -0.45, 0.45))
        blend = float(np.clip((abs_track_pos - GLOBAL_ASSIST_CENTER_START) / 0.45, 0.08, 0.60))
        raw_steer = float(np.clip(raw_steer * (1.0 - blend) + center_steer * blend, -0.95, 0.95))
        info["center_steer"] = center_steer
        info["center_blend"] = blend
        info["active"] = True
        info["reasons"].append("global_track_center")

    if abs_track_pos > GLOBAL_ASSIST_EDGE_START:
        accel_limit = max(0.10, 0.55 - (abs_track_pos - GLOBAL_ASSIST_EDGE_START) * 0.65)
        accel_cmd = min(accel_cmd, accel_limit)
        if speed_x > 85.0:
            brake_cmd = max(brake_cmd, min(0.38, 0.06 + (abs_track_pos - GLOBAL_ASSIST_EDGE_START) * 0.32))
        info["active"] = True
        info["reasons"].append("global_edge_guard")

    if abs_track_pos > GLOBAL_ASSIST_OFFTRACK:
        recover_steer = float(np.clip(-np.sign(track_pos) * min(1.0, 0.45 + 0.25 * (abs_track_pos - 1.0)), -1.0, 1.0))
        raw_steer = recover_steer
        accel_cmd = 0.0
        if speed_x > 25.0:
            brake_cmd = max(brake_cmd, 0.28)
        info["recover_steer"] = recover_steer
        info["active"] = True
        info["reasons"].append("global_offtrack_recovery")

    if speed_x > 45.0 and (abs(speed_y) > 10.0 or abs(angle) > 0.55):
        speed_y_term = 0.0
        angle_term = 0.0
        if abs(speed_y) > 10.0:
            speed_y_term = -np.sign(speed_y) * min(0.60, abs(speed_y) * 0.020)
        if abs(angle) > 0.35:
            angle_term = -np.sign(angle) * min(0.28, abs(angle) * 0.18)
        stability_steer = float(np.clip(speed_y_term + angle_term, -0.70, 0.70))
        raw_steer = float(np.clip(raw_steer * 0.60 + stability_steer * 0.40, -0.95, 0.95))
        accel_cmd = min(accel_cmd, 0.12)
        if speed_x > 75.0:
            brake_cmd = max(brake_cmd, min(0.45, 0.08 + abs(speed_y) / 130.0 + max(0.0, abs(angle) - 0.55) * 0.15))
        info["stability_steer"] = stability_steer
        info["active"] = True
        info["reasons"].append("global_stability")

    if speed_x < 4.0 and abs_track_pos < 0.95 and abs(angle) < 0.50:
        accel_cmd = max(accel_cmd, 0.30)
        brake_cmd = min(brake_cmd, 0.05)
        info["active"] = True
        info["reasons"].append("global_unstuck_throttle")

    info["accel_after"] = accel_cmd
    info["brake_after"] = brake_cmd
    info["steer_after"] = raw_steer
    return raw_steer, accel_cmd, brake_cmd, info


def apply_guardian_assist(
        dist_ep, speed_x, speed_y, track_pos, angle, track,
        raw_steer, accel_cmd, brake_cmd, human_envelope=None):
    lap_dist = dist_ep % LAP_THRESHOLD if dist_ep >= 0.0 else 0.0
    info = {
        "active": False,
        "abort": False,
        "reasons": [],
        "lap_dist": lap_dist,
        "target_speed": None,
        "sensor_target_speed": None,
        "lookahead_clear": None,
        "overspeed": None,
        "brake_distance": None,
        "center_steer": None,
        "force_field_steer": None,
        "stability_steer": None,
        "slip_steer": None,
        "first_corner_trajectory": None,
        "human_envelope": None,
        "accel_before": accel_cmd,
        "brake_before": brake_cmd,
        "steer_before": raw_steer,
    }

    abs_track_pos = abs(track_pos)
    abs_angle = abs(angle)
    abs_speed_y = abs(speed_y)

    if abs_track_pos >= GUARDIAN_ABORT_TRACK:
        # In this project an off-track lap is invalid, so guardian mode stops
        # the run instead of trying to recover a lap the user would restart.
        accel_cmd = 0.0
        if speed_x > 20.0:
            brake_cmd = max(brake_cmd, 0.35)
        info["active"] = True
        info["abort"] = True
        info["reasons"].append("guardian_offtrack_abort")
        info["accel_after"] = accel_cmd
        info["brake_after"] = brake_cmd
        info["steer_after"] = raw_steer
        return raw_steer, accel_cmd, brake_cmd, info

    human_env = _human_envelope_at(human_envelope, lap_dist)
    human_line_confident = True
    human_envelope_violation = False
    human_speed_trusted = False
    if human_env is not None:
        track_ok = human_env["track_low"] <= track_pos <= human_env["track_high"]
        angle_ok = human_env["angle_low"] <= angle <= human_env["angle_high"]
        speed_y_ok = human_env["speed_y_low"] <= speed_y <= human_env["speed_y_high"]
        speed_ok = speed_x <= human_env["speed_soft"]
        major_angle_error = (not angle_ok) and abs_angle > HUMAN_ENVELOPE_MAJOR_ANGLE
        major_speed_y_error = (not speed_y_ok) and abs_speed_y > HUMAN_ENVELOPE_MAJOR_SPEED_Y
        human_speed_trusted = (
            track_ok and
            abs_track_pos < HUMAN_ENVELOPE_SPEED_TRUST_TRACK and
            not major_angle_error and
            not major_speed_y_error
        )
        human_line_confident = (
            track_ok and
            (angle_ok or abs_angle < GUARDIAN_RACING_LINE_ANGLE) and
            (speed_y_ok or abs_speed_y < GUARDIAN_RACING_LINE_SPEED_Y)
        )
        human_envelope_violation = (
            not track_ok or
            not speed_ok or
            major_angle_error or
            major_speed_y_error
        )
        info["human_envelope"] = {
            "active": True,
            "dist": human_env["dist"],
            "track_target": human_env["track_target"],
            "track_low": human_env["track_low"],
            "track_high": human_env["track_high"],
            "track_ok": track_ok,
            "angle_target": human_env["angle_target"],
            "angle_low": human_env["angle_low"],
            "angle_high": human_env["angle_high"],
            "angle_ok": angle_ok,
            "speed_y_target": human_env["speed_y_target"],
            "speed_y_low": human_env["speed_y_low"],
            "speed_y_high": human_env["speed_y_high"],
            "speed_y_ok": speed_y_ok,
            "speed_p50": human_env["speed_p50"],
            "speed_soft": human_env["speed_soft"],
            "speed_hard": human_env["speed_hard"],
            "speed_ok": speed_ok,
            "speed_trusted": human_speed_trusted,
            "major_angle_error": major_angle_error,
            "major_speed_y_error": major_speed_y_error,
            "steer_low": human_env["steer_low"],
            "steer_high": human_env["steer_high"],
            "steer_p50": human_env["steer_p50"],
            "brake_p50": human_env["brake_p50"],
            "brake_p90": human_env["brake_p90"],
            "brake_p95": human_env["brake_p95"],
            "brake_cap": human_env["brake_cap"],
            "accel_low": human_env["accel_low"],
            "accel_p50": human_env["accel_p50"],
            "accel_high": human_env["accel_high"],
            "line_confident": human_line_confident,
            "violation": human_envelope_violation,
        }

    target_speed, lookahead_clear = _guardian_target_speed_from_track(track)
    info["sensor_target_speed"] = target_speed
    info["lookahead_clear"] = lookahead_clear

    if GUARDIAN_FIRST_CORNER_SPEED_PROFILE[0][0] <= lap_dist <= GUARDIAN_FIRST_CORNER_SPEED_PROFILE[-1][0]:
        first_corner_target = _interp_profile(GUARDIAN_FIRST_CORNER_SPEED_PROFILE, lap_dist)
        target_speed = min(target_speed, first_corner_target) if target_speed is not None else first_corner_target
        info["first_corner_target_speed"] = first_corner_target

    if target_speed is not None:
        if abs_track_pos > 0.65:
            edge_cap = max(85.0, 240.0 - (abs_track_pos - 0.65) * 380.0)
            target_speed = min(target_speed, edge_cap)
            info["edge_speed_cap"] = edge_cap
        if abs_angle > 0.28 and speed_x > 45.0:
            angle_cap = max(80.0, 230.0 - (abs_angle - 0.28) * 260.0)
            target_speed = min(target_speed, angle_cap)
            info["angle_speed_cap"] = angle_cap
        if abs_speed_y > 8.0 and speed_x > 45.0:
            slip_cap = max(75.0, 230.0 - (abs_speed_y - 8.0) * 7.0)
            target_speed = min(target_speed, slip_cap)
            info["slip_speed_cap"] = slip_cap

        racing_line_confident = (
            abs_track_pos < GUARDIAN_RACING_LINE_TRACK and
            abs_angle < GUARDIAN_RACING_LINE_ANGLE and
            abs_speed_y < GUARDIAN_RACING_LINE_SPEED_Y and
            human_line_confident
        )
        if racing_line_confident:
            target_speed += GUARDIAN_RACING_LINE_TARGET_BONUS
            info["racing_line_confident"] = True
            info["racing_line_target_bonus"] = GUARDIAN_RACING_LINE_TARGET_BONUS
        else:
            info["racing_line_confident"] = False

        if human_env is not None:
            human_speed_cap = human_env["speed_hard"] if human_speed_trusted else human_env["speed_soft"]
            if human_speed_trusted:
                target_speed = max(target_speed, human_env["speed_p50"])
            target_speed = min(target_speed, human_speed_cap)
            info["human_envelope"]["speed_floor"] = human_env["speed_p50"] if human_speed_trusted else None
            info["human_envelope"]["speed_cap"] = human_speed_cap

        overspeed = speed_x - target_speed
        delta_v_ms = max(0.0, overspeed) / 3.6
        brake_distance = delta_v_ms * delta_v_ms / (2.0 * GUARDIAN_BRAKE_DECEL_MS2)
        info["target_speed"] = target_speed
        info["overspeed"] = overspeed
        info["brake_distance"] = brake_distance
        info["speed_guard_enabled"] = lap_dist >= GUARDIAN_SPEED_SUPPRESS_DIST
        lift_overspeed = (
            GUARDIAN_STRONG_LIFT_OVERSPEED if racing_line_confident
            else GUARDIAN_LIFT_OVERSPEED
        )
        info["speed_lift_overspeed"] = lift_overspeed

        late_for_corner = (
            lap_dist >= GUARDIAN_SPEED_SUPPRESS_DIST and
            lookahead_clear is not None and
            speed_x > target_speed and
            brake_distance > lookahead_clear
        )
        if (
            human_env is not None and
            human_speed_trusted and
            speed_x <= human_env["speed_soft"] and
            abs_track_pos < GUARDIAN_EDGE_TRACK
        ):
            if late_for_corner:
                info["human_envelope"]["suppressed_predictive_brake"] = True
            late_for_corner = False
        speed_veto = (
            lap_dist >= GUARDIAN_SPEED_SUPPRESS_DIST and
            speed_x > 35.0 and
            (overspeed > lift_overspeed or late_for_corner)
        )
        if speed_veto:
            accel_limit = 0.55
            if overspeed > GUARDIAN_STRONG_LIFT_OVERSPEED or late_for_corner or abs_track_pos > 0.72:
                accel_limit = 0.25
            if abs_track_pos > GUARDIAN_WARN_TRACK:
                accel_limit = 0.10
            accel_cmd = min(accel_cmd, accel_limit)
            info["active"] = True
            _add_reason(info, "guardian_speed_veto")

        if speed_x > 45.0 and late_for_corner:
            brake_strength = max(0.18, 0.08 + max(0.0, overspeed - GUARDIAN_LIFT_OVERSPEED) / 95.0)
            if abs_track_pos > GUARDIAN_WARN_TRACK:
                brake_strength = max(brake_strength, 0.38)
            brake_cmd = max(brake_cmd, min(0.78, brake_strength))
            accel_cmd = 0.0
            info["active"] = True
            _add_reason(info, "guardian_predictive_brake")

    if human_env is not None and human_envelope_violation:
        line_error = human_env["track_target"] - track_pos
        human_line_steer = float(np.clip(
            line_error * HUMAN_ENVELOPE_STEER_GAIN -
            speed_y * HUMAN_ENVELOPE_SPEED_Y_GAIN +
            angle * HUMAN_ENVELOPE_ANGLE_GAIN,
            -0.92, 0.92))

        if (
            track_pos < human_env["track_low"] or
            speed_y < human_env["speed_y_low"] or
            angle > human_env["angle_high"]
        ):
            raw_steer = max(raw_steer, human_line_steer)
        elif (
            track_pos > human_env["track_high"] or
            speed_y > human_env["speed_y_high"] or
            angle < human_env["angle_low"]
        ):
            raw_steer = min(raw_steer, human_line_steer)
        else:
            raw_steer = float(np.clip(raw_steer * 0.70 + human_line_steer * 0.30, -0.95, 0.95))

        accel_limit = 0.45
        if speed_x > human_env["speed_soft"]:
            accel_limit = 0.20
        if speed_x > human_env["speed_hard"]:
            brake_cmd = max(brake_cmd, 0.16)
            accel_limit = 0.05
        accel_cmd = min(accel_cmd, accel_limit)

        info["active"] = True
        info["human_envelope"]["line_error"] = line_error
        info["human_envelope"]["line_steer"] = human_line_steer
        info["human_envelope"]["accel_limit"] = accel_limit
        _add_reason(info, "guardian_human_envelope")

    if human_env is not None:
        action_clamp_allowed = (
            abs_track_pos < GUARDIAN_EDGE_TRACK and
            abs_speed_y < HUMAN_ENVELOPE_MAJOR_SPEED_Y and
            abs_angle < HUMAN_ENVELOPE_MAJOR_ANGLE
        )
        if action_clamp_allowed:
            steer_before_action_clip = raw_steer
            raw_steer = float(np.clip(raw_steer, human_env["steer_low"], human_env["steer_high"]))
            if abs(raw_steer - steer_before_action_clip) > 1e-4:
                info["active"] = True
                info["human_envelope"]["steer_before_action_clip"] = steer_before_action_clip
                info["human_envelope"]["steer_after_action_clip"] = raw_steer
                _add_reason(info, "guardian_action_steer_clip")

            brake_before_action_clip = brake_cmd
            if brake_cmd > human_env["brake_cap"]:
                brake_cmd = human_env["brake_cap"]
                info["active"] = True
                info["human_envelope"]["brake_before_action_clip"] = brake_before_action_clip
                info["human_envelope"]["brake_after_action_clip"] = brake_cmd
                _add_reason(info, "guardian_action_brake_clip")

    if (
        GUARDIAN_FIRST_CORNER_TRAJECTORY_START <= lap_dist <=
        GUARDIAN_FIRST_CORNER_TRAJECTORY_END
    ):
        if human_env is not None:
            first_corner_target_track = human_env["track_target"]
            first_corner_speed_ref = human_env["speed_soft"]
            first_corner_target_source = "human_envelope"
        elif FIRST_CORNER_LANE_PROFILE[0][0] <= lap_dist <= FIRST_CORNER_LANE_PROFILE[-1][0]:
            first_corner_target_track = _interp_profile(FIRST_CORNER_LANE_PROFILE, lap_dist)
            first_corner_speed_ref = _interp_profile(GUARDIAN_FIRST_CORNER_SPEED_PROFILE, lap_dist)
            first_corner_target_source = "profile"
        else:
            first_corner_target_track = None
            first_corner_speed_ref = None
            first_corner_target_source = None

        if first_corner_target_track is not None:
            line_error = first_corner_target_track - track_pos
            abs_line_error = abs(line_error)
            line_deviation = max(
                0.0,
                abs_line_error - GUARDIAN_FIRST_CORNER_TRAJECTORY_MARGIN)
            first_corner_steer = float(np.clip(
                line_error * GUARDIAN_FIRST_CORNER_TRAJECTORY_STEER_GAIN -
                speed_y * GUARDIAN_FIRST_CORNER_TRAJECTORY_SPEED_Y_GAIN +
                angle * GUARDIAN_FIRST_CORNER_TRAJECTORY_ANGLE_GAIN,
                -GUARDIAN_FIRST_CORNER_TRAJECTORY_STEER_LIMIT,
                GUARDIAN_FIRST_CORNER_TRAJECTORY_STEER_LIMIT))

            info["first_corner_trajectory"] = {
                "active": False,
                "source": first_corner_target_source,
                "target_track_pos": first_corner_target_track,
                "line_error": line_error,
                "line_deviation": line_deviation,
                "trajectory_steer": first_corner_steer,
                "speed_ref": first_corner_speed_ref,
                "steer_before": raw_steer,
                "brake_before": brake_cmd,
                "accel_before": accel_cmd,
            }

            trajectory_risk = (
                line_deviation > 0.0 or
                abs_speed_y > 7.5 or
                abs_angle > 0.22
            )
            if trajectory_risk:
                steer_blend = 0.34 + min(0.46, line_deviation * 0.95)
                if abs_line_error >= GUARDIAN_FIRST_CORNER_TRAJECTORY_STRONG_MARGIN:
                    steer_blend = max(steer_blend, 0.62)
                if abs_speed_y > 10.0 or abs_track_pos > 0.72:
                    steer_blend = max(steer_blend, 0.70)

                guided_steer = float(np.clip(
                    raw_steer * (1.0 - steer_blend) +
                    first_corner_steer * steer_blend,
                    -GUARDIAN_FIRST_CORNER_TRAJECTORY_STEER_LIMIT,
                    GUARDIAN_FIRST_CORNER_TRAJECTORY_STEER_LIMIT))
                if line_error < -GUARDIAN_FIRST_CORNER_TRAJECTORY_MARGIN:
                    raw_steer = min(guided_steer, first_corner_steer * 0.92)
                elif line_error > GUARDIAN_FIRST_CORNER_TRAJECTORY_MARGIN:
                    raw_steer = max(guided_steer, first_corner_steer * 0.92)
                else:
                    raw_steer = guided_steer

                info["active"] = True
                info["first_corner_trajectory"]["active"] = True
                info["first_corner_trajectory"]["steer_blend"] = steer_blend
                info["first_corner_trajectory"]["steer_after"] = raw_steer
                _add_reason(info, "guardian_first_corner_trajectory")

            if first_corner_speed_ref is not None:
                line_speed_cap = first_corner_speed_ref
                if line_deviation > 0.0:
                    line_speed_cap -= min(34.0, line_deviation * 70.0)
                if abs_speed_y > 8.0:
                    line_speed_cap -= min(30.0, (abs_speed_y - 8.0) * 4.0)
                if abs_angle > 0.24:
                    line_speed_cap -= min(24.0, (abs_angle - 0.24) * 90.0)
                line_speed_cap = max(92.0, line_speed_cap)
                trajectory_overspeed = speed_x - line_speed_cap
                info["first_corner_trajectory"]["speed_cap"] = line_speed_cap
                info["first_corner_trajectory"]["overspeed"] = trajectory_overspeed

                if trajectory_overspeed > 6.0 and (line_deviation > 0.0 or abs_speed_y > 7.5):
                    accel_cmd = min(accel_cmd, 0.18 if line_deviation < 0.30 else 0.04)
                    info["active"] = True
                    info["first_corner_trajectory"]["accel_after_speed"] = accel_cmd
                    _add_reason(info, "guardian_first_corner_speed")

                if (
                    trajectory_overspeed > 18.0 or
                    line_deviation > 0.34 or
                    abs_speed_y > 11.0
                ):
                    brake_strength = 0.18
                    brake_strength += min(0.42, max(0.0, trajectory_overspeed) / 95.0)
                    brake_strength += min(0.20, line_deviation * 0.35)
                    if abs_speed_y > 11.0:
                        brake_strength += min(0.16, (abs_speed_y - 11.0) / 70.0)
                    brake_cmd = max(brake_cmd, min(0.82, brake_strength))
                    accel_cmd = min(accel_cmd, 0.02)
                    info["active"] = True
                    info["first_corner_trajectory"]["brake_after_speed"] = brake_cmd
                    info["first_corner_trajectory"]["accel_after_brake"] = accel_cmd
                    _add_reason(info, "guardian_first_corner_brake")

            if lap_dist >= GUARDIAN_FIRST_CORNER_LATE_STEER_START:
                steer_cap = GUARDIAN_FIRST_CORNER_LATE_STEER_CAP
                if speed_x > 105.0:
                    steer_cap = min(steer_cap, 0.54)
                if abs_speed_y > 5.0:
                    steer_cap = min(steer_cap, 0.52)
                if abs_speed_y > 11.0:
                    steer_cap = min(steer_cap, GUARDIAN_FIRST_CORNER_LATE_SLIP_STEER_CAP)
                steer_before_cap = raw_steer
                raw_steer = float(np.clip(raw_steer, -steer_cap, steer_cap))
                if abs(raw_steer - steer_before_cap) > 1e-4:
                    info["active"] = True
                    info["first_corner_trajectory"]["late_steer_cap"] = steer_cap
                    info["first_corner_trajectory"]["steer_before_cap"] = steer_before_cap
                    info["first_corner_trajectory"]["steer_after_cap"] = raw_steer
                    _add_reason(info, "guardian_first_corner_steer_cap")

    if abs_track_pos >= GUARDIAN_FORCE_FIELD_TRACK:
        edge_strength = float(np.clip(
            (abs_track_pos - GUARDIAN_FORCE_FIELD_TRACK) /
            max(1e-6, 1.0 - GUARDIAN_FORCE_FIELD_TRACK),
            0.0, 1.0))
        force_steer = float(np.clip(
            -track_pos * (0.55 + 0.45 * edge_strength) + angle * 0.10,
            -0.95, 0.95))
        if abs_speed_y > 8.0:
            force_steer = float(np.clip(
                force_steer - np.sign(speed_y) * min(0.22, abs_speed_y * 0.011),
                -0.95, 0.95))
        max_high_speed_lock = float(np.clip(95.0 / max(35.0, abs(speed_x)), 0.32, 1.0))
        force_steer = float(np.clip(force_steer, -max_high_speed_lock, max_high_speed_lock))
        blend = 0.24 + edge_strength * 0.56
        raw_steer = float(np.clip(raw_steer * (1.0 - blend) + force_steer * blend, -0.95, 0.95))

        if speed_x > 70.0:
            accel_cmd = min(accel_cmd, max(0.0, 0.46 - edge_strength * 0.46))
        if speed_x > 90.0:
            brake_cmd = max(brake_cmd, min(0.62, 0.12 + edge_strength * 0.45))
        if abs_track_pos >= GUARDIAN_EDGE_TRACK and speed_x > 70.0:
            brake_cmd = max(brake_cmd, 0.55)
            accel_cmd = 0.0

        info["active"] = True
        info["force_field_steer"] = force_steer
        info["force_field_blend"] = blend
        info["edge_strength"] = edge_strength
        _add_reason(
            info,
            "guardian_edge_veto" if abs_track_pos >= GUARDIAN_EDGE_TRACK
            else "guardian_wall_repulsion")

    heading_risk = (
        lap_dist >= GUARDIAN_LAUNCH_SUPPRESS_DIST and
        abs_angle >= GUARDIAN_HEADING_ANGLE and
        abs_track_pos > 0.38 and
        speed_x > 35.0
    )
    if heading_risk:
        heading_steer = float(np.clip(-np.sign(angle) * min(0.60, abs_angle * 0.44), -0.60, 0.60))
        raw_steer = float(np.clip(raw_steer * 0.58 + heading_steer * 0.42, -0.95, 0.95))
        if speed_x > 80.0:
            accel_cmd = min(accel_cmd, 0.25)
            brake_cmd = max(brake_cmd, min(0.34, 0.08 + (abs_angle - GUARDIAN_HEADING_ANGLE) * 0.24))
        info["active"] = True
        info["stability_steer"] = heading_steer
        _add_reason(info, "guardian_heading_guard")

    slip_risk = abs_speed_y > 13.0 and speed_x > 45.0 and abs_track_pos > 0.40
    if slip_risk:
        slip_steer = float(np.clip(-np.sign(speed_y) * min(0.46, abs_speed_y * 0.020), -0.46, 0.46))
        raw_steer = float(np.clip(raw_steer * 0.66 + slip_steer * 0.34, -0.95, 0.95))
        accel_cmd = min(accel_cmd, 0.25)
        if speed_x > 85.0:
            brake_cmd = max(brake_cmd, min(0.32, 0.08 + (abs_speed_y - 13.0) / 90.0))
        info["active"] = True
        info["slip_steer"] = slip_steer
        _add_reason(info, "guardian_slip_guard")

    if (
        GUARDIAN_FIRST_CORNER_LATE_STEER_START <= lap_dist <=
        GUARDIAN_FIRST_CORNER_TRAJECTORY_END and
        info.get("first_corner_trajectory") is not None
    ):
        final_steer_cap = GUARDIAN_FIRST_CORNER_LATE_STEER_CAP
        if speed_x > 105.0:
            final_steer_cap = min(final_steer_cap, 0.54)
        if abs_speed_y > 5.0:
            final_steer_cap = min(final_steer_cap, 0.52)
        if abs_speed_y > 11.0:
            final_steer_cap = min(final_steer_cap, GUARDIAN_FIRST_CORNER_LATE_SLIP_STEER_CAP)
        final_steer_before_cap = raw_steer
        raw_steer = float(np.clip(raw_steer, -final_steer_cap, final_steer_cap))
        if abs(raw_steer - final_steer_before_cap) > 1e-4:
            info["active"] = True
            info["first_corner_trajectory"]["final_steer_cap"] = final_steer_cap
            info["first_corner_trajectory"]["steer_before_final_cap"] = final_steer_before_cap
            info["first_corner_trajectory"]["steer_after_final_cap"] = raw_steer
            _add_reason(info, "guardian_first_corner_final_steer_cap")

    info["accel_after"] = accel_cmd
    info["brake_after"] = brake_cmd
    info["steer_after"] = raw_steer
    return raw_steer, accel_cmd, brake_cmd, info


def apply_launch_lane_keeper(dist_ep, speed_x, speed_y, track_pos, angle, raw_steer, accel_cmd, brake_cmd):
    info = {
        "active": False,
        "reasons": [],
        "target_track_pos": None,
        "lower_track_pos": None,
        "upper_track_pos": None,
        "lane_error": None,
        "lane_steer": None,
        "center_lane_steer": None,
        "traction_accel_limit": None,
        "accel_before": accel_cmd,
        "brake_before": brake_cmd,
        "steer_before": raw_steer,
    }

    if dist_ep < LAUNCH_LANE_CORRIDOR[0][0] or dist_ep > 180.0 or speed_x > 190.0:
        info["accel_after"] = accel_cmd
        info["brake_after"] = brake_cmd
        info["steer_after"] = raw_steer
        return raw_steer, accel_cmd, brake_cmd, info

    lower_track_pos, upper_track_pos = _interp_corridor(LAUNCH_LANE_CORRIDOR, dist_ep)
    center_track_pos = (lower_track_pos + upper_track_pos) * 0.5
    info["target_track_pos"] = center_track_pos
    info["lower_track_pos"] = lower_track_pos
    info["upper_track_pos"] = upper_track_pos

    if dist_ep <= LAUNCH_TRACTION_DIST and speed_x < LAUNCH_TRACTION_SPEED:
        traction_limit = _interp_profile(
            LAUNCH_TRACTION_ACCEL_PROFILE,
            max(0.0, min(dist_ep, LAUNCH_TRACTION_DIST)))
        if abs(angle) > 0.08 or abs(speed_y) > 4.0:
            traction_limit = min(traction_limit, 0.62)
        if accel_cmd > traction_limit:
            accel_cmd = traction_limit
            info["active"] = True
            info["reasons"].append("launch_traction_control")
        info["traction_accel_limit"] = traction_limit

        center_error = center_track_pos - track_pos
        if abs(center_error) > 0.10 or abs(angle) > 0.06:
            center_lane_steer = float(np.clip(
                center_error * 0.20 + angle * 0.08,
                -0.20, 0.20))
            raw_steer = float(np.clip(raw_steer * 0.72 + center_lane_steer * 0.28, -0.95, 0.95))
            info["active"] = True
            info["reasons"].append("launch_center_lane")
            info["lane_error"] = center_error
            info["center_lane_steer"] = center_lane_steer

    soft_margin = 0.08
    if track_pos > upper_track_pos + soft_margin:
        lane_error = upper_track_pos - track_pos
        lane_steer = float(np.clip(lane_error * 0.30 + angle * 0.08, -0.26, 0.16))
        raw_steer = lane_steer
        info["active"] = True
        info["reasons"].append("launch_lane_right")
        info["lane_error"] = lane_error
        info["lane_steer"] = lane_steer
    elif track_pos < lower_track_pos - soft_margin:
        lane_error = lower_track_pos - track_pos
        lane_steer = float(np.clip(lane_error * 0.30 + angle * 0.08, -0.16, 0.26))
        raw_steer = lane_steer
        info["active"] = True
        info["reasons"].append("launch_lane_left")
        info["lane_error"] = lane_error
        info["lane_steer"] = lane_steer

    edge_limit = max(abs(lower_track_pos), abs(upper_track_pos), 0.72)
    if abs(track_pos) > edge_limit + 0.10 and speed_x > 100.0:
        accel_cmd = min(accel_cmd, 0.35)
        info["active"] = True
        info["reasons"].append("launch_exit_lift")
    if abs(track_pos) > edge_limit + 0.22 and speed_x > 120.0:
        accel_cmd = min(accel_cmd, 0.05)
        brake_cmd = max(brake_cmd, 0.14)
        info["active"] = True
        info["reasons"].append("launch_exit_brake")

    info["accel_after"] = accel_cmd
    info["brake_after"] = brake_cmd
    info["steer_after"] = raw_steer
    return raw_steer, accel_cmd, brake_cmd, info


def apply_corner_safety(dist_ep, speed_x, speed_y, track_pos, angle, raw_steer, accel_cmd, brake_cmd, front_track=None):
    lap_dist = dist_ep % LAP_THRESHOLD if dist_ep >= 0.0 else 0.0
    info = {
        "active": False,
        "reasons": [],
        "lap_dist": lap_dist,
        "speed_profile": None,
        "target_speed": None,
        "overspeed": None,
        "target_track_pos": None,
        "front_track": front_track,
        "accel_before": accel_cmd,
        "brake_before": brake_cmd,
        "steer_before": raw_steer,
    }

    accel_cmd, brake_cmd = _apply_speed_profile(
        info, "first_corner_speed", FIRST_CORNER_SPEED_PROFILE,
        lap_dist, speed_x, accel_cmd, brake_cmd)

    if FIRST_CORNER_LANE_PROFILE[0][0] <= lap_dist <= FIRST_CORNER_LANE_PROFILE[-1][0]:
        target_track_pos = _interp_profile(FIRST_CORNER_LANE_PROFILE, lap_dist)
        lane_error = target_track_pos - track_pos
        lane_steer = float(np.clip(
            lane_error * FIRST_CORNER_LANE_GAIN + angle * FIRST_CORNER_ANGLE_GAIN,
            -FIRST_CORNER_LANE_LIMIT, FIRST_CORNER_LANE_LIMIT))
        info["first_corner_target_track_pos"] = target_track_pos
        info["first_corner_lane_error"] = lane_error
        info["first_corner_lane_steer"] = lane_steer

        if track_pos > target_track_pos + FIRST_CORNER_LANE_MARGIN:
            raw_steer = lane_steer
            accel_cmd = min(accel_cmd, 0.35)
            info["active"] = True
            info["reasons"].append("first_corner_lane_right")
        elif track_pos < target_track_pos - FIRST_CORNER_LANE_MARGIN:
            raw_steer = lane_steer
            accel_cmd = min(accel_cmd, 0.35)
            info["active"] = True
            info["reasons"].append("first_corner_lane_left")

        if abs(lane_error) > 0.80 and speed_x > 150.0:
            accel_cmd = 0.0
            brake_cmd = max(brake_cmd, 0.14)
            info["active"] = True
            info["reasons"].append("first_corner_line_brake")

    accel_cmd, brake_cmd = _apply_speed_profile(
        info, "sector_700_820_speed", SECOND_SECTOR_SPEED_PROFILE,
        lap_dist, speed_x, accel_cmd, brake_cmd)

    accel_cmd, brake_cmd = _apply_speed_profile(
        info, "sector_1450_1660_speed", THIRD_SECTOR_SPEED_PROFILE,
        lap_dist, speed_x, accel_cmd, brake_cmd)

    accel_cmd, brake_cmd = _apply_speed_profile(
        info, "corkscrew_speed", CORKSCREW_SPEED_PROFILE,
        lap_dist, speed_x, accel_cmd, brake_cmd)

    if SECOND_SECTOR_LANE_PROFILE[0][0] <= lap_dist <= SECOND_SECTOR_LANE_PROFILE[-1][0]:
        target_track_pos = _interp_profile(SECOND_SECTOR_LANE_PROFILE, lap_dist)
        lane_error = target_track_pos - track_pos
        lane_steer = float(np.clip(
            lane_error * SECOND_SECTOR_LANE_GAIN + angle * SECOND_SECTOR_ANGLE_GAIN,
            -SECOND_SECTOR_LANE_LIMIT, SECOND_SECTOR_LANE_LIMIT))
        info["target_track_pos"] = target_track_pos
        info["lane_error"] = lane_error
        info["lane_steer"] = lane_steer

        if track_pos > target_track_pos + SECOND_SECTOR_LEFT_MARGIN:
            raw_steer = min(raw_steer, lane_steer)
            accel_cmd = min(accel_cmd, 0.35)
            info["active"] = True
            info["reasons"].append("sector_700_820_lane_left")
        elif track_pos < target_track_pos - SECOND_SECTOR_RIGHT_MARGIN:
            raw_steer = max(raw_steer, lane_steer)
            accel_cmd = min(accel_cmd, 0.55)
            info["active"] = True
            info["reasons"].append("sector_700_820_lane_right")

    if 815.0 <= lap_dist <= 850.0 and abs(track_pos) > SECOND_SECTOR_HEADING_EDGE:
        target_track_pos = _interp_profile(SECOND_SECTOR_LANE_PROFILE, lap_dist)
        heading_lane_error = target_track_pos - track_pos
        heading_steer = float(np.clip(heading_lane_error * 0.45 + angle * 0.40, -0.55, 0.55))
        info["target_track_pos"] = target_track_pos
        info["heading_lane_error"] = heading_lane_error
        info["heading_steer"] = heading_steer

        if track_pos > 0.65 and angle < -0.18:
            raw_steer = min(raw_steer, heading_steer)
            accel_cmd = min(accel_cmd, 0.35)
            info["active"] = True
            info["reasons"].append("sector_800_heading_left")
        elif track_pos < -0.65 and angle > 0.18:
            raw_steer = max(raw_steer, heading_steer)
            accel_cmd = min(accel_cmd, 0.35)
            info["active"] = True
            info["reasons"].append("sector_800_heading_right")

    if THIRD_SECTOR_LANE_PROFILE[0][0] <= lap_dist <= THIRD_SECTOR_LANE_PROFILE[-1][0]:
        target_track_pos = _interp_profile(THIRD_SECTOR_LANE_PROFILE, lap_dist)
        lane_error = target_track_pos - track_pos
        lane_steer = float(np.clip(
            lane_error * THIRD_SECTOR_LANE_GAIN + angle * THIRD_SECTOR_ANGLE_GAIN,
            -THIRD_SECTOR_LANE_LIMIT, THIRD_SECTOR_LANE_LIMIT))
        info["target_track_pos"] = target_track_pos
        info["third_sector_lane_error"] = lane_error
        info["third_sector_lane_steer"] = lane_steer

        if track_pos > target_track_pos + THIRD_SECTOR_LANE_MARGIN:
            raw_steer = min(raw_steer, lane_steer)
            accel_cmd = min(accel_cmd, 0.35)
            info["active"] = True
            info["reasons"].append("sector_1450_1660_lane_right")
        elif track_pos < target_track_pos - THIRD_SECTOR_LANE_MARGIN:
            raw_steer = max(raw_steer, lane_steer)
            accel_cmd = min(accel_cmd, 0.35)
            info["active"] = True
            info["reasons"].append("sector_1450_1660_lane_left")

        if abs(track_pos) > 0.82 and speed_x > 120.0:
            accel_cmd = min(accel_cmd, 0.10)
            brake_cmd = max(brake_cmd, 0.10)
            info["active"] = True
            info["reasons"].append("sector_1450_1660_edge_lift")

    if speed_x > 45.0 and (abs(speed_y) > 11.0 or abs(angle) > 0.65):
        speed_y_term = 0.0
        angle_term = 0.0
        if abs(speed_y) > 11.0:
            speed_y_term = -np.sign(speed_y) * min(0.75, abs(speed_y) * 0.025)
        if abs(angle) > 0.35:
            angle_term = -np.sign(angle) * min(0.30, abs(angle) * 0.20)
        stability_steer = float(np.clip(speed_y_term + angle_term, -0.85, 0.85))
        raw_steer = float(np.clip(raw_steer * 0.35 + stability_steer * 0.65, -0.95, 0.95))
        accel_cmd = min(accel_cmd, 0.05)
        if speed_x > 80.0:
            brake_cmd = max(brake_cmd, min(0.55, 0.12 + abs(speed_y) / 110.0))
        info["active"] = True
        info["stability_steer"] = stability_steer
        info["speed_y"] = speed_y
        info["reasons"].append("stability_control")

    if abs(track_pos) > 1.0:
        raw_steer = float(np.clip(-np.sign(track_pos) * min(1.0, 0.45 + 0.20 * (abs(track_pos) - 1.0)), -1.0, 1.0))
        accel_cmd = 0.0
        if speed_x > 30.0:
            brake_cmd = max(brake_cmd, 0.35)
        info["active"] = True
        info["reasons"].append("offtrack_recovery")

    if abs(track_pos) > 0.90 and speed_x > 80.0:
        accel_cmd = min(accel_cmd, 0.10)
        info["active"] = True
        info["reasons"].append("near_edge_lift")

    front_clear = front_track is None or front_track > 35.0
    if 850.0 <= lap_dist <= 980.0 and abs(track_pos) < 0.85 and speed_x < 75.0 and abs(angle) < 0.35 and front_clear:
        recovery_accel = float(np.clip(0.30 + (75.0 - speed_x) / 120.0, 0.30, 0.60))
        accel_cmd = max(accel_cmd, recovery_accel)
        brake_cmd = min(brake_cmd, 0.05)
        info["active"] = True
        info["reasons"].append("post_recovery_throttle")

    info["accel_after"] = accel_cmd
    info["brake_after"] = brake_cmd
    info["steer_after"] = raw_steer
    return raw_steer, accel_cmd, brake_cmd, info


# ============================================================
# STATE: 17-dim
# ============================================================
def get_state(S, lap_start_dist=0.0):
    track = S.get('track', [100.0] * 19)

    dist_raced = float(S.get('distRaced', 0.0))
    if lap_start_dist is None:
        lap_start_dist = 0.0
    lap_distance = dist_raced - float(lap_start_dist)
    if not np.isfinite(lap_distance):
        lap_distance = 0.0
    # TORCS can report distFromStart near the lap end while the car is on the
    # grid. Episode-relative distRaced keeps lap_pos=0 at launch.
    if lap_distance < 0.0:
        lap_distance = max(0.0, dist_raced)

    lap_pos = (lap_distance % LAP_THRESHOLD) / LAP_THRESHOLD
    lap_angle = 2.0 * PI * lap_pos

    return np.array([
        float(S.get('angle', 0.0)) / PI,
        float(S.get('trackPos', 0.0)),
        float(S.get('speedX', 0.0)) / 300.0,
        float(S.get('speedY', 0.0)) / 300.0,
        float(S.get('rpm', 0.0)) / 10000.0,
        lap_pos,
        np.sin(lap_angle),
        np.cos(lap_angle),
        float(track[0]) / 200.0,
        float(track[3]) / 200.0,
        float(track[5]) / 200.0,
        float(track[7]) / 200.0,
        float(track[9]) / 200.0,
        float(track[11]) / 200.0,
        float(track[13]) / 200.0,
        float(track[16]) / 200.0,
        float(track[18]) / 200.0,
    ], dtype=np.float32)


def prepare_training_samples(raw_data, source_name):
    """Normalize fresh 17D and legacy 20D samples into the current 17D format."""
    prepared = []
    group_ids = []
    legacy_count = 0
    group_id = 0
    group_start_lap_pos = None
    prev_lap_pos = None

    for idx, sample in enumerate(raw_data):
        state = [float(x) for x in sample.get('state', [])]
        action = [float(x) for x in sample.get('action', [])]
        state_dim = len(state)

        if state_dim not in (STATE_DIM, LEGACY_STATE_DIM):
            raise ValueError(
                f"{source_name}[{idx}] has state_dim={state_dim}, "
                f"expected {STATE_DIM} or legacy {LEGACY_STATE_DIM}."
            )
        if len(action) != ACTION_DIM:
            raise ValueError(
                f"{source_name}[{idx}] has action_dim={len(action)}, "
                f"expected {ACTION_DIM}."
            )

        lap_pos_raw = state[5] if len(state) > 5 else 0.0
        if prev_lap_pos is not None and lap_pos_raw + 0.10 < prev_lap_pos:
            group_id += 1
            group_start_lap_pos = None

        if state_dim == LEGACY_STATE_DIM:
            legacy_count += 1
            if group_start_lap_pos is None:
                group_start_lap_pos = lap_pos_raw
            lap_pos = (lap_pos_raw - group_start_lap_pos) % 1.0
            state = state[:STATE_DIM]
            state[5] = lap_pos
            state[6] = float(np.sin(2.0 * PI * lap_pos))
            state[7] = float(np.cos(2.0 * PI * lap_pos))
        elif group_start_lap_pos is None:
            group_start_lap_pos = 0.0

        prepared.append({
            'state': state,
            'action': action,
            'source': source_name,
        })
        group_ids.append(f"{source_name}:{group_id}")
        prev_lap_pos = lap_pos_raw

    return prepared, group_ids, legacy_count


def split_by_lap_groups(group_ids, min_lap_samples=500, val_fraction=0.20):
    """Hold out whole lap-like groups so validation is not random-frame leakage."""
    group_order = []
    group_counts = {}
    for gid in group_ids:
        if gid not in group_counts:
            group_order.append(gid)
            group_counts[gid] = 0
        group_counts[gid] += 1

    eligible = [gid for gid in group_order if group_counts[gid] >= min_lap_samples]
    if not eligible and len(group_order) > 1:
        eligible = group_order
    if not eligible:
        return list(range(len(group_ids))), [], group_counts, []

    n_val = max(1, int(round(len(eligible) * val_fraction)))
    val_groups = set(eligible[-n_val:])
    train_idx = [i for i, gid in enumerate(group_ids) if gid not in val_groups]
    val_idx = [i for i, gid in enumerate(group_ids) if gid in val_groups]

    if not train_idx:
        split = max(1, int(len(group_ids) * (1.0 - val_fraction)))
        train_idx = list(range(split))
        val_idx = list(range(split, len(group_ids)))
        val_groups = set(group_ids[i] for i in val_idx)

    return train_idx, val_idx, group_counts, list(val_groups)


# ============================================================
# TARGET SPEED — unchanged
# ============================================================
def get_target_speed(track):
    front = [track[i] for i in range(7, 12)]
    front_min = min(front)
    wide_front = [track[i] for i in range(5, 14)]
    wide_min = min(wide_front)
    look_ahead = min(front_min, wide_min)

    if look_ahead > 180:   return 320
    elif look_ahead > 150: return 300
    elif look_ahead > 100: return 250
    elif look_ahead > 80:  return 220
    elif look_ahead > 50:  return 190
    elif look_ahead > 38:  return 160
    elif look_ahead > 30:  return 140
    elif look_ahead > 25:  return 130
    elif look_ahead > 18:  return 105
    elif look_ahead > 12:  return 85
    elif look_ahead > 8:   return 65
    else:                  return 45


def detect_corner_direction(track):
    left_sum  = sum(track[0:7])
    right_sum = sum(track[12:19])
    diff = right_sum - left_sum
    if abs(diff) < 30:
        return 0
    return 1 if diff > 0 else -1


# ============================================================
# BRAKE + GEARS — used only during collect, not in play
# ============================================================
def apply_brake_and_gears(S, R):
    speed_x    = float(S.get('speedX', 0))
    angle      = float(S.get('angle', 0))
    speed_y    = float(S.get('speedY', 0))
    track      = S.get('track', [100.0] * 19)
    wheel_spin = S.get('wheelSpinVel', [0, 0, 0, 0])

    target_speed = get_target_speed(track)
    front_min = min(track[7], track[8], track[9], track[10], track[11])

    if front_min < 50 and speed_x > 140:
        R['brake'] = 0.7; R['accel'] = 0.0
        _apply_gears(speed_x, R); return
    if front_min < 30 and speed_x > 100:
        R['brake'] = 0.9; R['accel'] = 0.0
        _apply_gears(speed_x, R); return

    if speed_x > target_speed:
        overspeed = speed_x - target_speed
        if overspeed > 80:
            R['brake'] = 1.0; R['accel'] = 0.0
        elif overspeed > 50:
            R['brake'] = 0.7; R['accel'] = 0.0
        elif overspeed > 25:
            R['brake'] = 0.4; R['accel'] = 0.0
        else:
            R['brake'] = 0.15
    else:
        R['brake'] = 0.0

    if abs(angle) > 0.5:
        R['brake'] = max(R.get('brake', 0), 0.7); R['accel'] = 0.0
    elif abs(angle) > 0.3:
        R['brake'] = max(R.get('brake', 0), 0.3)
        R['accel'] = min(R.get('accel', 0), 0.2)

    if abs(speed_y) > 30:
        R['brake'] = max(R.get('brake', 0), 0.2)
        R['accel'] = min(R.get('accel', 0), 0.3)

    if speed_x > 10:
        slip = sum(abs(wheel_spin[i] * 0.3 - speed_x) for i in range(4))
        if slip / 4 > speed_x * 0.5:
            R['brake'] = R.get('brake', 0) * 0.6

    if (wheel_spin[2] + wheel_spin[3]) - (wheel_spin[0] + wheel_spin[1]) > 8:
        R['accel'] = max(0.0, R.get('accel', 0) - 0.15)

    _apply_gears(speed_x, R)


def _apply_gears(speed_x, R):
    gear = 1
    if speed_x > 40:  gear = 2
    if speed_x > 70:  gear = 3
    if speed_x > 100: gear = 4
    if speed_x > 135: gear = 5
    if speed_x > 170: gear = 6
    R['gear'] = gear


# ============================================================
# RULE-BASED DRIVER — steer + accel (unchanged)
# ============================================================
STEER_LOCK = 0.366

def rule_based_steer_accel(S):
    angle     = float(S.get('angle', 0.0))
    track_pos = float(S.get('trackPos', 0.0))
    speed_x   = float(S.get('speedX', 0.0))
    track     = S.get('track', [100.0] * 19)

    target_speed = get_target_speed(track)

    steer = angle * 0.8 / STEER_LOCK - track_pos * 0.35

    corner_dir = detect_corner_direction(track)
    if corner_dir != 0 and speed_x > 40 and abs(track_pos) < 0.4:
        steer -= corner_dir * 0.1

    steer = max(-1.0, min(1.0, steer))

    speed_diff = target_speed - speed_x
    if speed_x < target_speed:
        if speed_diff > 60:    accel = 1.0
        elif speed_diff > 30:  accel = 0.8
        elif speed_diff > 10:  accel = 0.5
        else:                  accel = 0.3
    else:
        accel = 0.0

    if speed_x < 10: accel = max(accel, 1.0)
    if abs(angle) > 0.7: accel = 0.0
    elif abs(angle) > 0.4: accel = min(accel, 0.2)

    if abs(track_pos) > 0.95:
        steer = -track_pos * 0.5; accel = 0.15
    if abs(track_pos) > 1.0:
        steer = -track_pos * 0.8; accel = 0.05

    look = min(min(track[7:12]), min(track[5:14]))
    if abs(track_pos) > 0.7 and look < 33 and accel > 0.2:
        accel = 0.15

    return np.array([steer, accel], dtype=np.float32)


# ============================================================
# NEW: rule_based_action_with_brake
# Returns [steer, accel, brake] — used only for data collection
# ============================================================
def rule_based_action_with_brake(S):
    """Return [steer, accel, brake] for BC training."""
    action_2 = rule_based_steer_accel(S)
    steer = float(action_2[0])
    accel = float(action_2[1])

    # Compute brake using apply_brake_and_gears logic
    R = {'steer': steer, 'accel': accel, 'brake': 0.0}
    apply_brake_and_gears(S, R)
    brake = float(R.get('brake', 0.0))
    # Sync accel with what apply_brake_and_gears decided
    accel = float(R.get('accel', accel))

    return np.array([steer, accel, brake], dtype=np.float32)


def rule_based_drive_full(S, R):
    """Full rule-based drive for collect — no print."""
    action = rule_based_action_with_brake(S)
    R['steer'] = float(action[0])
    R['accel'] = float(action[1])
    R['brake'] = float(action[2])
    _apply_gears(float(S.get('speedX', 0)), R)


# ============================================================
# PHASE 1: COLLECT DATA
# ============================================================
def collect_data(num_laps=50, max_steps=500000):
    print("\n" + "=" * 60)
    print("  PHASE 1: Collecting data (aggressive driver)")
    print("  Make sure TORCS is running with Corkscrew track!")
    print("=" * 60)

    
    C = snakeoil3.Client(p=PORT)
    C.MAX_STEPS = max_steps

    # Load existing data if available
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            all_data = json.load(f)
        if all_data and len(all_data[0].get('state', [])) not in (STATE_DIM, LEGACY_STATE_DIM):
            print(f"  Incompatible existing {DATA_FILE}: state_dim={len(all_data[0].get('state', []))}, expected {STATE_DIM}.")
            print("  Move/delete old data before collecting in the new format.")
            C.shutdown()
            return
        print(f"  Loaded {len(all_data)} existing samples, continuing...")
    else:
        all_data = []


    lap_count = 0
    prev_last_lap = 0.0
    lap_start_dist = None

    for step in range(max_steps, 0, -1):
        C.get_servers_input()
        S = C.S.d
        if lap_start_dist is None:
            lap_start_dist = float(S.get('distRaced', 0.0))

        state = get_state(S, lap_start_dist)
        action = rule_based_action_with_brake(S)  # [steer, accel, brake]
        all_data.append({
            'state': state.tolist(),
            'action': action.tolist(),
        })

        rule_based_drive_full(S, C.R.d)

        dist  = float(S.get('distRaced', 0.0))
        speed = float(S.get('speedX', 0.0))
        cur_step = max_steps - step

        if cur_step % 500 == 0 and cur_step > 0:
            print(f"    step {cur_step:5d} | dist={dist:6.0f}m | "
                  f"speed={speed:5.1f}km/h | "
                  f"tpos={S.get('trackPos', 0):+.3f}")

        last_lap = float(S.get('lastLapTime', 0.0))
        if last_lap > 0 and last_lap != prev_last_lap:
            lap_count += 1
            prev_last_lap = last_lap
            print(f"\n  Lap {lap_count} completed! Time: {last_lap:.2f}s | "
                  f"Total dist: {dist:.0f}m")
            if lap_count >= num_laps:
                print(f"  Collected {num_laps} laps, stopping.")
                break

        C.respond_to_server()

    C.shutdown()

    with open(DATA_FILE, 'w') as f:
        json.dump(all_data, f)
    print(f"\n  Collected {len(all_data)} samples -> {DATA_FILE}")
    # Verify brake distribution
    import json as _json
    with open(DATA_FILE) as f:
        d = _json.load(f)
    brakes = [x['action'][2] for x in d]
    brake_nonzero = sum(1 for b in brakes if b > 0.05)
    print(f"  Brake>0.05 samples: {brake_nonzero} ({100*brake_nonzero/len(brakes):.1f}%)")


# ============================================================
# PHASE 2: BEHAVIORAL CLONING — unchanged except action_dim=3
# ============================================================
def train_bc(epochs=500, batch_size=256):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from model import Actor

    print("\n" + "=" * 60)
    print("  PHASE 2: Behavioral Cloning")
    print("=" * 60)

    if not os.path.exists(DATA_FILE):
        print(f"  No data! Run: python train.py collect")
        return

    with open(DATA_FILE, 'r') as f:
        base_data = json.load(f)
    if not base_data:
        print(f"  Empty data file: {DATA_FILE}")
        return

    try:
        base_prepared, base_group_ids, base_legacy = prepare_training_samples(
            base_data, DATA_FILE)
    except ValueError as e:
        print(f"  Incompatible data: {e}")
        return

    train_idx, val_idx, group_counts, val_groups = split_by_lap_groups(
        base_group_ids)
    train_data = [base_prepared[i] for i in train_idx]
    val_data = [base_prepared[i] for i in val_idx]

    print(f"  Loaded {len(base_prepared)} base samples")
    print(f"  Current state dim: {STATE_DIM} (legacy {LEGACY_STATE_DIM}D samples are converted)")
    if base_legacy:
        print(f"  Converted legacy base samples: {base_legacy} (dropped prev_action, rebased lap_pos)")
    print(f"  Lap-like groups: {len(group_counts)} | validation groups: {len(val_groups)}")
    if val_groups:
        val_sizes = [group_counts[g] for g in val_groups]
        print(f"  Validation samples: {len(val_data)} | group sizes: {val_sizes}")
    else:
        print("  Validation disabled: not enough lap groups")

    if os.path.exists(CORRECTION_FILE):
        with open(CORRECTION_FILE, 'r') as f:
            correction_data = json.load(f)
        if correction_data:
            try:
                correction_prepared, _, correction_legacy = prepare_training_samples(
                    correction_data, CORRECTION_FILE)
            except ValueError as e:
                print(f"  Incompatible corrections: {e}")
                return
            train_data.extend(correction_prepared * CORRECTION_REPEAT)
            print(f"  Loaded {len(correction_prepared)} correction samples")
            if correction_legacy:
                print(f"  Converted legacy correction samples: {correction_legacy}")
            print(f"  Correction weight: x{CORRECTION_REPEAT}")
            print(f"  Effective training samples: {len(train_data)}")
        else:
            print(f"  Correction file is empty: {CORRECTION_FILE}")
    else:
        print(f"  No correction file found: {CORRECTION_FILE}")

    use_guardian_training = "--with-guardian" in COMMAND_FLAGS
    if os.path.exists(GUARDIAN_FILE):
        if use_guardian_training:
            with open(GUARDIAN_FILE, 'r') as f:
                guardian_data = json.load(f)
            if guardian_data:
                try:
                    guardian_prepared, _, guardian_legacy = prepare_training_samples(
                        guardian_data, GUARDIAN_FILE)
                except ValueError as e:
                    print(f"  Incompatible guardian samples: {e}")
                    return
                train_data.extend(guardian_prepared * GUARDIAN_REPEAT)
                print(f"  Loaded {len(guardian_prepared)} guardian intervention samples")
                if guardian_legacy:
                    print(f"  Converted legacy guardian samples: {guardian_legacy}")
                print(f"  Guardian weight: x{GUARDIAN_REPEAT}")
                print(f"  Effective training samples: {len(train_data)}")
            else:
                print(f"  Guardian file is empty: {GUARDIAN_FILE}")
        else:
            print(f"  Guardian samples found but not used: {GUARDIAN_FILE}")
            print("  To include them deliberately, run: python train.py bc --with-guardian")
    else:
        print(f"  No guardian file found: {GUARDIAN_FILE}")

    if os.path.exists(CORRECTION_PENDING_FILE):
        print(f"  Pending corrections found: {CORRECTION_PENDING_FILE}")
        print(f"  Run: python combine_corrections.py before final correction training")

    print(f"  Action dim: {len(train_data[0]['action'])} (steer, accel, brake)")

    train_states  = torch.FloatTensor([d['state']  for d in train_data])
    train_actions = torch.FloatTensor([d['action'] for d in train_data])
    val_states = torch.FloatTensor([d['state'] for d in val_data]) if val_data else None
    val_actions = torch.FloatTensor([d['action'] for d in val_data]) if val_data else None

    # ── Loss weighting ────────────────────────────────────────────────────
    # Human braking is rare (~6% of frames) but sharp (~0.66 when present), so
    # plain per-frame MSE averages it toward zero and the brake head goes dead
    # (it predicts ~0 through corner-entry braking zones -> overspeed -> off).
    # Two corrections:
    #   1. per-sample weight rises with brake magnitude, so braking frames are
    #      not drowned by the mass of straight-line throttle frames;
    #   2. per-output weight emphasizes the chronically under-predicted brake
    #      head relative to steer/accel.
    brake_sample_weight = get_int_flag("--brake-weight", 10)
    brake_output_weight = get_int_flag("--brake-out-weight", 3)
    output_weights = torch.FloatTensor([1.0, 1.0, float(brake_output_weight)])

    train_weights = 1.0 + float(brake_sample_weight) * train_actions[:, 2]
    braking_mask = train_actions[:, 2] > 0.05
    n_braking = int(braking_mask.sum().item())
    n_total = len(train_actions)
    mean_w_brake = float(train_weights[braking_mask].mean()) if n_braking else 1.0
    mean_w_rest = float(train_weights[~braking_mask].mean()) if n_braking < n_total else 1.0
    print(f"  Loss weighting: per-sample w = 1 + {brake_sample_weight}*brake "
          f"| brake output x{brake_output_weight}")
    print(f"  Braking frames (brake>0.05): {n_braking}/{n_total} "
          f"({100.0 * n_braking / max(1, n_total):.1f}%) "
          f"| mean weight braking={mean_w_brake:.2f} non-braking={mean_w_rest:.2f}")

    actor     = Actor(STATE_DIM)
    optimizer = optim.Adam(actor.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)
    best_score = float('inf')

    def evaluate(states_t, actions_t):
        actor.eval()
        sum_sq = torch.zeros(ACTION_DIM)
        count = 0
        with torch.no_grad():
            for i in range(0, len(states_t), 4096):
                batch_s = states_t[i:i + 4096]
                batch_a = actions_t[i:i + 4096]
                diff_sq = (actor(batch_s) - batch_a) ** 2
                sum_sq += diff_sq.sum(dim=0).cpu()
                count += diff_sq.shape[0]
        comp = sum_sq / max(1, count)
        return float(comp.mean().item()), [float(x) for x in comp.tolist()]

    def fmt_components(values):
        return f"steer={values[0]:.6f} accel={values[1]:.6f} brake={values[2]:.6f}"

    ow = [float(x) for x in output_weights.tolist()]

    def weighted_score(comp):
        # Model-selection metric: weight the brake component so we don't save a
        # checkpoint that minimizes average MSE by quietly never braking.
        return sum(c * w for c, w in zip(comp, ow)) / sum(ow)

    for epoch in range(epochs):
        actor.train()
        perm      = torch.randperm(len(train_states))
        states_s  = train_states[perm]
        actions_s = train_actions[perm]
        weights_s = train_weights[perm]
        total_loss = 0
        n_batches  = 0

        for i in range(0, len(states_s), batch_size):
            batch_s = states_s[i:i + batch_size]
            batch_a = actions_s[i:i + batch_size]
            batch_w = weights_s[i:i + batch_size]
            pred    = actor(batch_s)
            se          = (pred - batch_a) ** 2 * output_weights   # [B, 3] per-output weighted
            per_sample  = se.mean(dim=1)                           # [B]
            loss        = (per_sample * batch_w).sum() / batch_w.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        batch_loss = total_loss / n_batches

        train_loss, train_comp = evaluate(train_states, train_actions)
        train_score = weighted_score(train_comp)
        if val_states is not None:
            val_loss, val_comp = evaluate(val_states, val_actions)
            val_score = weighted_score(val_comp)
            score = val_score
        else:
            val_loss, val_comp = None, None
            score = train_score

        if epoch % 10 == 0 or score < best_score:
            marker = ""
            if score < best_score:
                best_score = score
                torch.save(actor.state_dict(), "bc_model.pth")
                marker = " <- BEST saved"
            msg = (
                f"  epoch {epoch:4d} | batch={batch_loss:.6f} "
                f"| train={train_loss:.6f} ({fmt_components(train_comp)})"
            )
            if val_loss is not None:
                msg += f" | val={val_loss:.6f} ({fmt_components(val_comp)})"
            msg += f" | sel_score={score:.6f}"
            print(msg + marker)

    score_name = "validation" if val_states is not None else "training"
    print(f"\n  BC complete! Best weighted {score_name} score={best_score:.6f} -> bc_model.pth")



# ============================================================
# PLAY MODE — AI controls steer, accel, brake. Code: gears only.
# ============================================================
def play_model():
    import torch
    from model import Actor

    print("\n" + "=" * 60)
    print("  PLAY MODE — AI controls steer, accel, brake")
    print("=" * 60)

    actor = Actor(STATE_DIM)

    loaded = False
    model_file = None
    for fname in ["bc_model.pth"]:
        if os.path.exists(fname):
            try:
                if fname == "bc_model.pth":
                    actor.load_state_dict(
                        torch.load(fname, map_location='cpu', weights_only=False))
                else:
                    ck = torch.load(fname, map_location='cpu', weights_only=False)
                    actor.load_state_dict(ck['actor'])
            except RuntimeError as e:
                print(f"  Incompatible model file: {fname}")
                print(f"  Current STATE_DIM={STATE_DIM}. Retrain with fresh data: python train.py bc")
                print(f"  Details: {e}")
                return
            print(f"  Loaded: {fname}")
            loaded = True
            model_file = fname
            break

    if not loaded:
        print("  NO MODEL FOUND!")
        return

    actor.eval()

    use_hybrid = "--hybrid" in COMMAND_FLAGS
    use_racing_line_guardian = (
        "--racing-line-guardian" in COMMAND_FLAGS or
        "--rlg" in COMMAND_FLAGS
    ) and not use_hybrid
    start_guard_requested = "--start-guard" in COMMAND_FLAGS
    use_start_guard = start_guard_requested or use_racing_line_guardian or use_hybrid
    use_steer_smoothing = "--raw-steer" not in COMMAND_FLAGS
    use_guardian_assist = "--guardian-assist" in COMMAND_FLAGS and not use_racing_line_guardian and not use_hybrid
    launch_lane_keeper_requested = "--launch-lane-keeper" in COMMAND_FLAGS
    use_launch_lane_keeper = (
        launch_lane_keeper_requested or
        use_guardian_assist
    )
    guardian_collect_requested = "--guardian-collect" in COMMAND_FLAGS
    use_guardian_collect = (
        (use_guardian_assist or use_racing_line_guardian) and
        guardian_collect_requested
    )
    use_global_assist = (
        ("--global-assist" in COMMAND_FLAGS or "--assist" in COMMAND_FLAGS) and
        not use_guardian_assist and
        not use_racing_line_guardian and
        not use_hybrid
    )
    use_corner_safety = (
        "--corner-safety" in COMMAND_FLAGS or "--speed-profile" in COMMAND_FLAGS
    ) and not use_global_assist and not use_guardian_assist and not use_racing_line_guardian and not use_hybrid
    human_envelope = (
        load_human_racing_envelope()
        if (use_guardian_assist or use_racing_line_guardian)
        else None
    )
    if use_guardian_assist or use_racing_line_guardian:
        if human_envelope is None:
            print("  Human racing envelope: unavailable")
        else:
            sectors = human_envelope.get("sectors", {})
            points = sectors.get("whole_track", {}).get("points", [])
            if use_racing_line_guardian and not use_guardian_assist:
                print(f"  Human speed/safety profile: ON ({len(points)} whole-track points)")
            else:
                print(f"  Human racing envelope: ON ({len(points)} whole-track points)")
    if use_racing_line_guardian:
        print("  Racing Line Guardian: simple rule-based track supervisor ON")

    C = snakeoil3.Client(p=PORT)
    C.MAX_STEPS = 50000
    C.get_servers_input()
    S = C.S.d

    dist_start = float(S.get('distRaced', 0))
    lap_start_dist = dist_start
    steps = 0
    prev_steer = 0.0
    prev_cmd = [0.0, 0.0, 0.0]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(PLAY_LOG_DIR, exist_ok=True)
    play_log_path = os.path.join(PLAY_LOG_DIR, f"play_{run_id}.jsonl")
    latest_log_path = os.path.join(PLAY_LOG_DIR, "play_latest.jsonl")
    play_log_files = [
        open(play_log_path, "w", encoding="utf-8", buffering=1),
        open(latest_log_path, "w", encoding="utf-8", buffering=1),
    ]

    def write_play_log(event):
        event = dict(event)
        event.setdefault("time", datetime.now().isoformat(timespec="milliseconds"))
        line = json.dumps(event, ensure_ascii=False)
        for handle in play_log_files:
            handle.write(line + "\n")

    def close_play_logs():
        for handle in play_log_files:
            if not handle.closed:
                handle.close()

    import atexit
    atexit.register(close_play_logs)

    guardian_samples = []
    guardian_samples_saved = False

    def save_guardian_samples():
        nonlocal guardian_samples_saved
        if guardian_samples_saved or not use_guardian_collect or not guardian_samples:
            return
        existing_samples = []
        if os.path.exists(GUARDIAN_FILE):
            try:
                with open(GUARDIAN_FILE, "r", encoding="utf-8") as f:
                    loaded_samples = json.load(f)
                if isinstance(loaded_samples, list):
                    existing_samples = loaded_samples
                else:
                    print(f"  Guardian save skipped: {GUARDIAN_FILE} is not a JSON list")
                    return
            except (OSError, json.JSONDecodeError) as e:
                print(f"  Guardian save skipped: cannot read {GUARDIAN_FILE}: {e}")
                return
        existing_samples.extend(guardian_samples)
        with open(GUARDIAN_FILE, "w", encoding="utf-8") as f:
            json.dump(existing_samples, f)
        guardian_samples_saved = True
        print(f"  Guardian samples saved: +{len(guardian_samples)} -> {os.path.abspath(GUARDIAN_FILE)}")

    atexit.register(save_guardian_samples)

    write_play_log({
        "type": "meta",
        "run_id": run_id,
        "command": " ".join(["train.py"] + COMMAND_ARGS),
        "state_dim": STATE_DIM,
        "model_file": model_file,
        "flags": sorted(COMMAND_FLAGS),
        "start_guard": use_start_guard,
        "start_guard_requested": start_guard_requested,
        "start_guard_forced_by_racing_line_guardian": (
            use_racing_line_guardian and not start_guard_requested
        ),
        "start_guard_forced_by_hybrid": (
            use_hybrid and not start_guard_requested
        ),
        "steer_smoothing": use_steer_smoothing,
        "launch_lane_keeper": use_launch_lane_keeper,
        "launch_lane_keeper_requested": launch_lane_keeper_requested,
        "launch_lane_keeper_forced_by_guardian": (
            use_guardian_assist and
            not launch_lane_keeper_requested
        ),
        "racing_line_guardian": use_racing_line_guardian,
        "racing_line_guardian_meta": (
            {
                "type": "simple_rule_based_track_supervisor",
                "uses_human_envelope": human_envelope is not None,
                "uses_human_speed_profile": human_envelope is not None,
                "follows_human_line": False,
                "controls": ["steer", "accel", "brake"],
                "purpose": "prevent off-track and slow unstable corner entries",
            }
            if use_racing_line_guardian else None
        ),
        "hybrid_assist": use_hybrid,
        "hybrid_launch_end": HYBRID_LAUNCH_END,
        "hybrid_first_corner_start": HYBRID_FIRST_CORNER_START,
        "hybrid_first_corner_end": HYBRID_FIRST_CORNER_END,
        "hybrid_first_corner_speed_profile": HYBRID_FIRST_CORNER_SPEED_PROFILE if use_hybrid else None,
        "hybrid_first_corner_lane_profile": HYBRID_FIRST_CORNER_LANE_PROFILE if use_hybrid else None,
        "hybrid_steer_rate": HYBRID_STEER_RATE,
        "hybrid_hard_steer_rate": HYBRID_HARD_STEER_RATE,
        "guardian_assist": use_guardian_assist,
        "guardian_collect": use_guardian_collect,
        "guardian_collect_requested_without_assist": (
            guardian_collect_requested and
            not (use_guardian_assist or use_racing_line_guardian)
        ),
        "global_assist": use_global_assist,
        "corner_safety": use_corner_safety,
        "corner_safety_profiles_disabled_by_global_or_guardian_assist": (
            (use_global_assist or use_guardian_assist or use_racing_line_guardian or use_hybrid) and
            ("--corner-safety" in COMMAND_FLAGS or "--speed-profile" in COMMAND_FLAGS)
        ),
        "corner_speed_profile": FIRST_CORNER_SPEED_PROFILE if use_corner_safety else None,
        "first_corner_lane_profile": FIRST_CORNER_LANE_PROFILE if use_corner_safety else None,
        "sector_700_820_speed_profile": SECOND_SECTOR_SPEED_PROFILE if use_corner_safety else None,
        "sector_700_820_lane_profile": SECOND_SECTOR_LANE_PROFILE if use_corner_safety else None,
        "sector_1450_1660_speed_profile": THIRD_SECTOR_SPEED_PROFILE if use_corner_safety else None,
        "sector_1450_1660_lane_profile": THIRD_SECTOR_LANE_PROFILE if use_corner_safety else None,
        "corkscrew_speed_profile": CORKSCREW_SPEED_PROFILE if use_corner_safety else None,
        "launch_lane_corridor": LAUNCH_LANE_CORRIDOR if use_launch_lane_keeper else None,
        "base_steer_rate": BASE_STEER_RATE,
        "launch_steer_rate": LAUNCH_STEER_RATE,
        "global_assist_steer_rate": GLOBAL_ASSIST_STEER_RATE,
        "global_assist_center_start": GLOBAL_ASSIST_CENTER_START,
        "global_assist_edge_start": GLOBAL_ASSIST_EDGE_START,
        "guardian_file": GUARDIAN_FILE,
        "guardian_force_field_track": GUARDIAN_FORCE_FIELD_TRACK,
        "guardian_warn_track": GUARDIAN_WARN_TRACK,
        "guardian_edge_track": GUARDIAN_EDGE_TRACK,
        "guardian_abort_track": GUARDIAN_ABORT_TRACK,
        "guardian_heading_angle": GUARDIAN_HEADING_ANGLE,
        "guardian_launch_suppress_dist": GUARDIAN_LAUNCH_SUPPRESS_DIST,
        "guardian_speed_suppress_dist": GUARDIAN_SPEED_SUPPRESS_DIST,
        "guardian_lift_overspeed": GUARDIAN_LIFT_OVERSPEED,
        "guardian_strong_lift_overspeed": GUARDIAN_STRONG_LIFT_OVERSPEED,
        "guardian_racing_line_track": GUARDIAN_RACING_LINE_TRACK,
        "guardian_racing_line_angle": GUARDIAN_RACING_LINE_ANGLE,
        "guardian_racing_line_speed_y": GUARDIAN_RACING_LINE_SPEED_Y,
        "guardian_racing_line_target_bonus": GUARDIAN_RACING_LINE_TARGET_BONUS,
        "guardian_steer_rate": GUARDIAN_STEER_RATE,
        "guardian_hard_steer_rate": GUARDIAN_HARD_STEER_RATE,
        "racing_line_guardian_emergency_steer_rate": RACING_LINE_GUARDIAN_EMERGENCY_STEER_RATE,
        "simple_trajectory_min_dist": SIMPLE_TRAJECTORY_MIN_DIST,
        "racing_line_guardian_emergency_reasons": list(RACING_LINE_GUARDIAN_EMERGENCY_REASONS),
        "guardian_brake_decel_ms2": GUARDIAN_BRAKE_DECEL_MS2,
        "guardian_first_corner_trajectory_start": GUARDIAN_FIRST_CORNER_TRAJECTORY_START,
        "guardian_first_corner_trajectory_end": GUARDIAN_FIRST_CORNER_TRAJECTORY_END,
        "guardian_first_corner_trajectory_margin": GUARDIAN_FIRST_CORNER_TRAJECTORY_MARGIN,
        "guardian_first_corner_trajectory_strong_margin": GUARDIAN_FIRST_CORNER_TRAJECTORY_STRONG_MARGIN,
        "guardian_first_corner_late_steer_start": GUARDIAN_FIRST_CORNER_LATE_STEER_START,
        "guardian_first_corner_late_steer_cap": GUARDIAN_FIRST_CORNER_LATE_STEER_CAP,
        "guardian_first_corner_late_slip_steer_cap": GUARDIAN_FIRST_CORNER_LATE_SLIP_STEER_CAP,
        "human_envelope_speed_trust_track": HUMAN_ENVELOPE_SPEED_TRUST_TRACK,
        "human_envelope_major_angle": HUMAN_ENVELOPE_MAJOR_ANGLE,
        "human_envelope_major_speed_y": HUMAN_ENVELOPE_MAJOR_SPEED_Y,
        "human_action_steer_margin": HUMAN_ACTION_STEER_MARGIN,
        "human_action_brake_margin": HUMAN_ACTION_BRAKE_MARGIN,
        "human_action_accel_margin": HUMAN_ACTION_ACCEL_MARGIN,
        "launch_traction_dist": LAUNCH_TRACTION_DIST,
        "launch_traction_speed": LAUNCH_TRACTION_SPEED,
        "launch_traction_accel_profile": LAUNCH_TRACTION_ACCEL_PROFILE,
        "guardian_first_corner_speed_profile": GUARDIAN_FIRST_CORNER_SPEED_PROFILE if use_guardian_assist else None,
        "human_envelope_file": HUMAN_ENVELOPE_FILE,
        "human_envelope_enabled": human_envelope is not None,
        "human_envelope_summary": summarize_human_racing_envelope(human_envelope),
        "safety_steer_rate": SAFETY_STEER_RATE,
        "recovery_steer_rate": RECOVERY_STEER_RATE,
        "second_sector_lane_gain": SECOND_SECTOR_LANE_GAIN,
        "second_sector_angle_gain": SECOND_SECTOR_ANGLE_GAIN,
        "second_sector_left_margin": SECOND_SECTOR_LEFT_MARGIN,
        "second_sector_right_margin": SECOND_SECTOR_RIGHT_MARGIN,
        "second_sector_heading_edge": SECOND_SECTOR_HEADING_EDGE,
        "lap_threshold": LAP_THRESHOLD,
    })

    print(f"  Start guard: {'ON' if use_start_guard else 'off'}")
    if use_racing_line_guardian and not start_guard_requested:
        print("  Start guard forced by Racing Line Guardian")
    if use_hybrid and not start_guard_requested:
        print("  Start guard forced by Hybrid")
    print(f"  Steer smoothing: {'ON' if use_steer_smoothing else 'off'}")
    print(f"  Launch lane keeper: {'ON' if use_launch_lane_keeper else 'off'}")
    if use_guardian_assist and not launch_lane_keeper_requested:
        print("  Launch lane keeper forced by guardian")
    print(f"  Racing Line Guardian: {'ON' if use_racing_line_guardian else 'off'}")
    print(f"  Hybrid assist: {'ON' if use_hybrid else 'off'}")
    print(f"  Guardian assist: {'ON' if use_guardian_assist else 'off'}")
    if guardian_collect_requested and not (use_guardian_assist or use_racing_line_guardian):
        print("  Guardian collect ignored: add --racing-line-guardian or --guardian-assist to enable it")
    print(f"  Guardian collect: {'ON' if use_guardian_collect else 'off'}")
    print(f"  Global assist: {'ON' if use_global_assist else 'off'}")
    print(f"  Corner safety: {'ON' if use_corner_safety else 'off'}")
    print(f"  Play log: {os.path.abspath(play_log_path)}")

    print("  Driving...")

    dist_ep = 0.0
    stop_reason = "normal"

    while True:
        prev_cmd_used = [float(x) for x in prev_cmd]
        state = get_state(S, lap_start_dist)
        with torch.no_grad():
            a = actor(torch.FloatTensor(state).unsqueeze(0)).numpy()[0]

        speed_x = float(S.get('speedX', 0))
        speed_y = float(S.get('speedY', 0))
        track_pos = float(S.get('trackPos', 0))
        angle = float(S.get('angle', 0))
        rpm = float(S.get('rpm', 0))
        dist_raced = float(S.get('distRaced', 0))
        dist_from_start = float(S.get('distFromStart', 0))
        cur_lap_time = float(S.get('curLapTime', 0))
        dist_ep_before = dist_raced - dist_start
        raw_track = S.get('track', [])
        if not isinstance(raw_track, (list, tuple)):
            raw_track = []
        track = [float(x) for x in raw_track]
        abs_track_pos = abs(track_pos)
        off_track = abs_track_pos > 1.0
        near_edge = 0.85 <= abs_track_pos <= 1.0
        if track_pos > 0.05:
            track_side = "right"
        elif track_pos < -0.05:
            track_side = "left"
        else:
            track_side = "center"

        model_steer = float(np.clip(a[0], -1, 1))
        raw_steer = model_steer
        accel_cmd = float(np.clip(a[1], 0, 1))
        brake_cmd = float(np.clip(a[2], 0, 1))
        launch_helper_info = {"active": False}
        racing_line_guardian_info = {
            "active": False,
            "abort": False,
            "mode": "PASSIVE",
            "reasons": [],
        }
        guardian_assist_info = {"active": False, "abort": False}
        hybrid_assist_info = {"active": False, "abort": False, "mode": "PASSIVE", "reasons": []}
        global_assist_info = {"active": False}
        corner_safety_info = {"active": False}

        if use_start_guard and dist_ep_before < 8.0 and speed_x < 45.0:
            raw_steer = 0.0

        if use_launch_lane_keeper:
            raw_steer, accel_cmd, brake_cmd, launch_helper_info = apply_launch_lane_keeper(
                dist_ep_before, speed_x, speed_y, track_pos, angle, raw_steer, accel_cmd, brake_cmd)

        if use_hybrid:
            raw_steer, accel_cmd, brake_cmd, hybrid_assist_info = apply_hybrid_assist(
                dist_ep_before, speed_x, speed_y, track_pos, angle,
                raw_steer, accel_cmd, brake_cmd)

        if use_racing_line_guardian:
            raw_steer, accel_cmd, brake_cmd, racing_line_guardian_info = apply_rule_based_track_supervisor(
                dist_ep_before, speed_x, speed_y, track_pos, angle, track,
                raw_steer, accel_cmd, brake_cmd,
                model_action=[model_steer, float(np.clip(a[1], 0, 1)), float(np.clip(a[2], 0, 1))],
                human_envelope=human_envelope)

        if use_guardian_assist:
            raw_steer, accel_cmd, brake_cmd, guardian_assist_info = apply_guardian_assist(
                dist_ep_before, speed_x, speed_y, track_pos, angle, track,
                raw_steer, accel_cmd, brake_cmd, human_envelope=human_envelope)

        if use_global_assist:
            raw_steer, accel_cmd, brake_cmd, global_assist_info = apply_global_assist(
                speed_x, speed_y, track_pos, angle, track, raw_steer, accel_cmd, brake_cmd)

        if use_corner_safety:
            front_track = track[9] if len(track) > 9 else None
            raw_steer, accel_cmd, brake_cmd, corner_safety_info = apply_corner_safety(
                dist_ep_before, speed_x, speed_y, track_pos, angle, raw_steer, accel_cmd, brake_cmd, front_track)

        steer_cmd = raw_steer
        if use_steer_smoothing:
            steer_rate = BASE_STEER_RATE
            if launch_helper_info.get("active"):
                steer_rate = max(steer_rate, LAUNCH_STEER_RATE)
            hybrid_reasons = hybrid_assist_info.get("reasons", [])
            if use_hybrid and hybrid_assist_info.get("active"):
                if (
                    hybrid_assist_info.get("abort") or
                    "hybrid_launch_edge" in hybrid_reasons or
                    "hybrid_first_corner_edge" in hybrid_reasons or
                    "hybrid_first_corner_lane" in hybrid_reasons
                ):
                    steer_rate = max(steer_rate, HYBRID_HARD_STEER_RATE)
                else:
                    steer_rate = max(steer_rate, HYBRID_STEER_RATE)
            rlg_reasons = racing_line_guardian_info.get("reasons", [])
            if use_racing_line_guardian and racing_line_guardian_info.get("active"):
                rlg_mode = racing_line_guardian_info.get("mode")
                if (
                    racing_line_guardian_info.get("abort") or
                    any(reason in rlg_reasons for reason in RACING_LINE_GUARDIAN_EMERGENCY_REASONS)
                ):
                    steer_rate = max(steer_rate, RACING_LINE_GUARDIAN_EMERGENCY_STEER_RATE)
                elif rlg_mode in ("VETO", "RETURN") or "rlg_edge_veto" in rlg_reasons or "rlg_slip_veto" in rlg_reasons:
                    steer_rate = max(steer_rate, GUARDIAN_HARD_STEER_RATE)
                else:
                    steer_rate = max(steer_rate, GUARDIAN_STEER_RATE)
            guardian_reasons = guardian_assist_info.get("reasons", [])
            if use_guardian_assist and guardian_assist_info.get("active"):
                if "guardian_offtrack_abort" in guardian_reasons:
                    steer_rate = RECOVERY_STEER_RATE
                elif (
                    "guardian_edge_veto" in guardian_reasons or
                    "guardian_wall_repulsion" in guardian_reasons or
                    "guardian_slip_guard" in guardian_reasons or
                    "guardian_first_corner_trajectory" in guardian_reasons or
                    "guardian_first_corner_steer_cap" in guardian_reasons or
                    "guardian_first_corner_final_steer_cap" in guardian_reasons
                ):
                    steer_rate = max(steer_rate, GUARDIAN_HARD_STEER_RATE)
                else:
                    steer_rate = max(steer_rate, GUARDIAN_STEER_RATE)
            global_reasons = global_assist_info.get("reasons", [])
            if use_global_assist and global_assist_info.get("active"):
                if "global_offtrack_recovery" in global_reasons:
                    steer_rate = RECOVERY_STEER_RATE
                else:
                    steer_rate = max(steer_rate, GLOBAL_ASSIST_STEER_RATE)
            safety_reasons = corner_safety_info.get("reasons", [])
            if use_corner_safety and corner_safety_info.get("active"):
                if "offtrack_recovery" in safety_reasons:
                    steer_rate = RECOVERY_STEER_RATE
                else:
                    steer_rate = max(steer_rate, SAFETY_STEER_RATE)
            launch_helper_info["steer_rate"] = steer_rate
            hybrid_assist_info["steer_rate"] = steer_rate
            racing_line_guardian_info["steer_rate"] = steer_rate
            guardian_assist_info["steer_rate"] = steer_rate
            global_assist_info["steer_rate"] = steer_rate
            corner_safety_info["steer_rate"] = steer_rate
            steer_delta = np.clip(raw_steer - prev_steer, -steer_rate, steer_rate)
            steer_cmd = float(np.clip(prev_steer + steer_delta, -1, 1))
        prev_steer = steer_cmd
        prev_cmd = [steer_cmd, accel_cmd, brake_cmd]

        guardian_sample_info = (
            racing_line_guardian_info if use_racing_line_guardian
            else guardian_assist_info
        )
        if (
            use_guardian_collect and
            guardian_sample_info.get("active") and
            not guardian_sample_info.get("abort") and
            not off_track
        ):
            guardian_samples.append({
                "state": state.tolist(),
                "action": [steer_cmd, accel_cmd, brake_cmd],
                "source": (
                    "racing_line_guardian" if use_racing_line_guardian
                    else "guardian_assist"
                ),
                "dist_ep": dist_ep_before,
                "dist_from_start": dist_from_start,
                "speed_x": speed_x,
                "speed_y": speed_y,
                "track_pos": track_pos,
                "angle": angle,
                "reasons": list(guardian_sample_info.get("reasons", [])),
                "mode": guardian_sample_info.get("mode"),
                "risk_score": guardian_sample_info.get("risk_score"),
            })

        R = C.R.d
        R['steer'] = steer_cmd
        R['accel'] = accel_cmd
        R['brake'] = brake_cmd

        # Gears only — AI controls everything else
        gear = 1
        if speed_x > 40:  gear = 2
        if speed_x > 70:  gear = 3
        if speed_x > 100: gear = 4
        if speed_x > 135: gear = 5
        if speed_x > 170: gear = 6
        R['gear'] = gear

        write_play_log({
            "type": "step",
            "step": steps + 1,
            "dist_ep": dist_ep_before,
            "dist_raced": dist_raced,
            "dist_from_start": dist_from_start,
            "cur_lap_time": cur_lap_time,
            "speed_x": speed_x,
            "speed_y": speed_y,
            "track_pos": track_pos,
            "off_track": off_track,
            "near_edge": near_edge,
            "track_side": track_side,
            "angle": angle,
            "rpm": rpm,
            "track": track,
            "state": state.tolist(),
            "prev_cmd": prev_cmd_used,
            "model_steer": model_steer,
            "raw_steer": raw_steer,
            "cmd_steer": steer_cmd,
            "accel": accel_cmd,
            "brake": brake_cmd,
            "launch_helper": launch_helper_info,
            "hybrid_assist": hybrid_assist_info,
            "racing_line_guardian": racing_line_guardian_info,
            "guardian_assist": guardian_assist_info,
            "global_assist": global_assist_info,
            "corner_safety": corner_safety_info,
            "gear": gear,
        })

        if use_racing_line_guardian and racing_line_guardian_info.get("abort"):
            stop_reason = "racing_line_guardian_offtrack_abort"
            dist_ep = dist_ep_before
            print(f"\n  Racing Line Guardian abort: off track at {dist_ep_before:.0f}m "
                  f"(trackPos={track_pos:+.3f}). Restart run instead of recovering.")
            break

        if use_hybrid and hybrid_assist_info.get("abort"):
            stop_reason = "hybrid_offtrack_abort"
            dist_ep = dist_ep_before
            print(f"\n  Hybrid abort: off track at {dist_ep_before:.0f}m "
                  f"(trackPos={track_pos:+.3f}). Restart run instead of recovering.")
            break

        if use_guardian_assist and guardian_assist_info.get("abort"):
            stop_reason = "guardian_offtrack_abort"
            dist_ep = dist_ep_before
            print(f"\n  Guardian abort: off track at {dist_ep_before:.0f}m "
                  f"(trackPos={track_pos:+.3f}). Restart run instead of recovering.")
            break

        C.respond_to_server()
        C.get_servers_input()
        S = C.S.d

        dist_ep = float(S.get('distRaced', 0)) - dist_start
        steps  += 1

        if steps <= 300 and (steps <= 30 or steps % 25 == 0):
            print(f"  dbg {steps:3d} | dist={dist_ep_before:6.1f}m | "
                  f"v={speed_x:6.1f} | tpos={track_pos:+.3f} | "
                  f"ang={angle:+.3f} | model_steer={model_steer:+.3f} | "
                  f"raw_steer={raw_steer:+.3f} | "
                  f"cmd_steer={steer_cmd:+.3f} | acc={accel_cmd:.2f} | brk={brake_cmd:.2f}")
        if use_corner_safety and steps <= 2500 and corner_safety_info.get("active") and steps % 25 == 0:
            target_speed = corner_safety_info.get("target_speed")
            overspeed = corner_safety_info.get("overspeed")
            target_txt = f"{target_speed:5.1f}" if target_speed is not None else "  n/a"
            over_txt = f"{overspeed:+5.1f}" if overspeed is not None else "  n/a"
            reasons = ",".join(corner_safety_info.get("reasons", []))
            print(f"  cs  {steps:3d} | dist={dist_ep_before:6.1f}m | "
                  f"v={speed_x:6.1f}/{target_txt} | "
                  f"over={over_txt} | acc={accel_cmd:.2f} | brk={brake_cmd:.2f} | {reasons}")
        if use_guardian_assist and steps <= 2500 and guardian_assist_info.get("active") and steps % 25 == 0:
            target_speed = guardian_assist_info.get("target_speed")
            overspeed = guardian_assist_info.get("overspeed")
            lookahead = guardian_assist_info.get("lookahead_clear")
            target_txt = f"{target_speed:5.1f}" if target_speed is not None else "  n/a"
            over_txt = f"{overspeed:+5.1f}" if overspeed is not None else "  n/a"
            look_txt = f"{lookahead:5.1f}" if lookahead is not None else "  n/a"
            reasons = ",".join(guardian_assist_info.get("reasons", []))
            print(f"  gd  {steps:3d} | dist={dist_ep_before:6.1f}m | "
                  f"v={speed_x:6.1f}/{target_txt} | over={over_txt} | "
                  f"look={look_txt} | tpos={track_pos:+.3f} | "
                  f"ang={angle:+.3f} | acc={accel_cmd:.2f} | "
                  f"brk={brake_cmd:.2f} | {reasons}")
        if use_racing_line_guardian and steps <= 2500 and racing_line_guardian_info.get("active") and steps % 25 == 0:
            target = racing_line_guardian_info.get("target") or {}
            risk = racing_line_guardian_info.get("risk") or {}
            target_speed = target.get("target_speed")
            target_track = target.get("track")
            risk_score = racing_line_guardian_info.get("risk_score")
            mode = racing_line_guardian_info.get("mode")
            target_txt = f"{target_speed:5.1f}" if target_speed is not None else "  n/a"
            track_txt = f"{target_track:+.3f}" if target_track is not None else "  n/a"
            risk_txt = f"{risk_score:4.2f}" if risk_score is not None else " n/a"
            reasons = ",".join(racing_line_guardian_info.get("reasons", []))
            print(f"  rlg {steps:3d} | dist={dist_ep_before:6.1f}m | "
                  f"mode={mode:7s} | risk={risk_txt} | "
                  f"v={speed_x:6.1f}/{target_txt} | "
                  f"tpos={track_pos:+.3f}->{track_txt} | "
                  f"line={risk.get('line_error', 0.0):+.3f} | "
                  f"acc={accel_cmd:.2f} | brk={brake_cmd:.2f} | {reasons}")
        if use_hybrid and steps <= 2500 and hybrid_assist_info.get("active") and steps % 25 == 0:
            target_speed = hybrid_assist_info.get("target_speed")
            target_track = hybrid_assist_info.get("target_track_pos")
            target_txt = f"{target_speed:5.1f}" if target_speed is not None else "  n/a"
            track_txt = f"{target_track:+.3f}" if target_track is not None else "  n/a"
            reasons = ",".join(hybrid_assist_info.get("reasons", []))
            print(f"  hyb {steps:3d} | dist={dist_ep_before:6.1f}m | "
                  f"mode={hybrid_assist_info.get('mode', 'PASSIVE'):22s} | "
                  f"v={speed_x:6.1f}/{target_txt} | "
                  f"tpos={track_pos:+.3f}->{track_txt} | "
                  f"err={hybrid_assist_info.get('lane_error', 0.0):+.3f} | "
                  f"acc={accel_cmd:.2f} | brk={brake_cmd:.2f} | {reasons}")
        if use_global_assist and steps <= 2500 and global_assist_info.get("active") and steps % 25 == 0:
            target_speed = global_assist_info.get("target_speed")
            overspeed = global_assist_info.get("overspeed")
            target_txt = f"{target_speed:5.1f}" if target_speed is not None else "  n/a"
            over_txt = f"{overspeed:+5.1f}" if overspeed is not None else "  n/a"
            reasons = ",".join(global_assist_info.get("reasons", []))
            print(f"  ga  {steps:3d} | dist={dist_ep_before:6.1f}m | "
                  f"v={speed_x:6.1f}/{target_txt} | "
                  f"over={over_txt} | tpos={track_pos:+.3f} | "
                  f"acc={accel_cmd:.2f} | brk={brake_cmd:.2f} | {reasons}")

        if steps % 100 == 0:
            print(f"    {steps:5d} | {dist_ep:6.0f}m | "
                  f"{float(S.get('speedX', 0)):5.1f}km/h | "
                  f"lap={float(S.get('curLapTime', 0)):.1f}s")

        if dist_ep > LAP_THRESHOLD * 2:
            print(f"\n  Done! Distance: {dist_ep:.0f}m")
            break
        if steps > 30000:
            break

    write_play_log({
        "type": "stop",
        "reason": stop_reason,
        "steps": steps,
        "distance": dist_ep,
        "guardian_samples_pending": len(guardian_samples),
    })
    save_guardian_samples()
    C.shutdown()
    close_play_logs()
    print(f"  Play log saved: {os.path.abspath(play_log_path)}")
    print(f"  Latest log alias: {os.path.abspath(latest_log_path)}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    if COMMAND == "collect":
        collect_data(num_laps=50)
    elif COMMAND == "human_collect":
        from human_drive import human_collect_data
        human_collect_data(
            num_laps=get_int_flag("--laps", 50),
            auto_reset_each_lap=("--reset-each-lap" in COMMAND_FLAGS),
        )
    elif COMMAND == "correction_collect":
        from human_drive import human_collect_data
        correction_output = (
            CORRECTION_PENDING_FILE if os.path.exists(CORRECTION_FILE)
            else CORRECTION_FILE
        )
        human_collect_data(
            output_file=correction_output,
            restart_on_save=True,
            allow_low_progress=("--allow-start" in COMMAND_FLAGS),
        )
    elif COMMAND == "bc":
        train_bc(epochs=500)
    elif COMMAND == "offline_validate":
        from offline_validate import main as offline_validate_main
        offline_validate_main(COMMAND_ARGS[1:])
    elif COMMAND == "play":
        try:
            play_model()
        except KeyboardInterrupt:
            print("\n  Play interrupted by user.")
            print(f"  Latest log alias: {os.path.abspath(os.path.join(PLAY_LOG_DIR, 'play_latest.jsonl'))}")
    else:
        print("TORCS Corkscrew AI Training Pipeline (Human-in-the-loop)")
        print("=" * 45)
        print("  python train.py collect         # Phase 1: collect data using aggressive bot")
        print("  python train.py human_collect   # Phase 1: collect data manually (Play the game!)")
        print("  python train.py human_collect --reset-each-lap # Phase 1: reset car after every completed lap")
        print("  python train.py human_collect --laps=10 --reset-each-lap # Phase 1: collect 10 clean reset laps")
        print("  python train.py correction_collect # Phase 4: collect targeted correction samples")
        print("  python train.py correction_collect --allow-start # Corrections that include 0-15 km/h start")
        print("  python train.py bc              # Phase 2: Behavioral Cloning")
        print("  python train.py offline_validate # Offline model-vs-human validation")
        print("  python train.py play            # Run trained model")
        print("  python train.py play --hybrid   # Run BC with sector-based launch + first-corner support")
        print("  python train.py play --racing-line-guardian # Target mode: BC + simple rule-based track supervisor")
        print("  python train.py play --racing-line-guardian --guardian-collect # Save guardian interventions")
        print("  python train.py play --global-assist # Run BC with whole-track stability assist")
        print("  python train.py play --guardian-assist # Run BC with late-intervention guardian")
        print("  python train.py play --guardian-assist --guardian-collect # Save guardian interventions")
        print("  python train.py play --start-guard --launch-lane-keeper # Optional start helpers")
        print("  python train.py play --corner-safety # Test speed/lane safety profiles")
        print("  python train.py bc --with-guardian # Deliberately train with guardian_data.json")
        print("  python train.py bc --brake-weight=10 --brake-out-weight=3 # Tune brake loss weighting")
        print("  Note: current model expects 17-dim states; legacy 20-dim data is converted during training.")
        print()
        print("Order: human_collect -> bc -> play -> correction_collect -> combine_corrections.py -> bc -> play")
