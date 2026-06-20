"""
Racing-line guardian for TORCS BC inference.

The BC actor remains the primary driver. This layer only intervenes when the
current trajectory is likely to leave the human driving envelope or when the
model enters a state where closed-loop BC usually cascades into an off-track.
"""

import math


PASSIVE = "PASSIVE"
NUDGE = "NUDGE"
VETO = "VETO"
RETURN = "RETURN"
ABORT = "ABORT"


def _clip(value, low, high):
    return max(low, min(high, value))


def _finite(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _add_reason(info, reason):
    if reason not in info["reasons"]:
        info["reasons"].append(reason)


def _track_value(track, idx, default=100.0):
    if not isinstance(track, (list, tuple)) or idx < 0 or idx >= len(track):
        return default
    value = _finite(track[idx], default)
    if value <= 0.0:
        return default
    return value


def _track_min(track, indexes, default=100.0):
    values = [_track_value(track, idx, default) for idx in indexes]
    return min(values) if values else default


class HumanRacingLineProfile:
    def __init__(self, envelope, lap_threshold=3600.0):
        self.envelope = envelope if isinstance(envelope, dict) else {}
        self.lap_threshold = float(lap_threshold)
        sectors = self.envelope.get("sectors", {})
        whole_track = sectors.get("whole_track", {})
        points = whole_track.get("points", [])
        self.points = sorted(
            [p for p in points if isinstance(p, dict) and "dist" in p],
            key=lambda p: float(p["dist"]))

    def available(self):
        return len(self.points) >= 2

    def summary(self):
        return {
            "available": self.available(),
            "points": len(self.points),
            "lap_threshold": self.lap_threshold,
            "source_file": self.envelope.get("source_file"),
            "source_size": self.envelope.get("source_size"),
            "generated_at": self.envelope.get("generated_at"),
            "version": self.envelope.get("version"),
        }

    def at(self, dist):
        if not self.points:
            return None
        lap_dist = _finite(dist, 0.0) % self.lap_threshold
        if lap_dist <= self.points[0]["dist"]:
            return dict(self.points[0])

        prev = self.points[0]
        for cur in self.points[1:]:
            if lap_dist <= cur["dist"]:
                span = max(1e-6, _finite(cur["dist"]) - _finite(prev["dist"]))
                t = _clip((lap_dist - _finite(prev["dist"])) / span, 0.0, 1.0)
                return self._blend(prev, cur, t, lap_dist)
            prev = cur
        return dict(self.points[-1])

    def _blend(self, a, b, t, dist):
        out = {"dist": dist}
        keys = set(a.keys()) | set(b.keys())
        for key in keys:
            if key == "dist":
                continue
            av = a.get(key)
            bv = b.get(key)
            if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
                out[key] = float(av) + (float(bv) - float(av)) * t
            elif key in b:
                out[key] = bv
            else:
                out[key] = av
        return out


class RacingLineGuardian:
    def __init__(self, envelope, lap_threshold=3600.0):
        self.profile = HumanRacingLineProfile(envelope, lap_threshold)
        self.lap_threshold = float(lap_threshold)
        self.mode = PASSIVE
        self.return_ticks = 0

        # Tuned to be a guardian, not a second driver.
        self.abort_track = 1.02
        self.edge_track = 0.92
        self.warn_track = 0.82
        self.brake_decel_ms2 = 12.5
        self.max_horizon_s = 0.80
        self.min_horizon_m = 18.0
        self.max_horizon_m = 82.0
        self.launch_suppress_dist = 240.0
        self.pre_corner_dist = 300.0
        self.full_guard_dist = 330.0
        self.transition_track_delta = 0.28
        self.transition_speed_drop = 28.0
        self.edge_recovery_track = 0.86
        self.emergency_track = 0.96
        self.first_corner_entry_start = 285.0
        self.first_corner_entry_end = 345.0
        self.first_corner_left_soft = -0.68
        self.first_corner_left_hard = -0.80
        self.post_apex_stabilize_start = 488.0
        self.post_apex_stabilize_end = 536.0
        self.post_apex_exit_hold_end = 516.0
        self.post_apex_cross_damper_end = 526.0
        self.post_apex_cross_min_track = 0.06
        self.post_apex_left_soft = 0.18
        self.low_speed_release = 9.0
        self.edge_shield_soft_track = 0.72
        self.edge_shield_project_track = 0.78
        self.edge_shield_hard_track = 0.88
        self.progress_release_speed = 58.0
        self.progress_safe_track = 0.80

    def available(self):
        return self.profile.available()

    def meta(self):
        return {
            "profile": self.profile.summary(),
            "abort_track": self.abort_track,
            "edge_track": self.edge_track,
            "warn_track": self.warn_track,
            "brake_decel_ms2": self.brake_decel_ms2,
            "max_horizon_s": self.max_horizon_s,
            "launch_suppress_dist": self.launch_suppress_dist,
            "pre_corner_dist": self.pre_corner_dist,
            "full_guard_dist": self.full_guard_dist,
            "transition_track_delta": self.transition_track_delta,
            "transition_speed_drop": self.transition_speed_drop,
            "edge_recovery_track": self.edge_recovery_track,
            "emergency_track": self.emergency_track,
            "first_corner_entry_start": self.first_corner_entry_start,
            "first_corner_entry_end": self.first_corner_entry_end,
            "first_corner_left_soft": self.first_corner_left_soft,
            "first_corner_left_hard": self.first_corner_left_hard,
            "post_apex_stabilize_start": self.post_apex_stabilize_start,
            "post_apex_stabilize_end": self.post_apex_stabilize_end,
            "post_apex_exit_hold_end": self.post_apex_exit_hold_end,
            "post_apex_cross_damper_end": self.post_apex_cross_damper_end,
            "post_apex_cross_min_track": self.post_apex_cross_min_track,
            "post_apex_left_soft": self.post_apex_left_soft,
            "low_speed_release": self.low_speed_release,
            "edge_shield_soft_track": self.edge_shield_soft_track,
            "edge_shield_project_track": self.edge_shield_project_track,
            "edge_shield_hard_track": self.edge_shield_hard_track,
            "progress_release_speed": self.progress_release_speed,
            "progress_safe_track": self.progress_safe_track,
        }

    def apply(self, dist_ep, speed_x, speed_y, track_pos, angle, track,
              steer_cmd, accel_cmd, brake_cmd, model_action=None):
        lap_dist = _finite(dist_ep, 0.0) % self.lap_threshold
        speed_x = _finite(speed_x)
        speed_y = _finite(speed_y)
        track_pos = _finite(track_pos)
        angle = _finite(angle)
        steer_cmd = _clip(_finite(steer_cmd), -1.0, 1.0)
        accel_cmd = _clip(_finite(accel_cmd), 0.0, 1.0)
        brake_cmd = _clip(_finite(brake_cmd), 0.0, 1.0)

        info = {
            "active": False,
            "abort": False,
            "mode": PASSIVE,
            "previous_mode": self.mode,
            "reasons": [],
            "lap_dist": lap_dist,
            "risk_score": 0.0,
            "risk": {},
            "target": None,
            "lookahead": None,
            "command_before": {
                "steer": steer_cmd,
                "accel": accel_cmd,
                "brake": brake_cmd,
            },
            "model_action": model_action,
        }

        if not self.available():
            info["mode"] = PASSIVE
            _add_reason(info, "rlg_no_profile")
            return steer_cmd, accel_cmd, brake_cmd, info

        if abs(track_pos) >= self.abort_track:
            accel_cmd = 0.0
            if speed_x > 20.0:
                brake_cmd = max(brake_cmd, 0.35)
            info["active"] = True
            info["abort"] = True
            info["mode"] = ABORT
            info["risk_score"] = 99.0
            _add_reason(info, "rlg_offtrack_abort")
            self.mode = ABORT
            info["command_after"] = {
                "steer": steer_cmd,
                "accel": accel_cmd,
                "brake": brake_cmd,
            }
            return steer_cmd, accel_cmd, brake_cmd, info

        if lap_dist < self.launch_suppress_dist:
            info["mode"] = PASSIVE
            info["launch_suppressed"] = True
            info["command_after"] = {
                "steer": steer_cmd,
                "accel": accel_cmd,
                "brake": brake_cmd,
            }
            self.mode = PASSIVE
            return steer_cmd, accel_cmd, brake_cmd, info

        current = self.profile.at(lap_dist)
        horizon_m = _clip((max(0.0, speed_x) / 3.6) * self.max_horizon_s,
                          self.min_horizon_m, self.max_horizon_m)
        future = self.profile.at(lap_dist + horizon_m)
        target = self._target_from_profile(
            current, future, track, lap_dist, track_pos, angle, speed_y)
        info["target"] = target

        predicted_track_pos = self._predict_track_pos(
            track_pos, angle, speed_y, speed_x)
        risk = self._risk(
            lap_dist, speed_x, speed_y, track_pos, angle, track,
            steer_cmd, target, predicted_track_pos)
        risk_score = risk["score"]
        info["risk"] = risk
        info["risk_score"] = risk_score
        info["lookahead"] = {
            "horizon_m": horizon_m,
            "predicted_track_pos": predicted_track_pos,
            "front": target["front_clear"],
            "center_clear": target["center_clear"],
            "wide_clear": target["wide_clear"],
        }

        mode = self._choose_mode(lap_dist, risk_score, risk, track_pos, speed_x)
        info["mode"] = mode

        if mode == PASSIVE:
            self.mode = PASSIVE
            info["command_after"] = {
                "steer": steer_cmd,
                "accel": accel_cmd,
                "brake": brake_cmd,
            }
            return steer_cmd, accel_cmd, brake_cmd, info

        info["active"] = True
        steer_cmd, accel_cmd, brake_cmd = self._apply_control(
            mode, risk, target, lap_dist, track_pos, angle, speed_x, speed_y,
            steer_cmd, accel_cmd, brake_cmd, info)

        self.mode = mode
        info["command_after"] = {
            "steer": steer_cmd,
            "accel": accel_cmd,
            "brake": brake_cmd,
        }
        return steer_cmd, accel_cmd, brake_cmd, info

    def _target_from_profile(self, current, future, track, lap_dist, track_pos,
                             angle, speed_y):
        current = current or {}
        future = future or current or {}
        cur_low = _finite(current.get("track_low"), -0.85)
        cur_high = _finite(current.get("track_high"), 0.85)
        fut_low = _finite(future.get("track_low"), cur_low)
        fut_high = _finite(future.get("track_high"), cur_high)
        speed_p50 = _finite(current.get("speed_p50"), 135.0)
        speed_soft = _finite(current.get("speed_soft"), speed_p50 + 18.0)
        speed_hard = _finite(current.get("speed_hard"), speed_soft + 15.0)
        future_soft = _finite(future.get("speed_soft"), speed_soft)
        future_hard = _finite(future.get("speed_hard"), speed_hard)

        current_track = _finite(current.get("track_target"))
        future_track = _finite(future.get("track_target"), current_track)
        profile_shift = future_track - current_track
        profile_speed_drop = max(0.0, speed_soft - future_soft)

        if lap_dist < self.pre_corner_dist:
            future_weight = 0.0
            phase = "pre_corner"
        elif lap_dist < self.full_guard_dist:
            future_weight = 0.18
            phase = "transition"
        else:
            future_weight = 0.35
            phase = "full"

        if lap_dist >= self.pre_corner_dist:
            if abs(profile_shift) >= self.transition_track_delta:
                shift_factor = _clip(
                    (abs(profile_shift) - self.transition_track_delta) / 0.45,
                    0.0, 1.0)
                future_weight = max(future_weight, 0.50 + 0.18 * shift_factor)
            if profile_speed_drop >= self.transition_speed_drop:
                drop_factor = _clip(
                    (profile_speed_drop - self.transition_speed_drop) / 80.0,
                    0.0, 1.0)
                future_weight = max(future_weight, 0.46 + 0.16 * drop_factor)

        if (
            self.post_apex_stabilize_start <= lap_dist <= self.post_apex_exit_hold_end and
            track_pos > 0.42 and
            profile_shift < 0.0
        ):
            future_weight = min(future_weight, 0.18 if track_pos > 0.55 else 0.25)

        target_track = current_track * (1.0 - future_weight) + future_track * future_weight
        if future_weight <= 0.0:
            low = cur_low
            high = cur_high
        else:
            low = min(cur_low, fut_low)
            high = max(cur_high, fut_high)
        corridor_half = _clip((high - low) * 0.5, 0.16, 0.42)
        guide_low = _clip(target_track - corridor_half, -0.98, 0.98)
        guide_high = _clip(target_track + corridor_half, -0.98, 0.98)

        front = _track_value(track, 9, 100.0)
        center_clear = _track_min(track, range(7, 12), front)
        wide_clear = _track_min(track, range(5, 14), front)
        sensor_clear = min(front, center_clear * 1.08, wide_clear * 1.22)
        sensor_speed = self._speed_from_clear(sensor_clear)
        future_margin = 4.0 if profile_speed_drop >= self.transition_speed_drop else 8.0
        braking_target_speed = min(speed_soft, future_hard + future_margin)
        if profile_speed_drop >= self.transition_speed_drop:
            speed_floor = min(speed_p50, future_hard + 18.0)
            target_speed = min(speed_hard, max(speed_floor, braking_target_speed))
        else:
            target_speed = min(speed_hard, max(speed_p50, braking_target_speed))

        line_trusted = (
            cur_low - 0.12 <= track_pos <= cur_high + 0.12 and
            abs(angle) < 0.34 and
            abs(speed_y) < 10.5
        )
        sensor_cap_enabled = (
            not line_trusted or
            abs(track_pos) > self.warn_track or
            abs(angle) > 0.40 or
            abs(speed_y) > 12.0 or
            (lap_dist >= self.pre_corner_dist and sensor_clear < 18.0)
        )
        if lap_dist < self.pre_corner_dist and line_trusted:
            sensor_cap_enabled = False
        if sensor_cap_enabled:
            sensor_margin = 44.0 if line_trusted else 30.0
            target_speed = min(target_speed, sensor_speed + sensor_margin)

        return {
            "track": target_track,
            "phase": phase,
            "future_weight": future_weight,
            "track_low": low,
            "track_high": high,
            "current_track_low": cur_low,
            "current_track_high": cur_high,
            "guide_low": guide_low,
            "guide_high": guide_high,
            "corridor_half": corridor_half,
            "speed_p50": speed_p50,
            "speed_soft": speed_soft,
            "speed_hard": speed_hard,
            "future_speed_soft": future_soft,
            "target_speed": target_speed,
            "sensor_speed": sensor_speed,
            "future_track": future_track,
            "profile_shift": profile_shift,
            "profile_speed_drop": profile_speed_drop,
            "braking_target_speed": braking_target_speed,
            "sensor_cap_enabled": sensor_cap_enabled,
            "line_trusted": line_trusted,
            "front_clear": front,
            "center_clear": center_clear,
            "wide_clear": wide_clear,
            "sensor_clear": sensor_clear,
            "steer_p50": _finite(current.get("steer_p50")),
            "brake_p90": _finite(current.get("brake_p90")),
            "brake_cap": _finite(current.get("brake_cap"), 0.15),
            "samples": int(_finite(current.get("samples"), 0)),
        }

    def _speed_from_clear(self, clear):
        clear = _finite(clear, 100.0)
        if clear > 180.0:
            return 285.0
        if clear > 150.0:
            return 265.0
        if clear > 120.0:
            return 240.0
        if clear > 100.0:
            return 215.0
        if clear > 80.0:
            return 185.0
        if clear > 60.0:
            return 158.0
        if clear > 48.0:
            return 138.0
        if clear > 38.0:
            return 118.0
        if clear > 28.0:
            return 100.0
        if clear > 20.0:
            return 82.0
        return 65.0

    def _predict_track_pos(self, track_pos, angle, speed_y, speed_x):
        horizon = 0.58 if speed_x < 170.0 else 0.48
        lateral_term = (speed_y / 300.0) * horizon * 2.15
        heading_term = angle * _clip(speed_x / 260.0, 0.15, 0.75) * 0.36
        return track_pos + lateral_term + heading_term

    def _predict_edge_track_pos(self, track_pos, speed_y, speed_x):
        speed_factor = _clip(speed_x / 100.0, 0.55, 1.20)
        lateral_term = (speed_y / 38.0) * speed_factor
        return track_pos + lateral_term

    def _risk(self, lap_dist, speed_x, speed_y, track_pos, angle, track,
              steer_cmd, target, predicted_track_pos):
        line_error = target["track"] - track_pos
        future_error = target["track"] - predicted_track_pos
        guide_violation = 0.0
        if track_pos < target["guide_low"]:
            guide_violation = target["guide_low"] - track_pos
        elif track_pos > target["guide_high"]:
            guide_violation = track_pos - target["guide_high"]

        future_violation = 0.0
        if predicted_track_pos < target["guide_low"]:
            future_violation = target["guide_low"] - predicted_track_pos
        elif predicted_track_pos > target["guide_high"]:
            future_violation = predicted_track_pos - target["guide_high"]

        overspeed = speed_x - target["target_speed"]
        delta_v_ms = max(0.0, overspeed) / 3.6
        brake_distance = (delta_v_ms * delta_v_ms) / (2.0 * self.brake_decel_ms2)
        braking_room = max(12.0, target["sensor_clear"] * 0.78)
        brake_room_risk = max(0.0, brake_distance - braking_room) / 35.0

        line_risk = max(0.0, abs(line_error) - target["corridor_half"] * 0.72) / 0.42
        future_risk = max(0.0, abs(future_error) - target["corridor_half"] * 0.65) / 0.42
        guide_risk = max(guide_violation, future_violation) / 0.34
        speed_risk = max(0.0, overspeed - 8.0) / 42.0
        slip_risk = max(0.0, abs(speed_y) - 7.0) / 13.0
        angle_risk = max(0.0, abs(angle) - 0.20) / 0.45
        edge_risk = max(0.0, abs(track_pos) - self.warn_track) / 0.18
        transition_error = target["future_track"] - predicted_track_pos
        transition_scale = _clip(
            abs(target["profile_shift"]) / max(1e-6, self.transition_track_delta),
            0.0, 2.2)
        if target["phase"] == "pre_corner":
            transition_risk = 0.0
        else:
            transition_risk = (
                max(0.0, abs(transition_error) - target["corridor_half"] * 0.55) /
                0.38 * transition_scale)
        outward_speed = max(0.0, track_pos * speed_y)
        outward_risk = max(0.0, outward_speed - 2.0) / 10.0
        edge_projected_track = self._predict_edge_track_pos(track_pos, speed_y, speed_x)
        edge_current_risk = 0.0
        if abs(track_pos) > self.edge_shield_hard_track or outward_speed > 0.9:
            edge_current_risk = (
                max(0.0, abs(track_pos) - self.edge_shield_soft_track) /
                max(1e-6, self.edge_shield_hard_track - self.edge_shield_soft_track))
        edge_projected_risk = (
            max(0.0, abs(edge_projected_track) - self.edge_shield_project_track) /
            max(1e-6, 1.0 - self.edge_shield_project_track))
        kinetic_edge_risk = 0.0
        if abs(track_pos) > 0.24 and speed_x > 35.0:
            kinetic_edge_risk = max(0.0, outward_speed - 0.9) / 5.2
        edge_shield_risk = max(edge_current_risk, edge_projected_risk, kinetic_edge_risk)

        entry_left_violation = 0.0
        if (
            self.first_corner_entry_start <= lap_dist <= self.first_corner_entry_end and
            speed_x > 150.0 and
            track_pos < self.first_corner_left_soft
        ):
            entry_left_violation = (
                (self.first_corner_left_soft - track_pos) /
                max(1e-6, self.first_corner_left_soft - self.first_corner_left_hard))

        post_apex_left_risk = 0.0
        if self.post_apex_stabilize_start <= lap_dist <= self.post_apex_stabilize_end:
            soft_left = self.post_apex_left_soft
            if lap_dist > self.post_apex_exit_hold_end:
                soft_left = max(-0.22, soft_left - (lap_dist - self.post_apex_exit_hold_end) * 0.018)
            left_projection = min(track_pos, predicted_track_pos)
            left_violation = max(0.0, soft_left - left_projection) / 0.32
            yaw_swing = 0.0
            if track_pos < 0.58 and speed_x > 70.0:
                yaw_swing = max(0.0, abs(angle) - 0.36) / 0.26
            lateral_swing = 0.0
            if track_pos < 0.35 and speed_x > 70.0:
                lateral_swing = max(0.0, abs(speed_y) - 3.0) / 9.0
            post_apex_left_risk = max(left_violation, yaw_swing * 0.85 + lateral_swing * 0.55)

        post_apex_cross_risk = 0.0
        if (
            self.post_apex_stabilize_start <= lap_dist <= self.post_apex_cross_damper_end and
            self.post_apex_cross_min_track < track_pos < self.warn_track and
            speed_x > 55.0
        ):
            yaw_left_risk = max(0.0, angle - 0.03) / 0.34
            lateral_left_risk = max(0.0, speed_y - 0.2) / 5.0
            early_cut_risk = max(0.0, track_pos - 0.22) / 0.48
            if angle > 0.025 or speed_y > -0.2:
                post_apex_cross_risk = max(
                    yaw_left_risk,
                    min(1.55, early_cut_risk * 0.72 +
                        yaw_left_risk * 0.72 +
                        lateral_left_risk * 0.38))

        sensor_risk = 0.0
        if target["sensor_cap_enabled"]:
            sensor_risk = max(0.0, speed_x - target["sensor_speed"] - 22.0) / 50.0
        elif target["phase"] == "pre_corner":
            sensor_risk = max(0.0, speed_x - target["sensor_speed"] - 58.0) / 80.0

        action_away = 0.0
        if abs(line_error) > 0.18 and abs(steer_cmd) > 0.04:
            if line_error * steer_cmd < 0.0:
                action_away = min(1.0, abs(steer_cmd) * 1.5)

        score = (
            line_risk * 0.85 +
            future_risk * 1.15 +
            guide_risk * 1.35 +
            speed_risk * 0.95 +
            brake_room_risk * 1.30 +
            slip_risk * 1.15 +
            angle_risk * 0.85 +
            edge_risk * 1.60 +
            transition_risk * 1.20 +
            outward_risk * 1.10 +
            edge_shield_risk * 1.85 +
            entry_left_violation * 1.70 +
            post_apex_left_risk * 1.45 +
            post_apex_cross_risk * 1.35 +
            sensor_risk * (0.25 if target["phase"] == "pre_corner" else 0.70) +
            action_away * 0.65)

        return {
            "score": score,
            "line_error": line_error,
            "future_error": future_error,
            "guide_violation": guide_violation,
            "future_violation": future_violation,
            "line_risk": line_risk,
            "future_risk": future_risk,
            "guide_risk": guide_risk,
            "overspeed": overspeed,
            "speed_risk": speed_risk,
            "brake_distance": brake_distance,
            "braking_room": braking_room,
            "brake_room_risk": brake_room_risk,
            "slip_risk": slip_risk,
            "angle_risk": angle_risk,
            "edge_risk": edge_risk,
            "transition_error": transition_error,
            "transition_risk": transition_risk,
            "outward_speed": outward_speed,
            "outward_risk": outward_risk,
            "edge_projected_track": edge_projected_track,
            "edge_shield_risk": edge_shield_risk,
            "entry_left_violation": entry_left_violation,
            "post_apex_left_risk": post_apex_left_risk,
            "post_apex_cross_risk": post_apex_cross_risk,
            "sensor_risk": sensor_risk,
            "action_away_risk": action_away,
            "lap_dist": lap_dist,
        }

    def _choose_mode(self, lap_dist, risk_score, risk, track_pos, speed_x):
        pre_corner = lap_dist < self.pre_corner_dist
        if pre_corner:
            hard_condition = (
                abs(track_pos) > self.edge_track or
                risk["slip_risk"] > 0.95 or
                risk["edge_risk"] > 0.55 or
                risk["edge_shield_risk"] > 0.40 or
                risk["entry_left_violation"] > 0.32 or
                (risk["guide_violation"] > 0.38 and abs(track_pos) > 0.78))
            if hard_condition or risk_score >= 2.35:
                return VETO
            if risk["entry_left_violation"] > 0.04:
                return NUDGE
            if risk_score >= 0.78:
                return NUDGE
            if self.mode in (VETO, RETURN, NUDGE) and risk_score > 0.40 and speed_x > 40.0:
                return RETURN
            return PASSIVE

        hard_condition = (
            abs(track_pos) > self.edge_track or
            risk["guide_violation"] > 0.20 or
            risk["future_violation"] > 0.25 or
            risk["transition_risk"] > 0.55 or
            risk["brake_room_risk"] > 0.25 or
            risk["slip_risk"] > 0.65 or
            risk["edge_risk"] > 0.40 or
            risk["edge_shield_risk"] > 0.34 or
            risk["entry_left_violation"] > 0.18 or
            risk["post_apex_left_risk"] > 0.48 or
            risk["post_apex_cross_risk"] > 0.42 or
            (abs(track_pos) > 0.78 and risk["outward_risk"] > 0.25))
        if hard_condition or risk_score >= 1.35:
            return VETO
        if risk_score >= 0.55:
            return RETURN if self.mode in (VETO, RETURN) and risk_score < 0.90 else NUDGE
        if self.mode in (VETO, RETURN, NUDGE) and risk_score > 0.28 and speed_x > 40.0:
            return RETURN
        return PASSIVE

    def _apply_control(self, mode, risk, target, lap_dist, track_pos, angle, speed_x,
                       speed_y, steer_cmd, accel_cmd, brake_cmd, info):
        line_error = risk["line_error"]
        future_error = risk["future_error"]
        transition_error = risk.get("transition_error", 0.0)
        pre_corner = lap_dist < self.pre_corner_dist
        entry_left_violation = risk.get("entry_left_violation", 0.0)
        post_apex_left_risk = risk.get("post_apex_left_risk", 0.0)
        post_apex_cross_risk = risk.get("post_apex_cross_risk", 0.0)
        edge_shield_risk = risk.get("edge_shield_risk", 0.0)
        edge_projected_track = risk.get("edge_projected_track", track_pos)
        post_apex_window = (
            self.post_apex_stabilize_start <= lap_dist <= self.post_apex_stabilize_end)
        post_apex_exit_hold = (
            self.post_apex_stabilize_start <= lap_dist <= self.post_apex_exit_hold_end and
            0.42 < track_pos < self.warn_track)
        post_apex_cross_damper = (
            self.post_apex_stabilize_start <= lap_dist <= self.post_apex_cross_damper_end and
            self.post_apex_cross_min_track < track_pos < self.warn_track and
            speed_x > 55.0 and
            (angle > 0.025 or post_apex_cross_risk > 0.22))

        if pre_corner:
            steer_need = line_error * 0.36 + future_error * 0.12
            steer_need -= speed_y * 0.014
            steer_need += angle * 0.06
            steer_need = _clip(steer_need, -0.42, 0.42)
        else:
            steer_need = line_error * 0.50 + future_error * 0.28
            steer_need += transition_error * _clip(risk["transition_risk"] * 0.10, 0.0, 0.30)
            steer_need -= speed_y * 0.028
            steer_need += angle * 0.12
            steer_need = _clip(steer_need, -0.86, 0.86)

        if mode == NUDGE:
            if pre_corner:
                blend = _clip(0.12 + risk["score"] * 0.10, 0.12, 0.28)
            else:
                blend = _clip(0.18 + risk["score"] * 0.18, 0.18, 0.42)
            _add_reason(info, "rlg_nudge")
        elif mode == RETURN:
            if pre_corner:
                blend = _clip(0.18 + risk["score"] * 0.12, 0.16, 0.34)
            else:
                blend = _clip(0.26 + risk["score"] * 0.20, 0.24, 0.52)
            _add_reason(info, "rlg_return")
        else:
            if pre_corner:
                blend = _clip(0.30 + risk["score"] * 0.08, 0.30, 0.48)
            else:
                blend = _clip(0.48 + risk["score"] * 0.16, 0.52, 0.82)
            _add_reason(info, "rlg_veto")

        steer_after = _clip(steer_cmd * (1.0 - blend) + steer_need * blend,
                            -0.95, 0.95)
        if line_error < -0.16:
            steer_after = min(steer_after, steer_need * 0.75)
        elif line_error > 0.16:
            steer_after = max(steer_after, steer_need * 0.75)

        if post_apex_exit_hold and steer_after < 0.0:
            hold_floor = -0.04
            if steer_cmd > 0.0:
                hold_floor = _clip(steer_cmd * 0.30, 0.02, 0.12)
            steer_after = max(steer_after, hold_floor)
            _add_reason(info, "rlg_post_apex_exit_hold")

        if post_apex_cross_damper:
            countersteer_floor = _clip(
                0.08 + max(0.0, angle) * 0.82 +
                max(0.0, speed_y) * 0.035 +
                max(0.0, 0.44 - track_pos) * 0.20,
                0.08, 0.62)
            if steer_after < countersteer_floor:
                steer_after = countersteer_floor
                _add_reason(info, "rlg_post_apex_cross_damper")

        if abs(track_pos) > self.warn_track:
            edge_steer = _clip(-track_pos * 0.72 + angle * 0.10 - speed_y * 0.012,
                               -0.90, 0.90)
            edge_blend = _clip((abs(track_pos) - self.warn_track) / 0.18,
                               0.22, 0.80)
            if pre_corner:
                edge_blend = min(edge_blend, 0.36)
            steer_after = _clip(steer_after * (1.0 - edge_blend) +
                                edge_steer * edge_blend, -0.95, 0.95)
            _add_reason(info, "rlg_edge_veto")

        if edge_shield_risk > 0.16:
            side = -1.0 if (track_pos < 0.0 or edge_projected_track < 0.0) else 1.0
            shield_steer = _clip(
                -side * (0.46 + edge_shield_risk * 0.24) -
                speed_y * 0.028 - track_pos * 0.16,
                -0.98, 0.98)
            if side < 0.0:
                steer_after = max(steer_after, shield_steer)
            else:
                steer_after = min(steer_after, shield_steer)
            _add_reason(info, "rlg_edge_shield")

        if entry_left_violation > 0.0:
            entry_steer = _clip(
                0.22 + entry_left_violation * 0.34 +
                max(0.0, -speed_y) * 0.020 + max(0.0, angle) * 0.10,
                0.22, 0.88)
            steer_after = max(steer_after, entry_steer)
            _add_reason(info, "rlg_first_corner_entry_guard")

        if post_apex_left_risk > 0.12:
            post_apex_steer = _clip(0.24 + post_apex_left_risk * 0.25, 0.24, 0.78)
            steer_after = max(steer_after, post_apex_steer)
            _add_reason(info, "rlg_post_apex_left_guard")

        if abs(speed_y) > 10.5 and speed_x > 45.0:
            slip_steer = _clip(-math.copysign(min(0.58, abs(speed_y) * 0.024), speed_y),
                               -0.58, 0.58)
            slip_blend = _clip(steer_after * 0.52 + slip_steer * 0.48,
                               -0.95, 0.95)
            if edge_shield_risk > 0.30:
                shield_side = -1.0 if (track_pos < 0.0 or edge_projected_track < 0.0) else 1.0
                if shield_side < 0.0:
                    steer_after = max(steer_after, slip_blend)
                else:
                    steer_after = min(steer_after, slip_blend)
            else:
                steer_after = slip_blend
            _add_reason(info, "rlg_slip_veto")

        overspeed = risk["overspeed"]
        if overspeed > 6.0 or risk["line_risk"] > 0.0 or risk["future_risk"] > 0.0:
            if mode == NUDGE:
                accel_limit = 0.78 if pre_corner else 0.58
            elif mode == RETURN:
                accel_limit = 0.62 if pre_corner else 0.42
            else:
                accel_limit = 0.45 if pre_corner else 0.20
            if risk["guide_risk"] > 0.25 or abs(track_pos) > self.warn_track:
                accel_limit = min(accel_limit, 0.08)
                if pre_corner and abs(track_pos) < self.edge_track:
                    accel_limit = max(accel_limit, 0.32)
            if risk["slip_risk"] > 0.25:
                accel_limit = min(accel_limit, 0.24 if pre_corner else 0.12)
            accel_cmd = min(accel_cmd, accel_limit)
            _add_reason(info, "rlg_lift")

        brake_needed = 0.0
        if overspeed > (34.0 if pre_corner else 18.0):
            brake_needed = max(brake_needed, 0.10 + min(0.45, (overspeed - 18.0) / 95.0))
        if risk["brake_room_risk"] > (0.35 if pre_corner else 0.0):
            brake_needed = max(brake_needed, 0.16 + min(0.42, risk["brake_room_risk"] * 0.28))
        if risk["guide_risk"] > (0.78 if pre_corner else 0.38):
            brake_needed = max(brake_needed, 0.18 + min(0.28, risk["guide_risk"] * 0.16))
        if risk["slip_risk"] > 0.55:
            brake_needed = max(brake_needed, 0.18 + min(0.22, risk["slip_risk"] * 0.12))
        if (
            target["brake_p90"] > 0.30 and
            (overspeed > -4.0 or risk["transition_risk"] > 0.25)
        ):
            brake_needed = max(
                brake_needed,
                min(0.88, 0.24 + target["brake_p90"] * 0.54))
        if target["profile_speed_drop"] > self.transition_speed_drop and overspeed > -2.0:
            brake_needed = max(
                brake_needed,
                min(0.78, 0.22 + target["profile_speed_drop"] / 150.0))
        if entry_left_violation > 0.0:
            brake_needed = max(
                brake_needed,
                min(0.86, 0.22 + entry_left_violation * 0.32 +
                    max(0.0, speed_x - 180.0) / 150.0))
        if post_apex_left_risk > 0.18:
            brake_needed = max(
                brake_needed,
                min(0.74, 0.22 + post_apex_left_risk * 0.30 +
                    max(0.0, abs(angle) - 0.42) * 0.35))
        elif post_apex_window and track_pos < 0.35 and abs(angle) > 0.42:
            brake_needed = max(brake_needed, min(0.46, 0.18 + abs(angle) * 0.30))
        if post_apex_cross_damper and speed_x > 70.0:
            cross_brake = 0.0
            if angle > 0.12 or track_pos < 0.46 or speed_y > 1.0:
                cross_brake = min(
                    0.62,
                    0.16 + max(0.0, angle - 0.10) * 0.88 +
                    max(0.0, 0.46 - track_pos) * 0.28 +
                    max(0.0, speed_y - 0.5) * 0.035)
            if cross_brake > 0.0:
                brake_needed = max(brake_needed, cross_brake)
                _add_reason(info, "rlg_post_apex_cross_brake")
        if edge_shield_risk > 0.16 and speed_x > 35.0:
            brake_needed = max(
                brake_needed,
                min(0.90, 0.28 + edge_shield_risk * 0.34 +
                    max(0.0, abs(speed_y) - 4.0) * 0.026))
        if abs(track_pos) > self.edge_track and speed_x > 70.0:
            brake_needed = max(brake_needed, 0.42)
        if (
            abs(track_pos) > self.edge_recovery_track and
            risk["outward_risk"] > 0.20 and
            speed_x > 55.0
        ):
            brake_needed = max(brake_needed, 0.68)
        if abs(track_pos) > self.emergency_track and speed_x > 45.0:
            brake_needed = max(brake_needed, 0.82)

        if brake_needed > 0.0:
            if entry_left_violation > 0.0:
                brake_limit = 0.86
            elif edge_shield_risk > 0.70:
                brake_limit = 0.90
            else:
                brake_limit = 0.34 if pre_corner and abs(track_pos) < self.edge_track else 0.82
            brake_cmd = max(brake_cmd, min(brake_limit, brake_needed))
            accel_cmd = min(accel_cmd, 0.18 if pre_corner else 0.02)
            _add_reason(info, "rlg_brake")
            if target["brake_p90"] > 0.30:
                _add_reason(info, "rlg_human_brake_zone")
            if entry_left_violation > 0.0:
                accel_cmd = 0.0
                _add_reason(info, "rlg_first_corner_entry_brake")
            if post_apex_left_risk > 0.18:
                accel_cmd = 0.0
                _add_reason(info, "rlg_post_apex_stability_brake")
            if post_apex_cross_damper:
                accel_cmd = 0.0
            if edge_shield_risk > 0.16:
                accel_cmd = 0.0 if edge_shield_risk > 0.55 else min(accel_cmd, 0.08)
                _add_reason(info, "rlg_edge_shield_brake")
            if risk["outward_risk"] > 0.20 and abs(track_pos) > self.edge_recovery_track:
                accel_cmd = 0.0
                _add_reason(info, "rlg_edge_emergency_brake")

        # Prevent a post-apex steering spike from creating a lateral slide.
        if abs(speed_y) > 6.0 or abs(track_pos) > 0.70 or risk["future_risk"] > 0.0:
            steer_cap = 0.70
            if pre_corner:
                steer_cap = min(steer_cap, 0.46)
            if speed_x > 140.0:
                steer_cap = min(steer_cap, 0.58)
            if abs(speed_y) > 11.0:
                steer_cap = min(steer_cap, 0.50)
            inward_command = steer_after * (-track_pos) > 0.0
            if edge_shield_risk > 0.30 and inward_command:
                steer_cap = max(steer_cap, 0.78)
                if edge_shield_risk > 0.70:
                    steer_cap = max(steer_cap, 0.95)
                _add_reason(info, "rlg_edge_shield_cap")
            if abs(track_pos) > self.edge_recovery_track and inward_command:
                steer_cap = max(steer_cap, 0.76)
                if abs(track_pos) > self.emergency_track:
                    steer_cap = max(steer_cap, 0.86)
                elif abs(speed_y) > 12.0:
                    steer_cap = max(steer_cap, 0.66)
                _add_reason(info, "rlg_edge_recovery_cap")
            steer_after = _clip(steer_after, -steer_cap, steer_cap)
            info["steer_cap"] = steer_cap
            _add_reason(info, "rlg_steer_cap")

        safe_progress = (
            speed_x < self.progress_release_speed and
            abs(track_pos) < self.progress_safe_track and
            abs(speed_y) < 5.0 and
            edge_shield_risk < 0.20
        )
        if safe_progress:
            if speed_x < 32.0:
                min_accel = 0.56
            else:
                min_accel = 0.34
            if abs(angle) > 0.55:
                min_accel = min(min_accel, 0.28)
            brake_cmd = min(brake_cmd, 0.04)
            accel_cmd = max(accel_cmd, min_accel)
            steer_after = _clip(steer_after, -0.72, 0.72)
            _add_reason(info, "rlg_safe_progress_release")

        if (
            speed_x < self.low_speed_release and
            abs(track_pos) < 0.76 and
            abs(angle) < 0.62 and
            edge_shield_risk < 0.20
        ):
            brake_cmd = min(brake_cmd, 0.02)
            accel_cmd = max(accel_cmd, 0.48 if speed_x < 4.0 else 0.34)
            steer_after = _clip(steer_after, -0.58, 0.58)
            _add_reason(info, "rlg_low_speed_release")
        elif (
            speed_x < self.low_speed_release * 1.7 and
            abs(track_pos) < 0.80 and
            abs(angle) < 0.45 and
            brake_cmd > 0.18 and
            edge_shield_risk < 0.20
        ):
            brake_cmd = min(brake_cmd, 0.10)
            accel_cmd = max(accel_cmd, 0.14)
            _add_reason(info, "rlg_low_speed_brake_release")

        info["steer_need"] = steer_need
        info["steer_blend"] = blend
        info["target_speed"] = target["target_speed"]
        info["overspeed"] = overspeed
        return steer_after, accel_cmd, brake_cmd
