//! High-level role assignment and off-ball positioning.
//!
//! The planner is intentionally small and deterministic. It provides one
//! primary challenger, one genuine cover player, and a transition/support
//! role for the remaining teammate(s). Shot reachability remains owned by
//! shot_planner, not by these role heuristics.

use crate::{
    math::Vector,
    prediction::BallTrajectory,
    state::{AgentFrame, CarState},
};

fn team_side(team: u32) -> f32 {
    if team == 0 { -1.0 } else { 1.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Challenge,
    Cover,
    Support,
    KickoffCheat,
    KickoffBoost,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TacticalPlan {
    pub role: Role,
    pub challenger: usize,
    pub cover_owner: Option<usize>,
    pub urgent_save: bool,
    pub challenger_eta: f32,
}

fn arrival_time(car: CarState, target: Vector) -> f32 {
    let displacement = (target - car.location).flatten();
    let distance = displacement.magnitude();
    if distance <= 1.0 {
        return 0.0;
    }

    let direction = displacement / distance;
    let forward = car.forward().flatten().normalize();
    let alignment = if forward.magnitude() > 0.0 {
        forward.dot(direction).clamp(-1.0, 1.0)
    } else {
        -1.0
    };
    let turn_angle = alignment.acos();
    let speed = car.velocity.flatten().magnitude();
    let turn_rate = if speed > 1400.0 { 2.0 } else { 3.5 };
    let turn_time = turn_angle / turn_rate;
    let initial_speed = car.velocity.flatten().dot(direction).max(0.0);

    let boost_accel = if car.boost > 0.0 && turn_angle < 0.9 {
        1600.0
    } else {
        850.0
    };
    let boost_time = if car.boost > 0.0 && turn_angle < 0.9 {
        (car.boost / 33.3333).min(((2300.0 - initial_speed) / boost_accel).max(0.0))
    } else {
        0.0
    };
    let boost_distance = initial_speed * boost_time
        + 0.5 * boost_accel * boost_time * boost_time;
    let speed_after_boost = (initial_speed + boost_accel * boost_time).min(2300.0);
    let remaining = (distance - boost_distance).max(0.0);

    let accel_distance = ((2300.0_f32.powi(2) - speed_after_boost.powi(2))
        / (2.0 * 850.0)).max(0.0);
    let drive_time = if remaining <= accel_distance {
        ((speed_after_boost.powi(2) + 1700.0 * remaining).sqrt() - speed_after_boost)
            / 850.0
    } else {
        ((2300.0 - speed_after_boost) / 850.0)
            + (remaining - accel_distance) / 2300.0
    };

    (turn_time + boost_time + drive_time).max(0.0)
}

pub fn urgent_save(frame: &AgentFrame, prediction: &BallTrajectory) -> bool {
    let side = team_side(frame.me.team);
    let ball = frame.ball;
    let own_depth = ball.location.y * side;
    let moving_to_own_goal = ball.velocity.y * side > 180.0;

    if own_depth > 2200.0 && moving_to_own_goal {
        return true;
    }

    let own_goal_y = side * 5120.0;
    prediction.slices.iter().any(|slice| {
        if slice.time > 2.6 || slice.time < 0.05 {
            return false;
        }
        let reaches_goal_plane = (slice.location.y - own_goal_y).abs() < 420.0
            || slice.location.y * side > 4740.0;
        reaches_goal_plane
            && slice.location.x.abs() < 1050.0
            && slice.location.z < 850.0
            && slice.velocity.y * side > 100.0
    })
}

pub fn kickoff_kicker(cars: &[CarState], ball: Vector, team: u32) -> Option<usize> {
    cars.iter()
        .filter(|car| car.team == team && !car.demolished)
        .min_by(|a, b| {
            let score = |car: &CarState| {
                arrival_time(*car, ball)
                    + if car.velocity.magnitude() > 100.0 { -0.03 } else { 0.0 }
            };
            score(a).total_cmp(&score(b))
        })
        .map(|car| car.index)
}

pub fn build_plan(frame: &AgentFrame, prediction: &BallTrajectory) -> TacticalPlan {
    let me = frame.me;
    let urgent = urgent_save(frame, prediction);
    let teammates: Vec<CarState> = frame.cars.iter().copied()
        .filter(|car| car.team == me.team && !car.demolished)
        .collect();

    let challenger = teammates.iter().min_by(|a, b| {
        arrival_time(**a, frame.ball.location)
            .total_cmp(&arrival_time(**b, frame.ball.location))
    }).map(|car| car.index).unwrap_or(me.index);

    let challenger_eta = teammates.iter()
        .find(|car| car.index == challenger)
        .map(|car| arrival_time(*car, frame.ball.location))
        .unwrap_or_else(|| arrival_time(me, frame.ball.location));

    let side = team_side(me.team);
    let cover_owner = if teammates.len() <= 1 {
        None
    } else {
        teammates.iter()
            .filter(|car| car.index != challenger)
            .max_by(|a, b| {
                (a.location.y * side).total_cmp(&(b.location.y * side))
            })
            .map(|car| car.index)
    };

    // During an emergency, let a near-best teammate contest even if the
    // ordinary role allocator selected somebody else. This prevents a rigid
    // role assignment from blocking an actual last-second save.
    let role = if urgent && arrival_time(me, frame.ball.location) <= challenger_eta + 0.28 {
        Role::Challenge
    } else if me.index == challenger {
        Role::Challenge
    } else if cover_owner == Some(me.index) {
        Role::Cover
    } else {
        Role::Support
    };

    TacticalPlan {
        role,
        challenger,
        cover_owner,
        urgent_save: urgent,
        challenger_eta,
    }
}

pub fn cover_target(frame: &AgentFrame) -> Vector {
    let side = team_side(frame.me.team);
    let own_goal_y = side * 5120.0;
    let own_depth = (frame.ball.location.y * side).clamp(0.0, 4200.0);
    // As the ball approaches our goal, the cover car retreats with it.
    // Keep the target between the ball and goal instead of stepping out
    // toward midfield when the threat is already deep in the box.
    let outward = (1100.0 - own_depth * 0.20).clamp(250.0, 1100.0);
    let ball_x = frame.ball.location.x * 0.62
        + frame.ball.velocity.x.clamp(-500.0, 500.0) * 0.22;
    Vector::new(
        ball_x.clamp(-1050.0, 1050.0),
        own_goal_y - side * outward,
        0.0,
    )
}

/// Port of the Python support.get_target() routine. It positions from the
/// most advanced living teammate, adjusts the lane for own-half possession,
/// and applies the source's small per-frame target smoothing.
pub fn support_target(frame: &AgentFrame, previous_target: Option<Vector>) -> Vector {
    let side_value = team_side(frame.me.team);
    let ball = frame.ball.location;
    let ball_x = ball.x;
    let ball_y = ball.y * side_value;

    let friends: Vec<CarState> = frame.cars.iter()
        .copied()
        .filter(|car| car.team == frame.me.team && car.index != frame.me.index && !car.demolished)
        .collect();
    let my_y = frame.me.location.y * side_value;
    let lead = friends.iter()
        .filter(|friend| friend.location.y * side_value < my_y - 250.0)
        .min_by(|a, b| {
            (a.location.y * side_value).total_cmp(&(b.location.y * side_value))
        });

    let (mut target_x, mut target_y) = if let Some(lead) = lead {
        // Python uses a 900-unit gap in 2v2 and 1150 in 3v3.
        let support_gap = if friends.len() <= 1 { 900.0 } else { 1150.0 };
        let lead_x = lead.location.x;
        let lead_y = lead.location.y * side_value;
        let target_x = lead_x * 0.58 + ball_x * 0.42;
        let target_y = (lead_y + support_gap)
            .min(ball_y + 1450.0)
            .max(ball_y + 350.0);
        (target_x, target_y)
    } else if ball_y < -2500.0 {
        (ball_x * 0.22, ball_y + 1150.0)
    } else if ball_y < 640.0 {
        (ball_x * 0.32, ball_y + 900.0)
    } else {
        (ball_x * 0.42, ball_y + 700.0)
    };

    // In our half the support bot becomes a wide outlet / second defender.
    if ball_y > 900.0 {
        let lane = if frame.me.index % 2 == 0 { 1.0 } else { -1.0 };
        target_x = lane * if ball_y > 2600.0 { 2250.0 } else { 1900.0 };
        target_y = (ball_y - if ball_y > 2600.0 { 850.0 } else { 650.0 }).max(-500.0);
    }

    target_y = target_y.clamp(-4300.0, 4100.0);
    target_x = target_x.clamp(-2550.0, 2550.0);
    let mut target = Vector::new(target_x, target_y * side_value, 0.0);

    if let Some(previous) = previous_target {
        let shift = target - previous;
        if shift.magnitude() < 500.0 {
            target = previous + shift * 0.35;
        }
    }
    target
}

pub fn kickoff_cheat_target(frame: &AgentFrame) -> Vector {
    let side = team_side(frame.me.team);
    let ball = frame.ball.location;
    let x = (ball.x * 0.35).clamp(-900.0, 900.0);
    Vector::new(x, ball.y + side * 760.0, 0.0)
}

/// Select a boost pad on the route to the requested target. Full pads receive
/// priority, but a close small pad can still be worth taking at very low boost.
pub fn choose_boost_pad(
    frame: &AgentFrame,
    target: Vector,
    opponent_near_ball: bool,
) -> Option<Vector> {
    let me = frame.me;
    if me.boost >= 35.0 || opponent_near_ball {
        return None;
    }

    let route = (target - me.location).flatten().normalize();
    if route.magnitude() == 0.0 {
        return None;
    }

    frame.boost_pads.iter()
        .filter_map(|pad| {
            let to_pad = (pad.location - me.location).flatten();
            let distance = to_pad.magnitude();
            if distance > if me.boost < 12.0 { 2600.0 } else { 1700.0 } {
                return None;
            }
            // Use RLBot's respawn countdown exactly as the Python boost-pad
            // tracker does: don't detour when the pad becomes active too late.
            if !pad.active && pad.timer > distance / 2300.0 {
                return None;
            }
            let direction = to_pad.normalize();
            let route_alignment = direction.dot(route);
            if route_alignment < -0.10 {
                return None;
            }
            let value = distance / (if pad.large { 1.0 } else { 0.75 })
                + (1.0 - route_alignment.max(0.0)) * 550.0;
            Some((value, pad.location))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, location)| location)
}

pub fn opponent_near_ball(frame: &AgentFrame, radius: f32) -> bool {
    frame.cars.iter().any(|car| {
        car.team != frame.me.team
            && !car.demolished
            && car.location.flat_distance(frame.ball.location) < radius
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_car(index: usize, team: u32, location: Vector) -> CarState {
        use crate::math::Matrix3;
        use crate::state::Hitbox;
        use rlbot::flat::AirState;
        CarState {
            index, team, location, velocity: Vector::ZERO,
            orientation: Matrix3::new(0.0, 0.0, 0.0),
            angular_velocity: Vector::ZERO, demolished: false,
            airborne: false, supersonic: false, jumped: false,
            doublejumped: false, air_state: AirState::OnGround,
            has_jumped: false, has_double_jumped: false, has_dodged: false,
            dodge_timeout: 0.0, dodge_elapsed: 0.0, dodge_dir: Vector::ZERO,
            flip_reset_available: false, boost: 50.0, hitbox: Hitbox::default(),
        }
    }

    fn frame(me: CarState, cars: Vec<CarState>, ball: Vector) -> AgentFrame {
        AgentFrame {
            me, cars, ball: crate::state::BallState { location: ball, velocity: Vector::ZERO },
            match_state: crate::state::MatchState::default(),
            boost_pads: Vec::new(),
        }
    }

    #[test]
    fn support_target_uses_the_most_advanced_living_teammate() {
        let me = make_car(0, 0, Vector::new(0.0, -1000.0, 17.0));
        let lead = make_car(1, 0, Vector::new(500.0, 0.0, 17.0));
        let demolished = CarState { demolished: true, ..make_car(2, 0, Vector::new(-500.0, 100.0, 17.0)) };
        let state = frame(
            me,
            vec![me, lead, demolished],
            Vector::new(200.0, 400.0, 92.75),
        );
        let target = support_target(&state, None);
        assert!((target.x - 374.0).abs() < 0.01);
        assert!((target.y + 900.0).abs() < 0.01);
    }

    #[test]
    fn support_target_uses_wide_own_half_lane() {
        let me = make_car(1, 0, Vector::new(0.0, -2000.0, 17.0));
        let state = frame(
            me, vec![me],
            Vector::new(300.0, -3500.0, 92.75),
        );
        let target = support_target(&state, None);
        assert_eq!(target.x, -2250.0);
        assert_eq!(target.y, -2650.0);
    }

    #[test]
    fn support_target_smooths_only_small_target_changes() {
        let me = make_car(0, 0, Vector::new(0.0, -2000.0, 17.0));
        let state = frame(
            me, vec![me],
            Vector::new(200.0, -1000.0, 92.75),
        );
        let raw = support_target(&state, None);
        let previous = raw + Vector::new(100.0, 0.0, 0.0);
        let smoothed = support_target(&state, Some(previous));
        assert!((smoothed.x - (previous.x + (raw.x - previous.x) * 0.35)).abs() < 0.01);
        assert!((smoothed.y - raw.y).abs() < 0.01);
    }

    #[test]
    fn team_side_convention_matches_goal_coordinates() {
        assert_eq!(team_side(0), -1.0);
        assert_eq!(team_side(1), 1.0);
    }
}
