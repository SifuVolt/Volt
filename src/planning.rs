use crate::math::Vector;
use crate::prediction::BallSlice;
use crate::state::CarState;

/// Which wall family the Noob Black unfolding calculation uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WallTransform {
    Ground,
    XWall,
    YWall,
}

/// A vertical arena wall near the predicted ball.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WallSurface {
    /// true means an x-coordinate side wall; false means a y-coordinate back wall.
    pub x_axis: bool,
    pub side: i8,
    pub gap: f32,
}

/// The contact family selected by the Noob Black reachability scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WallDriveMode {
    SingleJump,
    DoubleJump,
    Aerial,
}

/// A wall contact which passed the source reachability checks.
///
/// intercept_time is relative to the current packet, not an absolute game time.
/// Execution (Volt's wall_approach routine) is intentionally not part of this
/// planner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WallDrivePlan {
    pub intercept_time: f32,
    pub ball_slice: BallSlice,
    pub mode: WallDriveMode,
    pub transform: WallTransform,
    pub surface: WallSurface,
}

/// Returns the sign of a team: blue=-1, orange=+1, matching Volt's utils.side.
pub fn team_side(team: u32) -> i8 {
    if team == 0 { -1 } else { 1 }
}

/// Find the nearest eligible wall family for a ball position.
///
/// Mirrors tools for volt::wall_surface_for_ball, including the opponent
/// back-wall and goal-mouth exclusions.
pub fn wall_surface_for_ball(
    location: Vector,
    max_gap: f32,
    team: Option<u32>,
) -> Option<WallSurface> {
    let mut candidates = [
        WallSurface {
            x_axis: true,
            side: if location.x >= 0.0 { 1 } else { -1 },
            gap: 4096.0 - location.x.abs() - 92.75,
        },
        WallSurface {
            x_axis: false,
            side: if location.y >= 0.0 { 1 } else { -1 },
            gap: 5120.0 - location.y.abs() - 92.75,
        },
    ];
    candidates.sort_by(|a, b| a.gap.total_cmp(&b.gap));

    for candidate in candidates {
        if candidate.gap < -5.0 || candidate.gap > max_gap {
            continue;
        }

        if !candidate.x_axis {
            if location.x.abs() < 900.0 && location.z < 700.0 {
                continue;
            }
            if team.is_some_and(|team| candidate.side == team_side(team)) {
                continue;
            }
        }

        return Some(WallSurface {
            gap: candidate.gap.max(0.0),
            ..candidate
        });
    }

    None
}

/// Detect whether the car is attached to an eligible vertical wall.
pub fn car_wall_surface(car: &CarState, team: u32) -> Option<WallSurface> {
    if car.location.x.abs() > 3720.0
        && car.up().x.abs() > 0.34
        && car.up().x * car.location.x < 0.0
    {
        return Some(WallSurface {
            x_axis: true,
            side: if car.location.x >= 0.0 { 1 } else { -1 },
            gap: 0.0,
        });
    }

    let y_wall_side = if car.location.y >= 0.0 { 1 } else { -1 };
    if car.location.y.abs() > 4740.0
        && car.up().y.abs() > 0.34
        && car.up().y * car.location.y < 0.0
        && team_side(team) != y_wall_side
    {
        return Some(WallSurface {
            x_axis: false,
            side: if car.location.y >= 0.0 { 1 } else { -1 },
            gap: 0.0,
        });
    }

    None
}

/// Noob Black's wall-specific jump-time estimate, including the low-height
/// quadratic branch and its single/double-jump variants.
pub fn noob_wall_jump_time(height: f32, double_jump: u8) -> (f32, u8) {
    let height = height.max(0.0);
    let single_time = (height + 7.0) / 538.988_710_937_5;

    if single_time >= 24.0 / 120.0 {
        if single_time >= 25.0 / 120.0 && double_jump > 0 {
            if double_jump == 1 {
                return ((height + 407.0 / 6.0) / 830.988_710_937_5, 3);
            }
            return ((height + 667.0 / 6.0) / 1038.988_710_937_5, 2);
        }
        return (single_time, 1);
    }

    let (a, b, c) = (717.335_937_5_f32, 272.762_695_312_5_f32, 17.671_875_f32);
    let root = (height - c) / a + b * b / (4.0 * a * a);
    if root < 0.0 {
        return (0.0, 1);
    }

    (-b / (2.0 * a) + root.sqrt(), 1)
}

fn noob_wall_get_dist(position: Vector) -> f32 {
    (4096.0 - position.x.abs()).min(5120.0 - position.y.abs())
}

/// Build Noob Black's 120 Hz floor-to-wall unfolding transform for the ball.
fn ball_from_ground(position: Vector) -> Vector {
    let dx = 4096.0 - position.x.abs();
    let dy = 5120.0 - position.y.abs();

    if dx < dy {
        Vector::new(
            position.x.signum() * (4096.0 + position.z),
            position.y,
            0.0,
        )
    } else {
        Vector::new(
            position.x,
            position.y.signum() * (5120.0 + position.z),
            0.0,
        )
    }
}

/// Unfold a ball position from an x-wall route into the adjacent y-wall plane.
fn ball_x_to_y(origin: Vector, position: Vector) -> Vector {
    let dx = 4096.0 - position.x.abs();
    let dy = 5120.0 - position.y.abs();

    if dx < dy {
        return Vector::new(position.x.signum() * 4096.0, position.y, position.z);
    }

    let origin_x = if origin.x >= 0.0 { 1.0 } else { -1.0 };
    Vector::new(
        origin_x * 4096.0,
        position.y.signum() * (5120.0 + (position.x - origin_x * 4096.0).abs()),
        position.z,
    )
}

/// Unfold a ball position from a y-wall route into the adjacent x-wall plane.
fn ball_y_to_x(origin: Vector, position: Vector) -> Vector {
    let dx = 4096.0 - position.x.abs();
    let dy = 5120.0 - position.y.abs();

    if dx < dy {
        let origin_y = if origin.y >= 0.0 { 1.0 } else { -1.0 };
        return Vector::new(
            position.x.signum() * (4096.0 + (position.y - origin_y * 5120.0).abs()),
            origin_y * 5120.0,
            position.z,
        );
    }

    Vector::new(position.x, position.y.signum() * 5120.0, position.z)
}

fn car_on_x_wall(position: Vector) -> Vector {
    Vector::new(position.x.signum() * 4096.0, position.y, position.z)
}

fn car_on_y_wall(position: Vector) -> Vector {
    Vector::new(position.x, position.y.signum() * 5120.0, position.z)
}

/// Noob Black's 600-step, 120 Hz ground travel profile.
///
/// This is intentionally not replaced by Volt's simpler throttle_acceleration:
/// Noob's profile models the high-speed 1400-1410 transition and boost use
/// explicitly, matching the wall-shot reachability reference.
fn noob_wall_travel_profile(car: &CarState) -> (Vec<f32>, Vec<f32>) {
    let speed = car.velocity.flatten().magnitude();
    let boost = car.boost.max(0.0);
    let mut distances = Vec::with_capacity(601);
    let mut velocities = Vec::with_capacity(601);
    distances.push(0.0);
    velocities.push(speed);

    for tick in 0..600 {
        let velocity = *velocities.last().unwrap();
        let boost_used = tick as f32 / 3.6 < boost;
        let boost_accel = if boost_used { 2975.0 / 3.0 } else { 0.0 };

        let next_velocity = if velocity <= 1400.0 {
            velocity + (1600.0 - 1440.0 / 1400.0 * velocity + boost_accel) / 120.0
        } else if velocity < 1410.0 {
            velocity + (22560.0 - 16.0 * velocity + boost_accel) / 120.0
        } else {
            (velocity + (2975.0 / 360.0) * if boost_used { 1.0 } else { 0.0 })
                .min(2300.0)
        };

        let next_velocity = next_velocity.min(2300.0);
        velocities.push(next_velocity);
        distances.push(distances.last().unwrap() + next_velocity / 120.0);
    }

    (distances, velocities)
}

#[derive(Debug, Clone, Copy)]
struct ReachableContact {
    time: f32,
    slice: BallSlice,
}

fn find_wall_drive_inner(
    car: &CarState,
    slices: &[BallSlice],
    team: u32,
    foe_goal: Vector,
    max_time: f32,
) -> Option<WallDrivePlan> {
    if car.airborne && car_wall_surface(car, team).is_none() {
        return None;
    }
    if slices.len() < 2 {
        return None;
    }

    let max_t = max_time.clamp(0.5, 5.0);
    let max_index = 300_usize.min((max_t * 60.0) as usize);
    let (travel_d, travel_v) = noob_wall_travel_profile(car);

    let car_location = car.location;
    let car_velocity = car.velocity;
    let transform = if car.location.z <= 50.0 {
        WallTransform::Ground
    } else if 4096.0 - car.location.x.abs() < 5120.0 - car.location.y.abs() {
        WallTransform::XWall
    } else {
        WallTransform::YWall
    };
    let car_surface = match transform {
        WallTransform::Ground => car_location.flatten(),
        WallTransform::XWall => car_on_x_wall(car_location),
        WallTransform::YWall => car_on_y_wall(car_location),
    };

    let target_height = |ball_pos: Vector| {
        let deviation = (ball_pos.y - foe_goal.y).abs();
        deviation * ((deviation / 5120.0).min(1.0) * std::f32::consts::FRAC_PI_4).tan()
    };

    let mut single: Option<ReachableContact> = None;
    let mut double: Option<ReachableContact> = None;
    let mut aerial: Option<ReachableContact> = None;
    let mut wait_for_fall = [-1.0_f32; 4];
    let mut air_t = 0.0_f32;

    let jumped = car.jumped;
    let double_jumped = car.doublejumped;
    let jump_vec = 292.0 * if !double_jumped { 1.0 } else { 0.0 }
        + 146.0 * if !jumped { 1.0 } else { 0.0 };
    let up_direction = car.up();

    for i in 0..=max_index {
        let time_value = i as f32 / 60.0;
        let slice_index = (i * 2).min(slices.len() - 1);
        let ball_slice = slices[slice_index];
        let ball_pos = ball_slice.location;

        if ball_pos.y.abs() > 5250.0 {
            break;
        }

        let surface = wall_surface_for_ball(ball_pos, 1800.0, Some(team));
        if surface.is_none() || !(0.0..=2200.0).contains(&ball_pos.z) {
            if single.is_none() || double.is_none() {
                air_t += (2 * i + 1) as f32 / 3600.0;
            }
            continue;
        }
        let surface = surface.unwrap();

        let wall_velocity = (if surface.x_axis {
            ball_slice.velocity.x
        } else {
            ball_slice.velocity.y
        }) * surface.side as f32;

        let wall_jump_time = noob_wall_jump_time(
            (4096.0 - ball_pos.x.abs()).min(5120.0 - ball_pos.y.abs()),
            0,
        ).0;
        let stable_wall_ball = wall_jump_time <= 1.45
            && wall_velocity.abs() < 150.0
            && (400.0..=1900.0).contains(&ball_pos.z);
        let rolling_up_wall_ball = ball_pos.z
            + ball_slice.velocity.z.powi(2) / 1300.0 <= 129.0
            && wall_velocity >= 720.0;

        if !(stable_wall_ball || rolling_up_wall_ball) {
            air_t += (2 * i + 1) as f32 / 3600.0;
            continue;
        }

        let target_point = Vector::new(foe_goal.x, foe_goal.y, target_height(ball_pos));
        let shot_offset = (target_point - ball_pos).normalize();

        if shot_offset.magnitude() > 0.001 {
            if single.is_none() {
                let approach = ball_pos - shot_offset * 134.85;
                let jump_time = noob_wall_jump_time(noob_wall_get_dist(approach), 0).0;
                let mut jump_target = approach;
                jump_target.z += 325.0 * jump_time * jump_time;
                let transformed_target = match transform {
                    WallTransform::Ground => ball_from_ground(jump_target),
                    WallTransform::XWall => ball_x_to_y(car_location, jump_target),
                    WallTransform::YWall => ball_y_to_x(car_location, jump_target),
                };
                let required_distance = transformed_target.distance(car_surface);
                let ticks_charging = (time_value * 120.0).round().clamp(0.0, 600.0) as usize;
                let ticks_driving = ((time_value - jump_time).max(0.0).min(5.0) * 120.0)
                    .round().clamp(0.0, 600.0) as usize;
                let available_distance = travel_d[ticks_driving]
                    + travel_v[ticks_driving] * (ticks_charging as f32 - ticks_driving as f32) / 120.0;

                if available_distance >= required_distance {
                    if jump_time > 1.45 {
                        if wait_for_fall[0] < 0.0 {
                            wait_for_fall[0] = time_value;
                        }
                    } else {
                        single = Some(ReachableContact { time: time_value, slice: ball_slice });
                    }
                }
            }

            if double.is_none() {
                let approach = ball_pos - shot_offset * 92.75;
                let jump_time = noob_wall_jump_time(noob_wall_get_dist(approach), 1).0;
                let mut jump_target = approach;
                jump_target.z += 325.0 * jump_time * jump_time;
                let transformed_target = match transform {
                    WallTransform::Ground => ball_from_ground(jump_target),
                    WallTransform::XWall => ball_x_to_y(car_location, jump_target),
                    WallTransform::YWall => ball_y_to_x(car_location, jump_target),
                };
                let required_distance = transformed_target.distance(car_surface);
                let ticks_charging = (time_value * 120.0).round().clamp(0.0, 600.0) as usize;
                let ticks_driving = ((time_value - jump_time).max(0.0).min(5.0) * 120.0)
                    .round().clamp(0.0, 600.0) as usize;
                let available_distance = travel_d[ticks_driving]
                    + travel_v[ticks_driving] * (ticks_charging as f32 - ticks_driving as f32) / 120.0;

                if available_distance >= required_distance {
                    if jump_time > 1.45 {
                        if wait_for_fall[1] < 0.0 {
                            wait_for_fall[1] = time_value;
                        }
                    } else {
                        double = Some(ReachableContact { time: time_value, slice: ball_slice });
                    }
                }
            }
        }

        if aerial.is_none() {
            if shot_offset.magnitude() <= 0.001 {
                air_t += (2 * i + 1) as f32 / 3600.0;
                continue;
            }
            let contact = ball_pos - shot_offset * 92.75;
            let predicted_car = car_location
                + car_velocity * time_value
                + Vector::new(0.0, 0.0, -325.0) * air_t
                + up_direction * jump_vec;
            let required_distance = contact.distance(predicted_car);

            if required_distance <= (2975.0 / 6.0) * air_t {
                aerial = Some(ReachableContact { time: time_value, slice: ball_slice });
            } else {
                air_t += (2 * i + 1) as f32 / 3600.0;
            }
        }
    }

    // Python uses max_t both as the scan horizon and as the "not found"
    // sentinel. A contact exactly at max_t is therefore not selectable.
    let single = single.filter(|contact| contact.time < max_t);
    let double = double.filter(|contact| contact.time < max_t);
    let aerial = aerial.filter(|contact| contact.time < max_t);

    // Preserve the source convention: once the scan completes, the single
    // route is considered eligible for the aerial budget check.
    wait_for_fall[0] = 1.0;

    let jump_index = if single.is_some() { 0 } else { 1 };
    if single.is_none() && double.is_none() {
        return None;
    }
    let aerial_available = aerial.is_some()
        && wait_for_fall[jump_index] >= 0.0
        && car.boost / 100.0 * 3.0
            >= aerial.unwrap().time + if car.airborne { 0.0 } else { 0.1 };

    let selected_mode = if aerial_available {
        WallDriveMode::Aerial
    } else if single.is_some() {
        WallDriveMode::SingleJump
    } else {
        WallDriveMode::DoubleJump
    };

    let selected = match selected_mode {
        WallDriveMode::Aerial => aerial.unwrap(),
        WallDriveMode::SingleJump => single.unwrap(),
        WallDriveMode::DoubleJump => double.unwrap(),
    };

    let surface = wall_surface_for_ball(selected.slice.location, 1800.0, Some(team))?;
    Some(WallDrivePlan {
        intercept_time: selected.time,
        ball_slice: selected.slice,
        mode: selected_mode,
        transform,
        surface,
    })
}

/// Find the earliest Noob Black wall contact within the requested horizon.
///
/// slices must be uniformly sampled at 120 Hz, matching the Python
/// prediction indexing. The finder evaluates candidates at 60 Hz by reading
/// every second slice. The function is pure and does not mutate the car.
pub fn find_wall_drive(
    car: &CarState,
    slices: &[BallSlice],
    team: u32,
    foe_goal: Vector,
    max_time: f32,
) -> Option<WallDrivePlan> {
    if max_time.is_nan() {
        return None;
    }
    find_wall_drive_inner(car, slices, team, foe_goal, max_time)
}


#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WallApproachGeometry {
    pub ball: Vector,
    pub shot_direction: Vector,
    pub approach: Vector,
    pub jump_time: f32,
    pub target_position: Vector,
    pub car_position: Vector,
    pub distance: f32,
    pub local_target: Vector,
    pub speed_required: f32,
    pub local_angle: f32,
    pub on_wall: bool,
    pub surface_up: Option<Vector>,
}

/// Rebuild Noob Black's wall_approach geometry from the current car state.
pub fn wall_approach_geometry(
    car: &CarState,
    ball_slice: BallSlice,
    team: u32,
    foe_goal: Vector,
    mode: WallDriveMode,
    intercept_time: f32,
) -> Option<WallApproachGeometry> {
    let ball = ball_slice.location;
    let surface = wall_surface_for_ball(ball, 1800.0, Some(team))?;
    let wall_velocity = (if surface.x_axis { ball_slice.velocity.x } else { ball_slice.velocity.y })
        * surface.side as f32;
    let wall_jump_time = noob_wall_jump_time(
        (4096.0 - ball.x.abs()).min(5120.0 - ball.y.abs()), 0,
    ).0;
    let stable_wall_ball = wall_jump_time <= 1.45
        && wall_velocity.abs() < 150.0
        && (400.0..=1900.0).contains(&ball.z);
    let rolling_up_wall_ball = ball.z + ball_slice.velocity.z.powi(2) / 1300.0 <= 129.0
        && wall_velocity >= 720.0;
    if !(stable_wall_ball || rolling_up_wall_ball) {
        return None;
    }

    let deviation = (ball.y - foe_goal.y).abs();
    let target_height = deviation
        * ((deviation / 5120.0).min(1.0) * std::f32::consts::FRAC_PI_4).tan();
    let shot_target = Vector::new(foe_goal.x, foe_goal.y, target_height);
    let shot_direction = (shot_target - ball).normalize();
    if shot_direction == Vector::ZERO {
        return None;
    }

    let contact_offset = match mode {
        WallDriveMode::SingleJump => 134.85,
        WallDriveMode::DoubleJump | WallDriveMode::Aerial => 92.75,
    };
    let approach = ball - shot_direction * contact_offset;
    let jump_index = if mode == WallDriveMode::DoubleJump { 1 } else { 0 };
    let jump_time = noob_wall_jump_time(
        (4096.0 - approach.x.abs()).min(5120.0 - approach.y.abs()), jump_index,
    ).0;
    let mut target_og = approach;
    target_og.z += 325.0 * jump_time * jump_time;

    let (target_position, car_position) = if car.location.z <= 50.0 {
        (ball_from_ground(target_og), car.location.flatten())
    } else if 4096.0 - car.location.x.abs() < 5120.0 - car.location.y.abs() {
        (ball_x_to_y(car.location, target_og), car_on_x_wall(car.location))
    } else {
        (ball_y_to_x(car.location, target_og), car_on_y_wall(car.location))
    };
    let distance = target_position.distance(car_position);
    let local_target = car.local_location(target_position);
    let speed_required = (distance / (intercept_time - jump_time).max(0.12)).clamp(500.0, 2200.0);
    let local_angle = local_target.y.atan2(local_target.x).abs();
    let on_wall = car_wall_surface(car, team).is_some_and(|wall| {
        wall.x_axis == surface.x_axis && wall.side == surface.side
    });
    let surface_up = if on_wall {
        Some(if surface.x_axis {
            Vector::new(-surface.side as f32, 0.0, 0.0)
        } else {
            Vector::new(0.0, -surface.side as f32, 0.0)
        })
    } else {
        None
    };

    Some(WallApproachGeometry {
        ball, shot_direction, approach, jump_time, target_position, car_position,
        distance, local_target, speed_required, local_angle, on_wall, surface_up,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Matrix3;
    use crate::state::Hitbox;
    use rlbot::flat::AirState;

    fn car(location: Vector, velocity: Vector, boost: f32) -> CarState {
        CarState {
            index: 0,
            team: 0,
            location,
            velocity,
            orientation: Matrix3::new(0.0, 0.0, 0.0),
            angular_velocity: Vector::ZERO,
            demolished: false,
            airborne: false,
            supersonic: false,
            jumped: false,
            doublejumped: false,
            air_state: AirState::OnGround,
            has_jumped: false,
            has_double_jumped: false,
            has_dodged: false,
            dodge_timeout: 0.0,
            dodge_elapsed: 0.0,
            dodge_dir: Vector::ZERO,
            flip_reset_available: false,
            boost,
            hitbox: Hitbox::default(),
        }
    }

    #[test]
    fn wall_surface_excludes_own_back_wall_and_goal_mouth() {
        assert!(wall_surface_for_ball(Vector::new(0.0, -5000.0, 100.0), 1800.0, Some(0)).is_none());
        assert!(wall_surface_for_ball(Vector::new(0.0, 5000.0, 100.0), 1800.0, Some(1)).is_none());
        assert!(wall_surface_for_ball(Vector::new(0.0, 5000.0, 100.0), 1800.0, Some(0)).is_none());
    }

    #[test]
    fn wall_surface_accepts_side_wall_and_opponent_back_wall() {
        let side = wall_surface_for_ball(Vector::new(3900.0, 0.0, 500.0), 1800.0, Some(0)).unwrap();
        assert!(side.x_axis);
        let back = wall_surface_for_ball(Vector::new(0.0, 5000.0, 800.0), 1800.0, Some(0)).unwrap();
        assert!(!back.x_axis);
        assert_eq!(back.side, 1);
    }

    #[test]
    fn noob_wall_jump_time_matches_low_and_high_branches() {
        let low = noob_wall_jump_time(20.0, 0);
        let high_single = noob_wall_jump_time(800.0, 0);
        let high_double = noob_wall_jump_time(800.0, 1);
        assert!(low.0 < 0.05);
        assert!(high_single.0 > 1.0);
        assert!(high_double.0 > 0.0);
        assert!(high_double.0 < high_single.0);
    }

    #[test]
    fn wall_unfolding_transforms_match_reference_geometry() {
        let origin = Vector::new(4000.0, 0.0, 500.0);
        let ball = Vector::new(3900.0, 1000.0, 600.0);
        let xwall = ball_x_to_y(origin, ball);
        assert_eq!(xwall.x, 4096.0);
        assert_eq!(xwall.y, 1000.0);
        assert_eq!(xwall.z, 600.0);

        let ground = ball_from_ground(Vector::new(4000.0, 0.0, 300.0));
        assert_eq!(ground.x, 4396.0);
        assert_eq!(ground.z, 0.0);
    }

    #[test]
    fn noob_wall_travel_profile_has_120hz_samples_and_uses_boost() {
        let no_boost = car(Vector::ZERO, Vector::ZERO, 0.0);
        let with_boost = car(Vector::ZERO, Vector::ZERO, 1.0);
        let (d0, v0) = noob_wall_travel_profile(&no_boost);
        let (d1, v1) = noob_wall_travel_profile(&with_boost);

        assert_eq!(d0.len(), 601);
        assert_eq!(v0.len(), 601);
        assert!(d1[120] > d0[120]);
        assert!(v1[120] > v0[120]);
    }

    #[test]
    fn finder_returns_first_reachable_single_jump_wall_contact() {
        let car = car(Vector::ZERO, Vector::ZERO, 0.0);
        let slices: Vec<BallSlice> = (0..=600)
            .map(|index| BallSlice {
                time: index as f32 / 120.0,
                location: Vector::new(4000.0, 0.0, 800.0),
                velocity: Vector::ZERO,
            })
            .collect();

        let plan = find_wall_drive(
            &car,
            &slices,
            0,
            Vector::new(0.0, 5120.0, 0.0),
            5.0,
        )
        .unwrap();

        assert_eq!(plan.mode, WallDriveMode::SingleJump);
        assert!(plan.intercept_time > 4.0 && plan.intercept_time < 5.0);
        assert!(plan.surface.x_axis);
        assert!(plan.ball_slice.location.x > 3900.0);
    }

    #[test]
    fn airborne_car_must_be_attached_to_wall() {
        let mut airborne = car(Vector::ZERO, Vector::ZERO, 50.0);
        airborne.airborne = true;
        let slices = vec![
            BallSlice::default(),
            BallSlice { time: 1.0 / 60.0, ..BallSlice::default() },
        ];
        assert!(find_wall_drive(&airborne, &slices, 0, Vector::new(0.0, 5120.0, 0.0), 5.0).is_none());
    }
}
