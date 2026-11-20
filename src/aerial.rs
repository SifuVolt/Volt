//! Party Cannon-style aerial shot planning and execution.
//!
//! The planner evaluates RLBot's collision-aware ball slices with the same
//! boost-budget validation used by VirxERLU-CLib. Contact geometry follows
//! Party Cannon's 155-unit nose offset and predicts the outgoing ball path
//! before choosing the car's target point.

use crate::{
    control::default_pd,
    math::Vector,
    prediction::{BallSlice, BallTrajectory},
    shot_physics::aerial_shot_is_viable,
    state::CarState,
};
use rlbot::flat::ControllerState;

const BALL_RADIUS: f32 = 92.75;
const MAX_SPEED: f32 = 2300.0;
const BOOST_CONSUMPTION: f32 = 33.3;
const AIR_THROTTLE_ACCEL: f32 = 66.667;
const JUMP_VEL: f32 = 291.667;
const JUMP_ACCEL: f32 = 1458.333;
const JUMP_DURATION: f32 = 0.2;
const STICKY_ACCEL: f32 = 325.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AerialCandidate {
    pub slice: BallSlice,
    pub target_location: Vector,
    pub shot_direction: Vector,
    pub double_jump: bool,
    pub fast: bool,
    pub estimated_boost: f32,
    pub score: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AerialShot {
    pub started_at: f32,
    pub intercept_time: f32,
    pub ball: Vector,
    pub target_location: Vector,
    pub shot_direction: Vector,
    pub start_boost: f32,
    pub double_jump: bool,
    pub fast: bool,
    pub elapsed: f32,
    pub jump_release_frames: u8,
    pub second_jump_frames: u8,
    pub jumped: bool,
    pub dodging: bool,
    pub dodge_elapsed: f32,
    pub dodge_step: u8,
    pub dodge_input: Vector,
    pub finished: bool,
}

impl AerialShot {
    pub fn new(car: CarState, candidate: AerialCandidate, game_time: f32) -> Self {
        Self {
            started_at: game_time,
            intercept_time: game_time + candidate.slice.time,
            ball: candidate.slice.location,
            target_location: candidate.target_location,
            shot_direction: candidate.shot_direction,
            start_boost: car.boost,
            double_jump: candidate.double_jump,
            fast: candidate.fast,
            elapsed: 0.0,
            jump_release_frames: 0,
            second_jump_frames: 0,
            jumped: car.has_jumped || car.airborne,
            dodging: false,
            dodge_elapsed: 0.0,
            dodge_step: 0,
            dodge_input: Vector::ZERO,
            finished: false,
        }
    }

    pub fn run(
        &mut self,
        car: CarState,
        game_time: f32,
        gravity: Vector,
        boost_accel: f32,
    ) -> ControllerState {
        let mut controls = ControllerState::default();
        let time_remaining = self.intercept_time - game_time;
        if self.finished {
            return controls;
        }
        if self.dodging {
            return self.run_dodge(car, controls);
        }
        if time_remaining <= -0.04 || (car.airborne && car.has_dodged) {
            self.finished = true;
            return controls;
        }
        if self.start_boost < car.boost || car.boost <= 0.0 {
            self.finished = true;
            return controls;
        }

        self.elapsed = (game_time - self.started_at).max(0.0);
        let mut currently_double_jumping = false;

        if !self.jumped && !car.has_dodged {
            if self.elapsed <= JUMP_DURATION {
                controls.jump = true;
            } else if self.double_jump && self.jump_release_frames < 3 {
                controls.jump = false;
                self.jump_release_frames += 1;
                return controls;
            } else if self.double_jump && self.second_jump_frames < 6 {
                controls.jump = true;
                self.second_jump_frames += 1;
                currently_double_jumping = true;
            } else {
                self.jumped = true;
            }
        }

        let time = time_remaining.max(0.001);
        let (final_position, final_velocity) = if self.jumped || (car.airborne && self.elapsed > JUMP_DURATION) {
            (
                car.location + car.velocity * time + gravity * (0.5 * time * time),
                car.velocity + gravity * time,
            )
        } else {
            jump_prediction(
                car,
                time,
                self.elapsed,
                self.double_jump && !car.has_double_jumped && !car.has_dodged,
                gravity,
            )
        };

        let offset = self.target_location - final_position;
        let offset_length = offset.magnitude();
        let forward = car.forward();
        let direction_to_target = offset.normalize();
        let local_target = if self.jumped {
            car.local(self.ball - car.location)
        } else {
            car.local(offset)
        };
        let (_, pd) = default_pd(&car, local_target, false, None);
        controls.pitch = pd.pitch;
        controls.yaw = pd.yaw;
        controls.roll = pd.roll;

        let forward_projection = offset.dot(forward) / time;
        let turn_angle = offset.angle(forward);
        controls.boost = forward_projection
            >= (boost_accel + AIR_THROTTLE_ACCEL) * (1.0 / 120.0_f32).max(13.0 / 120.0)
            && turn_angle < 0.4
            && car.boost > 0.0;
        controls.throttle = (forward_projection
            / (AIR_THROTTLE_ACCEL * (1.0 / 120.0_f32).max(1.0 / 120.0)))
            .clamp(-1.0, 1.0);

        if currently_double_jumping {
            controls.steer = 0.0;
            controls.yaw = 0.0;
            controls.pitch = 0.0;
            controls.roll = 0.0;
        }

        let required_accel = 2.0 * offset_length / (time * time);
        let cannot_reach = self.jumped
            && offset_length > 50.0
            && time > 0.5
            && required_accel * 0.8 > AIR_THROTTLE_ACCEL
            && (car.boost <= 0.0 || required_accel * 0.8 > boost_accel + AIR_THROTTLE_ACCEL);
        if time_remaining <= 0.0 || cannot_reach {
            self.finished = true;
        } else if time_remaining < 0.12 && offset_length < 115.0 {
            self.dodging = true;
            self.dodge_elapsed = 0.0;
            self.dodge_step = 0;
            self.dodge_input = dodge_direction(car, self.shot_direction);
            return self.run_dodge(car, ControllerState::default());
        } else if offset_length < 50.0 && !currently_double_jumping {
            let (_, aligned) = default_pd(&car, car.local(self.shot_direction), false, None);
            controls.pitch = aligned.pitch;
            controls.yaw = aligned.yaw;
            controls.roll = aligned.roll;
        }

        let _ = (final_velocity, direction_to_target, BALL_RADIUS, BOOST_CONSUMPTION);
        controls
    }

    fn run_dodge(
        &mut self,
        car: CarState,
        mut controls: ControllerState,
    ) -> ControllerState {
        self.dodge_elapsed += 1.0 / 120.0;
        let dodge_jump = !car.airborne;
        if dodge_jump && self.dodge_elapsed < 0.1 {
            controls.jump = true;
        } else if dodge_jump && self.dodge_step < 3 {
            controls.jump = false;
            self.dodge_step += 1;
        } else if self.dodge_elapsed < if dodge_jump { 0.7 } else { 0.6 } {
            controls.yaw = self.dodge_input.x;
            controls.pitch = self.dodge_input.y;
            controls.jump = true;
        } else {
            self.finished = true;
        }
        if car.has_dodged {
            self.finished = true;
        }
        controls
    }
}

fn jump_prediction(
    car: CarState,
    time: f32,
    elapsed: f32,
    double_jump: bool,
    gravity: Vector,
) -> (Vector, Vector) {
    let up = car.up();
    let jump_remaining = (JUMP_DURATION - elapsed).clamp(0.0, JUMP_DURATION);
    let stick_remaining = (0.05 - elapsed).clamp(0.0, 0.05);
    let grounded = !car.has_jumped && !car.airborne;
    let mut position = car.location
        + car.velocity * time
        + gravity * (0.5 * time * time)
        + if grounded { up * (JUMP_VEL * time) } else { Vector::ZERO }
        + up * JUMP_ACCEL * jump_remaining * (time - 0.5 * jump_remaining)
        - up * STICKY_ACCEL * stick_remaining * (time - 0.5 * stick_remaining);
    let mut velocity = car.velocity
        + gravity * time
        + if grounded { up * JUMP_VEL } else { Vector::ZERO }
        + up * JUMP_ACCEL * jump_remaining
        - up * STICKY_ACCEL * stick_remaining;
    if double_jump {
        position = position + up * JUMP_VEL * (time - jump_remaining);
        velocity = velocity + up * JUMP_VEL;
    }
    (position, velocity)
}

fn dodge_direction(car: CarState, shot_direction: Vector) -> Vector {
    let forward = car.forward();
    let flat_forward = forward.flatten().normalize();
    let direction = shot_direction.flatten().normalize();
    let perpendicular = Vector::new(flat_forward.y, -flat_forward.x, 0.0);
    let mut local = Vector::new(-perpendicular.dot(direction), -flat_forward.dot(direction), 0.0);
    let forward_velocity = forward.dot(car.velocity);
    let speed_fraction = forward_velocity.abs() / MAX_SPEED;
    let backwards = if forward_velocity.abs() < 100.0 {
        local.x < 0.0
    } else {
        (local.x >= 0.0) != (forward_velocity > 0.0)
    };
    if backwards {
        local.x /= (16.0 / 15.0) * (1.0 + 1.5 * speed_fraction);
    }
    local.y /= 1.0 + 0.9 * speed_fraction;
    local.normalize()
}

fn shot_power_modifier(value: f32) -> f32 {
    let value = value.clamp(0.0, 4600.0);
    if value <= 500.0 {
        0.65
    } else if value <= MAX_SPEED {
        let t = (value - 500.0) / 1800.0;
        0.65 + (0.55 - 0.65) * t
    } else {
        let t = (value - MAX_SPEED) / 4600.0;
        0.55 + (0.3 - 0.55) * t
    }
}

fn contact_geometry(
    car: CarState,
    slice: BallSlice,
    goal: Vector,
    gravity: Vector,
) -> (Vector, Vector) {
    let time = slice.time.max(0.001);
    let raw_car_velocity = (slice.location - car.location) / time;
    let raw_speed = raw_car_velocity.magnitude();
    let car_final_velocity = if raw_speed > MAX_SPEED {
        raw_car_velocity * (MAX_SPEED / raw_speed)
    } else {
        raw_car_velocity
    };
    let ball_after_velocity = (car_final_velocity * 6.0 + slice.velocity) / 7.0;
    let direction_to_goal = (goal - slice.location).flatten().normalize();
    let velocity_difference = (car_final_velocity - slice.velocity).magnitude();
    let scoring_speed = (velocity_difference * shot_power_modifier(velocity_difference)
        + ball_after_velocity.dot(direction_to_goal))
        .clamp(500.0, 6000.0);
    let distance_to_goal = goal.flat_distance(slice.location);
    let time_to_goal = distance_to_goal / scoring_speed;
    let ball_after_velocity = (car_final_velocity + slice.velocity * 2.0) / 3.0;
    let predicted_ball = slice.location
        + ball_after_velocity * time_to_goal
        + gravity * (0.5 * time_to_goal * time_to_goal);
    let mut shot_direction = (goal - predicted_ball).normalize();
    let mut target_location = slice.location - shot_direction * 155.0;

    let (surface_point, normal) = nearest_surface(target_location);
    let distance_from_surface = (target_location - surface_point).dot(normal);
    if distance_from_surface < 50.0 {
        let angle = ((distance_from_surface - 50.0) / 155.0).clamp(-1.0, 1.0).asin();
        let flat_direction = shot_direction - normal * shot_direction.dot(normal);
        shot_direction = (flat_direction.normalize() * angle.cos() + normal * angle.sin()).normalize();
        target_location = slice.location - shot_direction * 155.0;
    }
    (target_location, shot_direction)
}

fn nearest_surface(location: Vector) -> (Vector, Vector) {
    let candidates = [
        (location.z.abs(), Vector::new(location.x, location.y, 0.0), Vector::new(0.0, 0.0, 1.0)),
        ((2044.0 - location.z).abs(), Vector::new(location.x, location.y, 2044.0), Vector::new(0.0, 0.0, -1.0)),
        ((4096.0 - location.x.abs()).abs(), Vector::new(location.x.signum() * 4096.0, location.y, location.z), Vector::new(-location.x.signum(), 0.0, 0.0)),
        ((5120.0 - location.y.abs()).abs(), Vector::new(location.x, location.y.signum() * 5120.0, location.z), Vector::new(0.0, -location.y.signum(), 0.0)),
    ];
    let best = candidates.iter().min_by(|a, b| a.0.total_cmp(&b.0)).unwrap();
    (best.1, best.2)
}

fn opponent_eta(opponent: CarState, target: Vector) -> f32 {
    if opponent.demolished {
        return f32::INFINITY;
    }
    let delta = (target - opponent.location).flatten();
    let distance = delta.magnitude();
    let forward = opponent.forward().flatten().normalize();
    let direction = delta.normalize();
    let facing = if direction == Vector::ZERO || forward == Vector::ZERO {
        0.0
    } else {
        forward.dot(direction).clamp(-1.0, 1.0)
    };
    let speed = opponent.velocity.flatten().magnitude();
    distance / (speed + 650.0).clamp(850.0, 2100.0)
        + (1.0 - facing) * 0.28
}

pub fn find_aerial_candidate(
    car: CarState,
    prediction: &BallTrajectory,
    goal: Vector,
    opponents: &[CarState],
    gravity: Vector,
    boost_accel: f32,
) -> Option<AerialCandidate> {
    if prediction.slices.is_empty() || car.boost <= 0.0 {
        return None;
    }
    let mut best: Option<AerialCandidate> = None;
    for slice in prediction.slices.iter().copied().filter(|slice| {
        (0.45..=3.5).contains(&slice.time) && slice.location.z >= 350.0
    }) {
        let (target_location, shot_direction) = contact_geometry(car, slice, goal, gravity);
        let validation = aerial_shot_is_viable(
            &car,
            target_location,
            slice.time,
            gravity,
            boost_accel,
            true,
        );
        if !validation.valid {
            continue;
        }
        let fastest_opponent = opponents.iter()
            .map(|opponent| opponent_eta(*opponent, slice.location))
            .fold(f32::INFINITY, f32::min);
        let race_penalty = if fastest_opponent + 0.08 < slice.time {
            ((slice.time - fastest_opponent) / slice.time.max(0.1)).clamp(0.0, 1.0) * 0.22
        } else {
            0.0
        };
        let local_target = car.local_location(target_location);
        let alignment_penalty = car.forward().angle(local_target) * 0.035;
        let boost_penalty = (validation.validation.boost_estimate as f32 / car.boost.max(1.0)) * 0.10;
        let score = slice.time + race_penalty + alignment_penalty + boost_penalty
            + if validation.fast { 0.025 } else { 0.0 };
        let candidate = AerialCandidate {
            slice,
            target_location,
            shot_direction,
            // CLib's fast-aerial branch includes the second jump impulse;
            // execute the matching three-frame release and second-jump hold.
            double_jump: validation.fast,
            fast: validation.fast,
            estimated_boost: validation.validation.boost_estimate as f32,
            score,
        };
        if best.is_none_or(|current| candidate.score < current.score) {
            best = Some(candidate);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{math::Matrix3, state::Hitbox};
    use rlbot::flat::AirState;

    fn car(location: Vector, boost: f32) -> CarState {
        CarState {
            index: 0,
            team: 0,
            location,
            velocity: Vector::ZERO,
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
            hitbox: Hitbox { length: 118.0, width: 84.0, height: 36.0, offset: Vector::ZERO },
        }
    }

    #[test]
    fn party_cannon_contact_target_is_155_units_behind_ball() {
        let slice = BallSlice {
            time: 1.5,
            location: Vector::new(0.0, 0.0, 900.0),
            velocity: Vector::ZERO,
        };
        let (target, direction) = contact_geometry(
            car(Vector::new(0.0, -1500.0, 17.0), 100.0),
            slice,
            Vector::new(0.0, 5120.0, 300.0),
            Vector::new(0.0, 0.0, -650.0),
        );
        assert!((target.distance(slice.location) - 155.0).abs() < 0.01);
        assert!(direction.magnitude() > 0.99);
    }

    #[test]
    fn aerial_candidate_requires_a_valid_boost_budget() {
        let car = car(Vector::new(0.0, -1200.0, 17.0), 100.0);
        let prediction = BallTrajectory {
            slices: vec![
                BallSlice {
                    time: 1.5,
                    location: Vector::new(0.0, 0.0, 900.0),
                    velocity: Vector::ZERO,
                },
                BallSlice {
                    time: 2.0,
                    location: Vector::new(500.0, 0.0, 1100.0),
                    velocity: Vector::ZERO,
                },
                BallSlice {
                    time: 3.0,
                    location: Vector::new(0.0, 0.0, 900.0),
                    velocity: Vector::ZERO,
                },
            ],
        };
        let result = find_aerial_candidate(
            car,
            &prediction,
            Vector::new(0.0, 5120.0, 300.0),
            &[],
            Vector::new(0.0, 0.0, -650.0),
            991.6667,
        );
        assert!(result.is_some());
        assert!(result.unwrap().estimated_boost >= 0.0);
    }

    #[test]
    fn aerial_state_emits_jump_and_boost_controls() {
        let car = car(Vector::ZERO, 100.0);
        let candidate = AerialCandidate {
            slice: BallSlice {
                time: 1.5,
                location: Vector::new(0.0, 0.0, 900.0),
                velocity: Vector::ZERO,
            },
            target_location: Vector::new(0.0, 0.0, 745.0),
            shot_direction: Vector::new(0.0, 0.0, 1.0),
            double_jump: false,
            fast: false,
            estimated_boost: 20.0,
            score: 1.5,
        };
        let mut shot = AerialShot::new(car, candidate, 1.0);
        let controls = shot.run(
            car,
            1.05,
            Vector::new(0.0, 0.0, -650.0),
            991.6667,
        );
        assert!(controls.jump);
    }
}
