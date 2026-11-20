//! Source-driven standalone mechanics controllers ported from Volt's
//! `routines of volt`: `wave_dash` and `wall_dash`.
//!
//! These controllers deliberately expose an explicit stateful tick API so
//! they can be scheduled by the routine executor without embedding hidden
//! timing or packet access in the physics layer.

use crate::{
    control::default_pd,
    math::Vector,
    state::CarState,
};
use rlbot::flat::ControllerState;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MechanicsOutput {
    pub controller: ControllerState,
    pub finished: bool,
}

/// The Python wall_dash routine's four phases:
/// jump, jump-release, directional dodge, and wait for wall recontact.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WallDash {
    target: Vector,
    start_time: Option<f32>,
    phase_start: f32,
    phase: u8,
    last_wall_contact: Option<f32>,
    lost_wall_time: Option<f32>,
    dash_count: u32,
}

impl WallDash {
    pub const JUMP_TIME: f32 = 0.07;
    pub const RELEASE_TIME: f32 = 0.05;
    pub const DODGE_TIME: f32 = 0.08;
    pub const LAND_TIMEOUT: f32 = 0.40;
    pub const LAND_SETTLE_TIME: f32 = 0.05;
    pub const MAX_TIME: f32 = 2.5;

    pub fn new(target: Vector) -> Self {
        Self {
            target,
            start_time: None,
            phase_start: 0.0,
            phase: 0,
            last_wall_contact: None,
            lost_wall_time: None,
            dash_count: 0,
        }
    }

    pub fn dash_count(&self) -> u32 {
        self.dash_count
    }

    pub fn on_side_wall(car: &CarState) -> bool {
        !car.airborne && car.location.x.abs() > 3500.0 && car.up().z.abs() < 0.45
    }

    pub fn tick(&mut self, car: CarState, now: f32) -> MechanicsOutput {
        let start = *self.start_time.get_or_insert_with(|| {
            self.phase_start = now;
            now
        });
        let elapsed = now - start;
        let phase_time = now - self.phase_start;
        if elapsed > Self::MAX_TIME {
            return MechanicsOutput { finished: true, ..Default::default() };
        }

        let distance = car.location.flatten().distance(self.target.flatten());
        if distance < 1200.0 {
            return MechanicsOutput { finished: true, ..Default::default() };
        }

        let mut output = ControllerState::default();
        let local_target = car.local_location(self.target);
        let (_, orientation) = default_pd(&car, local_target, false, None);
        output.pitch = orientation.pitch;
        output.yaw = orientation.yaw;
        output.roll = orientation.roll;
        output.steer = orientation.steer;
        output.throttle = 1.0;
        output.boost = car.boost > 0.0 && distance > 1800.0;

        let wall_contact = Self::on_side_wall(&car);
        if wall_contact {
            self.last_wall_contact = Some(now);
        }

        match self.phase {
            0 => {
                if !wall_contact {
                    if self.last_wall_contact.is_none_or(|last| now - last > 0.15) {
                        return MechanicsOutput { finished: true, ..Default::default() };
                    }
                    return MechanicsOutput { controller: output, finished: false };
                }
                output.jump = true;
                if phase_time >= Self::JUMP_TIME {
                    self.phase = 1;
                    self.phase_start = now;
                }
            }
            1 => {
                output.jump = false;
                if phase_time >= Self::RELEASE_TIME {
                    self.phase = 2;
                    self.phase_start = now;
                }
            }
            2 => {
                output.jump = true;
                output.pitch = -1.0;
                output.yaw = if local_target.y.abs() > 100.0 {
                    if local_target.y > 0.0 { 0.6 } else { -0.6 }
                } else {
                    0.0
                };
                if phase_time >= Self::DODGE_TIME {
                    self.phase = 3;
                    self.phase_start = now;
                    self.lost_wall_time = Some(now);
                    self.dash_count += 1;
                }
            }
            _ => {
                output.jump = false;
                output.pitch = 0.0;
                output.yaw = 0.0;
                if wall_contact {
                    if phase_time >= Self::LAND_SETTLE_TIME {
                        self.phase = 0;
                        self.phase_start = now;
                    }
                } else if self.lost_wall_time.is_some_and(|lost| now - lost > Self::LAND_TIMEOUT) {
                    return MechanicsOutput { finished: true, ..Default::default() };
                }
            }
        }

        MechanicsOutput { controller: output, finished: false }
    }
}

/// The Python wave_dash routine. Its step counter advances once per routine
/// tick, while the two-second timeout is measured using game time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaveDash {
    target: Option<Vector>,
    direction: u8,
    start_time: Option<f32>,
    step: u32,
}

impl WaveDash {
    pub fn new(target: Option<Vector>) -> Self {
        Self { target, direction: 0, start_time: None, step: 0 }
    }

    pub fn tick(&mut self, car: CarState, now: f32) -> MechanicsOutput {
        let start = *self.start_time.get_or_insert(now);
        if let Some(target) = self.target {
            self.direction = if target.x.abs() > target.y.abs() { 0 } else { 1 };
            if (self.direction == 0 && target.x < 0.0)
                || (self.direction == 1 && target.y < 0.0)
            {
                self.direction += 2;
            }
        }

        let forward_target = car.velocity.flatten().normalize() * 100.0;
        let target_switch = match self.direction {
            0 => forward_target + Vector::new(0.0, 0.0, 50.0),
            1 => forward_target,
            2 => forward_target - Vector::new(0.0, 0.0, 50.0),
            _ => forward_target,
        };
        let target_up = match self.direction {
            0 | 2 => Vector::new(0.0, 0.0, 1.0),
            1 => Vector::new(0.0, -50.0, 1.0),
            _ => Vector::new(0.0, 50.0, 1.0),
        };

        let (_, orientation) = default_pd(
            &car,
            car.local(target_switch),
            false,
            Some(car.local(target_up)),
        );
        let mut output = ControllerState::default();
        output.pitch = orientation.pitch;
        output.yaw = orientation.yaw;
        output.roll = orientation.roll;
        output.steer = orientation.steer;

        match self.direction {
            0 => output.throttle = 1.0,
            2 => output.throttle = -1.0,
            _ => output.handbrake = true,
        }

        if self.step < 1 {
            output.jump = true;
        } else if self.step < 4 {
            // Explicit jump release window.
        } else if !car.airborne {
            return MechanicsOutput { finished: true, ..Default::default() };
        } else if now - start > 2.0 {
            return MechanicsOutput { finished: true, ..Default::default() };
        } else if car.location.z + car.velocity.z * 0.2 < 5.0 {
            output.jump = true;
            output.yaw = 0.0;
            if self.direction == 0 || self.direction == 2 {
                output.roll = 0.0;
                output.pitch = if self.direction == 0 { -1.0 } else { 1.0 };
            } else {
                output.roll = if self.direction == 1 { -1.0 } else { 1.0 };
                output.pitch = 0.0;
            }
        }

        self.step += 1;
        MechanicsOutput { controller: output, finished: false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{math::Matrix3, state::Hitbox};
    use rlbot::flat::AirState;

    fn car(location: Vector, velocity: Vector, airborne: bool, up: Vector) -> CarState {
        let mut orientation = Matrix3::new(0.0, 0.0, 0.0);
        orientation.up = up;
        CarState {
            index: 0,
            team: 0,
            location,
            velocity,
            orientation,
            angular_velocity: Vector::ZERO,
            demolished: false,
            airborne,
            supersonic: false,
            jumped: airborne,
            doublejumped: false,
            air_state: if airborne { AirState::InAir } else { AirState::OnGround },
            has_jumped: airborne,
            has_double_jumped: false,
            has_dodged: false,
            dodge_timeout: 0.0,
            dodge_elapsed: 0.0,
            dodge_dir: Vector::ZERO,
            flip_reset_available: false,
            boost: 50.0,
            hitbox: Hitbox::default(),
        }
    }

    #[test]
    fn wall_dash_requires_actual_side_wall_contact_before_jump() {
        let mut dash = WallDash::new(Vector::new(0.0, 3000.0, 0.0));
        let grounded_floor = car(Vector::ZERO, Vector::ZERO, false, Vector::new(0.0, 0.0, 1.0));
        assert!(dash.tick(grounded_floor, 1.0).finished);
    }

    #[test]
    fn wall_dash_runs_jump_release_and_dodge_phases() {
        let mut dash = WallDash::new(Vector::new(0.0, 3000.0, 0.0));
        let wall_car = car(
            Vector::new(3900.0, 0.0, 500.0),
            Vector::new(0.0, 1000.0, 0.0),
            false,
            Vector::new(-1.0, 0.0, 0.0),
        );
        assert!(dash.tick(wall_car, 1.0).controller.jump);
        assert!(dash.tick(wall_car, 1.08).controller.jump);
        assert!(!dash.tick(wall_car, 1.14).controller.jump);
        let dodge = dash.tick(wall_car, 1.23).controller;
        assert!(dodge.jump);
        assert_eq!(dodge.pitch, -1.0);
        assert_eq!(dash.dash_count(), 1);
    }

    #[test]
    fn wave_dash_releases_jump_then_repeats_landing_input() {
        let mut dash = WaveDash::new(None);
        let airborne = car(
            Vector::new(0.0, 0.0, 20.0),
            Vector::new(1000.0, 0.0, -100.0),
            true,
            Vector::new(0.0, 0.0, 1.0),
        );
        assert!(dash.tick(airborne, 1.0).controller.jump);
        assert!(!dash.tick(airborne, 1.01).controller.jump);
        assert!(!dash.tick(airborne, 1.02).controller.jump);
        assert!(!dash.tick(airborne, 1.03).controller.jump);
    }

    #[test]
    fn wave_dash_selects_backward_direction_from_target_sign() {
        let mut dash = WaveDash::new(Some(Vector::new(-500.0, 0.0, 0.0)));
        let grounded = car(Vector::ZERO, Vector::new(1000.0, 0.0, 0.0), false, Vector::new(0.0, 0.0, 1.0));
        let output = dash.tick(grounded, 0.0);
        assert_eq!(output.controller.throttle, -1.0);
    }
}
