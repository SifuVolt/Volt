//! Source-faithful port of Volt's Python `grid_defense` ground routine.
//!
//! This module deliberately keeps the original branch structure and numeric
//! thresholds. Stateful previous-front tracking belongs to the routine, just
//! as it does in Python.

use crate::{
    math::Vector,
    state::{BallState, BoostPadState, CarState},
};
use rlbot::flat::ControllerState;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DefenseOutput {
    pub controller: ControllerState,
    pub start_flip: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GridDefense {
    previous_front: Option<Vector>,
}

impl GridDefense {
    fn sign(value: f32) -> f32 {
        if value < 0.0 { -1.0 } else if value > 0.0 { 1.0 } else { 0.0 }
    }

    fn angle(a: Vector, b: Vector) -> f32 {
        if a.magnitude() < 1.0 || b.magnitude() < 1.0 {
            return std::f32::consts::FRAC_PI_2;
        }
        a.normalize().dot(b.normalize()).clamp(-1.0, 1.0).acos()
    }

    fn steer(car: CarState, target: Vector) -> f32 {
        let local = car.local_location(target);
        (local.y.atan2(local.x) * 5.0).clamp(-1.0, 1.0)
    }

    fn roll(car: CarState) -> f32 {
        (-car.right().z).atan2(car.up().z)
    }

    fn angular_speed(car: CarState) -> f32 {
        car.angular_velocity.magnitude()
    }

    fn choose_pad(
        car: CarState,
        goal: Vector,
        pads: &[BoostPadState],
        fast: bool,
    ) -> Option<BoostPadState> {
        if car.boost >= 100.0 || fast {
            return None;
        }

        let car_goal_distance = car.location.distance(-goal);
        pads.iter()
            .copied()
            .filter(|pad| {
                (pad.location.x.abs() > 3000.0 || pad.location.y.abs() > 4000.0)
                    && pad.location.distance(-goal) < car_goal_distance
                    && Self::sign(pad.location.x) == Self::sign(car.location.x)
                    && pad.active
            })
            .min_by(|a, b| {
                // Python sorts (not boost.large, score): every full pad is
                // preferred to every small pad, then shortest route wins.
                let a_key = (
                    !a.large,
                    car.location.distance(a.location) + a.location.distance(-goal),
                );
                let b_key = (
                    !b.large,
                    car.location.distance(b.location) + b.location.distance(-goal),
                );
                a_key.0.cmp(&b_key.0).then_with(|| a_key.1.total_cmp(&b_key.1))
            })
    }

    fn prepare_for_attack(
        &mut self,
        car: CarState,
        ball: BallState,
        goal: Vector,
        lane_side: f32,
    ) -> ControllerState {
        let mut controller = ControllerState::default();
        let mut target = Vector::new(4000.0 * lane_side, 0.0, 0.0);
        let direction = target + goal;
        let distance = target.distance(-goal) / 5120.0
            * (ball.location.y * goal.y + 1000.0).clamp(1000.0, 5120.0);
        if direction.magnitude() > 1.0 {
            target = -goal + direction.normalize() * distance;
        }

        let target_distance = target.distance(car.location.flatten());
        if car.location.flatten().distance(target) <= 100.0 {
            let throttle_value = -Self::sign(
                std::f32::consts::FRAC_PI_2 - Self::angle(car.velocity, car.forward()),
            ) * car.velocity.magnitude().powi(2) / 6000.0
                + Self::sign(
                    std::f32::consts::FRAC_PI_2
                        - Self::angle(target - car.location.flatten(), car.forward()),
                ) * target.distance(car.location.flatten());
            controller.throttle = Self::sign(throttle_value);
            if controller.throttle == 0.0 { controller.throttle = 1.0; }
        } else {
            controller.throttle =
                Self::sign(target_distance - car.velocity.magnitude().powi(2) / 6000.0);
        }

        let previous_front = self.previous_front.unwrap_or(car.forward());
        let speed = car.velocity.magnitude();
        controller.handbrake =
            Self::angle(car.velocity, car.forward()) <= std::f32::consts::FRAC_PI_4
                && Self::angle(car.forward(), target - car.location.flatten())
                    > std::f32::consts::FRAC_PI_4
                && Self::angular_speed(car) >= (speed / 500.0).min(2.0)
                && Self::angle(target - car.location.flatten(), car.forward())
                    < Self::angle(target - car.location.flatten(), previous_front);
        controller.steer = Self::steer(car, target);
        controller.boost = false;
        self.previous_front = Some(car.forward());
        controller
    }

    pub fn run(
        &mut self,
        car: CarState,
        cars: &[CarState],
        ball: BallState,
        pads: &[BoostPadState],
        goal: Vector,
        goalie: bool,
        fast: bool,
    ) -> DefenseOutput {
        let side = if car.team == 0 { -1.0 } else { 1.0 };
        let goal_sign = if Self::sign(goal.y) == 0.0 { 1.0 } else { Self::sign(goal.y) };
        let ball_depth = ball.location.y * side;
        let car_depth = car.location.y * side;

        // Python grid_defense's prepare_for_attack transition.
        let should_prepare = !goalie
            && ball_depth < 0.0
            && car_depth >= -900.0
            && (car.forward().y * side < 0.0 || car_depth > 4120.0);
        if should_prepare {
            let nearest_opponent = cars
                .iter()
                .copied()
                .filter(|other| other.team != car.team && !other.demolished)
                .min_by(|a, b| {
                    a.location.distance(ball.location)
                        .total_cmp(&b.location.distance(ball.location))
                });
            let lane_side = nearest_opponent
                .map(|foe| Self::sign(ball.location.x - foe.location.x) * 0.5)
                .unwrap_or(0.0);
            return DefenseOutput {
                controller: self.prepare_for_attack(car, ball, goal, lane_side),
                start_flip: false,
            };
        }

        let goalie_fast = goalie && (car.location.y + goal.y).abs() > 1000.0;
        let best_pad = Self::choose_pad(car, goal, pads, fast || goalie_fast);
        let target = if let Some(pad) = best_pad {
            pad.location
        } else if goalie {
            Vector::new(0.0, -goal.y - goal_sign * 150.0, 0.0)
        } else {
            Vector::new(Self::sign(car.location.x) * 900.0, -goal_sign * 5050.0, 0.0)
        };

        let mut controller = ControllerState::default();
        controller.boost = false;
        controller.handbrake = false;

        if car.location.distance(-goal) <= 1500.0 {
            let depth = car.location.y * goal_sign;
            let speed = car.velocity.magnitude();

            if speed <= 100.0
                && -5320.0 + if goalie { 0.0 } else { 200.0 } > depth
                && Self::angle(car.forward(), target - car.location) > std::f32::consts::FRAC_PI_4
            {
                controller.handbrake = true;
                controller.throttle = -Self::sign(
                    std::f32::consts::FRAC_PI_2 - Self::angle(car.forward(), car.velocity),
                );
                controller.steer = Self::steer(car, target) * controller.throttle;
            } else if speed <= 100.0
                && depth >= -5320.0 + if goalie { 0.0 } else { 200.0 }
                && depth <= -5120.0 + if goalie { 0.0 } else { 100.0 }
                && Self::angle(car.forward(), ball.location - car.location) >= 0.2
            {
                controller.handbrake = true;
                controller.throttle = -Self::sign(
                    std::f32::consts::FRAC_PI_2 - Self::angle(car.forward(), car.velocity),
                );
                let steer_target = if goalie {
                    let ball_lane = ball.location.flatten()
                        - Vector::new(ball.location.x.clamp(-900.0, 900.0), 0.0, 0.0)
                        + goal;
                    if Self::angle(ball_lane, goal) <= std::f32::consts::FRAC_PI_3 {
                        ball.location
                    } else {
                        -goal + Vector::new(
                            ball.location.x,
                            Self::sign(ball.location.y + goal.y)
                                / std::f32::consts::FRAC_PI_3.tan()
                                * (ball.location.x.abs() - 900.0),
                            0.0,
                        )
                    }
                } else {
                    Vector::new(0.0, car.location.y, 0.0)
                };
                controller.steer = Self::steer(car, steer_target) * controller.throttle;
            } else {
                let throttle_value = -Self::sign(
                    std::f32::consts::FRAC_PI_2 - Self::angle(car.velocity, car.forward()),
                ) * speed.powi(2) / 6000.0
                    + Self::sign(
                        std::f32::consts::FRAC_PI_2
                            - Self::angle(target - car.location, car.forward()),
                    ) * target.distance(car.location);
                controller.throttle = Self::sign(throttle_value);
                if controller.throttle == 0.0 { controller.throttle = 1.0; }
                controller.steer = Self::steer(car, target);

                if !goalie && car.velocity.flatten().magnitude() > 100.0 {
                    let lane_x = Self::sign(car.location.x) * 900.0;
                    let own_lane = Vector::new(lane_x, -goal_sign * 5050.0, 0.0);
                    let opposite_lane = Vector::new(-lane_x, -goal_sign * 5050.0, 0.0);
                    let lane_distance = (target - car.location).flatten().magnitude();
                    let threshold = ((25000.0 / 11.0 * lane_distance).min(1.0).asin() * 2.0)
                        .min(1.0);
                    let angle_current = Self::angle(
                        (own_lane - car.location).flatten(),
                        car.velocity.flatten(),
                    );
                    let angle_opposite = Self::angle(
                        (opposite_lane - car.location).flatten(),
                        car.velocity.flatten(),
                    );
                    if angle_current > threshold && angle_opposite > threshold {
                        controller.throttle = -Self::sign(
                            std::f32::consts::FRAC_PI_2
                                - Self::angle(car.velocity, car.forward()),
                        );
                    }
                }
            }
        } else {
            let speed = car.velocity.flatten().magnitude();
            let previous_front = self.previous_front.unwrap_or(car.forward());
            controller.throttle = 1.0;
            controller.handbrake = speed < 1000.0
                && Self::angle(car.velocity, car.forward()) <= std::f32::consts::FRAC_PI_4
                && Self::angle(car.forward(), target - car.location)
                    > std::f32::consts::FRAC_PI_4
                && Self::angular_speed(car) >= (speed / 500.0).min(2.0)
                && Self::angle(target - car.location, car.forward())
                    < Self::angle(target - car.location, previous_front);
            controller.boost = best_pad.is_some_and(|pad| pad.large)
                && !car.airborne
                && !controller.handbrake;

            let mut steer_target = target;
            if !car.airborne && Self::roll(car).abs() > 0.2 {
                steer_target = car.location.flatten();
            }
            controller.steer = Self::steer(car, steer_target);

            let flip_distance = speed * 1.4 + 500.0 * 1.5 + speed.powi(2) / 7000.0;
            let should_flip = !car.airborne
                && controller.steer.abs() <= 0.01
                && speed >= 1000.0
                && (car.location.flatten().distance(target.flatten()) >= flip_distance
                    || (best_pad.is_some_and(|pad| pad.large)
                        && (car.location.y + car.forward().y * flip_distance).abs() < 5000.0));
            if should_flip {
                self.previous_front = Some(car.forward());
                return DefenseOutput { controller, start_flip: true };
            }
        }

        self.previous_front = Some(car.forward());
        DefenseOutput { controller, start_flip: false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Matrix3;
    use crate::state::Hitbox;
    use rlbot::flat::AirState;

    fn car(team: u32, location: Vector) -> CarState {
        CarState {
            index: 0, team, location, velocity: Vector::ZERO,
            orientation: Matrix3::new(0.0, 0.0, 0.0),
            angular_velocity: Vector::ZERO, demolished: false,
            airborne: false, supersonic: false, jumped: false,
            doublejumped: false, air_state: AirState::OnGround,
            has_jumped: false, has_double_jumped: false, has_dodged: false,
            dodge_timeout: 0.0, dodge_elapsed: 0.0, dodge_dir: Vector::ZERO,
            flip_reset_available: false, boost: 0.0, hitbox: Hitbox::default(),
        }
    }

    #[test]
    fn goalie_holds_python_goal_line_target() {
        let mut defense = GridDefense::default();
        let me = car(0, Vector::new(0.0, -4000.0, 0.0));
        let output = defense.run(
            me, &[me], BallState::default(), &[],
            Vector::new(0.0, 5120.0, 300.0), true, false,
        );
        assert!(output.controller.steer.is_finite());
        assert_eq!(output.controller.boost, false);
    }

    #[test]
    fn non_goalie_uses_python_transition_lane() {
        let mut defense = GridDefense::default();
        let me = car(0, Vector::new(1000.0, -3000.0, 0.0));
        let output = defense.run(
            me, &[me], BallState::default(), &[],
            Vector::new(0.0, 5120.0, 300.0), false, false,
        );
        assert_eq!(output.controller.throttle, 1.0);
        assert!(output.controller.steer.is_finite());
    }
}
