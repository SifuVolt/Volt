use crate::{
    control::default_pd,
    math::Vector,
    state::{BallState, CarState},
};
use rlbot::flat::ControllerState;

/// Persistent port of Python Volt's speedflip_kickoff routine.
///
/// The state deliberately survives packet ticks. The speed_flip mechanic is
/// run by BotBrain between phase 0 and phase 1, just as the Python routine
/// pushes speed_flip onto its routine stack.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeedFlipKickoff {
    phase: u8,
    phase3_start_time: Option<f32>,
    phase3_start_ball: Vector,
    phase3_start_velocity: Vector,
    phase3_flip_start: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KickoffOutput {
    pub controller: ControllerState,
    pub start_speedflip: Option<f32>,
    pub finished: bool,
}

impl Default for SpeedFlipKickoff {
    fn default() -> Self {
        Self {
            phase: 0,
            phase3_start_time: None,
            phase3_start_ball: Vector::ZERO,
            phase3_start_velocity: Vector::ZERO,
            phase3_flip_start: None,
        }
    }
}

impl SpeedFlipKickoff {
    pub fn tick(&mut self, car: CarState, ball: BallState, game_time: f32) -> KickoffOutput {
        let mut controller = ControllerState::default();

        match self.phase {
            // Python phase 0: accelerate toward the ball until >800 uu/s,
            // provided the ball is still more than 1800 uu away.
            0 => {
                let local_ball = car.local_location(ball.location);
                let (_, pd) = default_pd(&car, local_ball, false, None);
                controller.steer = pd.steer;
                controller.pitch = pd.pitch;
                controller.yaw = pd.yaw;
                controller.roll = pd.roll;
                controller.throttle = 1.0;
                controller.boost = true;
                controller.jump = false;

                let speed = car.velocity.magnitude();
                let distance = car.location.flat_distance(ball.location);
                if speed > 800.0 && distance > 1800.0 {
                    let right_handed =
                        car.location.x * -team_side(car.team) > 0.0;
                    self.phase = 1;
                    return KickoffOutput {
                        controller,
                        start_speedflip: Some(if right_handed { 1.0 } else { -1.0 }),
                        finished: false,
                    };
                }
            }
            // Python phase 1 waits for the pushed speed_flip routine to pop,
            // then waits until wheel contact before resuming the kickoff.
            1 => {
                if !car.airborne {
                    self.phase = 2;
                }
            }
            // Python phase 2: full-speed drive to the ball; retain the source
            // 2290 speed threshold and 1000-unit transition.
            2 => {
                let local_ball = car.local_location(ball.location);
                let (_, pd) = default_pd(&car, local_ball, false, None);
                controller.steer = pd.steer;
                controller.pitch = pd.pitch;
                controller.yaw = pd.yaw;
                controller.roll = pd.roll;
                controller.throttle = 1.0;
                controller.boost = car.velocity.magnitude() < 2290.0;

                if car.location.flat_distance(ball.location) < 1000.0 {
                    self.phase = 3;
                    self.phase3_start_time = Some(game_time);
                    self.phase3_start_ball = ball.location;
                    self.phase3_start_velocity = ball.velocity;
                    self.phase3_flip_start = Some(game_time);
                }
            }
            // Python phase 3: drive through the ball and use a one-frame
            // jump, three-frame release, one-frame forward dodge sequence.
            3 => {
                let local_ball = car.local_location(ball.location);
                let (_, pd) = default_pd(&car, local_ball, false, None);
                controller.steer = pd.steer;
                controller.pitch = pd.pitch;
                controller.yaw = pd.yaw;
                controller.roll = pd.roll;
                controller.throttle = 1.0;
                controller.boost = car.boost > 0.0
                    && local_ball.y.atan2(local_ball.x).abs() < 0.35;

                let flip_elapsed = game_time
                    - self.phase3_flip_start.unwrap_or(game_time);
                if flip_elapsed < 1.0 / 60.0 {
                    controller.jump = true;
                    controller.pitch = 0.0;
                } else if flip_elapsed < 4.0 / 60.0 {
                    controller.jump = false;
                    controller.pitch = 0.0;
                } else if flip_elapsed < 5.0 / 60.0 {
                    controller.jump = true;
                    controller.pitch = -1.0;
                } else {
                    controller.jump = false;
                    controller.pitch = 0.0;
                }

                let displacement = ball.location.distance(self.phase3_start_ball);
                let velocity_change = (ball.velocity - self.phase3_start_velocity).magnitude();
                let likely_contact =
                    (displacement > 100.0 && velocity_change > 250.0)
                    || velocity_change > 500.0;
                let timed_out = game_time
                    - self.phase3_start_time.unwrap_or(game_time)
                    > 2.0;
                if likely_contact || timed_out {
                    return KickoffOutput {
                        controller,
                        start_speedflip: None,
                        finished: true,
                    };
                }
            }
            _ => {
                return KickoffOutput {
                    controller,
                    start_speedflip: None,
                    finished: true,
                };
            }
        }

        KickoffOutput {
            controller,
            start_speedflip: None,
            finished: false,
        }
    }

    #[cfg(test)]
    pub fn phase(&self) -> u8 {
        self.phase
    }
}

fn team_side(team: u32) -> f32 {
    if team == 0 { -1.0 } else { 1.0 }
}
