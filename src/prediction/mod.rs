use crate::math::Vector;
use crate::physics::PhysicsConfig;

/// A single point on a free-flight ball trajectory.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BallSlice {
    /// Time relative to the current packet, in seconds.
    pub time: f32,
    pub location: Vector,
    pub velocity: Vector,
}

/// Ballistic prediction while the ball is airborne and collision-free.
///
/// This is intentionally only the free-flight segment. It does not invent
/// wall/floor bounce coefficients or goal-post collision behaviour. A later
/// collision solver can split this trajectory into physically valid segments.
#[derive(Debug, Clone, PartialEq)]
pub struct BallTrajectory {
    pub slices: Vec<BallSlice>,
}

/// One absolute-time sample copied from RLBot's ball prediction.
/// This small value type also makes timestamp conversion independently testable.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AbsoluteBallSlice {
    pub game_seconds: f32,
    pub location: Vector,
    pub velocity: Vector,
}

impl BallTrajectory {
    /// Convert absolute-time RLBot prediction samples into packet-relative time.
    ///
    /// Past samples are discarded. The remaining samples retain RLBot's
    /// collision-aware trajectory and original 120 Hz sampling; no physics
    /// are re-simulated here.
    pub fn from_absolute_slices(
        samples: impl IntoIterator<Item = AbsoluteBallSlice>,
        current_game_seconds: f32,
    ) -> Self {
        if !current_game_seconds.is_finite() {
            return Self { slices: Vec::new() };
        }

        let slices = samples
            .into_iter()
            .filter_map(|sample| {
                let time = sample.game_seconds - current_game_seconds;
                (time.is_finite() && time >= 0.0).then_some(BallSlice {
                    time,
                    location: sample.location,
                    velocity: sample.velocity,
                })
            })
            .collect();

        Self { slices }
    }

    /// Copy RLBot's authoritative prediction into Volt's packet-relative
    /// trajectory representation.
    pub fn from_rlbot_prediction(
        prediction: &rlbot::flat::BallPrediction,
        current_game_seconds: f32,
    ) -> Self {
        Self::from_absolute_slices(
            prediction.slices.iter().map(|slice| AbsoluteBallSlice {
                game_seconds: slice.game_seconds,
                location: Vector::new(
                    slice.physics.location.x,
                    slice.physics.location.y,
                    slice.physics.location.z,
                ),
                velocity: Vector::new(
                    slice.physics.velocity.x,
                    slice.physics.velocity.y,
                    slice.physics.velocity.z,
                ),
            }),
            current_game_seconds,
        )
    }

    /// Generate samples from t=0 through horizon, including the endpoint.
    /// A positive finite sample rate is required (normally 60 Hz).
    pub fn free_flight(
        location: Vector,
        velocity: Vector,
        config: PhysicsConfig,
        horizon: f32,
        sample_rate: f32,
    ) -> Self {
        if !horizon.is_finite() || horizon < 0.0 || !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Self { slices: Vec::new() };
        }

        let interval = 1.0 / sample_rate;
        let count = (horizon / interval).ceil() as usize;
        let mut slices = Vec::with_capacity(count.saturating_add(1));

        for index in 0..=count {
            let time = (index as f32 * interval).min(horizon);
            slices.push(BallSlice {
                time,
                location: location + velocity * time + config.gravity * (0.5 * time * time),
                velocity: velocity + config.gravity * time,
            });
        }

        Self { slices }
    }

    /// Interpolate a state between adjacent samples; returns None outside the horizon.
    pub fn at_time(&self, time: f32) -> Option<BallSlice> {
        if self.slices.is_empty() || !time.is_finite() || time < 0.0 {
            return None;
        }

        let first = *self.slices.first()?;
        let last = *self.slices.last()?;
        if time < first.time || time > last.time {
            return None;
        }

        let upper = self.slices.partition_point(|slice| slice.time < time);
        if upper == 0 {
            return Some(first);
        }
        if upper >= self.slices.len() {
            return Some(last);
        }

        let a = self.slices[upper - 1];
        let b = self.slices[upper];
        let span = b.time - a.time;
        if span <= f32::EPSILON {
            return Some(a);
        }

        let alpha = ((time - a.time) / span).clamp(0.0, 1.0);
        Some(BallSlice {
            time,
            location: a.location + (b.location - a.location) * alpha,
            velocity: a.velocity + (b.velocity - a.velocity) * alpha,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absolute_prediction_samples_become_packet_relative() {
        let trajectory = BallTrajectory::from_absolute_slices(
            [
                AbsoluteBallSlice {
                    game_seconds: 9.9,
                    location: Vector::new(-1.0, 0.0, 0.0),
                    velocity: Vector::ZERO,
                },
                AbsoluteBallSlice {
                    game_seconds: 10.0,
                    location: Vector::new(10.0, 20.0, 30.0),
                    velocity: Vector::new(100.0, 200.0, 300.0),
                },
                AbsoluteBallSlice {
                    game_seconds: 10.25,
                    location: Vector::new(35.0, 50.0, 70.0),
                    velocity: Vector::new(100.0, 120.0, 160.0),
                },
            ],
            10.0,
        );

        assert_eq!(trajectory.slices.len(), 2);
        assert!((trajectory.slices[0].time).abs() < 0.0001);
        assert!((trajectory.slices[1].time - 0.25).abs() < 0.0001);
        assert_eq!(trajectory.slices[0].location, Vector::new(10.0, 20.0, 30.0));
        assert_eq!(trajectory.slices[1].velocity, Vector::new(100.0, 120.0, 160.0));
    }

    #[test]
    fn absolute_prediction_rejects_invalid_clock_and_pre_first_sample_queries() {
        let invalid = BallTrajectory::from_absolute_slices(
            [AbsoluteBallSlice {
                game_seconds: 1.0,
                location: Vector::ZERO,
                velocity: Vector::ZERO,
            }],
            f32::NAN,
        );
        assert!(invalid.slices.is_empty());

        let future_start = BallTrajectory::from_absolute_slices(
            [AbsoluteBallSlice {
                game_seconds: 2.0,
                location: Vector::new(5.0, 0.0, 0.0),
                velocity: Vector::ZERO,
            }],
            1.5,
        );
        assert!(future_start.at_time(0.0).is_none());
        assert!(future_start.at_time(0.5).is_some());
    }

    #[test]
    fn free_flight_matches_ballistic_equations() {
        let config = PhysicsConfig::from_world_gravity(-650.0);
        let trajectory = BallTrajectory::free_flight(
            Vector::new(0.0, 0.0, 500.0),
            Vector::new(100.0, 0.0, 200.0),
            config,
            1.0,
            60.0,
        );
        let end = trajectory.at_time(1.0).unwrap();
        assert!((end.location.x - 100.0).abs() < 0.001);
        assert!((end.location.z - 375.0).abs() < 0.001);
        assert!((end.velocity.z + 450.0).abs() < 0.001);
    }

    #[test]
    fn trajectory_includes_zero_and_requested_endpoint() {
        let config = PhysicsConfig::from_world_gravity(-650.0);
        let trajectory = BallTrajectory::free_flight(Vector::ZERO, Vector::ZERO, config, 0.25, 60.0);
        assert_eq!(trajectory.slices.first().unwrap().time, 0.0);
        assert!((trajectory.slices.last().unwrap().time - 0.25).abs() < 0.0001);
    }

    #[test]
    fn interpolation_returns_ball_state_between_samples() {
        let config = PhysicsConfig::from_world_gravity(0.0);
        let trajectory = BallTrajectory::free_flight(
            Vector::ZERO, Vector::new(120.0, 0.0, 0.0), config, 1.0, 60.0);
        let middle = trajectory.at_time(0.025).unwrap();
        assert!((middle.location.x - 3.0).abs() < 0.001);
        assert!((middle.velocity.x - 120.0).abs() < 0.001);
    }

    #[test]
    fn invalid_sampling_inputs_return_empty_trajectory() {
        let config = PhysicsConfig::from_world_gravity(-650.0);
        assert!(BallTrajectory::free_flight(Vector::ZERO, Vector::ZERO, config, 1.0, 0.0).slices.is_empty());
    }
}