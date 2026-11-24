//! Curvature-constrained ground approach planning using VirxEC's Dubins paths.
//!
//! A Dubins path connects the car's current ground pose to a target pose while
//! respecting a minimum turning radius. The returned waypoint is a short
//! distance ahead on that path and is intended to be tracked by Volt's existing
//! default_drive controller. This module does not replace throttle/PD control.

use dubins_paths::f32::{DubinsPath, PosRot};

use crate::{
    math::Vector,
    physics::drive::turn_radius,
    state::CarState,
};

/// A ground route from the current car pose to a target pose.
#[derive(Clone, Copy, Debug)]
pub struct Approach {
    path: DubinsPath,
    /// First look-ahead point on the planned route, in world coordinates.
    pub waypoint: Vector,
    /// Total distance along the curvature-constrained route.
    pub path_length: f32,
    /// Minimum turning radius used to construct this route.
    pub turn_radius: f32,
    /// Distance sampled ahead of the car to obtain the steering waypoint.
    pub lookahead: f32,
}

impl Approach {
    /// Calculate a Dubins approach for a grounded car.
    ///
    /// final_direction is the direction the car should face when it reaches
    /// target (for a shot, this is the intended outgoing shot direction).
    /// Returns None when the car is airborne, reversing, or the path cannot
    /// be calculated; callers should then retain their existing direct-drive
    /// behavior.
    pub fn new(
        car: &CarState,
        target: Vector,
        final_direction: Vector,
    ) -> Option<Self> {
        if car.airborne || car.local_velocity().x < -100.0 {
            return None;
        }

        Self::from_pose(
            car.location,
            car.forward().y.atan2(car.forward().x),
            target,
            final_direction,
            car.velocity.flatten().magnitude(),
        )
    }

    /// Pose-based constructor, also exposed for deterministic unit tests.
    pub fn from_pose(
        start: Vector,
        start_heading: f32,
        target: Vector,
        final_direction: Vector,
        speed: f32,
    ) -> Option<Self> {
        let target = target.flatten();
        let direction = final_direction.flatten().normalize();
        if !start.x.is_finite()
            || !start.y.is_finite()
            || !target.x.is_finite()
            || !target.y.is_finite()
            || direction == Vector::ZERO
        {
            return None;
        }

        // Volt's curvature table is speed-dependent. Clamp the low-speed
        // input to the first useful driving band; the Dubins solver requires
        // a strictly positive radius.
        let radius = turn_radius(speed.abs().max(500.0)).clamp(120.0, 900.0);
        let end_heading = direction.y.atan2(direction.x);
        let path = DubinsPath::shortest_from(
            PosRot::from_floats(start.x, start.y, start_heading),
            PosRot::from_floats(target.x, target.y, end_heading),
            radius,
        )
        .ok()?;

        let path_length = path.length();
        if !path_length.is_finite() {
            return None;
        }

        let lookahead = (250.0 + speed.abs() * 0.16).clamp(300.0, 600.0);
        let sample = path.sample(lookahead.min(path_length));
        Some(Self {
            path,
            waypoint: Vector::new(sample.x(), sample.y(), 0.0),
            path_length,
            turn_radius: radius,
            lookahead,
        })
    }

    /// Sample any distance along this route, clamped to its endpoint.
    pub fn sample(&self, distance: f32) -> Vector {
        let sample = self.path.sample(distance.clamp(0.0, self.path_length));
        Vector::new(sample.x(), sample.y(), 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dubins_approach_respects_start_and_end_pose() {
        let start = Vector::new(-1500.0, -1000.0, 0.0);
        let target = Vector::new(1000.0, 2000.0, 0.0);
        let direction = Vector::new(0.0, 1.0, 0.0);
        let approach = Approach::from_pose(start, 0.0, target, direction, 1400.0)
            .expect("a valid ground path should exist");

        assert!(approach.path_length >= start.flat_distance(target));
        assert!(approach.turn_radius > 0.0);
        assert!(approach.waypoint.x.is_finite());
        assert!(approach.waypoint.y.is_finite());
        assert!(approach.sample(approach.path_length).flat_distance(target) < 0.1);
    }

    #[test]
    fn dubins_approach_lookahead_stays_inside_path() {
        let approach = Approach::from_pose(
            Vector::ZERO,
            0.0,
            Vector::new(200.0, 0.0, 0.0),
            Vector::new(1.0, 0.0, 0.0),
            1000.0,
        )
        .expect("a valid short path should exist");

        assert!(approach.lookahead > 0.0);
        assert!(approach.waypoint.flat_distance(Vector::ZERO) <= approach.path_length + 0.1);
        assert!(approach.sample(approach.path_length + 1000.0)
            .flat_distance(Vector::new(200.0, 0.0, 0.0)) < 0.1);
    }
}
