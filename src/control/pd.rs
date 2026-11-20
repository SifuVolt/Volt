use crate::math::Vector;
use crate::physics::drive::steer_pd;
use crate::state::CarState;

/// Target alignment angles returned by Volt's Python defaultPD.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct OrientationAngles {
    pub pitch: f32,
    pub yaw: f32,
    pub roll: f32,
}

/// Controller outputs produced by Volt's Python defaultPD.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct OrientationControls {
    pub steer: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub roll: f32,
}

/// Pure Rust port of Volt's Python defaultPD.
///
/// local_target is already expressed in car-local forward/right/up space.
/// An explicit up vector is also expected to be local, matching Python.
pub fn default_pd(
    car: &CarState,
    local_target: Vector,
    upside_down: bool,
    up: Option<Vector>,
) -> (OrientationAngles, OrientationControls) {
    let up = up.unwrap_or_else(|| {
        car.local(Vector::new(0.0, 0.0, if upside_down { -1.0 } else { 1.0 }))
    });

    let angles = OrientationAngles {
        pitch: local_target.z.atan2(local_target.x),
        yaw: local_target.y.atan2(local_target.x),
        roll: up.y.atan2(up.z),
    };

    // VirxERLU stores angular_velocity in local coordinates. Python's
    // defaultPD localizes it once more for the steer damping term only;
    // preserve that exact behaviour rather than silently normalizing it.
    let steer_rate = -car.local(car.angular_velocity).z * 0.01;

    let controls = OrientationControls {
        steer: steer_pd(angles.yaw, steer_rate),
        pitch: steer_pd(angles.pitch, car.angular_velocity.y / 4.0),
        yaw: steer_pd(angles.yaw, -car.angular_velocity.z / 4.0),
        roll: steer_pd(angles.roll, car.angular_velocity.x / 4.0),
    };

    (angles, controls)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Matrix3;
    use crate::state::Hitbox;
    use rlbot::flat::AirState;

    fn test_car() -> CarState {
        CarState {
            index: 0,
            team: 0,
            location: Vector::default(),
            velocity: Vector::default(),
            orientation: Matrix3::new(0.0, 0.0, 0.0),
            angular_velocity: Vector::default(),
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
            dodge_dir: Vector::default(),
            flip_reset_available: false,
            boost: 0.0,
            hitbox: Hitbox::default(),
        }
    }

    #[test]
    fn default_pd_calculates_local_target_angles() {
        let car = test_car();
        let (angles, _) = default_pd(
            &car,
            Vector::new(1.0, 1.0, 1.0),
            false,
            None,
        );

        assert!((angles.pitch - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
        assert!((angles.yaw - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
        assert_eq!(angles.roll, 0.0);
    }

    #[test]
    fn default_pd_upside_down_reverses_roll_reference() {
        let car = test_car();
        let (normal, _) = default_pd(&car, Vector::new(1.0, 0.0, 0.0), false, None);
        let (inverted, _) = default_pd(&car, Vector::new(1.0, 0.0, 0.0), true, None);

        assert_eq!(normal.roll, 0.0);
        assert!((inverted.roll.abs() - std::f32::consts::PI).abs() < 1e-6);
    }

    #[test]
    fn default_pd_maps_local_angular_velocity_to_axis_controls() {
        let mut car = test_car();
        car.angular_velocity = Vector::new(4.0, 8.0, 12.0);

        let (angles, controls) = default_pd(
            &car,
            Vector::new(1.0, 0.0, 0.0),
            false,
            None,
        );

        assert_eq!(angles.pitch, 0.0);
        assert_eq!(angles.yaw, 0.0);
        assert_eq!(controls.steer, steer_pd(0.0, -0.12));
        assert_eq!(controls.pitch, steer_pd(0.0, 2.0));
        assert_eq!(controls.yaw, steer_pd(0.0, -3.0));
        assert_eq!(controls.roll, steer_pd(0.0, 1.0));
    }
}
