use crate::math::Vector;
use crate::state::{BallState, CarState};

pub const PYTHON_BEST_SHOT_VALUE: f32 = 92.75;
pub const PYTHON_MAX_SPEED: f32 = 2300.0;

/// Output of the distance/time calculation at the start of Python
/// routines.ground_shot.run().
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundDrivePlan {
    pub close_contact: bool,
    pub final_target: Vector,
    pub distance_remaining: f32,
    pub speed_required: f32,
}

/// Port of the target and requested-speed calculation in Python
/// ground_shot.run(). approach_target is the result of the Python contact
/// label helper followed by _shot_approach_target. The caller applies
/// cap_in_field only after this calculation, matching Python's order.
pub fn calculate_ground_drive_plan(
    car: CarState,
    ball: BallState,
    shot_direction: Vector,
    approach_target: Vector,
    intercept_time: f32,
) -> GroundDrivePlan {
    // Source uses Agent.local(world_ball_position), not local_location().
    let local_ball = car.local(ball.location);
    let close_contact = local_ball.y.abs()
        < PYTHON_BEST_SHOT_VALUE + car.hitbox.width * 0.5
        && local_ball.z.abs()
            < PYTHON_BEST_SHOT_VALUE + car.hitbox.height * 0.5;

    let final_target = if close_contact {
        ball.location - shot_direction * PYTHON_BEST_SHOT_VALUE
    } else {
        approach_target.flatten()
    };

    let distance_remaining = final_target.flat_distance(car.location)
        - car.hitbox.length / 3.0;
    let time_remaining = intercept_time.clamp(0.000001, 6.0);
    let mut speed_required = if close_contact {
        PYTHON_MAX_SPEED
    } else {
        distance_remaining / time_remaining
    };

    let ball_speed_toward_shot = ball.velocity.dot(shot_direction);
    if !close_contact
        && speed_required < 1900.0
        && car.boost > 50.0
        && intercept_time > 2.0
        && distance_remaining > 2560.0
        && ball_speed_toward_shot > 1000.0
    {
        speed_required *= 0.75;
    }

    GroundDrivePlan {
        close_contact,
        final_target,
        distance_remaining,
        speed_required,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Matrix3;
    use crate::state::Hitbox;
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
            hitbox: Hitbox::default(),
        }
    }

    fn ball(location: Vector, velocity: Vector) -> BallState {
        BallState { location, velocity }
    }

    #[test]
    fn close_contact_uses_python_max_speed_override() {
        let car = car(Vector::ZERO, 20.0);
        let ball = ball(Vector::new(100.0, 0.0, 92.75), Vector::ZERO);
        let direction = Vector::new(1.0, 0.0, 0.0);
        let plan = calculate_ground_drive_plan(
            car, ball, direction, Vector::new(0.0, 1000.0, 0.0), 0.8,
        );
        assert!(plan.close_contact);
        assert_eq!(plan.speed_required, PYTHON_MAX_SPEED);
        assert_eq!(plan.final_target, ball.location - direction * 92.75);
    }

    #[test]
    fn distant_contact_requests_distance_over_intercept_time() {
        let car = car(Vector::ZERO, 20.0);
        let ball = ball(Vector::new(2000.0, 0.0, 300.0), Vector::ZERO);
        let approach = Vector::new(1700.0, 0.0, 0.0);
        let plan = calculate_ground_drive_plan(
            car, ball, Vector::new(1.0, 0.0, 0.0), approach, 2.0,
        );
        assert!(!plan.close_contact);
        assert!((plan.distance_remaining - (1700.0 - car.hitbox.length / 3.0)).abs() < 1e-4);
        assert!((plan.speed_required - plan.distance_remaining / 2.0).abs() < 1e-4);
    }

    #[test]
    fn python_speed_building_exception_uses_ball_velocity_toward_shot() {
        let car = car(Vector::ZERO, 60.0);
        let ball = ball(
            Vector::new(3000.0, 0.0, 500.0),
            Vector::new(1200.0, 0.0, 0.0),
        );
        let direction = Vector::new(1.0, 0.0, 0.0);
        let approach = Vector::new(2800.0, 0.0, 0.0);
        let plan = calculate_ground_drive_plan(car, ball, direction, approach, 2.5);
        assert!(!plan.close_contact);
        let base = plan.distance_remaining / 2.5;
        assert!((plan.speed_required - base * 0.75).abs() < 1e-4);
    }

    #[test]
    fn slowing_ball_does_not_reduce_ground_speed_request_by_itself() {
        let car = car(Vector::ZERO, 20.0);
        let location = Vector::new(2000.0, 0.0, 400.0);
        let approach = Vector::new(1700.0, 0.0, 0.0);
        let direction = Vector::new(1.0, 0.0, 0.0);
        let slow = calculate_ground_drive_plan(
            car, ball(location, Vector::new(100.0, 0.0, 0.0)),
            direction, approach, 1.5,
        );
        let stopped = calculate_ground_drive_plan(
            car, ball(location, Vector::ZERO),
            direction, approach, 1.5,
        );
        assert_eq!(slow.speed_required, stopped.speed_required);
    }
}
