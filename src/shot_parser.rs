//! Exact VirxERLU-CLib shot-slice parser port.
//!
//! Source: VirxEC/VirxERLU-CLib/virxrlcu.c (master, 2026-11-21).
//! This is the low-level candidate validator used by Volt's Python tools.
//! Keep its 60 Hz reachability simulation, turn-delay model, post correction,
//! field gates and contact offsets separate from higher-level candidate scoring.

use crate::{
    math::Vector,
    shot_physics::{
        aerial_shot_is_viable, double_jump_shot_is_viable, ground_shot_is_viable,
        jump_shot_is_viable, ShotType,
    },
    state::{CarState, Hitbox},
};
use std::f64::consts::PI;

const SIM_DT: f64 = 1.0 / 60.0;
const MAX_SHOT_TIME: f64 = 6.0;
const MAX_SHOT_SLOPE: f64 = 3.0;
const MAX_SHOT_0_SLOPE: f64 = 10.0;
const GROUND_MIN_SLOPE: f64 = 0.5;
const JUMP_MIN_SLOPE: f64 = 0.75;
const DOUBLE_MIN_SLOPE: f64 = 0.75;
const AERIAL_MIN_SLOPE: f64 = 1.5;
const NO_ADJUST: f64 = 0.001;
const MIN_ADJUST: f64 = 0.5;
const MAX_ADJUST: f64 = 2.0;
const BRAKE_FORCE: f64 = 3500.0;
const MAX_SPEED: f64 = 2300.0;
const SAFE_MAX_SPEED: f64 = 2100.0;
const MAX_SPEED_NO_BOOST: f64 = 1410.0;
const BOOST_CONSUMPTION: f64 = 100.0 / 3.0;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ShotTargets {
    pub left: Vector,
    pub right: Vector,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ParsedShot {
    pub found: bool,
    pub fast: bool,
    pub shot_type: Option<ShotType>,
    pub targets: ShotTargets,
    pub ball_offset: Vector,
    pub shot_vector: Vector,
    pub approach_target: Vector,
    pub time: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct V2 { x: f64, y: f64 }

impl V2 {
    fn new(x: f64, y: f64) -> Self { Self { x, y } }
    fn from(v: Vector) -> Self { Self::new(v.x as f64, v.y as f64) }
    fn add(self, b: Self) -> Self { Self::new(self.x + b.x, self.y + b.y) }
    fn sub(self, b: Self) -> Self { Self::new(self.x - b.x, self.y - b.y) }
    fn mul(self, s: f64) -> Self { Self::new(self.x * s, self.y * s) }
    fn dot(self, b: Self) -> f64 { self.x * b.x + self.y * b.y }
    fn mag(self) -> f64 { self.dot(self).sqrt() }
    fn norm(self) -> Self { let m=self.mag(); if m == 0.0 { Self::default() } else { self.mul(1.0/m) } }
    fn cross_z(self, sign: f64) -> Self { Self::new(self.y * sign, -self.x * sign) }
    fn rotate(self, a: f64) -> Self { let (s,c)=a.sin_cos(); Self::new(c*self.x-s*self.y,s*self.x+c*self.y) }
    fn dist(self, b: Self) -> f64 { self.sub(b).mag() }
}

fn sign(value: f64) -> f64 { if value > 0.0 { 1.0 } else if value < 0.0 { -1.0 } else { 0.0 } }
fn dot3(a: Vector, b: Vector) -> f64 { a.x as f64*b.x as f64 + a.y as f64*b.y as f64 + a.z as f64*b.z as f64 }
fn cross3(a: Vector, b: Vector) -> Vector { a.cross(b) }
fn normalize3(v: Vector) -> Vector { v.normalize() }
fn angle3(a: Vector, b: Vector) -> f64 {
    let aa=a.normalize(); let bb=b.normalize();
    (dot3(aa,bb)).clamp(-1.0,1.0).acos()
}
fn angle2(a: Vector, b: Vector) -> f64 {
    let aa=V2::from(a).norm(); let bb=V2::from(b).norm();
    aa.dot(bb).clamp(-1.0,1.0).acos()
}

fn clib_clamp2d(v: Vector, start: Vector, end: Vector) -> Vector {
    let s=v.normalize();
    let down=Vector::new(0.0,0.0,-1.0);
    let right=s.dot(end.cross(down)) < 0.0;
    let left=s.dot(start.cross(down)) > 0.0;
    let inside=if end.dot(start.cross(down)) > 0.0 { right && left } else { right || left };
    if inside { return v; }
    if start.dot(s) < end.dot(s) { end } else { start }
}

// This reproduces C clamp(), including its horizontal rescale when the
// requested z component lies outside the post-ray bounds.
fn clib_clamp(v: Vector, start: Vector, end: Vector) -> Vector {
    let mut s=clib_clamp2d(v,start,end);
    if s.z < start.z {
        s.z=0.0;
        s=s.normalize()*(1.0-start.z);
        s.z=start.z;
    } else if s.z > end.z {
        s.z=0.0;
        s=s.normalize()*(1.0-end.z);
        s.z=end.z;
    }
    s
}

fn correct_for_posts(ball: Vector, targets: ShotTargets) -> (ShotTargets, bool) {
    let radius=95.0_f32;
    let goal_perp=(targets.right-targets.left).cross(Vector::new(0.0,0.0,1.0));
    let left_adjusted=targets.left+cross3((targets.left-ball).normalize(),Vector::new(0.0,0.0,-radius));
    let right_adjusted=targets.right+cross3((targets.right-ball).normalize(),Vector::new(0.0,0.0,radius));
    let left=if (left_adjusted-targets.left).dot(goal_perp)>0.0 { targets.left } else { left_adjusted };
    let right=if (right_adjusted-targets.right).dot(goal_perp)>0.0 { targets.right } else { right_adjusted };
    let line=right-left;
    let width=line.magnitude();
    let goal_line=line.normalize();
    let goal_perp=goal_line.cross(Vector::new(0.0,0.0,1.0));
    let center=left+goal_line*(width*0.5);
    let ball_to_goal=(center-ball).normalize();
    let fits=(width as f64)*(dot3(goal_perp,ball_to_goal).abs()) > (radius*2.0) as f64;
    (ShotTargets{left,right},fits)
}

fn in_quarter_field(x: f64, y: f64, hitbox: Hitbox) -> bool {
    x <= 4096.0-hitbox.length as f64/2.0
        && ((x >= 893.0-hitbox.width as f64 && y <= 5120.0-hitbox.length as f64/2.0) || x <= 893.0)
}
fn in_field_2d(point: Vector, hitbox: Hitbox) -> bool {
    in_quarter_field((point.x as f64).abs(),(point.y as f64).abs(),hitbox)
}
fn in_vertical_field(z: f64, height: f64) -> bool { height/2.0 <= z && z <= 2044.0-height }

fn curvature(v: f64) -> f64 {
    if (0.0..500.0).contains(&v) { 0.0069-5.84e-6*v }
    else if (500.0..1000.0).contains(&v) { 0.00561-3.26e-6*v }
    else if (1000.0..1500.0).contains(&v) { 0.0043-1.95e-6*v }
    else if (1500.0..1750.0).contains(&v) { 0.003025-1.1e-6*v }
    else if (1750.0..2500.0).contains(&v) { 0.0018-4e-7*v }
    else { 0.0 }
}
fn throttle_acceleration(v: f64) -> f64 {
    let x=v.abs();
    if x>=MAX_SPEED_NO_BOOST { return 0.0; }
    if x<1400.0 { (-36.0/35.0)*x+1600.0 } else { -16.0*(x-1400.0)+160.0 }
}
fn turn_radius(v: f64) -> f64 { 1.0/curvature(v) }

fn can_reach_target(max_time: f64, distance: f64, boost_accel: f64, velocity: f64, boost: f64, direction: f64, safe: bool) -> f64 {
    let mut d=distance;
    let mut t_left=max_time;
    let mut b=boost;
    let mut v=velocity;
    let ba_dt=boost_accel*SIM_DT;
    let bc_dt=BOOST_CONSUMPTION*SIM_DT;
    let bk_dt=BRAKE_FORCE*SIM_DT;
    let limited_boost=boost<=100.0;
    let forwards=direction==1.0;
    let mut required=max_time/d*direction;
    let max_allowed=|b:f64,v:f64| if b<bc_dt && v<=MAX_SPEED_NO_BOOST { MAX_SPEED_NO_BOOST } else { MAX_SPEED };
    if required>max_allowed(b,v) || (b<bc_dt && v>MAX_SPEED_NO_BOOST && required>v) { return -1.0; }
    let mut error=(required-v).round() as i32;
    if error.abs()<100 { return max_time; }
    while t_left>=-SIM_DT && !(required>max_allowed(b,v) || (b<bc_dt && v>MAX_SPEED_NO_BOOST && required>v)) {
        let accel_sign=sign(error as f64);
        v += if accel_sign == -sign(v) { bk_dt*accel_sign } else { throttle_acceleration(v)*SIM_DT*accel_sign };
        if v>=0.0 && b>bc_dt {
            v+=ba_dt;
            if limited_boost { b-=bc_dt; }
        }
        d-=v*SIM_DT;
        t_left-=SIM_DT;
        error=(required-v).round() as i32;
        // Preserve the original C conditional-expression precedence:
        // ((d<=0 || |error|<100 || forwards) ? v>required : v<required).
        let reached=if d<=0.0 || error.abs()<100 || forwards { v>required } else { v<required };
        if reached { return t_left; }
        if safe { required=t_left/d*direction; }
    }
    -1.0
}

/// Runtime reachability check used by Python jump_shot.run. This is the
/// CLib generic_is_viable call made before jump_shot_is_viable; it includes
/// turn time, braking, boost, and the actual remaining approach distance.
pub fn runtime_ground_reachability(
    time: f32,
    boost_accel: f32,
    car: &CarState,
    direction: Vector,
    distance: f32,
) -> Option<f32> {
    let remaining = generic_is_viable(
        time as f64,
        boost_accel as f64,
        car,
        direction,
        distance as f64,
        false,
    );
    (remaining >= 0.0).then_some(remaining as f32)
}

fn generic_is_viable(time: f64, boost_accel: f64, car: &CarState, direction: Vector, distance: f64, safe: bool) -> f64 {
    if distance < car.hitbox.length as f64*0.05 { return time; }
    if time<=0.0 || distance/time>MAX_SPEED { return -1.0; }
    let forward_angle=angle2(direction,car.forward().flatten()).abs();
    let backward_angle=PI-forward_angle;
    let true_speed=dot3(car.forward(),car.velocity);
    let car_speed=V2::from(car.velocity).mag();
    let (mut forward_time,mut backward_time)=if car_speed>400.0 {
        let radius=turn_radius(car_speed);
        (time-forward_angle*radius/car_speed,time-backward_angle*radius/car_speed)
    } else {
        (time-1.8*(forward_angle/PI),time-1.8*(backward_angle/PI))
    };
    if forward_angle<1.6 || distance/backward_time>MAX_SPEED_NO_BOOST {
        if forward_time>0.0 {
            forward_time=can_reach_target(forward_time,distance,boost_accel,true_speed,car.boost as f64,1.0,safe);
            if forward_time!=-1.0 { return forward_time; }
        }
    } else if backward_time>0.0 {
        backward_time=can_reach_target(backward_time,distance,boost_accel,true_speed,0.0,-1.0,safe);
        if backward_time!=-1.0 { return backward_time; }
    }
    -1.0
}

fn ray_intersects_line(origin: V2, direction: V2, p1: V2, p2: V2) -> f64 {
    let v1=origin.sub(p1);
    let v2=p2.sub(p1);
    let v3=V2::new(-direction.y,direction.x);
    let denominator=v2.dot(v3);
    // CLib's flat_cross returns a 3-component Vector and the source calls
    // its full 3D magnitude here; retain that exact calculation.
    let cross_x=v2.y-v1.y;
    let cross_y=v1.x-v2.x;
    let cross_z=v2.x*v1.y-v2.y*v1.x;
    let t1=(cross_x*cross_x+cross_y*cross_y+cross_z*cross_z).sqrt()/denominator;
    if t1<0.0 { return -1.0; }
    let t2=v1.dot(v3)/denominator;
    if t1>=0.0 && t2<=1.0 { t1 } else { -1.0 }
}

fn get_travel_distance(ball: Vector, offset_ball: Vector, car: Vector, shot: Vector, hitbox_width: f64) -> f64 {
    let ball=V2::from(ball); let offset=V2::from(offset_ball); let car=V2::from(car); let shot=V2::from(shot);
    let car_to_ball=ball.sub(car);
    let side=sign(shot.cross_z(1.0).dot(car_to_ball));
    let ray_direction=shot.mul(-1.0).rotate(side*-MIN_ADJUST);
    let car_to_offset_perp=offset.sub(car).cross_z(side).norm();
    let adjust_target=offset.sub(shot.mul(2560.0));
    let turn_distance=ray_intersects_line(ball,ray_direction,car,adjust_target);
    let final_target=offset.add(car_to_offset_perp.mul(hitbox_width));
    if turn_distance==-1.0 { return car.dist(final_target); }
    let turn_point=ball.add(ray_direction.mul(turn_distance));
    car.dist(turn_point)+turn_point.dist(final_target)
}

fn get_shot_vector(ball: Vector, car: Vector, targets: ShotTargets) -> Vector {
    let direction=(ball-car).normalize();
    clib_clamp(direction,(targets.left-ball).normalize(),(targets.right-ball).normalize())
}

/// Parse one predicted ball slice using CLib parse_slice_for_shot_with_target.
pub fn parse_slice_for_shot_with_target(
    car: &CarState,
    ball: Vector,
    ball_velocity: Vector,
    time_remaining: f32,
    ball_radius: f32,
    boost_accel: f32,
    gravity: Vector,
    targets: ShotTargets,
    do_ground: bool,
    do_jump: bool,
    do_double_jump: bool,
    do_aerial: bool,
    safe: bool,
    can_large_adjust: bool,
) -> ParsedShot {
    let mut result=ParsedShot::default();
    let car_to_ball=ball-car.location;
    let (corrected,fits)=correct_for_posts(ball,targets);
    if !fits { return result; }
    result.targets=corrected;
    let shot_vector=get_shot_vector(ball,car.location,corrected);
    result.shot_vector=shot_vector;
    let ball_offset=ball-shot_vector*ball_radius;
    let angle_to_target=angle2(car_to_ball,shot_vector).abs();
    if angle_to_target>=MAX_ADJUST { return result; }
    let min_adjust=angle_to_target<MIN_ADJUST;
    if !(can_large_adjust || min_adjust) { return result; }

    let offset_width=car.hitbox.width as f64*0.5;
    let car_flat=car.location.flatten();
    let mut final_target=ball_offset.flatten();
    if angle_to_target>NO_ADJUST {
        let side=sign(dot3(cross3(shot_vector,Vector::new(0.0,0.0,1.0)),car_to_ball));
        let to_offset=final_target-car_flat;
        let perp=Vector::new(to_offset.x,to_offset.y,0.0).cross(Vector::new(0.0,0.0,side as f32)).normalize();
        final_target=if min_adjust {
            final_target+perp*offset_width as f32
        } else {
            final_target-shot_vector.flatten()*(2560.0-ball_radius)
        };
    }
    if !in_field_2d(final_target,car.hitbox) { return result; }

    let mut found_time=time_remaining as f64;
    if do_ground || do_jump || do_double_jump {
        let distance=get_travel_distance(ball.flatten(),ball_offset.flatten(),car_flat,shot_vector.flatten(),offset_width)
            -(car.hitbox.length as f64*0.45);
        let direction=(final_target-car_flat).normalize();
        let viable_time=generic_is_viable(found_time,boost_accel as f64,car,direction,distance,safe);
        if viable_time!=-1.0 {
            found_time=viable_time;
            if do_ground {
                result.found=ground_shot_is_viable(car.hitbox.height,car.airborne,ball_offset.z);
                if result.found { result.shot_type=Some(ShotType::Ground); }
            }
            if do_jump && !result.found {
                result.found=jump_shot_is_viable(found_time as f32,gravity.z,car.velocity.z,car.hitbox.height,car.airborne,ball_offset.z,safe);
                if result.found { result.shot_type=Some(ShotType::Jump); }
            }
            if do_double_jump && !result.found {
                result.found=double_jump_shot_is_viable(found_time as f32,gravity.z,car.velocity.z,car.hitbox.height*0.5,car.airborne,ball_offset.z,safe);
                if result.found { result.shot_type=Some(ShotType::DoubleJump); }
            }
        }
    }

    if do_aerial && !result.found && time_remaining>0.2 && car.boost>0.0
        && angle_to_target<MIN_ADJUST && in_vertical_field(ball_offset.z as f64,car.hitbox.height as f64)
    {
        let aerial_target=Vector::new(final_target.x,final_target.y,ball_offset.z);
        let validation=aerial_shot_is_viable(car,aerial_target,time_remaining,gravity,boost_accel,safe);
        result.found=validation.valid;
        result.fast=validation.fast;
        if result.found { result.shot_type=Some(ShotType::Aerial); }
    }

    result.ball_offset=ball_offset;
    result.approach_target=final_target;
    result.time=found_time as f32;
    let _=ball_velocity; // retained in signature to match the slice parser call site
    result
}

/// Parse one predicted ball slice using CLib parse_slice_for_shot (untargeted).
pub fn parse_slice_for_shot(
    car: &CarState,
    ball: Vector,
    time_remaining: f32,
    ball_radius: f32,
    boost_accel: f32,
    gravity: Vector,
    do_ground: bool,
    do_jump: bool,
    do_double: bool,
    do_aerial: bool,
    safe: bool,
) -> ParsedShot {
    let mut result=ParsedShot::default();
    let direction=(ball-car.location).normalize();
    result.shot_vector=direction;
    let ball_offset=ball-direction*ball_radius;
    if !in_field_2d(ball_offset,car.hitbox) { return result; }
    let mut found_time=time_remaining as f64;
    if do_ground || do_jump || do_double {
        let distance=V2::from(ball_offset).dist(V2::from(car.location))-car.hitbox.length as f64*0.45;
        let viable=generic_is_viable(found_time,boost_accel as f64,car,direction.flatten().normalize(),distance,safe);
        if viable!=-1.0 {
            found_time=viable;
            if do_ground {
                result.found=ground_shot_is_viable(car.hitbox.height,car.airborne,ball_offset.z);
                if result.found { result.shot_type=Some(ShotType::Ground); }
            }
            if do_jump && !result.found {
                result.found=jump_shot_is_viable(found_time as f32,gravity.z,car.velocity.z,car.hitbox.height,car.airborne,ball_offset.z,safe);
                if result.found { result.shot_type=Some(ShotType::Jump); }
            }
            if do_double && !result.found {
                result.found=double_jump_shot_is_viable(found_time as f32,gravity.z,car.velocity.z,car.hitbox.height*0.5,car.airborne,ball_offset.z,safe);
                if result.found { result.shot_type=Some(ShotType::DoubleJump); }
            }
        }
    }
    if do_aerial && !result.found && in_vertical_field((ball_offset.z as f64).abs(),car.hitbox.height as f64) {
        let validation=aerial_shot_is_viable(car,ball_offset,time_remaining,gravity,boost_accel,safe);
        result.found=validation.valid;
        result.fast=validation.fast;
        if result.found { result.shot_type=Some(ShotType::Aerial); }
    }
    result.ball_offset=ball_offset;
    result.approach_target=ball_offset;
    result.time=found_time as f32;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Matrix3;
    use rlbot::flat::AirState;

    fn car(boost:f32, location:Vector, velocity:Vector, orientation:Matrix3) -> CarState {
        CarState {
            index:0, team:0, location, velocity, orientation,
            angular_velocity:Vector::ZERO, demolished:false, airborne:false,
            supersonic:false, jumped:false, doublejumped:false, air_state:AirState::OnGround,
            has_jumped:false, has_double_jumped:false, has_dodged:false,
            dodge_timeout:0.0,dodge_elapsed:0.0,dodge_dir:Vector::ZERO,
            flip_reset_available:false,boost,
            hitbox:Hitbox{length:118.0,width:84.0,height:36.0,offset:Vector::ZERO},
        }
    }

    #[test]
    fn clib_travel_distance_has_direct_and_turn_path_cases() {
        let direct=get_travel_distance(Vector::new(0.0,1000.0,0.0),Vector::new(0.0,900.0,0.0),Vector::new(0.0,0.0,0.0),Vector::new(0.0,1.0,0.0),42.0);
        assert!(direct.is_finite() && direct>0.0);
        let turn=get_travel_distance(Vector::new(1000.0,1000.0,0.0),Vector::new(900.0,900.0,0.0),Vector::new(-1000.0,0.0,0.0),Vector::new(0.0,1.0,0.0),42.0);
        assert!(turn.is_finite() && turn>0.0);
    }

    #[test]
    fn generic_reachability_rejects_zero_time_and_over_speed() {
        let c=car(50.0,Vector::ZERO,Vector::ZERO,Matrix3::new(0.0,0.0,0.0));
        assert_eq!(generic_is_viable(0.0,991.6667,&c,Vector::new(0.0,1.0,0.0),1000.0,true),-1.0);
        assert_eq!(generic_is_viable(0.1,991.6667,&c,Vector::new(0.0,1.0,0.0),5000.0,true),-1.0);
    }

    #[test]
    fn parser_returns_ground_candidate_for_valid_low_ball() {
        let c=car(50.0,Vector::new(0.0,-1000.0,17.0),Vector::new(0.0,1200.0,0.0),Matrix3::new(0.0,1.5707963,0.0));
        let targets=ShotTargets{left:Vector::new(-700.0,5120.0,350.0),right:Vector::new(700.0,5120.0,350.0)};
        let result=parse_slice_for_shot_with_target(&c,Vector::new(0.0,0.0,100.0),Vector::ZERO,1.0,92.75,991.6667,Vector::new(0.0,0.0,-650.0),targets,true,true,true,false,true,true);
        assert!(result.time.is_finite());
        assert!(result.ball_offset.x.is_finite() && result.ball_offset.y.is_finite() && result.ball_offset.z.is_finite());
    }

    #[test]
    fn untargeted_parser_preserves_no_target_aerial_validation() {
        let c=car(100.0,Vector::ZERO,Vector::ZERO,Matrix3::new(0.0,0.0,0.0));
        let result=parse_slice_for_shot(&c,Vector::new(0.0,0.0,900.0),1.8,92.75,991.6667,Vector::new(0.0,0.0,-650.0),false,false,false,true,true);
        assert!(result.found);
        assert_eq!(result.shot_type,Some(ShotType::Aerial));
    }
}
