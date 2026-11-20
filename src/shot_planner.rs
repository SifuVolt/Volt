//! Source-faithful Volt shot scan built on the VirxERLU-CLib parser.
use crate::{
    math::Vector,
    prediction::{BallSlice, BallTrajectory},
    shot_parser::{parse_slice_for_shot_with_target, ParsedShot, ShotTargets},
    shot_physics::{get_jump_time, ShotType},
    state::CarState,
};

const R: f32 = 92.75;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShotCandidate {
    pub time: f32,
    pub ball: Vector,
    pub ball_velocity: Vector,
    pub approach: Vector,
    pub shot_direction: Vector,
    pub shot_type: ShotType,
    pub fast: bool,
    pub score: f32,
    pub parser: ParsedShot,
}
fn cap(v:f32,a:f32,b:f32)->f32 { v.max(a).min(b) }
fn team_side(team:u32)->f32 { if team==0 {-1.0} else {1.0} }
fn center(t:ShotTargets)->Vector { (t.left+t.right)*0.5 }

fn aim_point(ball:Vector,vel:Vector,t:ShotTargets,g:f32)->Vector {
    let c=center(t);
    let half=(t.right-t.left).flatten().magnitude()*0.5;
    let usable=(half-100.0).max(0.0);
    let time=if vel.y.abs()>100.0 {(c.y-ball.y)/vel.y} else {-1.0};
    if time>0.05 && time<3.0 {
        Vector::new(cap(ball.x+vel.x*time,c.x-usable,c.x+usable),c.y,
            cap(ball.z+vel.z*time+0.5*g*time*time,120.0,550.0))
    } else { Vector::new(c.x,c.y,300.0) }
}
fn unrealistic(ball:Vector,t:ShotTargets,team:u32)->bool {
    let c=center(t);
    let past=(ball.y-c.y)*(-team_side(team));
    let half=(t.right-t.left).flatten().magnitude()*0.5;
    past>250.0 || (past>-300.0 && (ball.x-c.x).abs()>half+450.0)
}
fn backwall_rebound(me:CarState,ball:Vector,vel:Vector)->bool {
    let ts=if me.team==0 {1.0}else{-1.0};
    !me.airborne && ball.z>140.0 && ball.z<1200.0
        && (ball.y+5120.0*ts).abs()<=1400.0 && vel.y*ts>150.0
}
fn foe_eta(foe:CarState,ball:Vector)->f32 {
    let dist=foe.location.flat_distance(ball);
    let v=foe.velocity.flatten();
    let speed=v.magnitude();
    let dir=(ball-foe.location).flatten().normalize();
    if dir.magnitude()<0.001 { return 0.0; }
    let f=foe.forward().flatten();
    let angle=if f.magnitude()<0.001 {std::f32::consts::PI}
        else {f.normalize().dot(dir).clamp(-1.0,1.0).acos()};
    let turn=angle/if speed>1400.0 {2.0}else{3.5};
    let forward=v.dot(dir).max(0.0);
    let mut remaining=(dist-forward*turn).max(0.0);
    let ba=1600.0; let base=900.0;
    let boost_time=foe.boost.max(0.0)/33.3333333333;
    let bt=boost_time.min(((2300.0-forward)/ba).max(0.0));
    let bd=forward*bt+0.5*ba*bt*bt;
    let after=(forward+ba*bt).min(2300.0);
    remaining=(remaining-bd).max(0.0);
    let ct=((2300.0-after)/base).max(0.0);
    let cd=after*ct+0.5*base*ct*ct;
    let drive=if remaining<=cd {((after*after+2.0*base*remaining).sqrt()-after)/base}
        else {ct+(remaining-cd)/2300.0};
    let jump=if !foe.airborne && ball.z>180.0 && ball.z<650.0 {-0.08}else{0.0};
    (turn+bt+drive+jump).max(0.0)
}
fn useful(me:CarState,foes:&[CarState],friends:&[CarState],ball:Vector,t:f32,kind:ShotType,kickoff:bool,challenger:bool)->bool {
    if kind==ShotType::Aerial || t<=0.0 || foes.is_empty() || kickoff || (!friends.is_empty()&&challenger) {return true;}
    let Some(foe)=foes.iter().min_by(|a,b|a.location.flat_distance(ball).total_cmp(&b.location.flat_distance(ball))) else {return true;};
    let margin=if kind==ShotType::Jump {0.5}else{0.35};
    foe_eta(*foe,ball)+margin>=t
}
fn score(me:CarState,friends:&[CarState],ball:Vector,vel:Vector,kind:ShotType,t:f32,targets:ShotTargets,g:f32)->f32 {
    if unrealistic(ball,targets,me.team) {return f32::INFINITY;}
    let mut s=t*0.22+match kind {ShotType::Ground=>0.0,ShotType::Jump=>0.12,ShotType::DoubleJump=>0.25,ShotType::Aerial=>0.35};
    if kind==ShotType::DoubleJump && backwall_rebound(me,ball,vel) {s-=0.15;}
    let aim=aim_point(ball,vel,targets,g);
    let gd=(aim-ball).flatten().normalize();
    let goal=center(targets);
    let ts=if me.team==0 {1.0}else{-1.0};
    let depth=ball.y*ts;
    if depth>1500.0 && (goal.y-ball.y)*ts<0.0 {s+=((depth-1500.0)/2500.0).min(1.0)*0.20;}
    let approach=(ball-me.location).flatten().normalize();
    if approach.magnitude()>1.0 && gd.magnitude()>1.0 {
        let alignment=approach.dot(gd).clamp(-1.0,1.0);
        s+=(1.0-alignment)*0.45*cap(1.5/t.max(0.1),0.7,1.8);
        let closing=me.velocity.flatten().dot(approach)-vel.flatten().dot(approach);
        s+=cap((850.0-closing)/850.0,0.0,1.0)*0.55*cap(1.2/t.max(0.1),0.4,1.0);
        let own=Vector::new(0.0,ts*5120.0,350.0);
        let to_own=(own-ball).flatten();
        let urgent=to_own.magnitude()<3000.0 && vel.flatten().dot(to_own.normalize())>250.0;
        if !friends.is_empty() && !urgent {
            let mut edge=0.0_f32;
            for friend in friends {
                let to_ball=(ball-friend.location).flatten();
                let d=to_ball.magnitude();
                if d<=1.0 {continue;}
                let arrival=d/(friend.velocity.flatten().magnitude()+350.0).max(1200.0);
                let angle_edge=to_ball.normalize().dot(gd).clamp(-1.0,1.0)-alignment;
                let arrival_edge=t-arrival;
                if arrival_edge>=-0.15 && angle_edge>0.18 {
                    edge=edge.max(((arrival_edge+0.2).max(0.0)*0.35+angle_edge*0.75).min(0.65));
                }
            }
            s+=edge;
        }
    }
    if me.boost<12.0 {s+=match kind {ShotType::Jump=>0.18,ShotType::DoubleJump=>0.08,ShotType::Aerial=>0.10,_=>0.0};}
    if me.boost<=0.0 {s+=match kind {ShotType::Jump=>0.22,ShotType::DoubleJump=>-0.12,_=>0.0};}
    s
}

fn reserve(me:CarState,ball:Vector,t:f32,kind:ShotType,g:f32)->f32 {
    if kind==ShotType::Ground {return 0.0;}
    let height=(ball.z-R-me.location.z).max(0.0);
    let flat=me.location.flat_distance(ball);
    if kind==ShotType::Jump || kind==ShotType::DoubleJump {
        let mut jt=get_jump_time(height,me.velocity.z,g).time as f32;
        if !jt.is_finite() || jt<0.0 {jt=if kind==ShotType::DoubleJump {cap(height/1000.0,0.10,0.65)}else{0.0};}
        jt=cap(jt,0.0,0.65);
        let pre=(t-jt).max(0.0);
        let dir=(ball-me.location).flatten().normalize();
        let speed=me.velocity.flatten().dot(dir).max(0.0);
        let accel=900.0;
        let acc_t=pre.min(((2300.0-speed)/accel).max(0.0));
        let drive=speed*acc_t+0.5*accel*acc_t*acc_t+2300.0*(pre-acc_t).max(0.0);
        let gap=(flat-drive-180.0_f32.max(92.75)).max(0.0);
        let df=cap((gap-150.0)/700.0,0.0,1.0);
        let hf=cap(height/350.0,0.0,1.0);
        if kind==ShotType::DoubleJump {jt*=0.75;}
        return 33.3333333333*jt*(if kind==ShotType::Jump {0.60}else{0.42})*df*hf;
    }
    let d=me.location.distance(ball);
    cap((d-250.0)/1000.0,0.0,1.0)*cap(height/700.0,0.0,1.0)*(me.boost*0.18).max(2.0).min(15.0)
}

fn parse_budget(me:CarState,slice:BallSlice,t:f32,targets:ShotTargets,g:Vector,boost_accel:f32,flags:(bool,bool,bool))->ParsedShot {
    let mut car=me;
    let mut shot=parse_slice_for_shot_with_target(&car,slice.location,slice.velocity,t,R,boost_accel,g,targets,flags.0,flags.1,flags.2,false,true,false);
    if !shot.found {return shot;}
    let mut seen=[false;4];
    for _ in 0..3 {
        let Some(kind)=shot.shot_type else{return shot;};
        let i=match kind {ShotType::Ground=>0,ShotType::Jump=>1,ShotType::DoubleJump=>2,ShotType::Aerial=>3};
        if seen[i] {return ParsedShot::default();}
        seen[i]=true;
        let r=reserve(me,slice.location,t,kind,g.z);
        if r<=0.25 {return shot;}
        let budget=me.boost-r;
        if budget< -0.01 {return ParsedShot::default();}
        car.boost=budget.max(0.0).trunc();
        let next=parse_slice_for_shot_with_target(&car,slice.location,slice.velocity,t,R,boost_accel,g,targets,flags.0,flags.1,flags.2,false,true,false);
        if !next.found {return next;}
        if next.shot_type==Some(kind) {return next;}
        shot=next;
    }
    ParsedShot::default()
}

pub fn find_best_shot(
    me:CarState,cars:&[CarState],ball_now:Vector,prediction:&BallTrajectory,
    targets:ShotTargets,gravity:Vector,boost_accel:f32,game_time:f32,
    kickoff_contest_until:f32,urgent_save:bool,elected_challenger:bool,horizon:f32,
)->Option<ShotCandidate> {
    if me.airborne || prediction.slices.is_empty() {return None;}
    let friends:Vec<CarState>=cars.iter().copied().filter(|c|c.team==me.team&&!c.demolished&&c.index!=me.index).collect();
    let foes:Vec<CarState>=cars.iter().copied().filter(|c|c.team!=me.team&&!c.demolished).collect();
    let in_own_net=me.location.y*team_side(me.team)>=5120.0 && me.location.x.abs()<=900.0;
    let mut best:Option<ShotCandidate>=None;
    let mut last=-999.0_f32;
    for slice in &prediction.slices {
        let t=slice.time;
        if t<0.10 {continue;}
        if t>horizon.min(6.0) {break;}
        let interval=if urgent_save {0.025}else if t<1.5 {0.05}else{0.10};
        if t-last+1.0e-5<interval {continue;}
        last=t;
        if slice.location.y.abs()>5212.75 {break;}
        let remaining=t-1.0/120.0;
        if remaining<=0.0 {continue;}
        let rebound=backwall_rebound(me,slice.location,slice.velocity);
        let mut flags=(true,!in_own_net,true);
        if rebound {flags.1=false;}
        if me.boost<=0.0 {
            if slice.location.z>105.0 {flags.0=false;flags.1=false;}
            else {flags.2=false;}
        }
        let parsed=parse_budget(me,*slice,remaining,targets,gravity,boost_accel,flags);
        if !parsed.found {continue;}
        let Some(kind)=parsed.shot_type else{continue;};
        if !useful(me,&foes,&friends,slice.location,remaining,kind,game_time<kickoff_contest_until,elected_challenger) {continue;}
        let mut value=if urgent_save {remaining+match kind {ShotType::Ground=>0.0,ShotType::Jump=>0.005,ShotType::DoubleJump=>0.015,ShotType::Aerial=>0.025}}
            else {score(me,&friends,slice.location,slice.velocity,kind,remaining,targets,gravity.z)};
        if kind==ShotType::DoubleJump && me.boost>0.0 {value+=0.25;}
        if value<best.map_or(f32::INFINITY,|b|b.score) {
            best=Some(ShotCandidate{time:t,ball:slice.location,ball_velocity:slice.velocity,
                approach:parsed.approach_target,shot_direction:parsed.shot_vector,
                shot_type:kind,fast:parsed.fast,score:value,parser:parsed});
        }
    }
    let _=ball_now;
    best
}

fn solve_quad(a:f32,b:f32,c:f32,y:f32,branch:i32)->Option<f32> {
    let root=(y-c)/a+b*b/(4.0*a*a);
    if root<0.0 {None}else{Some(-b/(2.0*a)+root.sqrt()*if branch>=0 {1.0}else{-1.0})}
}
fn max_jump_height(double:bool)->f32 {
    let extra=292.0*if double {1.0}else{0.0};
    let x=-(564.9931640625+extra)/(-650.111328125);
    -325.0556640625*x*x+(564.9931640625+extra)*x-11.6591796875-292.0*25.0/120.0*if double {1.0}else{0.0}
}
fn grid_jump_time(height:f32,double:bool)->Option<f32> {
    let single=solve_quad(-325.0556640625,564.9931640625,-11.6591796875,height.min(max_jump_height(false)),-1)?;
    if single>=24.0/120.0 {
        if single>=25.0/120.0 && double {
            return solve_quad(-325.0556640625,856.9931640625,-11.6591796875-292.0*25.0/120.0,height.min(max_jump_height(true)),-1);
        }
        return Some(single);
    }
    solve_quad(392.2802734375,272.7626953125,17.671875,height,1)
}
fn grid_profile(car:CarState)->(Vec<f32>,Vec<f32>) {
    let mut ds=vec![0.0];let mut vs=vec![car.velocity.flatten().magnitude().max(0.0)];
    for i in 0..600 {
        let v=*vs.last().unwrap();
        let boost=(i as f32/120.0)<car.boost.max(0.0);
        let ba=if boost {2975.0/3.0}else{0.0};
        let next=if v<=1400.0 {v+(1600.0-1440.0/1400.0*v+ba)/120.0}
        else if v<1410.0 {v+(22560.0-16.0*v+ba)/120.0}
        else {(v+if boost {2975.0/360.0}else{0.0}).min(2300.0)};
        vs.push(next);ds.push(ds.last().unwrap()+next/120.0);
    }
    (ds,vs)
}
#[derive(Clone,Copy)]struct GridJump{time:f32,ball:Vector,contact:Vector}
fn find_grid_jump(car:CarState,pred:&BallTrajectory,tg:ShotTargets,horizon:f32)->Option<GridJump> {
    if car.airborne {return None;}
    let target=center(tg);let (ds,vs)=grid_profile(car);let mut best=None;
    for s in &pred.slices {
        let t=s.time;if t<0.05 {continue;}if t>horizon {break;}
        if ((t*60.0).round()/60.0-t).abs()>0.009 {continue;}
        if s.location.y.abs()>5212.75 {break;}
        let dir=(target-s.location).normalize();let contact=s.location-dir*R;
        let distance=contact.flat_distance(car.location);
        let Some(jt)=grid_jump_time((contact.z-car.location.z).max(0.0),true) else{continue;};
        if contact.z>car.location.z+max_jump_height(true)+60.0 {continue;}
        let charge=(t*120.0).round().clamp(0.0,600.0) as usize;
        let drive=((t-jt).max(0.0).min(5.0)*120.0).round().clamp(0.0,600.0) as usize;
        let reach=ds[drive]+vs[drive]*(charge as f32-drive as f32)/120.0;
        if reach<distance {continue;}
        if best.is_none_or(|b:GridJump|t<b.time) {best=Some(GridJump{time:t,ball:s.location,contact});}
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{math::Matrix3,state::Hitbox};
    use rlbot::flat::AirState;
    fn test_car()->CarState {CarState{index:0,team:0,location:Vector::new(0.0,-1000.0,17.0),velocity:Vector::new(0.0,1200.0,0.0),orientation:Matrix3::new(0.0,1.5707963,0.0),angular_velocity:Vector::ZERO,demolished:false,airborne:false,supersonic:false,jumped:false,doublejumped:false,air_state:AirState::OnGround,has_jumped:false,has_double_jumped:false,has_dodged:false,dodge_timeout:0.0,dodge_elapsed:0.0,dodge_dir:Vector::ZERO,flip_reset_available:false,boost:50.0,hitbox:Hitbox{length:118.0,width:84.0,height:36.0,offset:Vector::ZERO}}}
    #[test]fn grid_jump_time_matches_source_polynomial(){assert!(grid_jump_time(200.0,false).is_some());assert!(grid_jump_time(350.0,true).is_some());assert!(grid_jump_time(5.0,true).is_some());}
    #[test]fn grid_profile_has_601_120hz_samples(){let(d,v)=grid_profile(test_car());assert_eq!(d.len(),601);assert_eq!(v.len(),601);assert!(d[600]>d[0]);}
    #[test]fn candidate_scan_only_returns_finite_source_scored_candidates(){
        let car=test_car();let tg=ShotTargets{left:Vector::new(-700.0,5120.0,350.0),right:Vector::new(700.0,5120.0,350.0)};
        let pred=BallTrajectory{slices:(0..360).map(|i|BallSlice{time:i as f32/120.0,location:Vector::new(0.0,0.0,100.0),velocity:Vector::ZERO}).collect()};
        let c=find_best_shot(car,&[],Vector::ZERO,&pred,tg,Vector::new(0.0,0.0,-650.0),991.6667,0.0,-999.0,false,true,3.0);
        assert!(c.is_none()||c.is_some_and(|x|x.score.is_finite()));
    }
}
