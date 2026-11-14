#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
namespace cvdaw {
struct Vec {
 double x{},y{},z{};
 Vec operator+(Vec b)const{return {x+b.x,y+b.y,z+b.z};}
 Vec operator-(Vec b)const{return {x-b.x,y-b.y,z-b.z};}
 Vec operator*(double k)const{return {x*k,y*k,z*k};}
 double dot(Vec b)const{return x*b.x+y*b.y+z*b.z;}
 double norm()const{return std::sqrt(dot(*this));}
};
struct Euro {
 double min_cutoff=1.5,beta=0.08,derivative_cutoff=1.0;
 double raw{},filtered{},derivative{}; bool ready=false;
 static double alpha(double hz,double dt){return 1.0/(1.0+1.0/(2*std::numbers::pi*hz*dt));}
 double step(double x,double dt){
  if(!ready){ready=true;raw=filtered=x;return x;}
  double a=alpha(derivative_cutoff,dt);
  derivative+=a*((x-raw)/dt-derivative); raw=x;
  filtered+=alpha(min_cutoff+beta*std::abs(derivative),dt)*(x-filtered);return filtered;
 }
 void reset(){ready=false;derivative=0;}
};
using Pose=std::array<Vec,33>;
struct MotionFrame {
 std::array<Vec,4> position{},velocity{};
 std::array<bool,4> hit{};
 std::array<double,4> angle{},radius{};
 double hip_y{};
};
class Motion {
 std::array<std::array<Euro,3>,33> filters_{};
 std::array<Vec,4> previous_{},velocity_{};
 std::array<uint64_t,4> last_hit_{};
 uint64_t last_ns_{}; bool initialized_=false;
public:
 double min_speed=1.8,deceleration=18;
 void reset(){initialized_=false;last_ns_=0;for(auto& f:filters_)for(auto& a:f)a.reset();}
 bool process(Pose pose,uint64_t ns,MotionFrame& out){
  if(last_ns_ && ns<=last_ns_)return false;
  double dt=last_ns_?double(ns-last_ns_)*1e-9:1.0/60;
  if(dt>0.1){reset();dt=1.0/60;}
  // Input coordinates must have common metric: x, y*height/width, z.
  for(unsigned i=0;i<33;++i){auto& f=filters_[i];pose[i]={f[0].step(pose[i].x,dt),f[1].step(pose[i].y,dt),f[2].step(pose[i].z,dt)};}
  Vec hips=(pose[23]+pose[24])*0.5,shoulders=(pose[11]+pose[12])*0.5;
  double scale=(shoulders-hips).norm();
  if(!std::isfinite(scale)||scale<0.025){reset();return false;}
  out.hip_y=hips.y; constexpr std::array<unsigned,4> ids{15,16,27,28};
  for(unsigned i=0;i<4;++i){
   Vec p=(pose[ids[i]]-hips)*(1.0/scale);
   Vec v=initialized_?(p-previous_[i])*(1.0/dt):Vec{};
   Vec a=(v-velocity_[i])*(1.0/dt);double speed=velocity_[i].norm();
   out.hit[i]=initialized_ && speed>min_speed && a.dot(velocity_[i]*(1.0/std::max(speed,1e-9))) < -deceleration && ns-last_hit_[i]>=80000000;
   if(out.hit[i])last_hit_[i]=ns;
   out.position[i]=p;out.velocity[i]=v;out.angle[i]=std::atan2(-p.y,p.x);out.radius[i]=std::hypot(p.x,p.y);
   previous_[i]=p;velocity_[i]=v;
  }
  last_ns_=ns;initialized_=true;return true;
 }
};
// Cheap context fallback. A trained classifier may replace probabilities, never fast hits.
class Temporal {
 std::array<std::array<float,12>,30> ring_{};
 unsigned head_=0,count_=0; double mean_=0,last_y_=0,last_cross_=0,period_=0.5;
 bool below_=false;
public:
 float energy=0,bpm=120; bool full()const{return count_==30;}
 const auto& window()const{return ring_;} unsigned head()const{return head_;}
 void step(const MotionFrame& f,double seconds){
  double e=0;for(unsigned i=0;i<4;++i){auto v=f.velocity[i];ring_[head_][3*i]=v.x;ring_[head_][3*i+1]=v.y;ring_[head_][3*i+2]=v.z;e+=v.dot(v);}
  head_=(head_+1)%30;count_=std::min(count_+1,30u);
  energy+=0.1f*(float(std::clamp(e/80.0,0.0,1.0))-energy);
  mean_+=0.03*(f.hip_y-mean_);double y=f.hip_y-mean_;
  // Hysteretic phase detector + bounded loop filter (one bounce per quarter note).
  if(y<-0.008)below_=true;
  if(below_ && y>0.008 && y>last_y_){
   below_=false;double d=seconds-last_cross_;
   if(last_cross_>0 && d>=0.3 && d<=1.0){period_+=0.15*(d-period_);bpm=float(60/period_);}
   last_cross_=seconds;
  }
  last_y_=y;
 }
};
}
