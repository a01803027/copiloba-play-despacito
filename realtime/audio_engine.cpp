// Pinned single-header dependency; all backend/decoder setup happens before RT.
#define MA_NO_ENGINE
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MA_NO_FLAC
#define MA_NO_MP3
#ifdef __linux__
#define MA_ENABLE_ALSA
#elif defined(__APPLE__)
#define MA_ENABLE_COREAUDIO
#else
#define MA_ENABLE_NULL
#endif
#define MINIAUDIO_IMPLEMENTATION
#include "third_party/miniaudio.h"
#include "audio_engine.hpp"
#include "rt_startup.hpp"
#include "stem_worker.hpp"
#include "wav_writer.hpp"
#include "hardware_clock.hpp"
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>
using namespace cv_daw;
namespace {
volatile sig_atomic_t running=1;
void stop(int){running=0;}
std::vector<float> load_wav(const char* path,uint64_t max_frames){
 ma_decoder decoder;auto config=ma_decoder_config_init(ma_format_f32,2,sample_rate);
 if(ma_decoder_init_file(path,&config,&decoder)!=MA_SUCCESS)throw std::runtime_error(std::string("cannot decode WAV: ")+path);
 struct Close {ma_decoder* p;~Close(){ma_decoder_uninit(p);}} close{&decoder};
 ma_uint64 frames=0;if(ma_decoder_get_length_in_pcm_frames(&decoder,&frames)!=MA_SUCCESS || !frames || frames>max_frames)throw std::runtime_error("WAV length invalid/too large");
 std::vector<float> data(size_t(frames)*2);ma_uint64 read=0;
 ma_result result=ma_decoder_read_pcm_frames(&decoder,data.data(),frames,&read);
 if((result!=MA_SUCCESS && result!=MA_AT_END) || read!=frames)throw std::runtime_error("WAV decode truncated");
 for(float& x:data){if(!finite(x))throw std::runtime_error("WAV contains NaN/Inf");x=std::clamp(x,-1.f,1.f);}
 return data;
}
struct Assets {
 Wavetables tables;
 std::array<std::vector<float>,4> samples;
 Assets(){
  uint32_t rng=1234567;
  for(auto& data:samples)data.resize(24000*2);
  double kick_phase=0,prev_noise=0,snare_lp=0,snare_hp=0,hat_hp=0,hat_lp=0,click_lp=0;
  for(unsigned n=0;n<24000;++n){double t=double(n)/sample_rate;
   rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;double noise=double(rng)/UINT32_MAX*2-1;
   // Band-limited studio noise filters (zero raw Nyquist white-noise hiss!).
   double prev_n=prev_noise;prev_noise=noise;
   double hp1=noise-0.86*prev_n;
   click_lp+=0.34*(hp1-click_lp); // ~3.0 kHz warm beater click
   snare_hp=0.92*(snare_hp+noise-prev_n); // ~650 Hz HPF
   snare_lp+=0.42*(snare_hp-snare_lp); // ~4.2 kHz warm LPF (crisp snap, zero static hiss)
   hat_hp=0.68*(hat_hp+hp1); // ~3 kHz HPF
   hat_lp+=0.48*(hat_hp-hat_lp); // ~6.4 kHz silky LPF (kills >10 kHz hiss completely)
   // Smooth 1.2ms attack ramp guarantees zero sample-start click on every drum hit.
   double ramp=std::min(1.0,t/0.0012);
   // Punchy club Kick (165 Hz -> 46 Hz fast drop) + warm 3ms beater transient.
   kick_phase+=2*std::numbers::pi*(46+120*std::exp(-52*t))/sample_rate;
   double kick_body=std::sin(kick_phase)*std::exp(-13.5*t);
   double kick_punch=0.30*std::sin(2*std::numbers::pi*110*t)*std::exp(-36*t);
   double kick=ramp*0.74*std::tanh(1.40*(kick_body+kick_punch+0.26*click_lp*std::exp(-320*t)));
   // Warm tight Studio Snare / Clap (subtle 185 Hz wood body + smooth 650-4200 Hz band-passed clap tail, zero plastic pop).
   double clap_env=std::exp(-22*t)+(t>0.010?0.55*std::exp(-24*(t-0.010)):0.0)+(t>0.020?0.35*std::exp(-26*(t-0.020)):0.0);
   double snare_body=0.18*std::sin(2*std::numbers::pi*185*t)*std::exp(-20*t);
   double snare=ramp*0.72*std::tanh(1.15*(snare_body+0.68*snare_lp*clap_env));
   // Silky Studio Hi-Hat (warm metallic ratios + 3-6.4 kHz air, fast clean decay).
   double metal=0.45*std::sin(2*std::numbers::pi*3420*t)*std::sin(2*std::numbers::pi*4870*t)+0.35*std::sin(2*std::numbers::pi*6410*t);
   double hat=ramp*(0.45*hat_lp+0.35*metal)*std::exp(-46*t)*0.66;
   // Warm minor-9th chord stab.
   double stab=ramp*(std::sin(2*std::numbers::pi*523.25*t)+0.7*std::sin(2*std::numbers::pi*622.25*t)+0.6*std::sin(2*std::numbers::pi*783.99*t)+0.5*std::sin(2*std::numbers::pi*1174.66*t))*std::exp(-7.5*t)*0.30;
   std::array<float,4> value{float(kick),float(snare),float(hat),float(stab)};
   for(unsigned j=0;j<4;++j)samples[j][2*n]=samples[j][2*n+1]=value[j];
  }
 }
 std::array<SampleView,4> views()const{std::array<SampleView,4> result;for(unsigned i=0;i<4;++i)result[i]={samples[i].data(),uint32_t(samples[i].size()/2),2};return result;}
};
struct LiveTap {
 SpscQueue<Stereo,131072> fifo; // ~2.73 s preallocated lock-free buffer
 std::atomic<bool> stop{false};
 std::atomic<uint64_t> dropped{0};
 std::thread writer;
 void start(const char* path){
  auto wav=std::make_shared<WavWriter>(path);
  writer=std::thread([this,wav]{
   std::array<float,1024*2> chunk{};
   while(!stop.load(std::memory_order_acquire) || fifo.size_for_producer()>0){
    unsigned n=0;Stereo s{};
    while(n<1024 && fifo.try_pop(s)){chunk[2*n]=s.l;chunk[2*n+1]=s.r;++n;}
    if(n)wav->write(chunk.data(),n);
    else std::this_thread::sleep_for(std::chrono::milliseconds(5));
   }
  });
 }
 void finish(){stop.store(true,std::memory_order_release);if(writer.joinable())writer.join();}
 ~LiveTap(){finish();}
};
struct Runtime {
 AudioEngine* engine=nullptr;SharedMemory* shared=nullptr;LiveTap* tap=nullptr;std::atomic<int> startup_error{0},backend_error{0};
 bool profile=false;uint64_t calls=0,sum_ticks=0,max_ticks=0;
 ma_result (*original_start)(ma_device*)=nullptr;
};
void data_callback(ma_device* device,void* output,const void*,ma_uint32 frames){
 auto& runtime=*static_cast<Runtime*>(device->pUserData);
 flush_subnormals();const auto begin=runtime.profile?hardware_ticks():0;
 // Fixed 128 requested; defensive chunking never writes outside miniaudio's buffer.
 auto* out=static_cast<float*>(output);
 for(ma_uint32 offset=0;offset<frames;offset+=block_frames)runtime.engine->processBlock(out+2*offset,std::min(block_frames,frames-offset));
 if(runtime.tap){
  for(ma_uint32 i=0;i<frames;++i){
   if(!runtime.tap->fifo.try_push({out[2*i],out[2*i+1]}))runtime.tap->dropped.fetch_add(1,std::memory_order_relaxed);
  }
 }
 if(runtime.profile){uint64_t elapsed=hardware_ticks()-begin;++runtime.calls;runtime.sum_ticks+=elapsed;runtime.max_ticks=std::max(runtime.max_ticks,elapsed);
  auto& t=runtime.shared->telemetry;t.callback_count.store(runtime.calls,std::memory_order_relaxed);t.callback_sum_ticks.store(runtime.sum_ticks,std::memory_order_relaxed);t.callback_max_ticks.store(runtime.max_ticks,std::memory_order_relaxed);}
}
#ifdef __linux__
ma_result start_backend(ma_device* device){
 auto& r=*static_cast<Runtime*>(device->pUserData);
 int error=initialize_audio_thread();r.startup_error.store(error,std::memory_order_release);
 if(error)return MA_ERROR;
 return r.original_start?r.original_start(device):MA_SUCCESS;
}
// miniaudio normally recovers EPIPE internally. Intercept that recovery request
// and fail the stream so an underrun can never masquerade as a passing run.
Runtime* active_runtime=nullptr; // Set before start; cleared after device stop.
int fail_recovery(ma_snd_pcm_t*,int error,int){
 if(active_runtime){auto& t=active_runtime->shared->telemetry;
  if(error==-EPIPE)t.xruns.store(t.xruns.load(std::memory_order_relaxed)+1,std::memory_order_relaxed);
  active_runtime->backend_error.store(error,std::memory_order_release);
 }return error;
}
#endif
struct Context {ma_context value{};bool initialized=false;~Context(){if(initialized)ma_context_uninit(&value);}};
struct Device {ma_device value{};bool initialized=false;~Device(){if(initialized)ma_device_uninit(&value);}};
struct Unlock {bool locked=false;~Unlock(){if(locked)munlockall();}};
}
int main(int argc,char** argv){try{
 signal(SIGINT,stop);signal(SIGTERM,stop);
 const char* render_path=nullptr;const char* live_record_path=nullptr;bool demo=false,profile=false,performance=false,stretch_stems=false;double seconds=0;
 const char* sample_paths[4]{};const char* stem_paths[4]{};double source_bpm=120,initial_bpm=120;unsigned offline_blocks=0;
 for(int i=1;i<argc;++i){std::string option=argv[i];
  if(option=="--render" && i+1<argc)render_path=argv[++i];
  else if(option=="--live-record" && i+1<argc)live_record_path=argv[++i];
  else if(option=="--seconds" && i+1<argc){seconds=std::stod(argv[++i]);if(!(seconds>0&&seconds<=3600))throw std::runtime_error("seconds fuera de rango");}
  else if(option=="--performance" && i+1<argc){performance=true;initial_bpm=std::stod(argv[++i]);
   if((std::bit_cast<uint64_t>(initial_bpm)&0x7ff0000000000000ull)==0x7ff0000000000000ull || initial_bpm<60 || initial_bpm>200)throw std::runtime_error("performance BPM must be 60..200");}
  else if(option=="--stretch-stems")stretch_stems=true;
  else if(option=="--demo")demo=true;
  else if(option=="--profile")profile=true;
  else if(option=="--offline" && i+1<argc)offline_blocks=unsigned(std::stoul(argv[++i]));
  else if(option=="--samples" && i+4<argc)for(auto& p:sample_paths)p=argv[++i];
  else if(option=="--stems" && i+5<argc){source_bpm=std::stod(argv[++i]);for(auto& p:stem_paths)p=argv[++i];}
  else throw std::runtime_error("usage: cv-daw-engine [--performance BPM] [--samples kick.wav snare.wav hat.wav stab.wav] [--stems BPM drums.wav bass.wav vocals.wav other.wav] [--stretch-stems] [--offline BLOCKS]");
 }
 if((std::bit_cast<uint64_t>(source_bpm)&0x7ff0000000000000ull)==0x7ff0000000000000ull)throw std::runtime_error("source BPM must be finite");
 auto assets=std::make_unique<Assets>();for(unsigned i=0;i<4;++i)if(sample_paths[i])assets->samples[i]=load_wav(sample_paths[i],sample_rate*15);
 SharedMemoryMapping mapping(true);auto& shared=mapping.get();auto stems=std::make_unique<StemBus>();
 shared.controls.target_bpm.store(float(initial_bpm));stems->requested_bpm.store(float(initial_bpm));
 std::unique_ptr<StemWorker> worker;
 if(stem_paths[0]){std::array<std::vector<float>,4> data;for(unsigned i=0;i<4;++i)data[i]=load_wav(stem_paths[i],uint64_t(sample_rate)*60*15);
  worker=std::make_unique<StemWorker>(*stems,std::move(data),source_bpm,initial_bpm,performance&&!stretch_stems);worker->wait_for_prebuffer();}
 auto engine=std::make_unique<AudioEngine>(shared,assets->tables,assets->views(),*stems,performance);
 if(demo)shared.header.producer_claim.store(uint32_t(getpid()),std::memory_order_release);
 uint64_t last_demo_beat=UINT64_MAX;
 auto demo_step=[&]{
  // Deterministic DSP audition, separate from Python's expressive composer.
  uint64_t frame=shared.telemetry.total_samples_rendered.load(std::memory_order_acquire);constexpr unsigned beat=sample_rate;
  if(frame/beat!=last_demo_beat){last_demo_beat=frame/beat;constexpr unsigned chords[][3]{{48,51,55},{56,60,63},{53,56,60},{55,58,62}};const auto& chord=chords[(frame/(beat*4))%4];
   for(unsigned note:chord)shared.commands.try_push({LimbID::L_ARM,EventType::NOTE_ON,uint8_t(note),0,.45f,frame});}
 };
 if(offline_blocks || render_path){
  // Functional IPC harness, explicitly not hardware/RT qualification. No ALSA,
  // affinity or memory-lock privileges needed. Publish ready for Python sender.
  shared.header.ready.store(1,std::memory_order_release);flush_subnormals();std::array<float,block_frames*2> out{};
  std::unique_ptr<WavWriter> wav;if(render_path)wav=std::make_unique<WavWriter>(render_path);
  unsigned blocks=offline_blocks?offline_blocks:unsigned(std::ceil((seconds>0?seconds:8)*sample_rate/block_frames));
  uint64_t max_ticks=0,sum_ticks=0;auto frequency=hardware_frequency();
  std::cout<<"offline-ready"<<std::endl;
  auto deadline=std::chrono::steady_clock::now();
  for(unsigned i=0;i<blocks && running;++i){if(demo)demo_step();auto begin=hardware_ticks();engine->processBlock(out.data(),block_frames);auto elapsed=hardware_ticks()-begin;sum_ticks+=elapsed;max_ticks=std::max(max_ticks,elapsed);if(wav)wav->write(out.data(),block_frames);if(offline_blocks){deadline+=std::chrono::nanoseconds(2666667);std::this_thread::sleep_until(deadline);}}
  if(frequency)std::cout<<"offline DSP mean_us="<<1e6*sum_ticks/frequency/blocks<<" max_us="<<1e6*max_ticks/frequency<<std::endl;
  shared.header.ready.store(0,std::memory_order_release);
  std::cout<<"offline frames="<<engine->total_samples_rendered()<<" commands="<<shared.telemetry.commands_consumed.load()<<'\n';return 0;
 }
#if !defined(__linux__) && !defined(__APPLE__)
 throw std::runtime_error("Backend no disponible en esta plataforma");
#else
 Context context;
#ifdef __APPLE__
 ma_backend backend=ma_backend_coreaudio;
#else
 ma_backend backend=ma_backend_alsa;
#endif
 auto context_config=ma_context_config_init();
 if(ma_context_init(&backend,1,&context_config,&context.value)!=MA_SUCCESS)throw std::runtime_error("Audio context initialization failed");context.initialized=true;
 Runtime runtime;runtime.engine=engine.get();runtime.shared=&shared;runtime.profile=profile;
 std::unique_ptr<LiveTap> live_tap;
 if(live_record_path){live_tap=std::make_unique<LiveTap>();live_tap->start(live_record_path);runtime.tap=live_tap.get();}
 shared.telemetry.counter_frequency.store(hardware_frequency());
 auto config=ma_device_config_init(ma_device_type_playback);
#ifdef __linux__
 ma_device_id id{};std::strcpy(id.alsa,"hw:0,0");config.playback.pDeviceID=&id;
#endif
config.playback.format=ma_format_f32;config.playback.channels=2;
 #ifdef __linux__
 config.playback.shareMode=ma_share_mode_exclusive;
#endif
config.sampleRate=sample_rate;config.periodSizeInFrames=block_frames;config.periods=2;
 config.performanceProfile=ma_performance_profile_low_latency;config.dataCallback=data_callback;config.pUserData=&runtime;
 config.noClip=MA_TRUE;config.noPreSilencedOutputBuffer=MA_TRUE;config.noDisableDenormals=MA_TRUE;
 #ifdef __linux__
 config.alsa.noMMap=MA_TRUE;config.alsa.noAutoFormat=MA_TRUE;config.alsa.noAutoChannels=MA_TRUE;config.alsa.noAutoResample=MA_TRUE;
 #endif
 Device device;
 if(ma_device_init(&context.value,&config,&device.value)!=MA_SUCCESS)throw std::runtime_error("Audio device initialization failed");device.initialized=true;
 #ifdef __linux__
 const auto& playback=device.value.playback;
 // f32 is the callback contract. Codec-native integer conversion is permitted
 // in miniaudio, but native rate/channels/period geometry must match exactly.
 if(playback.internalSampleRate!=sample_rate || playback.internalChannels!=2 || playback.internalPeriodSizeInFrames!=block_frames || playback.internalPeriods!=2)throw std::runtime_error("ALSA negotiated incompatible geometry");
 runtime.original_start=context.value.callbacks.onDeviceStart;context.value.callbacks.onDeviceStart=start_backend;
 active_runtime=&runtime;context.value.alsa.snd_pcm_recover=reinterpret_cast<ma_proc>(fail_recovery);
 #endif
 signal(SIGINT,stop);signal(SIGTERM,stop);
 // Decoder, device, audio worker stack, stem worker and all DSP storage already
 // exist. Touch/warm DSP before locking; rebuild clock/voices before going live.
 {
  auto warm_shared=std::make_unique<SharedMemory>();auto warm_stems=std::make_unique<StemBus>();
  auto warm_engine=std::make_unique<AudioEngine>(*warm_shared,assets->tables,assets->views(),*warm_stems);
  std::array<float,block_frames*2> warm{};warm_engine->processBlock(warm.data(),block_frames);
 }
 #ifdef __linux__
 Unlock memory;int error=lock_process_memory();if(error)throw std::runtime_error(std::string("mlockall: ")+std::strerror(error));memory.locked=true;
 #endif
 ma_result start_result=ma_device_start(&device.value);
 if(start_result!=MA_SUCCESS){int startup=runtime.startup_error.load();throw std::runtime_error(std::string("audio start: ")+(startup?std::strerror(startup):ma_result_description(start_result)));}
 std::cout<<"audio-ready backend="<<(backend==ma_backend_coreaudio?"CoreAudio":"ALSA")<<" rate="<<device.value.sampleRate<<" callback_frames=128"<<std::endl;
 shared.header.ready.store(1,std::memory_order_release);
 auto until=std::chrono::steady_clock::now()+std::chrono::milliseconds(int64_t(seconds*1000));
 while(running && ma_device_is_started(&device.value) && !runtime.backend_error.load(std::memory_order_acquire) && !(worker&&worker->failed())){if(seconds>0&&std::chrono::steady_clock::now()>=until)break;if(demo)demo_step();std::this_thread::sleep_for(std::chrono::milliseconds(demo?1:100));}
 bool unexpected_stop=running && !ma_device_is_started(&device.value);
 shared.header.ready.store(0,std::memory_order_release);ma_device_stop(&device.value);
 if(live_tap){live_tap->finish();if(live_tap->dropped.load())std::cerr<<"live_tap_dropped="<<live_tap->dropped.load()<<'\n';}
 #ifdef __linux__
 active_runtime=nullptr;
 #endif
 auto& t=shared.telemetry;
 if(profile&&t.counter_frequency.load()&&runtime.calls)std::cout<<"callback_mean_us="<<1e6*runtime.sum_ticks/t.counter_frequency.load()/runtime.calls<<" callback_max_us="<<1e6*runtime.max_ticks/t.counter_frequency.load()<<" callback_count="<<runtime.calls<<std::endl;
 std::cout<<"frames="<<t.total_samples_rendered.load()<<" consumed="<<t.commands_consumed.load()<<" drops="<<t.producer_drops.load()<<" xruns="<<t.xruns.load()<<" late="<<t.late_commands.load()<<" pending_full="<<t.scheduler_overflows.load()<<" stem_starved="<<t.stem_starved_frames.load()<<'\n';
 return unexpected_stop || runtime.backend_error.load() || (worker&&worker->failed())?2:0;
#endif
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
