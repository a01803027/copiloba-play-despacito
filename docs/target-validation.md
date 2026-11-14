# Board acceptance plan

## Provisioning

1. Confirm the actual CPU frequency, RAM, camera interface, NPU driver/delegate
   versions, model support and `hw:0,0` codec formats on the installed BSP.
   Run `sh tools/board-audit.sh` and retain its output with the test report.
2. Use a PREEMPT_RT kernel. Start from `isolcpus=domain,managed_irq,1 nohz_full=1
   rcu_nocbs=1 irqaffinity=0`; verify these options on the deployed kernel. Keep
   general tasks, camera, NPU work and stretch workers on CPU 0. Disable or
   configure irqbalance so it does not undo routing. Managed/per-CPU IRQs may
   resist affinity requests. Verify actual `/proc/interrupts` deltas under load.
3. Confirm the performance governor, cooling and thermal stability at sustained
   load. `isolcpus` alone does not move workqueues or provide a deadline guarantee.
   Audit kernel threads, device IRQ priorities, RT throttling and audio IRQ
   service time with tracing. Keep required IRQ threads able to service ALSA.
4. Install the C++ executables and frontend; configure service account device
   access and systemd RT/memory limits. All IPC participants need the same UID,
   architecture, atomic ABI and shared-memory size/version. Set service CPU
   affinity explicitly. Never run the Python process on the isolated CPU.
5. Inspect sensor controls with `v4l2-ctl --list-ctrls`. The supplied exposure
   command assumes UVC `exposure_auto=1` and `exposure_absolute`; adapt it for
   other sensors. Exposure must fit a 16.667 ms frame. Verify negotiated 60 FPS,
   dropped frames, PTS semantics, rolling-shutter/readout latency, and row stride.
6. Verify vendor-compiled quantized pose/hand models with representative inputs,
   correct detector/crop tracking, visibility outputs and actual NPU execution.
   Compare decoded landmarks with reference results. Benchmark the complete
   pipeline; standalone model inference times are insufficient.

## Measurements

* Audio geometry: 128 frames = 2.667 ms, two periods = 5.333 ms; 256 frames =
  5.333 ms, two periods = 10.667 ms. Measure actual DAC latency and queue delay.
* Frame phase alone can cost almost 16.667 ms. Add exposure/readout, hand + pose
  inference, filtering/onset detection, IPC, quantization delay and DAC latency.
  Deceleration hits require a previous speed observation. A 60 Hz detector and
  strict full-range 40 ms early quantizer cannot establish a worst-case <35 ms.
* Measure physical motion to analog DAC output with an external synchronized
  marker/high-speed camera or instrumented motion rig and oscilloscope. Sweep
  motion phase relative to exposure and grid. Report worst case, p50/p95/p99,
  duration, sample count, clock error and sensor timestamp semantics. Test all
  limbs and simultaneous strikes. Capture-to-DAC telemetry is only an estimate:
  it excludes time before the driver capture PTS and uses an ALSA delay query,
  not an external hardware timestamp. Do not call it proof of physical latency.
* Stop audio gracefully to print XRUN, command overflow, scheduler overflow,
  late event, stem starvation, max render time and capture-to-DAC counters.
  A late counter includes immediate events arriving after their scheduled
  sample; investigate magnitude with additional board-side tracing as needed.
* Require zero XRUNs, zero event loss and zero stem starvation after prebuffering
  over an agreed sustained stress test (e.g. 8 h), both period sizes, all 32
  voices, simultaneous controls, four stems, camera/NPU, realistic I/O and
  thermal stress. Check runtime page faults, scheduling latency and render time
  against the period deadline. No amount of host testing proves WCET on A35.
* Test disconnects, camera occlusion, stale heartbeat, producer flood, stem worker
  failure, shutdown/restart and unsupported ALSA geometry. Camera loss suppresses
  new sounds and clears active voices after 250 ms (filter tail may remain).
  Command overflow drops the newest event and counts it. Stem starvation yields
  silence for that bus and counts missing frames. ALSA underrun stops with an
  error; restart requires a fresh session, not concealed recovery.

## Current evidence

Portable C++ tests build/run on the development Mac: SPSC full/empty/wrap and
200,000 concurrent transfers; POSIX shared memory across fork; filter reset;
torso scale/translation invariance; invalid/stale time handling; both quantizer
policies; heap order/capacity; sample-frame onset placement; bounded finite DSP
output; allocation guard around render; two-bar loop playback and mute behavior.
Python posture tests cover geometric states and rotation/scale/translation.
Linux target files have a CI build definition but were not compiled locally:
the development Docker daemon was unavailable. Board, NPU and analog latency
validation remain outstanding.
