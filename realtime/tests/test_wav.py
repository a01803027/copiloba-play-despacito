"""Test B: actual C++ DSP output, PCM16 stereo WAV, no device/camera."""
import array
import json
from pathlib import Path
import subprocess
import sys
import wave

engine=Path(sys.argv[1]).resolve()
path=Path(sys.argv[2]).resolve()
path.parent.mkdir(parents=True,exist_ok=True)
subprocess.run([str(engine),'--render',str(path),'--seconds','8','--demo'],check=True)
with wave.open(str(path),'rb') as f:
    assert (f.getnchannels(),f.getsampwidth(),f.getframerate())==(2,2,48000)
    frames=f.getnframes()
    samples=array.array('h',f.readframes(frames))
peak=max(abs(x) for x in samples)/32768
rms=(sum((x/32768)**2 for x in samples)/len(samples))**.5
assert frames>=8*48000 and .001<rms<1 and .01<peak<=1
print(json.dumps({'wav':str(path),'frames':frames,'peak':peak,'rms':rms},indent=2))
