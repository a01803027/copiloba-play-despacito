"""Full camera-free composer -> native IPC -> C++ DSP -> WAV regression test."""
import array
import json
from pathlib import Path
import subprocess
import sys
import wave
root=Path(__file__).resolve().parents[2]
output=root/'artifacts/composer-test.wav'
metrics=root/'artifacts/composer-test.json'
result=subprocess.run([sys.executable,'-m','composer.app','--mode','mock','--launch-engine',
                       '--engine',sys.argv[1],'--library',sys.argv[2],'--offline-audio',
                       '--record',str(output),'--seconds','3','--metrics',str(metrics)],
                      cwd=root,text=True,capture_output=True,timeout=15)
print(result.stdout)
if result.returncode:
    raise RuntimeError(result.stderr)
report=json.loads(metrics.read_text())
assert report['note_ons']>0 and report['events_sent']>report['note_ons']
assert report['command_drops']==report['invalid_commands']==report['scheduler_overflows']==0
with wave.open(str(output),'rb') as wav:
    values=array.array('h',wav.readframes(wav.getnframes()))
assert max(map(abs,values))>100
print('Composer IPC + WAV PASS')
