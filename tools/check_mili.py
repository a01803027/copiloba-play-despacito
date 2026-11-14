"""Test A: run the ORIGINAL main with real video/MediaPipe, silent dummy audio.
Only hardware adapters are redirected in this test process. Source is untouched.
"""
import os
os.environ['SDL_AUDIODRIVER']='dummy'
os.environ['MPLCONFIGDIR']=os.path.abspath('artifacts/matplotlib')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import BrazoIzquierdo
original_capture=cv2.VideoCapture
original_wait=cv2.waitKey
frames=0
video=Path(sys.argv[1] if len(sys.argv)>1 else 'brazo.mp4').resolve()
def key(_):
    global frames
    frames+=1
    return ord('q') if frames>=30 else -1
cv2.VideoCapture=lambda _:original_capture(str(video))
cv2.imshow=lambda *_:None
cv2.waitKey=key
BrazoIzquierdo.main()
if frames!=30:
    raise SystemExit('Test A incompleto: vídeo no entregó 30 imágenes')
print('Test A PASS: main original, 30 frames de vídeo, MediaPipe real, WAV cargados, salida SDL dummy; sin verificación física.')
