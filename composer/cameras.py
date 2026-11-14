"""Camera selection before audio startup. Matches pinned OpenCV 4.11 AVFoundation.
OpenCV sorts video/muxed devices by uniqueID before indexing. On this Mac,
system_profiler lists both video cameras with the same IDs. Never prefer iPhone.
"""
import json
import subprocess
import sys


def select_mac_camera(devices):
    cameras=sorted((d for d in devices if d.get('spcamera_unique-id')),
                   key=lambda d:d['spcamera_unique-id'])
    for keyword in ('macbook','facetime','built-in','integrada'):
        for index,device in enumerate(cameras):
            description=(device.get('_name','')+' '+device.get('spcamera_model-id','')).lower()
            if keyword in description and 'iphone' not in description:
                return index,device.get('_name','Webcam integrada')
    raise RuntimeError('No encontré la webcam integrada del Mac. Usa --camera INDICE para seleccionar una cámara manualmente.')


def resolve_camera(requested):
    if requested is not None:
        if requested<0:
            raise ValueError('--camera debe ser un índice no negativo')
        return requested,f'Cámara seleccionada manualmente ({requested})'
    if sys.platform!='darwin':
        return 0,'Cámara predeterminada (0)'
    try:
        result=subprocess.run(['/usr/sbin/system_profiler','SPCameraDataType','-json'],
                              capture_output=True,text=True,check=True,timeout=10)
        devices=json.loads(result.stdout).get('SPCameraDataType',[])
    except (subprocess.SubprocessError,ValueError,OSError) as exc:
        raise RuntimeError('No pude consultar las cámaras del Mac. Puedes usar --camera 1 para elegir su webcam en la configuración actual.') from exc
    return select_mac_camera(devices)
