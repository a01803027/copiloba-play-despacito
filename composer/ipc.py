"""Python mapping with native C++ acquire/release operations, one producer only."""
import ctypes as C
import os
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path
from realtime.vision_mock_sender import AudioCommand

class Bridge:
    def __init__(self, library):
        self.native=C.CDLL(str(Path(library).resolve()))
        signatures={
            'cv_daw_attach':([C.c_void_p,C.c_uint32],C.c_int),
            'cv_daw_claim':([C.c_void_p,C.c_uint32],C.c_int),
            'cv_daw_release':([C.c_void_p,C.c_uint32],None),
            'cv_daw_push':([C.c_void_p,C.POINTER(AudioCommand),C.c_uint32],C.c_int),
            'cv_daw_control':([C.c_void_p,C.c_uint32,C.c_uint32,C.c_float,C.c_uint32],C.c_int),
            'cv_daw_metric':([C.c_void_p,C.c_uint32],C.c_uint64),
        }
        for name,(arguments,result) in signatures.items():
            f=getattr(self.native,name);f.argtypes=arguments;f.restype=result
        self.memory=shared_memory.SharedMemory(name='cv_daw_shm',create=False,**({'track':False} if sys.version_info>=(3,13) else {}))
        if sys.version_info<(3,13):
            from multiprocessing import resource_tracker
            resource_tracker.unregister(self.memory._name,'shared_memory')
        self.pointer=C.c_void_p(C.addressof(C.c_char.from_buffer(self.memory.buf)))
        self.token=os.getpid()
        if not self.native.cv_daw_attach(self.pointer,self.memory.size):
            self.memory.close();raise RuntimeError('Motor no listo o ABI incompatible; recompila e inicia primero el motor.')
        if not self.native.cv_daw_claim(self.pointer,self.token):
            self.memory.close();raise RuntimeError('Ya hay otro controlador conectado. Cierra el mock antes de usar la cámara.')
        self.sent=0
        self.closed=False

    def metric(self, index):
        return self.native.cv_daw_metric(self.pointer,index)

    def send(self,event):
        command=AudioCommand(event.limb,event.kind,event.note,0,event.value,event.frame)
        result=self.native.cv_daw_push(self.pointer,C.byref(command),self.token)
        if result!=1:
            raise RuntimeError(f'Cola IPC rechazada ({result}); detener para evitar pérdida silenciosa.')
        self.sent+=1

    def controls(self, values):
        for group,key in ((0,'cutoff'),(1,'reverb'),(2,'bend'),(6,'gain')):
            for limb,value in enumerate(values[key]):
                if not self.native.cv_daw_control(self.pointer,group,limb,value,self.token):
                    raise RuntimeError(f'Control {key} rechazado; recompila la biblioteca IPC.')
        for group,key in ((3,'bpm'),(4,'energy')):
            if not self.native.cv_daw_control(self.pointer,group,0,values[key],self.token):
                raise RuntimeError('Control global rechazado')

    def probe(self):
        # Round trip includes callback interval and Python polling, NOT physical
        # motion-to-sound latency. Send a harmless preset to acknowledge ingestion.
        from .brain import Event
        before=self.metric(1)
        start=time.perf_counter_ns()
        self.send(Event(3,4,4,0.,0))
        deadline=time.monotonic()+1
        while self.metric(1)<=before:
            if time.monotonic()>deadline:
                raise RuntimeError('Motor sin avance: timeout de confirmación IPC')
            time.sleep(.0002)
        return (time.perf_counter_ns()-start)/1e6

    def close(self):
        if self.closed:
            return
        try:
            self.controls({'cutoff':[1000]*4,'reverb':[0]*4,'bend':[0]*4,'gain':[0]*4,'bpm':76,'energy':0})
        finally:
            self.native.cv_daw_release(self.pointer,self.token)
            self.memory.close()
            self.closed=True
