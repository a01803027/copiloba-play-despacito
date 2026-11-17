"""Shared per-tensor TFLite input/output and optional delegated execution."""
import numpy as np


class TFLiteModel:
    def __init__(self, path, delegate=None):
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate
        except ImportError:
            from tensorflow.lite import Interpreter
            from tensorflow.lite.experimental import load_delegate

        options = {'experimental_delegates': [load_delegate(delegate)]} if delegate else {}
        self.interpreter = Interpreter(model_path=str(path), num_threads=1, **options)
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        if len(inputs) != 1:
            raise ValueError('Se requiere un único tensor TFLite de entrada')
        self.input = inputs[0]
        self.outputs = self.interpreter.get_output_details()

    @staticmethod
    def quantization(details):
        scale, zero = details['quantization']
        if scale <= 0:
            raise ValueError('Se requiere cuantización TFLite por tensor')
        return scale, zero

    def invoke(self, values):
        if tuple(values.shape) != tuple(self.input['shape']):
            raise ValueError(f"Entrada TFLite esperada {self.input['shape']}, recibida {values.shape}")
        if self.input['dtype'] != np.float32:
            scale, zero = self.quantization(self.input)
            bounds = np.iinfo(self.input['dtype'])
            values = np.clip(np.rint(values / scale + zero), bounds.min, bounds.max)
        values = values.astype(self.input['dtype'], copy=False)
        self.interpreter.set_tensor(self.input['index'], values)
        self.interpreter.invoke()
        results = []
        for output in self.outputs:
            value = self.interpreter.get_tensor(output['index'])
            if output['dtype'] != np.float32:
                scale, zero = self.quantization(output)
                value = (value.astype(np.float32) - zero) * scale
            results.append(value)
        return results
