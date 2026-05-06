import os
import numpy as np
import onnxruntime as ort

path = 'data/model.onnx'
size = os.path.getsize(path)
print(f'File: {path}  {size:,} bytes  ({size/1024/1024:.2f} MB)')

sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
print('Inputs: ', [i.name for i in sess.get_inputs()])
print('Outputs:', [o.name for o in sess.get_outputs()])

obs = np.zeros((1, 455), dtype=np.float32)
out = sess.run(['action'], {'obs': obs})[0]
print(f'Inference OK — output shape: {out.shape}  range: [{out.min():.3f}, {out.max():.3f}]')
