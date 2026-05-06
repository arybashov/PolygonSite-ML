import os
import torch
import torch.nn as nn

OBS_DIM    = 455
ACTION_DIM = 45


class TeamPolicyNet(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, ACTION_DIM), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt   = torch.load('data/model.pt', map_location=device)
hidden = ckpt.get('hidden', 512)

model = TeamPolicyNet(hidden=hidden).to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()

dummy = torch.zeros(1, OBS_DIM, device=device)
out   = 'data/model.onnx'

torch.onnx.export(
    model, dummy, out,
    input_names=['obs'], output_names=['action'],
    dynamic_axes={'obs': {0: 'batch'}, 'action': {0: 'batch'}},
    opset_version=17,
)

size_kb = round(os.path.getsize(out) / 1024)
print(f'ONNX exported → {out}  ({size_kb} KB)  epoch={ckpt.get("epoch")}')

# Merge external data into a single self-contained file for browser use
import onnx
from onnx.external_data_helper import load_external_data_for_model

merged = onnx.load(out)
load_external_data_for_model(merged, os.path.dirname(out) or '.')
single = out.replace('.onnx', '_web.onnx')
onnx.save_model(merged, single, save_as_external_data=False)
size_mb = round(os.path.getsize(single) / 1024 / 1024, 2)
print(f'Single-file ONNX → {single}  ({size_mb} MB)')
