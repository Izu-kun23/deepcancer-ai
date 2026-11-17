import base64
import io
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

def png_url(array):
    buffer = io.BytesIO()
    Image.fromarray(array.astype(np.uint8)).save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()

def explain(model, target, tensor, original):
    values = {}
    def hook(module, inputs, output):
        values['activation'] = output
        output.register_hook(lambda grad: values.update(gradient=grad))
    handle = target.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        x = tensor.detach().clone().requires_grad_(True)
        logits = model(x)
        cls = logits.argmax(1).item()
        logits[0, cls].backward()
        weights = values['gradient'].mean((2, 3), keepdim=True)
        cam = F.relu((weights * values['activation']).sum(1))[0].detach().cpu().numpy()
        cam = cv2.resize(cam, original.size)
        cam = (cam - cam.min()) / max(float(cam.max() - cam.min()), 1e-8)
        heat = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        rgb = np.asarray(original.convert('RGB'))
        overlay = (.6 * rgb + .4 * heat).astype(np.uint8)
        probs = torch.softmax(logits.detach(), dim=1)[0].cpu().tolist()
        return cls, probs, {'heatmap': png_url(heat), 'overlay': png_url(overlay)}
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
