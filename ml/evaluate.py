import argparse
import hashlib
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from ml.data import audit
from ml.models import build, NAMES
from ml.preprocessing import TissueDataset

def evaluate(a):
    frame = pd.read_csv(a.manifest)
    audit(frame)
    checkpoint = torch.load(a.weights, map_location='cpu', weights_only=True)
    if checkpoint['manifest_sha256'] != hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest():
        raise ValueError('Manifest differs from training split')
    model, _ = build(checkpoint['architecture'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    loader = DataLoader(TissueDataset(frame[frame.split == 'test']), batch_size=32)
    labels, scores = [], []
    started = time.perf_counter()
    with torch.no_grad():
        for x, y in loader:
            scores.extend(model(x).softmax(1)[:, 1].tolist())
            labels.extend(y.tolist())
    elapsed = time.perf_counter() - started
    pred = (np.array(scores) >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    display = next(k for k, v in NAMES.items() if v == checkpoint['architecture'])
    result = dict(model=display, accuracy=accuracy_score(labels, pred), precision=precision_score(labels, pred, zero_division=0),
                  recall=recall_score(labels, pred, zero_division=0), specificity=float(tn / (tn + fp)),
                  f1=f1_score(labels, pred, zero_division=0), roc_auc=roc_auc_score(labels, scores),
                  confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]], samples=len(labels),
                  patients=int(frame[frame.split == 'test'].patient.nunique()), threshold=.5,
                  milliseconds_per_image=elapsed * 1000 / len(labels), weights_bytes=Path(a.weights).stat().st_size,
                  manifest_sha256=checkpoint['manifest_sha256'])
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, plotter, values in [('roc', RocCurveDisplay, scores), ('precision-recall', PrecisionRecallDisplay, scores), ('confusion', ConfusionMatrixDisplay, pred)]:
        plotter.from_predictions(labels, values)
        plt.savefig(out / (checkpoint['architecture'] + '-' + name + '.png'), bbox_inches='tight')
        plt.close()
    hist = Path(a.weights).with_name(checkpoint['architecture'] + '-history.json')
    if hist.exists():
        h = pd.read_json(hist)
        for key in ['loss', 'accuracy']:
            h.plot(x='epoch', y=['train_' + key, 'val_' + key])
            plt.savefig(out / (checkpoint['architecture'] + '-' + key + '.png'), bbox_inches='tight')
            plt.close()
    file = out / 'metrics.json'
    existing = json.loads(file.read_text()) if file.exists() else {'models': []}
    existing['models'] = [m for m in existing['models'] if m['model'] != display] + [result]
    if len({m['manifest_sha256'] for m in existing['models']}) != 1:
        raise ValueError('Cannot compare models evaluated on different manifests')
    file.write_text(json.dumps(existing, indent=2))
    pd.DataFrame(existing['models']).drop(columns=['confusion_matrix']).to_csv(out / 'comparison.csv', index=False)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--manifest', default='outputs/data/manifest.csv')
    p.add_argument('--output', default='outputs/evaluation')
    evaluate(p.parse_args())
