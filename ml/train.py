import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from ml.data import audit
from ml.models import build
from ml.preprocessing import TissueDataset

def train(a):
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    df = pd.read_csv(a.manifest)
    audit(df)
    frames = {s: df[df.split == s] for s in ['train', 'val']}
    loaders = {s: DataLoader(TissueDataset(f, s == 'train'), batch_size=a.batch_size, shuffle=s == 'train', num_workers=0) for s, f in frames.items()}
    model, _ = build(a.model, pretrained=not a.scratch and a.model != 'cnn')
    model.to(device)
    counts = frames['train'].label.value_counts().sort_index().to_numpy()
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr or (.001 if a.model == 'cnn' else .0001))
    best, stale, history = float('inf'), 0, []
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(a.epochs):
        log = {'epoch': epoch + 1}
        for split, loader in loaders.items():
            model.train(split == 'train')
            loss_sum, correct, total = 0, 0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(split == 'train'):
                    logits = model(x)
                    loss = loss_fn(logits, y)
                    if split == 'train':
                        loss.backward()
                        optimizer.step()
                loss_sum += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                total += len(y)
            log[split + '_loss'] = loss_sum / total
            log[split + '_accuracy'] = correct / total
        history.append(log)
        print(json.dumps(log), flush=True)
        if log['val_loss'] < best:
            best, stale = log['val_loss'], 0
            torch.save({'state_dict': model.state_dict(), 'architecture': a.model, 'classes': ['benign', 'malignant'],
                        'epoch': epoch + 1, 'seed': a.seed, 'input_size': 224,
                        'manifest_sha256': hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()}, out / (a.model + '.pt'))
        else:
            stale += 1
        (out / (a.model + '-history.json')).write_text(json.dumps(history, indent=2))
        if stale >= a.patience:
            break

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default='outputs/data/manifest.csv')
    p.add_argument('--model', choices=['cnn', 'resnet50', 'efficientnet_b0'], default='cnn')
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='ml/weights')
    p.add_argument('--scratch', action='store_true', help='Skip downloading ImageNet weights (smoke tests only)')
    train(p.parse_args())
