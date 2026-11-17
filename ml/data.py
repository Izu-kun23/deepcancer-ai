"""BreakHis manifest creation with patient-stratified, leakage-free splits."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PATTERN = re.compile(r'^SOB_([BM])_([A-Z]+)-(\d+-[^-]+)-(40|100|200|400)-(\d+)\.png$', re.I)

def parse_filename(path):
    m = PATTERN.match(Path(path).name)
    if not m:
        raise ValueError(f'Not a BreakHis filename: {Path(path).name}')
    label, subtype, patient, mag, _ = m.groups()
    return dict(path=str(Path(path).resolve()), label=int(label.upper() == 'M'),
                subtype=subtype.upper(), patient=patient, magnification=int(mag))

def audit(df):
    required = {'path', 'label', 'patient', 'split'}
    if not required.issubset(df.columns):
        raise ValueError('Manifest is missing columns')
    if df.empty or df.path.duplicated().any():
        raise ValueError('Empty manifest or duplicate paths')
    if not df.label.isin([0, 1]).all():
        raise ValueError('Labels must be 0 or 1')
    if (df.groupby('patient').label.nunique() > 1).any():
        raise ValueError('Conflicting labels for a patient')
    if (df.groupby('patient').split.nunique() > 1).any():
        raise ValueError('Patient leakage detected')
    for split in ['train', 'val', 'test']:
        if set(df[df.split == split].label) != {0, 1}:
            raise ValueError(f'{split} must contain both classes')
    return True

def prepare(root, output, seed=42):
    files = sorted(Path(root).rglob('SOB_*.png'))
    if not files:
        raise ValueError('No BreakHis PNG images found')
    df = pd.DataFrame([parse_filename(p) for p in files])
    if (df.groupby('patient').label.nunique() > 1).any():
        raise ValueError('Conflicting patient labels')
    patients = df[['patient', 'label']].drop_duplicates()
    train, rest = train_test_split(patients, test_size=.30, random_state=seed, stratify=patients.label)
    val, test = train_test_split(rest, test_size=.50, random_state=seed, stratify=rest.label)
    mapping = {p: s for s, frame in [('train', train), ('val', val), ('test', test)] for p in frame.patient}
    df['split'] = df.patient.map(mapping)
    audit(df)
    # Catch exact duplicated image content even under different filenames/patients.
    df['sha256'] = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in df.path]
    if (df.groupby('sha256').split.nunique() > 1).any():
        raise ValueError('Identical image content spans splits; resolve before training')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'manifest.csv', index=False)
    summary = {'seed': seed, 'patients': len(patients), 'images': len(df),
               'counts': df.groupby(['split', 'label']).size().unstack(fill_value=0).to_dict(),
               'patient_counts': df.groupby('split').patient.nunique().to_dict()}
    (out / 'split-summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return df

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--output', default='outputs/data')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    prepare(a.root, a.output, a.seed)
