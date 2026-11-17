# Architecture and implementation notes

DeepCancer AI separates the browser UI from Python model execution. Next.js statically exports a React/TypeScript dashboard built from shadcn/ui components. The layout follows the dashboard-01 block pattern: responsive sidebar, navigation, summary cards, analysis controls and a history table. No sample revenue charts or fabricated model metrics are included.

## Frontend

`frontend/app/page.tsx` owns the typed research workflow. React state tracks the selected file, architecture, magnification, current result and history. Browser storage retains result metadata and a custom inference URL. It never retains image bytes or Grad-CAM images in history. The UI uses shadcn Card, Button, Badge, Sidebar, Select, Dialog, Tabs, Input, Table, Alert and Progress components. Their source lives in the repository, with Radix primitives supplying keyboard and focus behaviour for interactive controls.

The dashboard validates PNG/JPEG type, decodability, minimum dimensions, pixel count and size. The Python API repeats validation independently. HTML file extension or MIME claims alone are insufficient. The original image uses a temporary object URL revoked when replaced.

A configured HTTPS base URL enables multipart `/predict` calls. Demonstration mode has an explicit fixed-output label. It never derives a cancer score from pixels and never fabricates a heatmap. The performance page loads `/metrics` and handles loading, unavailable and error states.

## ML pipeline

`ml/data.py` parses BreakHis filenames and stratifies patient groups into train, validation and test splits. It checks overlap, labels and exact file-content duplicates before saving a manifest. The recorded summary determines actual integer split sizes. File bytes are hashed; overlapping content across splits fails preparation.

`ml/models.py` supplies a three-block custom CNN, ResNet50 and EfficientNet-B0. Transfer models use ImageNet weights for initialisation; those weights are not trained cancer classifiers. Training fine-tunes all parameters, uses training-class weights and augmentation, and selects a checkpoint on validation loss. Test data is not used for selection.

`ml/evaluate.py` verifies the exact manifest hash recorded during training, computes image-level held-out metrics, exports charts and creates a comparison CSV. Threshold is fixed at 0.5. Metrics are not patient-level confidence intervals or evidence of clinical effectiveness.

`ml/gradcam.py` captures final-layer activations and gradients, computes nonnegative channel-weighted maps, resizes them and exports PNG data URLs. Hook cleanup runs in `finally`. The API serialises Grad-CAM operations with a process-local lock to avoid mixing hooks between requests.

## Backend and storage

FastAPI exposes `/health`, `/predict`, `/metrics` and `/docs`. Pydantic constrains response confidence to [0,1]. Checkpoints load on CPU with `weights_only=True`, architecture/class metadata is verified, and missing weights return HTTP 503. Model instances are cached; restart the process after weight replacement.

SQLAlchemy stores prediction UUID, class, confidence, architecture, magnification and timestamp. Uploaded images and original filenames are not stored. PostgreSQL is configured with `DATABASE_URL`; SQLite is the local fallback. Automatic table creation is a prototype convenience, not a migration system.

## Deployment boundary

Static hosting serves the frontend; a separate Python/PyTorch host serves the backend. CORS explicitly allows configured frontend origins but is not access authentication. This version lacks multi-user auth, rate limiting, tenant isolation, clinical governance and production migrations. Keep the backend private and use de-identified research data.

Dataset files, weights, environment secrets, databases, dependency folders and generated deployment output are excluded from GitHub source. Google Colab runtime artifacts are temporary; the notebook downloads checkpoint/report archives after each experiment.

## UI provenance

UI primitive source was retrieved from the official shadcn/ui `new-york` registry. The dashboard is adapted to this application's research workflow, following the dashboard-01 layout pattern rather than including its unrelated sample business data.
