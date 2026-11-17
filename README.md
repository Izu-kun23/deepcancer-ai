# DeepCancer AI

### Explainable breast histopathology classification with deep learning

DeepCancer AI is a full-stack **research prototype** that brings image classification, visual explanations and reproducible model experiments into one workflow. It focuses on a single problem: classifying breast tissue microscopy images as **benign or malignant**.

The intelligence comes from models trained and run in **PyTorch**. It does not require an OpenAI, Gemini or other paid AI inference API key.

[Open training notebook in Google Colab](https://colab.research.google.com/github/Izu-kun23/deepcancer-ai/blob/main/notebooks/DeepCancer_Colab.ipynb) · [Training notebook source](notebooks/DeepCancer_Colab.ipynb) · [Technical documentation](docs/architecture.md)

> **Research only.** This is not a diagnostic device, a substitute for a pathologist, or a validated clinical system. The demonstration interface returns clearly labelled illustrative data until it is connected to a trained inference backend. Confidence scores are not calibrated clinical probabilities.

## What it does

1. Accepts a de-identified PNG or JPEG histopathology image.
2. Validates the file format and size before processing.
3. Runs a selected PyTorch model through a separate FastAPI service when connected.
4. Displays the model classification and confidence score.
5. Generates actual **Grad-CAM** maps and overlays from model gradients.
6. Saves local research history and exports JSON reports.
7. Supports dataset preparation, training, checkpoint selection and held-out evaluation from a self-contained Colab notebook or the command line.

The web interface also offers an explicitly labelled demonstration mode and a synthetic illustrative tissue image for exploring the workflow. Those demonstration outputs are fixed examples, never medical predictions.

## Dashboard design

The dashboard uses **Next.js, React, TypeScript, Tailwind CSS and shadcn/ui**. Component source is kept in `frontend/components/ui`, allowing the interface to be inspected and customised with the project. Cards, buttons, badges, selects, tabs, dialogs, inputs, tables, alerts and progress indicators share a consistent visual system.

The workspace contains **Image analysis**, **Analysis history**, **Model performance** and **Methodology** views. The design uses restrained green accents, readable typography, clear status labels and responsive layouts.

## Technology stack in detail

| Layer | Technology | How this project uses it |
| --- | --- | --- |
| Frontend framework | Next.js 16, React 19 | App Router application, page metadata and static frontend export. |
| Frontend language | TypeScript | Typed research results, API responses, component props and client-side state. |
| Styling | Tailwind CSS 4 | Responsive layouts, spacing, colour tokens and component styling through PostCSS. |
| UI system | shadcn/ui | Locally owned UI components for dashboard controls, cards, dialogs, tabs and data presentation. |
| Icons | Lucide React | Consistent vector icons for navigation, upload, model states and exports. |
| Backend | Python + FastAPI | Separate HTTP service for model loading, image inference and evaluation reports. |
| Validation | Pydantic | Typed prediction response and confidence validation; FastAPI validates multipart request inputs. |
| ASGI server | Uvicorn | Runs the Python API locally or in a Python-capable deployment environment. |
| Deep learning | PyTorch | Model definitions, gradient computation, weighted loss, optimisation, checkpoint saving and inference. |
| Transfer learning | Torchvision | ResNet50 and EfficientNet-B0 with optional ImageNet initialisation; shared image transforms. |
| Explainability | Grad-CAM | Computes activation maps using gradients at the final convolutional feature layer. |
| Image decoding | Pillow | Validates image content and dimensions, converts RGB and encodes explanation images. |
| Image processing | OpenCV | Resizes Grad-CAM maps and applies heatmap colour maps for overlays. |
| Numeric processing | NumPy | Image arrays, prediction thresholding and numeric processing. |
| Dataset engineering | Pandas | Dataset manifests, label distributions, patient grouping, experiment histories and comparison exports. |
| Evaluation | Scikit-learn | Patient-stratified splits, accuracy, precision, recall, F1, specificity inputs, ROC-AUC and plot displays. |
| Research charts | Matplotlib | ROC and precision-recall curves, confusion matrices, training loss and accuracy plots. |
| Database abstraction | SQLAlchemy | Persists prediction metadata through a database-independent ORM. |
| Database | PostgreSQL 16 + Psycopg | Configurable persistent database; SQLite is the local development fallback. |
| Training environment | Google Colab | GPU-enabled notebook that prepares real BreakHis data, trains models and exports research artifacts. |
| Packaging | Docker + Docker Compose | API image and optional PostgreSQL service with persistent database volume. |
| Tests | Pytest + FastAPI TestClient | Engineering tests for split integrity, models, Grad-CAM, API validation and persistence. |
| Source control | Git + GitHub | Public source, documentation and reproducible project setup. |

The frontend lockfile pins resolved JavaScript dependencies. Python dependency ranges are in `requirements.txt`. The frontend host does **not** execute PyTorch; the Python backend must run separately.

## Architecture

```mermaid
flowchart TD
    User[Researcher] --> UI[Next.js and shadcn dashboard]
    UI -->|Multipart image upload| API[FastAPI service]
    API --> Model[PyTorch classifier]
    Model --> CAM[Grad-CAM explanation]
    CAM --> API
    API --> DB[(PostgreSQL or local SQLite)]
    API -->|Classification and explanation| UI
    Dataset[BreakHis dataset] --> Split[Patient-separated data pipeline]
    Split --> Train[Training and validation]
    Train --> Weights[Selected checkpoint]
    Weights --> Model
    Split --> Evaluate[Held-out evaluation]
    Weights --> Evaluate
    Evaluate --> Reports[Metrics, charts and comparison CSV]
```

The API exposes `/health`, `/predict`, `/metrics` and FastAPI's interactive `/docs`. Inference accepts only supported architectures, PNG/JPEG content, approved magnification values and images within its size limits. Missing weights return a clear **503** response; the API never substitutes a fabricated prediction.

## Models

| Model | Implementation | Purpose |
| --- | --- | --- |
| Custom CNN | Three convolution/ReLU/max-pooling blocks, adaptive pooling, dropout and a two-class linear head | Establish a trainable baseline. |
| ResNet50 | Torchvision ResNet50 with a replaced two-class classification head | Compare transfer learning against the baseline. |
| EfficientNet-B0 | Torchvision EfficientNet-B0 with a two-class classifier | Compare a more compact transfer-learning architecture. |

Transfer models initialise with ImageNet weights and fine-tune all parameters at a lower learning rate. ImageNet weights are **not cancer-classification weights**. Every architecture requires training on the intended tissue dataset.

### Training configuration

- RGB inputs resized to **224 × 224** with consistent ImageNet normalisation.
- Training-only horizontal/vertical flips and random rotations.
- Class-weighted cross-entropy based on the training split.
- Adam optimiser; default learning rate `0.001` for CNN and `0.0001` for transfer models.
- Default batch size `32`, up to `25` epochs and early-stopping patience `5`.
- Best checkpoint selected by **validation loss**.
- Checkpoints include architecture, class order, epoch, seed, input size and dataset-manifest hash.

## Dataset and leakage prevention

[BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-BreakHis/) provides real labelled breast histopathology images at 40×, 100×, 200× and 400× magnification. Obtain the dataset from its maintainers and comply with their usage and citation conditions. Images are not redistributed in this repository.

The data pipeline extracts the patient identifier from each filename, groups all associated images, and approximates a **70% training / 15% validation / 15% test split of patients** with stratification by binary class. Integer patient counts determine the final proportions.

Audits reject overlapping patients, duplicate paths, inconsistent patient labels, missing classes and identical image content spanning splits. Augmentation and class weights use training data only.

Do not replace this with a random image-level split: images from one patient can otherwise leak into both training and test sets.

## Run in Google Colab

1. Open the **Colab link** at the top of this README, or upload `notebooks/DeepCancer_Colab.ipynb` to Colab.
2. Choose **Runtime → Change runtime type → T4 GPU**, when available.
3. Run the setup cell. It includes the project training source, so no source ZIP upload is required.
4. Run dataset preparation. The official archive is approximately 4 GB.
5. Review the split summary, then run training and evaluation.
6. Save each downloaded checkpoint/report archive. Colab runtime files are temporary.

The notebook retains Colab's preinstalled CUDA PyTorch and installs missing analysis packages as needed. Free GPU availability and runtime duration are controlled by Colab. No paid AI inference API is required.

## Local setup

Use Python 3.11+ and Node.js 20.9+. Use an appropriate PyTorch build for your CPU, CUDA or Apple Silicon environment.

```bash
git clone https://github.com/Izu-kun23/deepcancer-ai.git
cd deepcancer-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`.

### Frontend

```bash
cd frontend
npm ci
npm run dev
```

Open `http://localhost:3000`.

### Backend

From the repository root in another terminal:

```bash
source .venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

API documentation: `http://localhost:8000/docs`.

For hosted inference, deploy the API behind HTTPS and enter its base URL in the dashboard's **Inference settings**. The frontend requires HTTPS for connected services. For localhost integration, use a local HTTPS reverse proxy or deliberately allow localhost HTTP in a development-only change.

### Prepare, train and evaluate

```bash
python -m ml.data --root data/BreaKHis_v1 --output outputs/data
python -m ml.train --model cnn
python -m ml.train --model resnet50
python -m ml.train --model efficientnet_b0
python -m ml.evaluate --weights ml/weights/cnn.pt
python -m ml.evaluate --weights ml/weights/resnet50.pt
python -m ml.evaluate --weights ml/weights/efficientnet_b0.pt
```

`--scratch` disables pretrained-weight downloads for engineering smoke tests. It does not produce a useful cancer classifier by itself.

## Evaluation and explainability

Evaluation reports **accuracy, precision, recall/sensitivity, specificity, F1 and ROC-AUC**, plus a confusion matrix, ROC curve, precision-recall curve, training curves, inference timing and weight size. Models are compared on the same manifest and test patients. The decision threshold is fixed at `0.5`; test data must not be used to tune it.

Outputs are written to `outputs/evaluation/metrics.json`, `comparison.csv` and PNG charts. The dashboard reads measured metrics from a connected backend and otherwise displays an unavailable state. No benchmark score is invented or inferred from demonstration data.

Current metrics are **image-level**. Patient-level aggregation, uncertainty intervals, calibration, subgroup analysis and external validation are future research requirements. Grad-CAM highlights regions contributing to a model score; it does not prove correctness or delineate a tumour.

## Storage and deployment

| Setting | Purpose | Default |
| --- | --- | --- |
| `DATABASE_URL` | SQLAlchemy database connection | Local SQLite file |
| `ALLOWED_ORIGINS` | Comma-separated permitted frontend origins | `http://localhost:3000` |
| `WEIGHTS_DIR` | Directory containing architecture `.pt` files | `ml/weights` |
| `METRICS_FILE` | Measured evaluation report | `outputs/evaluation/metrics.json` |

For PostgreSQL, use `postgresql+psycopg://USER:PASSWORD@HOST:5432/deepcancer`. Use environment variables for real credentials; do not commit them. `.env.example` contains placeholders only.

```bash
export POSTGRES_PASSWORD='choose-a-local-development-password'
docker compose up --build
```

The Compose configuration binds the API to localhost and mounts weights/reports read-only. The frontend can be statically hosted; the Python API needs a host capable of running PyTorch. The existing hosted workspace is separate from this public source repository, and its access settings are independent.

The API stores result metadata, not uploaded images or original filenames. Browser history is local to that browser. This prototype does not implement multi-user authentication, tenant isolation, rate limiting, production migrations or clinical data governance. Keep the backend private and add appropriate controls before any broader deployment. CORS is not authentication. Restart the API after replacing cached weights.

## Project structure

| Path | Contents |
| --- | --- |
| `frontend/app` | Next.js dashboard and global styling |
| `frontend/components/ui` | shadcn/ui component source |
| `frontend/lib` | Shared frontend helpers |
| `ml/data.py` | Filename parsing, manifests and patient-split audits |
| `ml/models.py` | Three neural-network architectures |
| `ml/preprocessing.py` | Dataset loading and consistent image transforms |
| `ml/train.py` | Training, validation, early stopping and checkpoints |
| `ml/evaluate.py` | Held-out metrics and research charts |
| `ml/gradcam.py` | Gradient-based heatmap and overlay generation |
| `backend/main.py` | FastAPI routes, model loading and metadata persistence |
| `notebooks` | Self-contained Colab training notebook |
| `tests` | Engineering verification |
| `docs` | Architecture, limitations and verification notes |
| `Dockerfile`, `compose.yaml` | API container and PostgreSQL setup |

## Verification and project status

The initial engineering suite passed **8 tests**, including forward passes for all three architectures, split/leakage checks, Grad-CAM hook cleanup, malformed upload rejection, missing-weight behaviour and inference/persistence with synthetic test weights. A separate synthetic smoke run completed preparation, training, checkpoint reload, evaluation and chart generation. These checks are **not cancer-classification performance evidence**.

A real BreakHis preparation run in Colab loaded **7,909 images** and produced 5,303 training, 1,120 validation and 1,486 test images using patient-separated groups. A GPU training workflow was started. Final trained-model metrics and weights are **not published here until independently retrieved and verified**.

Dataset images, trained weights, secrets, local database files, dependency folders, build output and hosting-specific identity files are excluded from this public source repository. PostgreSQL/Docker deployment and external Python-host deployment remain environment-specific validation tasks.

```bash
pytest -q
cd frontend
npm run build
```

## Research references

- Spanhol, F. A., Oliveira, L. S., Petitjean, C., & Heutte, L. (2016). *A Dataset for Breast Cancer Histopathological Image Classification*. IEEE Transactions on Biomedical Engineering, 63(7), 1455–1462. [Dataset and citation](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-BreakHis/).
- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. [Original paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf).
- [PyTorch transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).
- [shadcn/ui documentation](https://ui.shadcn.com/docs).

**Built for transparent, reproducible research. Not for clinical diagnosis.**
