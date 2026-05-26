# Face Recognition Security System

Face recognition and liveness-authentication project for Data Security and Privacy coursework. The project combines face embeddings, video authentication, liveness checks, threshold evaluation, notebook experiments, and a Flask-style web interface while keeping biometric data and private runtime files out of the public repository.

## Project Highlights

- Implements face authentication using Buffalo/InsightFace-style embeddings.
- Adds liveness-aware video authentication to reduce spoofing risk.
- Includes notebook experiments for inference, embedding comparison, liveness checks, and threshold sweeps.
- Provides a web application with registration, authentication, audit, results, and unknown-attempts pages.
- Separates public source code from private biometric data and large model checkpoints.

## Tech Stack

| Area | Tools |
| --- | --- |
| Programming | Python |
| Computer Vision | OpenCV, InsightFace/Buffalo model integration |
| Deep Learning | ONNX model assets, ViT-style liveness module |
| Web App | Flask-style Python app, HTML templates |
| Evaluation | Threshold sweep, embedding consistency checks, real/spoof video summaries |
| Privacy | Local-only biometric database and registered embeddings |

## Repository Structure

```text
.
|-- buffalo_auth_core.py              # Core face authentication and matching logic
|-- video_auth_pipeline.py            # Video authentication pipeline
|-- video_auth_webapp.py              # Web application entry point
|-- run_dashboard.py                  # Dashboard launcher
|-- threshold_sweep_eval.py           # Threshold evaluation utility
|-- exact_pair_sweep.py               # Pairwise comparison utility
|-- templates/                        # Web UI templates
|-- face_liveness_vit/                # Liveness inference module
|-- video/                            # Experiment summary CSV files
|-- *.ipynb                           # Training, inference, and evaluation notebooks
|-- Report.pdf                        # Final project report
```

## Web App Flow

The application is organized around a practical authentication workflow:

1. Register a user face locally.
2. Capture or upload authentication video/image evidence.
3. Generate embeddings and compare against enrolled identity.
4. Run liveness/spoof checks.
5. Display pass/fail result and keep audit information for review.

## How to Run

Install the required Python dependencies for the notebooks and web app, then launch:

```cmd
python video_auth_webapp.py
```

Depending on your local environment, you may also use:

```cmd
python run_dashboard.py
```

## Evaluation Artifacts

The `video/` folder includes summary CSV files for real and spoof clip groupings. These files support threshold tuning, embedding comparison, and evaluation of authentication behavior across multiple identities.

## Privacy and Large File Policy

The following files are intentionally not uploaded:

- `.deps/` dependency cache
- `app_data/` local authentication database and registered face embeddings
- `checkpoints/` training checkpoints
- large Buffalo ONNX files over GitHub's normal file-size limit
- Python cache and notebook checkpoint files

This keeps the repository safe for public review and avoids exposing biometric user data.

## Report

The final project report is included as `Report.pdf`.
