# Face Recognition Security

Face recognition and liveness-authentication project for Data Security and Privacy coursework. The project includes notebook experiments, a Flask-style web app, template pages, Buffalo/InsightFace model integration, threshold sweep utilities, and a final report.

## Contents

- `buffalo_auth_core.py` - core face authentication and matching logic
- `video_auth_pipeline.py` - video authentication pipeline
- `video_auth_webapp.py` - web application entry point
- `run_dashboard.py` - dashboard launcher
- `templates/` - web UI templates
- `buffalo/` - small ONNX model assets used by the app
- `face_liveness_vit/` - liveness inference module and small model
- `.vendor/insightface/` - vendored InsightFace source used by the project
- `video/` - video experiment summary CSV files
- `*.ipynb` - training, inference, embedding, and pipeline notebooks
- `Report.pdf` - final project report

## Files Kept Local

The following are intentionally not uploaded to GitHub:

- `.deps/` dependency cache
- `app_data/` local auth database and registered face embeddings
- `checkpoints/` training checkpoints
- `buffalo/1k3d68.onnx` and `buffalo/w600k_r50.onnx`, because they are over GitHub's normal 100 MB file limit
- Python cache and notebook checkpoint files

This keeps the public repository clean and avoids publishing biometric/user data.

## Run

Install the required Python dependencies for the notebooks and web app, then launch:

```cmd
python video_auth_webapp.py
```
