---
tags:
- face-liveness-detection
- anti-spoofing
- computer-vision
- pytorch
- vision-transformer
license: apache-2.0
---

# Face Liveness Detection Model

This model performs **face liveness detection** to distinguish between real faces and spoofing attempts (e.g., printed photos, video replay attacks, masks).

## Model Description

- **Architecture**: Vision Transformer (ViT) based encoder
- **Image Size**: 224x224
- **Patch Size**: 16x16
- **Embedding Dimension**: 128
- **Attention Heads**: 4
- **Transformer Layers**: 2
- **Classes**: Live (0) vs Spoof (1)
- **Optimal Threshold**: 0.1199

## Training Details

- **Dataset**: CelebA-Spoof
- **Batch Size**: 64
- **Learning Rate**: 0.0003
- **Weight Decay**: 0.05
- **Epochs**: 30
- **Optimizer**: AdamW
- **Mixed Precision**: True

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
import torchvision.transforms as transforms

# Download model
model_path = hf_hub_download(repo_id="hoangnguyenduc3009/face-liveness-vit", filename="model.pt")

# Load checkpoint
checkpoint = torch.load(model_path, map_location='cpu')

# Initialize model (you'll need the model class definition)
model = LivenessViT(
    img_size=224,
    patch_size=16,
    d_model=128,
    nhead=4,
    num_layers=2,
    num_classes=2
)
model.load_state_dict(checkpoint['model'])
model.eval()

# Prepare image
size = 224
transform = transforms.Compose([
    transforms.Resize(int(size * 1.14)),
    transforms.CenterCrop(size),
    # Ensure validation images are also grayscaled (keeps 3 channels)
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open("face.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Inference
with torch.no_grad():
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1)
    spoof_prob = probs[0, 1].item()
    
    # Use optimal threshold
    is_live = spoof_prob < 0.1199
    print(f"Live: {is_live}, Spoof probability: {spoof_prob:.4f}")
```

## Model Performance

The model was trained on CelebA-Spoof dataset with class balancing and data augmentation.
For detailed evaluation metrics, please refer to the training logs.

## Limitations

- Trained specifically on CelebA-Spoof dataset
- Performance may vary on different demographics or imaging conditions
- Should be used as part of a comprehensive security system, not as sole authentication

## Citation

If you use this model, please cite:

```bibtex
@misc{face-liveness-vit,
  author = {Your Name},
  title = {Face Liveness Detection with Vision Transformer},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/hoangnguyenduc3009/face-liveness-vit}}
}
```

## License

Apache 2.0
