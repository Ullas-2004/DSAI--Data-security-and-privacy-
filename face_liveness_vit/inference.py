import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=512, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x):
        return self.encoder(x)

class LivenessViT(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
        dim_feedforward: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size//patch_size)**2 + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        self.encoder = Encoder(d_model, nhead, num_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)
        cls_out = x[:, 0]
        return self.head(cls_out)


def load_model(checkpoint_path: str, device: str = 'cpu') -> Tuple[LivenessViT, dict]:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    cfg = checkpoint.get('cfg', {})
    model_cfg = cfg.get('model', {})
    
    model = LivenessViT(
        img_size=model_cfg.get('image_size', 224),
        patch_size=model_cfg.get('patch_size', 16),
        d_model=model_cfg.get('d_model', 128),
        nhead=model_cfg.get('nhead', 4),
        num_layers=model_cfg.get('num_layers', 2),
        num_classes=2,
        dropout=model_cfg.get('drop_rate', 0.0),
    )
    
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    return model, checkpoint


def predict(model: LivenessViT, image_path: str, threshold: float = 0.5, device: str = 'cpu') -> dict:
    """Predict if face is live or spoofed."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        spoof_prob = probs[0, 1].item()
    
    is_live = spoof_prob < threshold
    
    return {
        'is_live': is_live,
        'spoof_probability': spoof_prob,
        'live_probability': 1 - spoof_prob,
        'prediction': 'LIVE' if is_live else 'SPOOF'
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [checkpoint_path] [threshold]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else "model.pt"
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading model from {checkpoint_path}...")
    model, checkpoint = load_model(checkpoint_path, device)
    model = model.to(device)
    
    # Use threshold from checkpoint if available
    if 'threshold' in checkpoint:
        threshold = checkpoint['threshold']
        print(f"Using optimal threshold from checkpoint: {threshold:.4f}")
    
    print(f"Predicting on {image_path}...")
    result = predict(model, image_path, threshold, device)
    
    print(f"\nResult: {result['prediction']}")
    print(f"Live probability: {result['live_probability']:.4f}")
    print(f"Spoof probability: {result['spoof_probability']:.4f}")
