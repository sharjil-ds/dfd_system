import torch
import timm

# ===== Config =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"
EXPORT_PATH = "efficientnet_b0_deepfake.pt"

# ===== Define Model (same architecture as before) =====
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

# ===== Export TorchScript =====
example_input = torch.randn(1, 3, 224, 224).to(DEVICE)
traced_model = torch.jit.trace(model, example_input)
traced_model.save(EXPORT_PATH)

print(f"🚀 Exported model as TorchScript: {EXPORT_PATH}")
