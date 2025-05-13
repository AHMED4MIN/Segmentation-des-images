# model_check.py
import torch
from app import UNet  # Import your UNet from app.py

def check_model(model_path):
    # 1. Instantiate fresh model
    model = UNet()
    print("\nCurrent model architecture keys:")
    for k in model.state_dict().keys():
        print(f"- {k}")

    # 2. Load saved model weights
    try:
        state_dict = torch.load(model_path)
        print("\nSaved model keys:")
        for k in state_dict.keys():
            print(f"- {k}")
            
        # 3. Try loading weights
        model.load_state_dict(state_dict)
        print("\n✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"\n❌ Loading failed: {str(e)}")
        print("\n💡 Comparison:")
        missing = [k for k in model.state_dict().keys() if k not in state_dict]
        extra = [k for k in state_dict.keys() if k not in model.state_dict()]
        
        if missing:
            print(f"Missing keys in saved model: {missing}")
        if extra:
            print(f"Extra keys in saved model: {extra}")

if __name__ == "__main__":
    # Update this path to match your actual model file
    check_model("stored_models/segmentation/20250508154250_deeplabv3_resnet50.pth")