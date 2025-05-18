import torch
model_data = torch.load("stored_models\segmentation\20250510160945_deeplabv3_resnet101.pth", map_location='cpu')
print(type(model_data))
