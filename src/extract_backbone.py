import torch
import torch.nn as nn
import torchxrayvision as xrv
import os

# --- 1. Re-define the Imbalanced Model Architecture ---
# We need this to know the "shape" of the model we are loading.
class XRVTransferModel(nn.Module):
    def __init__(self, num_classes):
        super(XRVTransferModel, self).__init__()
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.backbone = model.features
        for name, param in self.backbone.named_parameters():
              if 'denseblock4' not in name:
                      param.requires_grad = False
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    def forward(self, x):
        features = self.backbone(x)
        pooled = self.pooling(features).view(features.size(0), -1)
        output = self.classifier(pooled)
        return output

# --- 2. Load Your 86% F1-Score Model ---
SOURCE_MODEL_PATH = 'best_imbalanced_model.pth'
BACKBONE_SAVE_PATH = 'finetuned_backbone_from_imbalanced_model.pth'

if not os.path.exists(SOURCE_MODEL_PATH):
    print(f"Error: Model file not found at {SOURCE_MODEL_PATH}")
    print("Please make sure 'best_imbalanced_model.pth' is in your project folder.")
else:
    # Initialize the full model structure
    # We must do this to load the weights
    full_model = XRVTransferModel(num_classes=4) 
    
    # Load the saved weights
    full_model.load_state_dict(torch.load(SOURCE_MODEL_PATH))
    
    # --- 3. Extract and Save the Backbone ---
    # This is the "brain" we want
    backbone_weights = full_model.backbone.state_dict()
    
    # Save these backbone-only weights to a new file
    torch.save(backbone_weights, BACKBONE_SAVE_PATH)
    
    print(f"Successfully extracted backbone from '{SOURCE_MODEL_PATH}'")
    print(f"Fine-tuned backbone saved to: '{BACKBONE_SAVE_PATH}'")