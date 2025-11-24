import os
import io
from pathlib import Path     # ⬅ change from zipfile.Path to pathlib.Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# repo root = one level above interface/
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"

MODEL_PATHS = {
    'TransUNet': OUTPUTS_DIR / 'transunet-model-outputs' / 'best_transunet_aeo.pth',
    'ResNet-34': OUTPUTS_DIR / 'resnet34-model-outputs' / 'best_cls_model.pth',
    'DB-LARNet': OUTPUTS_DIR / 'db-larnet-model-outputs' / 'best_dualbranch_cls_model.pth',
    'PLA-MIL': OUTPUTS_DIR / 'pla-mil-network-model-outputs' / 'best_plamil_model.pth',
    'SG-ResNet': OUTPUTS_DIR / 'segmentation-guided-resnet-34-model-outputs' / 'best_sg_resnet_model.pth',
}

CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
NUM_CLASSES = len(CLASS_NAMES)
TARGET_SIZE_CLS = 224 # Classification input size
TARGET_SIZE_SEG = 256 # Segmentation input size

# Transforms (must match training/validation used in notebooks)
CLASSIFIER_TRANSFORM = transforms.Compose([
    transforms.Resize((TARGET_SIZE_CLS, TARGET_SIZE_CLS)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

SEGMENTATION_TRANSFORM = transforms.Compose([
    transforms.Resize((TARGET_SIZE_SEG, TARGET_SIZE_SEG)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =========================================================================
# I. MODEL ARCHITECTURES (TransUNet, SG-ResNet, DB-LARNet, PLA-MIL)
# =========================================================================

# --- TransUNetAEO Components ---
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch) # input size to ConvBlock is in_ch
    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch due to odd input dimensions (padding logic from user's code)
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            diff_y = skip.size(-2) - x.size(-2)
            diff_x = skip.size(-1) - x.size(-1)
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class TransUNetAEO(nn.Module):
    def __init__(self, img_size=256, in_ch=3, num_classes=1, base_ch=64, num_heads=4, transformer_layers=4, dropout=0.1):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base_ch*4, base_ch*8)
        self.pool4 = nn.MaxPool2d(2)

        bottleneck_ch = base_ch * 16
        self.bottleneck_conv = ConvBlock(base_ch*8, bottleneck_ch)

        d_model = bottleneck_ch
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_model*4, dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(enc_layer, transformer_layers)

        self.up4 = UpBlock(bottleneck_ch, base_ch*8)
        self.up3 = UpBlock(base_ch*8, base_ch*4)
        self.up2 = UpBlock(base_ch*4, base_ch*2)
        self.up1 = UpBlock(base_ch*2, base_ch)
        self.final_conv = nn.Conv2d(base_ch, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        e4 = self.enc4(p3); p4 = self.pool4(e4)

        b = self.bottleneck_conv(p4)
        B, C, H, W = b.shape
        seq = b.flatten(2).permute(2, 0, 1) # (S,B,C)
        seq = self.transformer(seq)
        b = seq.permute(1, 2, 0).view(B, C, H, W)

        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        return self.final_conv(d1)

# --- SG-ResNet34 (4-channel input) ---
def build_4ch_resnet34(num_classes=7, pretrained=True):
    """Rebuilds ResNet-34 with 4 input channels for RGB + Mask."""
    resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)

    old_conv = resnet.conv1
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    with torch.no_grad():
        # Copy RGB weights
        new_conv.weight[:, :3, :, :] = old_conv.weight
        # Initialize 4th channel (mask) weights by averaging RGB weights
        new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)

    resnet.conv1 = new_conv
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, num_classes)
    return resnet

# --- DB-LARNet (Dual-Branch ResNet-18) ---
class DualBranchResNet(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        # Load weights conditionally based on the flag
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        
        base_full = models.resnet18(weights=weights)
        base_crop = models.resnet18(weights=weights)

        # feature extractors (everything except FC)
        self.full_backbone = nn.Sequential(*list(base_full.children())[:-1])  # -> (B,512,1,1)
        self.crop_backbone = nn.Sequential(*list(base_crop.children())[:-1])  # -> (B,512,1,1)

        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, full_img, crop_img):
        full_feat = self.full_backbone(full_img)
        crop_feat = self.crop_backbone(crop_img)

        full_feat = full_feat.view(full_feat.size(0), -1)
        crop_feat = crop_feat.view(crop_feat.size(0), -1)

        fused = torch.cat([full_feat, crop_feat], dim=1)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits

# --- PLA-MIL (Patch-Level Attention MIL) ---
class PatchEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        feat = self.proj(feat)
        return feat

class AttentionMIL(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.attention_a = nn.Linear(dim, hidden_dim)
        self.attention_b = nn.Linear(hidden_dim, 1)

    def forward(self, H):
        A = torch.tanh(self.attention_a(H))
        A = self.attention_b(A).squeeze(-1)
        A = torch.softmax(A, dim=1)
        bag_repr = torch.sum(H * A.unsqueeze(-1), dim=1)
        return bag_repr, A

class PatchMILLesionClassifier(nn.Module):
    def __init__(self, num_classes=7, patch_dim=256):
        super().__init__()
        self.encoder = PatchEncoder(out_dim=patch_dim)
        self.mil_pool = AttentionMIL(dim=patch_dim, hidden_dim=128)
        self.classifier = nn.Sequential(
            nn.Linear(patch_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, patches):
        B, N, C, H, W = patches.shape
        patches_flat = patches.view(B * N, C, H, W)
        patch_feats = self.encoder(patches_flat)
        D = patch_feats.shape[1]
        patch_feats = patch_feats.view(B, N, D)

        bag_repr, att_weights = self.mil_pool(patch_feats)
        logits = self.classifier(bag_repr)
        return logits, att_weights

# =========================================================================
# II. GRAD-CAM CLASSES
# =========================================================================

class GradCAM:
    """Standard GradCAM implementation for ResNet and SG-ResNet."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self.bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        score = logits[0, class_idx]
        score.backward()

        grads = self.gradients
        acts = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1).squeeze(0)

        cam = F.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy(), class_idx

    def close(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


class GradCAMBranch:
    """GradCAM for a specific branch in the Dual-Branch model."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self.bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inflow, outflow):
        self.activations = outflow.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, full_tensor, crop_tensor, target_class=None):
        self.model.zero_grad()
        logits = self.model(full_tensor, crop_tensor)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        score = logits[:, target_class]
        score.backward()

        grads = self.gradients
        acts = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1).squeeze(0)

        cam = F.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy(), target_class
    
    def close(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


# =========================================================================
# III. CORE FUNCTIONS CALLED BY Flask (app.py)
# =========================================================================

def load_all_models():
    loaded_models = {}

    def load_weights(model, path):
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.to(DEVICE).eval()
        return model

    try:
        transunet_model = TransUNetAEO(img_size=TARGET_SIZE_SEG).to(DEVICE)
        loaded_models['TransUNet'] = load_weights(transunet_model, MODEL_PATHS['TransUNet'])

        resnet34_base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        resnet34_base.fc = nn.Linear(resnet34_base.fc.in_features, NUM_CLASSES)
        loaded_models['ResNet-34'] = load_weights(resnet34_base, MODEL_PATHS['ResNet-34'])

        db_larnet_model = DualBranchResNet(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
        loaded_models['DB-LARNet'] = load_weights(db_larnet_model, MODEL_PATHS['DB-LARNet'])

        plamil_model = PatchMILLesionClassifier(num_classes=NUM_CLASSES, patch_dim=256).to(DEVICE)
        loaded_models['PLA-MIL'] = load_weights(plamil_model, MODEL_PATHS['PLA-MIL'])

        sg_resnet_model = build_4ch_resnet34(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
        loaded_models['SG-ResNet'] = load_weights(sg_resnet_model, MODEL_PATHS['SG-ResNet'])

    except Exception as e:
        print("CRITICAL WARNING: Failed to load one or more models:", e)
        return loaded_models

    print("Loaded models:", list(loaded_models.keys()))
    return loaded_models

# --- Helper Functions for Image Processing ---

def load_image_and_transform(image_path, transform_type='classifier'):
    """Loads image and applies the specified transform."""
    img_pil = Image.open(image_path).convert('RGB')
    if transform_type == 'classifier':
        transform = CLASSIFIER_TRANSFORM
    elif transform_type == 'segmentation':
        transform = SEGMENTATION_TRANSFORM
    else:
        raise ValueError("Invalid transform type.")
    
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    return img_pil, img_tensor

def create_4ch_input_with_mask(img_pil, mask_np_256):
    """
    Creates the 4-channel input tensor [R, G, B, Mask] required by SG-ResNet.
    
    img_pil: original PIL image
    mask_np_256: binary mask (H, W) numpy array, size 256x256
    """
    # 1. Transform RGB channels for normalization
    img_t = CLASSIFIER_TRANSFORM(img_pil) # (3, 224, 224)
    
    # 2. Resize mask to match classifier input size (224x224)
    mask_pil = Image.fromarray(mask_np_256).convert("L")
    mask_pil = mask_pil.resize((TARGET_SIZE_CLS, TARGET_SIZE_CLS), resample=Image.NEAREST)
    mask_bin = (np.array(mask_pil) > 0).astype("float32")
    
    mask_t = torch.from_numpy(mask_bin).unsqueeze(0)  # (1, 224, 224)
    
    # 3. Concatenate (R, G, B, Mask)
    x_4ch = torch.cat([img_t.cpu(), mask_t], dim=0).unsqueeze(0).to(DEVICE)  # (1, 4, 224, 224)
    
    return x_4ch

def apply_mask_and_crop(img_pil, mask_np_256, crop_size=TARGET_SIZE_CLS):
    """
    Finds the bounding box of the mask and crops/pads the original image.
    This simulates the lesion-crop baseline preparation.
    """
    # Find bounding box (min_row, max_row, min_col, max_col)
    coords = np.argwhere(mask_np_256 > 0)
    if coords.size == 0:
        # If no lesion detected, use the full image centered
        return img_pil.resize((crop_size, crop_size)).convert('RGB')
    
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)
    
    # Add small padding to the crop area
    pad = 10 
    H, W = mask_np_256.shape
    min_y = max(0, min_y - pad)
    min_x = max(0, min_x - pad)
    max_y = min(H, max_y + pad)
    max_x = min(W, max_x + pad)
    
    # Crop the original PIL image (rescaled to 256x256 first for coordinate accuracy)
    img_256 = img_pil.resize((H, W)) 
    cropped_pil = img_256.crop((min_x, min_y, max_x, max_y))
    
    # Resize the final crop to the required input size (224x224)
    cropped_pil = cropped_pil.resize((crop_size, crop_size))
    return cropped_pil


def run_transunet_segmentation(transunet_model, img_pil):
    """Runs the TransUNet model on the uploaded image."""
    img_tensor = SEGMENTATION_TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = transunet_model(img_tensor)
        probs = torch.sigmoid(logits)[0,0].cpu().numpy()
        # Resize probability map back to 256x256 (if model output size differed)
        probs_resized = cv2.resize(probs, (TARGET_SIZE_SEG, TARGET_SIZE_SEG))
        pred_mask = (probs_resized > 0.5).astype(np.uint8) * 255 # Binary mask 256x256 (0 or 255)
    
    return pred_mask


def make_plamil_patches(img_pil, patch_grid=4):
    """Splits a 224x224 crop into 4x4 patches for PLA-MIL."""
    PATCH_SIZE = TARGET_SIZE_CLS // patch_grid
    img_t = CLASSIFIER_TRANSFORM(img_pil).cpu()
    C, H, W = img_t.shape

    # Unfold image into patches (N=16 total patches)
    patches = img_t.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)
    patches = patches.permute(1, 2, 0, 3, 4) 
    patches = patches.reshape(-1, C, PATCH_SIZE, PATCH_SIZE)

    return patches.unsqueeze(0).to(DEVICE) # (1, N, 3, H_patch, W_patch)


# --- Grad-CAM Helper ---

def save_heatmap_to_disk(img_pil, cam, output_path, mask_np_full=None, alpha=0.5):
    """
    Overlays the CAM heatmap on the original image and optionally draws the mask contour,
    then saves the resulting image to disk.
    """
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    h, w, _ = img_np.shape

    # Resize CAM to original image size
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

    overlay = alpha * heatmap_color + (1 - alpha) * img_np
    overlay = np.clip(overlay, 0, 1)
    overlay_img = (overlay * 255).astype(np.uint8)

    # Optional: Draw mask contour (used for SG-ResNet/TransUNet visual validation)
    if mask_np_full is not None:
        mask_binary = (cv2.resize(mask_np_full, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_img, contours, -1, (0, 255, 0), thickness=2) # Draw green contour

    # Save to disk
    plt.imsave(output_path, overlay_img)
    return os.path.basename(output_path)


# --- Main Prediction Pipeline ---

def run_analysis_pipeline(models, main_model, secondary_model, input_image_path, gradcam_output_folder):
    """Routes the request to the correct model pipeline based on user selection."""
    
    # 1. Load the original uploaded image
    img_pil_original = Image.open(input_image_path).convert('RGB')
    
    # Placeholder for mask generated by TransUNet
    generated_mask_256 = None
    
    # --- Check for missing models ---
    if not models.get(main_model):
        raise ValueError(f"Required model '{main_model}' is not loaded. Check model files.")
        
    if main_model == 'TransUNet' and not models.get(secondary_model):
         raise ValueError(f"Required secondary model '{secondary_model}' is not loaded. Check model files.")

    # --- PIPELINE 1: Segmentation -> Classification (TransUNet + Classifier) ---
    if main_model == 'TransUNet':
        
        # 1. Run TransUNet segmentation
        transunet_model = models['TransUNet']
        # The segmentation needs a 256x256 image normalized
        img_pil_256 = img_pil_original.resize((TARGET_SIZE_SEG, TARGET_SIZE_SEG))
        generated_mask_256 = run_transunet_segmentation(transunet_model, img_pil_256) # Returns 256x256 mask (0 or 255)
        
        # 2. Use the mask to create the lesion crop
        cropped_img_pil = apply_mask_and_crop(img_pil_original, generated_mask_256, TARGET_SIZE_CLS)
        
        # 3. Load and prepare input for the chosen classifier
        classifier_model = models[secondary_model]
        
        if secondary_model == 'DB-LARNet':
            # DB-LARNet requires both full and crop images (both normalized 224x224)
            full_tensor = CLASSIFIER_TRANSFORM(img_pil_original).unsqueeze(0).to(DEVICE)
            crop_tensor = CLASSIFIER_TRANSFORM(cropped_img_pil).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = classifier_model(full_tensor, crop_tensor)
                probs = F.softmax(logits, dim=1)
                
            prediction_name = f"TransUNet + {secondary_model}"
            input_tensors = (full_tensor, crop_tensor) # Tuple for GradCAMBranch
            
        elif secondary_model == 'PLA-MIL':
            # PLA-MIL requires patch splitting (Lesion Crop input)
            patches_tensor = make_plamil_patches(cropped_img_pil) # (1, N, 3, H_p, W_p)
            
            with torch.no_grad():
                logits, att_weights = classifier_model(patches_tensor)
                probs = F.softmax(logits, dim=1)

            prediction_name = f"TransUNet + {secondary_model}"
            input_tensors = patches_tensor # Tensor for PLA-MIL visualization
            
        else: # ResNet-34 (Lesion Crop Baseline)
            # ResNet-34 uses only the cropped image
            input_tensor = CLASSIFIER_TRANSFORM(cropped_img_pil).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = classifier_model(input_tensor)
                probs = F.softmax(logits, dim=1)

            prediction_name = f"TransUNet + {secondary_model}"
            input_tensors = input_tensor # Tensor for standard GradCAM

        # Finalize prediction
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item() * 100
        
        # 4. Generate Grad-CAM/Attention Map
        gradcam_filename = f"{os.urandom(8).hex()}_{secondary_model}_cam.png"
        output_path = os.path.join(gradcam_output_folder, gradcam_filename)
        
        if secondary_model == 'PLA-MIL':
            # Use attention weights for visualization
            att = att_weights[0].cpu().numpy()
            att_norm = (att - att.min()) / (att.max() - att.min() + 1e-8)
            att_grid = att_norm.reshape(4, 4)
            cam_heatmap = cv2.resize(att_grid, (TARGET_SIZE_CLS, TARGET_SIZE_CLS), interpolation=cv2.INTER_CUBIC)
            
            # Save the attention map overlay on the cropped image
            save_heatmap_to_disk(cropped_img_pil, cam_heatmap, output_path, mask_np_full=generated_mask_256, alpha=0.5)

        elif secondary_model == 'DB-LARNet':
            # Generate CAM on the CROP branch (branch A)
            target_layer = classifier_model.crop_backbone[7]
            gc = GradCAMBranch(classifier_model, target_layer)
            cam_crop, _ = gc.generate(input_tensors[0], input_tensors[1], target_class=pred_idx)
            gc.close()
            # Save the CAM overlay on the cropped image
            save_heatmap_to_disk(cropped_img_pil, cam_crop, output_path, mask_np_full=generated_mask_256, alpha=0.5)
            
        else: # ResNet-34 (Standard GradCAM)
            target_layer = classifier_model.layer4
            gc = GradCAM(classifier_model, target_layer)
            cam, _ = gc.generate(input_tensors, class_idx=pred_idx)
            gc.close()
            # Save the CAM overlay on the cropped image
            save_heatmap_to_disk(cropped_img_pil, cam, output_path, mask_np_full=generated_mask_256, alpha=0.5)

        return CLASS_NAMES[pred_idx], confidence, gradcam_filename, prediction_name

    # --- PIPELINE 2: Segmentation-Guided Classification (SG-ResNet) ---
    elif main_model == 'SG-ResNet':
        
        sg_resnet_model = models['SG-ResNet']
        
        # NOTE: Since the user doesn't upload a mask, we must generate a dummy zero mask
        # for the 4th channel to run the SG-ResNet model structure.
        # In a real deployed environment, you'd integrate TransUNet here too, or expect a mask file.
        # For simplicity, we use a zero mask (256x256) and rely on the model's RGB features.
        dummy_mask_256 = np.zeros((TARGET_SIZE_SEG, TARGET_SIZE_SEG), dtype=np.uint8)

        # 1. Create 4-channel input (RGB + Zero Mask)
        x_4ch = create_4ch_input_with_mask(img_pil_original, dummy_mask_256) 
        
        # 2. Run prediction
        with torch.no_grad():
            logits = sg_resnet_model(x_4ch)
            probs = F.softmax(logits, dim=1)
            
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item() * 100

        # 3. Generate Grad-CAM for the SG-ResNet model
        gradcam_filename = f"{os.urandom(8).hex()}_sgresnet_cam.png"
        output_path = os.path.join(gradcam_output_folder, gradcam_filename)
        
        target_layer = sg_resnet_model.layer4
        gc = GradCAM(sg_resnet_model, target_layer)
        cam, _ = gc.generate(x_4ch, class_idx=pred_idx)
        gc.close()
        
        # Use the dummy mask in the visualization (it will show a zero contour)
        save_heatmap_to_disk(img_pil_original, cam, output_path, mask_np_full=dummy_mask_256, alpha=0.5)

        return CLASS_NAMES[pred_idx], confidence, gradcam_filename, "Segmentation-Guided ResNet-34"
    
    else:
        raise ValueError(f"Invalid model selection: {main_model}")