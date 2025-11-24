# Skin Lesion Analysis with Segmentation, Classification & Explainability

This repository contains a complete deep-learning pipeline for **automated skin lesion analysis** using the HAM10000 dermoscopic dataset.  
The pipeline includes:

- Lesion **segmentation** using a TransUNet-style architecture  
- Multiple **classification** models (baseline & advanced)  
- **Segmentation-guided** classification  
- **Patch-level attention MIL** for fine-grained reasoning  
- **Grad-CAM** and attention visualizations for model explainability  

All experiments were implemented and run in a Kaggle environment, but the code can be adapted to any PyTorch setup.

---

## 1. Models Implemented

### Segmentation

- **TransUNetAEO**  
  - U-Net style encoder–decoder with a Transformer encoder at the bottleneck  
  - Trained on dermoscopic images + binary lesion masks  
  - Input size: 256×256 RGB  
  - Loss: Dice + BCE (combined)

### Baseline RGB Classification Models

All use the 7 HAM10000 classes: `MEL, NV, BCC, AKIEC, BKL, DF, VASC`.

1. **ResNet-34 (Lesion Crop Baseline)**
   - Input: lesion-centered crops (224×224)
   - Pretrained on ImageNet  
   - Final FC replaced with 7-class head

2. **Dual-Branch LARNet (DB-LARNet)**
   - Branch A: full dermoscopic image  
   - Branch B: lesion crop  
   - Both branches: ResNet-18 backbones  
   - Fused feature vector → classifier head

3. **Patch-Level Attention MIL (PLA-MIL)**
   - Input: lesion crop (224×224), split into 4×4 patches (16 patches)
   - ResNet-18 feature encoding per patch  
   - Attention-based MIL pooling → image-level classifier

### Segmentation-Guided Classification

4. **Segmentation-Guided ResNet-34 (SG-ResNet)**
   - Input: 4-channel tensor `[R, G, B, Mask]`  
   - First conv modified to 4 input channels  
   - Lesion masks from ground-truth or TransUNet predictions  

---

## 2. Dataset

### HAM10000 Dermoscopic Dataset

Expected files:

- `images/` – original RGB dermoscopic images (`<image_id>.jpg`)  
- `masks/` – binary segmentation masks (`<image_id>_segmentation.png`)  
- `GroundTruth.csv` – one-hot encoded class labels  
- `ham_lesion_crops/` – lesion-centered cropped images  

### Class List (7 classes)

| Code  | Full Name |
|-------|-----------|
| MEL   | Melanoma |
| NV    | Melanocytic Nevus |
| BCC   | Basal Cell Carcinoma |
| AKIEC | Actinic Keratoses |
| BKL   | Benign Keratosis |
| DF    | Dermatofibroma |
| VASC  | Vascular Lesions |

### Train/Val Split

- Stratified by `label_idx`
- Default split: **80% training / 20% validation**

---

## 3. Technologies & Libraries

- **Deep Learning:** PyTorch, Torchvision  
- **Data Handling:** NumPy, Pandas  
- **Augmentation:** Albumentations, Torchvision  
- **Visualization:** Matplotlib, OpenCV  
- **Metrics:** scikit-learn  
- **Explainability:** Grad-CAM, Attention maps  
- **Documentation:** Markdown, LaTeX/TikZ  

---

## 4. Installation

Install core libraries:

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn matplotlib opencv-python-headless albumentations
```

If running outside Kaggle, replace hard-coded `/kaggle/...` paths accordingly.

---

## 5. Training & Evaluation

> Adjust script names based on your actual repo.

### 5.1 Segmentation — TransUNetAEO

Script: `transunet_segmentation.py`

Workflow:

- Loads RGB images + masks  
- Joint augmentations (resize, flips, rotations, noise)  
- Loss = Dice + BCE  
- Saves:
  - `training_history.csv`
  - `loss_curves.png`, `dice_curves.png`
  - `confusion_matrix.png`, `classification_report.txt`

**Validation Scores**

| Metric | Score |
|--------|-------|
| Dice   | 0.914 |
| IoU    | 0.858 |
| Pixel Accuracy | 0.955 |

---

### 5.2 ResNet-34 Classification (Lesion Crops)

Script: `resnet34_cls_lesion_crops.py`

- Input: 224×224 lesion crops  
- Augmentations: flips, rotations, ColorJitter  
- WeightedRandomSampler + class-weight CE loss  
- Optimizer: Adam (`lr=1e-4`)  
- Eval:
  - Confusion matrix  
  - ROC curves  
  - Classification report  

**Validation Accuracy: 0.8777**

---

### 5.3 Dual-Branch LARNet (Full Image + Crop)

Script: `dualbranch_dblarnet_cls.py`

- Input: full image + lesion crop  
- Two ResNet-18 backbones ⟶ concatenated features ⟶ classifier head  
- Weighted sampler + class weights  

**Validation Accuracy: 0.8762**

---

### 5.4 Patch-Level Attention MIL (PLA-MIL)

Script: `plamil_patchmil_cls.py`

- Split 224×224 crop into **16 patches**  
- Patch encoder = ResNet-18  
- Attention-based pooling → classifier  

**Validation Accuracy: 0.7304**  
(Provides high interpretability via patch attention)

---

### 5.5 Segmentation-Guided ResNet-34 (RGB + Mask 4-Channel)

Script: `sg_resnet34_4ch_cls.py`

- Modify first conv layer → `in_channels=4`  
- Input: `[R, G, B, Mask]` tensor  

**Validation Accuracy: 0.8707**

---

## 6. Explainability

### 6.1 Grad-CAM for ResNet-34

Script: `gradcam_resnet34.py`

Produces:

- Original Image  
- Pure Grad-CAM Heatmap  
- Overlay Heatmap  

Saved under: `resnet34_gradcam_outputs/`

---

### 6.2 Patch Attention Visualization for PLA-MIL

Script: `plamil_attention_vis.py`

Produces:

- Original  
- 4×4 Patch Attention Grid  
- Overlay Heatmap  

---

## 7. Results Summary

### 7.1 Segmentation

| Model     | Dice  | IoU   | Pixel Acc |
|-----------|-------|-------|-----------|
| TransUNet | 0.914 | 0.858 | 0.955     |

### 7.2 Classification

| Model                          | Input Type                     | Val Accuracy |
|--------------------------------|--------------------------------|-------------:|
| ResNet-34 (baseline)           | Lesion Crops                   | **0.8777**   |
| Dual-Branch LARNet             | Full Image + Crop              | 0.8762       |
| Segmentation-Guided ResNet-34  | RGB + Mask (4-ch)              | 0.8707       |
| PLA-MIL                        | Patch Grid (16 patches)        | 0.7304       |

---

## 8. How to Reproduce

1. **Prepare dataset folder:**
   - `images/`
   - `masks/`
   - `GroundTruth.csv`
   - `ham_lesion_crops/`

2. **Train segmentation**
   ```bash
   python transunet_segmentation.py
   ```

3. **Train classifiers**
   ```bash
   python resnet34_cls_lesion_crops.py
   python dualbranch_dblarnet_cls.py
   python plamil_patchmil_cls.py
   python sg_resnet34_4ch_cls.py
   ```

4. **Generate explainability outputs**
   ```bash
   python gradcam_resnet34.py
   python plamil_attention_vis.py
   ```

5. **Compare results**
   - Use confusion matrices  
   - ROC curves  
   - Grad-CAM/attention overlays  

---

## 9. Acknowledgements

- HAM10000 dataset creators  
- PyTorch & open-source libraries  
- Research literature on dermatology AI  

---

## 10. Running the Web Interface Locally

This section explains how to run the **Flask-based web interface** for:

- Skin lesion **segmentation**
- Lesion **classification**
- **Grad-CAM–based** explainability

---

### 10.1 Prerequisites

Make sure the following are installed on your machine:

- Python **3.10+**
- **Git**
- At least **8 GB RAM** (recommended, due to multiple PyTorch models)
- All trained model `.pth` files placed in the `outputs/` directory

---

### 10.2 Clone the Repository

Use the following commands in a terminal:

```bash
# Clone the GitHub repository
git clone https://github.com/dpavansekhar/Skin-Lesion-Detection-Using-Deep-Learning.git

# Move into the project root folder
cd Skin-Lesion-Detection-Using-Deep-Learning
````

---

### 10.3 Create and Activate a Virtual Environment (Recommended)

**Windows (PowerShell):**

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

---

### 10.4 Install Requirements

Install all required Python dependencies:

```bash
# Upgrade pip (optional but recommended)
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

---

### 10.5 Verify Project Structure

Ensure the repository structure matches the layout below (key folders only):

```text
interface/
   app/
      app.py
   models/
      model_utils.py
   static/
      uploads/
      gradcam/
   templates/
      index.html
      error.html
      results.html

outputs/
   transunet-model-outputs/
   resnet34-model-testing/
   db-larnet-model-outputs/
   pla-mil-network-model-outputs-ham10000/
   segmentation-guided-resnet-34-model-outputs-ham10000/
   (all best_*.pth model files)

requirements.txt
```

> ✅ All `.pth` model files **must** be located inside the `outputs/` folder.
> ✅ Folder and file names must match exactly those used in `interface/models/model_utils.py`.

---

### 10.6 Run the Flask Interface

From the **project root**, you can start the interface in one of two ways:

**Option A – Using Flask CLI**

```bash
# Run the Flask app using the Flask CLI
python -m flask --app interface.app.app run
```

**Option B – Running the script directly**

```bash
# Run the Flask app by calling the script directly
python interface/app/app.py
```

By default, the interface will be available at:

```text
http://127.0.0.1:5000
```

Open this URL in your web browser.

---

### 10.7 Using the Interface

Once the interface is running:

1. Upload a dermoscopic skin lesion image (`.jpg` / `.png`).
2. Choose the analysis pipeline:

   * **SG-ResNet** (segmentation-guided 4-channel model: RGB + mask)
   * **TransUNet + Classification** (segmentation followed by a classifier)
3. If you select **TransUNet**, also choose a secondary classifier:

   * `ResNet-34`
   * `DB-LARNet`
   * `PLA-MIL`
4. Click **Analyse**.
5. The results page will display:

   * Predicted **lesion class**
   * **Confidence score**
   * The original input image
   * A **Grad-CAM heatmap** overlay for visual explanation

---

### 10.8 Stopping the Server

To stop the Flask server, go back to the terminal where it is running and press:

```text
Ctrl + C
```

---

### 10.9 Troubleshooting

**1️⃣ Missing model files**

If you encounter errors related to model loading (e.g. *model not found*):

* Confirm that all required `.pth` files are present inside the `outputs/` folder.
* Ensure that path and filenames match those referenced in `model_utils.py`.

---

**2️⃣ Template not found**

If you see `TemplateNotFound` errors:

* Confirm that the `templates/` folder is located inside `interface/`.
* Check that the template filenames match exactly:

  * `index.html`
  * `error.html`
  * `results.html` (watch out for typos like `reuslts.html`).

---

**3️⃣ Import errors**

If Python packages are missing:

```bash
# Re-install all dependencies
pip install -r requirements.txt
```

---

**4️⃣ Port already in use**

If port `5000` is already in use, run the app on a different port:

```bash
# Run Flask app on port 5001 instead of 5000
python -m flask --app interface.app.app run --port 5001
```

---

