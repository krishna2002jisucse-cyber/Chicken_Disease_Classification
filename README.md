# 🐔 Chicken Disease Classification Using Deep Learning

> **VGG16-based Transfer Learning** with Grad-CAM Explainability, DVC MLOps Pipeline, and Flask Deployment  
> *M.Tech AIML — Symbiosis Institute of Technology, Pune | 2025–26*

---

## 📌 About the Project

This project automatically detects **Coccidiosis** (a common chicken disease) from fecal images using deep learning. A farmer simply uploads a photo of chicken feces and the system instantly predicts whether the chicken is **Diseased (Coccidiosis)** or **Healthy**.

**Key Results:**
- ✅ Validation Accuracy: **88.79%**
- ✅ Validation Loss: **0.2918**
- ✅ AUC Score: **0.935**

---

## 🏗️ Project Architecture

```
Fecal Image Input
       ↓
Data Preprocessing & Augmentation
       ↓
VGG16 Model (Transfer Learning from ImageNet)
       ↓
Prediction → Coccidiosis or Healthy
       ↓
Grad-CAM Heatmap (Explainability)
       ↓
Flask Web App (API Endpoints)
```

---

## 📁 Project Structure

```
Chicken_Disease_Classification/
│
├── config/
│   └── config.yaml              # paths and source URL
│
├── src/cnnClassifier/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── prepare_base_model.py
│   │   ├── training.py
│   │   └── evaluation.py
│   ├── pipeline/
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_prepare_base_model.py
│   │   ├── stage_03_training.py
│   │   ├── stage_04_evaluation.py
│   │   └── predict.py
│   └── utils/
│       ├── common.py
│       └── gradcam.py           # Grad-CAM heatmap generation
│
├── templates/
│   └── index.html               # web UI
│
├── artifacts/                   # auto-generated (model, data, results)
├── app.py                       # Flask web application
├── main.py                      # run full pipeline
├── params.yaml                  # model hyperparameters
├── dvc.yaml                     # DVC pipeline definition
├── requirements.txt
└── scores.json                  # evaluation results
```

---

## ⚙️ Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image Size | 224 × 224 × 3 |
| Batch Size | 16 |
| Epochs | 1 (increase for better results) |
| Optimizer | Adam |
| Loss | Categorical Cross-Entropy |
| Weights | ImageNet (frozen) |
| Classes | 2 (Coccidiosis, Healthy) |
| Augmentation | Yes (rotation, flip, zoom, shear) |

---

## 🚀 How to Run This Project

### Step 1 — Clone the Repository

```bash
git clone https://github.com/krishna2002jisucse-cyber/Chicken_Disease_Classification.git
cd Chicken_Disease_Classification
```

### Step 2 — Create a Virtual Environment

```bash
# Using conda
conda create -n chicken python=3.10 -y
conda activate chicken

# OR using venv
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the Full DVC Pipeline

```bash
# Run all 4 stages automatically
python main.py
```

This will:
1. **Download** the dataset from GitHub
2. **Prepare** the VGG16 base model
3. **Train** the model with augmentation
4. **Evaluate** and save results to `scores.json`

### Step 5 — Start the Flask Web App

```bash
python app.py
```

Then open your browser and go to: **http://localhost:8080**

---

## 🔗 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface — upload image from browser |
| `/predict` | POST | Send image → get prediction (JSON) |
| `/train` | GET/POST | Trigger full pipeline retraining |
| `/metrics` | GET | Get evaluation metrics as JSON |
| `/evaluation_images/<filename>` | GET | View evaluation charts |

### Example — Using /predict

```python
import requests, base64

with open("fecal_image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

response = requests.post("http://localhost:8080/predict",
                         json={"image": image_data})
print(response.json())
# Output: {"image": "Coccidiosis"} or {"image": "Healthy"}
```

---

## 📊 DVC Pipeline

This project uses **DVC (Data Version Control)** for a fully reproducible ML pipeline.

```bash
# Run the pipeline
dvc repro

# Visualize the pipeline DAG
dvc dag

# Check parameter changes
dvc params diff
```

**4 Pipeline Stages:**

| Stage | Script | Output |
|-------|--------|--------|
| 01 — Data Ingestion | `stage_01_data_ingestion.py` | `artifacts/data_ingestion/` |
| 02 — Base Model Prep | `stage_02_prepare_base_model.py` | `base_model_updated.h5` |
| 03 — Training | `stage_03_training.py` | `artifacts/training/model.h5` |
| 04 — Evaluation | `stage_04_evaluation.py` | `scores.json` |

---

## 🔍 Grad-CAM Explainability

The project includes **Grad-CAM** (Gradient-weighted Class Activation Mapping) which visually explains *why* the model made a prediction by highlighting the important regions in the fecal image.

- 🔴 **Red regions** = areas that caused the Coccidiosis prediction
- 🔵 **Blue regions** = less important areas

Generated via `src/cnnClassifier/utils/gradcam.py`

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| TensorFlow / Keras | Deep learning framework |
| VGG16 | Pre-trained CNN model |
| DVC | ML pipeline & experiment tracking |
| Flask | Web application & REST API |
| OpenCV | Grad-CAM heatmap generation |
| Matplotlib / Seaborn | Evaluation visualization |

---

## 📋 Requirements

```
tensorflow
flask
flask-cors
dvc
numpy
pandas
matplotlib
seaborn
scipy
python-box==6.0.2
pyYAML
tqdm
ensure==1.0.2
joblib
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🔧 Changing Hyperparameters

Edit `params.yaml` to change training settings:

```yaml
EPOCHS: 15          # increase for better accuracy
BATCH_SIZE: 16
IMAGE_SIZE: [224, 224, 3]
AUGMENTATION: True
LEARNING_RATE: 0.01
```

Then rerun:
```bash
python main.py
# or
dvc repro
```

---

## 📈 Results

```
Validation Loss     : 0.2918
Validation Accuracy : 88.79%
AUC Score           : 0.935
F1-Score            : 0.888
```

Evaluation outputs are saved to `artifacts/evaluation_results/`:
- `confusion_matrix.png`
- `roc_curve.png`
- `per_class_metrics.png`
- `evaluation_dashboard.png`
- `evaluation_metrics.json`

---

## 👤 Author

**Krishna Nandi**  
M.Tech — Artificial Intelligence and Machine Learning  
Symbiosis Institute of Technology, Pune  
📧 krishna.nandi.mtech2025@sitpune.edu.in  
🔗 [GitHub](https://github.com/krishna2002jisucse-cyber)

---

## 📄 License

This project is for academic purposes under Symbiosis Institute of Technology, Pune.

---

*If you find this project helpful, please ⭐ star the repository!*
