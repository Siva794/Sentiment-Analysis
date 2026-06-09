# 🚀 Emotion Intelligence Engine

### Efficient On-Device NLP Using DistilBERT, Mixed Precision Training & Quantization

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![DistilBERT](https://img.shields.io/badge/Model-DistilBERT-green)
![Accuracy](https://img.shields.io/badge/Accuracy-95.12%25-brightgreen)

</p>

---

## 📌 Project Overview

Emotion Intelligence Engine is a lightweight NLP system that detects human emotions from text using a fine-tuned DistilBERT transformer model.

The model is optimized using:

✅ Mixed Precision Training (AMP)

✅ Dynamic Quantization

✅ CPU-Friendly Deployment

✅ Lightweight Web Interface

The system classifies text into six emotional categories:

| Emotion     | Description                    |
| ----------- | ------------------------------ |
| 😊 Joy      | Positive and happy expressions |
| 😢 Sadness  | Negative emotional responses   |
| 😠 Anger    | Frustration and hostility      |
| 😨 Fear     | Anxiety and concern            |
| ❤️ Love     | Affection and care             |
| 😲 Surprise | Unexpected reactions           |

---

## 🎯 Key Results

| Metric               | Value               |
| -------------------- | ------------------- |
| Accuracy             | **95.12%**          |
| Weighted F1 Score    | **94.93%**          |
| Dataset Size         | **422,746 Samples** |
| Original Model Size  | **255.46 MB**       |
| Quantized Model Size | **132.29 MB**       |
| Size Reduction       | **48%**             |

---

## 🏗️ System Architecture

```text
Raw Text
    │
    ▼
DistilBERT Tokenizer
    │
    ▼
DistilBERT Encoder
    │
    ▼
Classification Head
    │
    ▼
Emotion Prediction
```

Optimization Pipeline:

```text
Training
   │
   ├── Mixed Precision Training (AMP)
   │
   ▼
Fine-Tuned DistilBERT
   │
   ▼
Dynamic Quantization
   │
   ▼
Compressed Deployment Model
```

---

## 🖥️ Demo Interface

Add screenshots here:

```text
assets/dashboard.png
assets/prediction.png
```

Example:

Input:

```text
I finally got the job I always wanted!
```

Prediction:

```text
😊 Joy
```

---

## 📂 Project Structure

```text
Sentiment-Analysis
│
├── app.py
├── train.py
├── evaluate.py
├── qat_train.py
├── requirements.txt
│
├── configs/
│   └── config.yaml
│
├── frontend/
│   ├── index.html
│   ├── emotion_app.html
│   └── styles.css
│
└── src/
    ├── preprocess.py
    ├── data_loader.py
    ├── dataloader_builder.py
    ├── train.py
    ├── evaluate.py
    ├── quantize.py
    ├── qat.py
    └── utils.py
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Siva794/Sentiment-Analysis.git
cd Sentiment-Analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train Model

```bash
python train.py
```

### Quantization-Aware Training

```bash
python qat_train.py
```

### Evaluate Model

```bash
python evaluate.py
```

### Launch Web Application

```bash
python app.py
```

---

## 🧠 Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* DistilBERT
* Flask
* AMP (Automatic Mixed Precision)
* Dynamic Quantization
* HTML/CSS

---

## 📊 Research Contributions

* Developed an efficient DistilBERT-based emotion classifier.
* Applied Mixed Precision Training for faster training.
* Reduced model size by 48% using quantization.
* Evaluated accuracy vs. efficiency trade-offs.
* Demonstrated practical deployment on CPU-based systems.

---

## 🔮 Future Improvements

* ONNX Runtime Deployment
* TensorRT Optimization
* Mobile Deployment
* Knowledge Distillation
* Real-Time API Integration

---

## 👨‍💻 Authors

**K. Siva Nagendra Prasad**

Department of Computational Intelligence

SRM Institute of Science and Technology

---

## ⭐ If you found this project useful

Give it a star on GitHub and share it with others!
