\# Efficient On-Device NLP Using Mixed Precision and Quantization for Tiny Transformer Models



\## Overview



This project presents an efficient emotion classification system based on DistilBERT for resource-constrained environments. The system classifies text into six emotion categories:



\* Joy

\* Sadness

\* Anger

\* Fear

\* Love

\* Surprise



The model is optimized using Mixed Precision Training (AMP) and Quantization techniques to reduce memory consumption while maintaining high classification accuracy.



This work was developed as part of research on efficient deployment of transformer-based Natural Language Processing (NLP) models. The corresponding IEEE paper reports a classification accuracy of \*\*95.12%\*\* and weighted \*\*F1-score of 94.93%\*\*. These results demonstrate that compact transformer models can achieve high performance while remaining suitable for deployment on edge and CPU-based systems.



\---



\## Features



\* DistilBERT-based emotion classification

\* Six-class emotion prediction

\* Mixed Precision Training (AMP)

\* Quantization-Aware Training experiments

\* Dynamic Post-Training Quantization

\* Lightweight web interface

\* Model evaluation and performance analysis

\* CPU-friendly deployment pipeline



\---



\## Dataset



The project uses the Emotion subset from the Sentiment and Emotion Analysis dataset available on Kaggle.



Dataset characteristics:



\* 422,746 text samples

\* 6 emotion classes

\* Balanced multi-class classification setup



Emotion labels:



| Label    |

| -------- |

| Joy      |

| Sadness  |

| Anger    |

| Fear     |

| Love     |

| Surprise |



\---



\## Model Architecture



\### Base Model



\* DistilBERT (`distilbert-base-uncased`)

\* Maximum sequence length: 128

\* Optimizer: AdamW

\* Loss Function: Cross Entropy Loss



\### Training Optimizations



\#### Mixed Precision Training



Automatic Mixed Precision (AMP) is used when CUDA-compatible GPUs are available.



Benefits:



\* Reduced GPU memory consumption

\* Faster training

\* Stable optimization using GradScaler



\#### Quantization



Dynamic post-training quantization is applied to reduce model size for deployment.



Benefits:



\* Reduced storage requirements

\* Faster model loading

\* Improved deployment efficiency



\---



\## Results



| Metric            | Baseline Model |

| ----------------- | -------------- |

| Accuracy          | 95.12%         |

| Weighted F1 Score | 94.93%         |

| FP32 Model Size   | 255.46 MB      |

| INT8 Model Size   | 132.29 MB      |



\### Model Compression



Quantization reduced storage requirements by approximately:



\*\*48%\*\*



without significant loss in classification performance.



\---



\## Project Structure



```text

Sentiment-Analysis

│

├── app.py

├── app\_tester.py

├── train.py

├── evaluate.py

├── evaluate\_qat.py

├── qat\_train.py

├── main.py

├── requirements.txt

│

├── configs/

│   └── config.yaml

│

├── frontend/

│   ├── index.html

│   ├── emotion\_app.html

│   └── styles.css

│

└── src/

&#x20;   ├── preprocess.py

&#x20;   ├── data\_loader.py

&#x20;   ├── dataloader\_builder.py

&#x20;   ├── train.py

&#x20;   ├── evaluate.py

&#x20;   ├── quantize.py

&#x20;   ├── qat.py

&#x20;   └── utils.py

```



\---



\## Installation



Clone the repository:



```bash

git clone https://github.com/Siva794/Sentiment-Analysis.git

cd Sentiment-Analysis

```



Install dependencies:



```bash

pip install -r requirements.txt

```



\---



\## Running the Project



\### Train Model



```bash

python train.py

```



\### Quantization-Aware Training



```bash

python qat\_train.py

```



\### Evaluate Model



```bash

python evaluate.py

```



\### Launch Web Application



```bash

python app.py

```



Open your browser and access the local application interface.



\---



\## Technologies Used



\* Python

\* PyTorch

\* Hugging Face Transformers

\* DistilBERT

\* Flask

\* Mixed Precision Training (AMP)

\* Quantization Techniques

\* HTML/CSS



\---



\## Research Highlights



\* Achieved 95.12% classification accuracy.

\* Reduced model size from 255.46 MB to 132.29 MB.

\* Demonstrated practical deployment of transformer models on resource-constrained devices.

\* Evaluated the trade-off between accuracy, model size, and inference efficiency.



\---



\## Future Work



\* ONNX Runtime deployment

\* TensorRT optimization

\* Mobile deployment

\* Improved Quantization-Aware Training support

\* Knowledge Distillation with TinyBERT and MobileBERT



\---



\## Authors



K. Siva Nagendra Prasad

S. Sai Ganesh

Ch. Harshith Sai

V. Narendra



Department of Computational Intelligence

SRM Institute of Science and Technology



\---



\## License



This project is intended for academic and research purposes.



