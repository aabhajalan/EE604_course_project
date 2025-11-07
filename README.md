# EE604_course_project
Deepfake detection using cnn
# 🎭 Temporal–Spatial Deepfake Detection using ResNeXt-LSTM and Transformer-Based Fusion Networks

**EE604 Course Project — IIT Kanpur**
By: *Sabeeh Muhammed MC*, *Aabha Jalan*, *Diyush S*, *Nikhil N*, *Abhinav Ghosh*

---

## 🧠 Overview

This repository contains our implementation for **Temporal–Spatial Deepfake Detection using ResNeXt-LSTM and Transformer-Based Fusion Networks**, developed as part of the *EE604: Machine Learning for Signal Processing* course at IIT Kanpur.

The project investigates how combining **spatial encoders (CNNs)** and **temporal sequence models (RNNs/Transformers)** improves robustness in deepfake video detection.

We evaluate three complementary architectures:

| Architecture                      | Type                | Purpose                                   |
| --------------------------------- | ------------------- | ----------------------------------------- |
| **ResNeXt-LSTM (ours)**           | CNN + LSTM          | Final high-accuracy spatio-temporal model |
| **MesoInception-4 + Transformer** | CNN + Attention     | Lightweight attention-based variant       |
| **MobileNetV2 + GRU**             | Efficient CNN + RNN | Resource-constrained efficient model      |

---

## 📂 Repository Structure

```
EE604_Course_Project/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── train_resnext_lstm.py
│   ├── train_meso_transformer.py
│   ├── train_mobilenet_gru.py
│   ├── inference/
│   │   └── predict_video.py
│   └── utils/
│       ├── data_loader.py
│       ├── metrics.py
│       └── visualization.py
│
├── app/                            # 🔹 Web interface (Streamlit)
│   ├── app.py
│   ├── model_loader.py
│   └── README.md
│
├── models/
│   ├── best_resnext_lstm.pth
│   ├── best_transformer_model.pth
│   └── best_mobilenetv2_gru.h5
│
├── figures/
│   ├── pipeline.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── qualitative.png
│
└── report/
    ├── Project_Report.pdf
    ├── main.tex
    ├── main.bib
    └── figures/
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/aabhajalan/EE604_course_project.git
cd EE604_course_project
pip install -r requirements.txt
```

For GPU systems, ensure PyTorch and torchvision are installed with CUDA support.

---

## 🚀 Training the Models

To train each model variant:

```bash
# Train the final ResNeXt-LSTM model
python src/train_resnext_lstm.py

# Train the MesoInception-4 + Transformer model
python src/train_meso_transformer.py

# Train the MobileNetV2 + GRU model
python src/train_mobilenet_gru.py
```

---

## 🧪 Evaluation

Evaluate a saved checkpoint on the test dataset:

```bash
python src/inference/predict_video.py --model models/best_resnext_lstm.pth --video sample_video.mp4
```

The script outputs:

* Predicted label (Real/Fake)
* Confidence score
* Optional visual overlay if enabled

---



## 📊 Experimental Results

All experiments were conducted on an NVIDIA T4 GPU with 15 GB VRAM using the **FaceForensics++**, **Celeb-DF v2**, and **DFDC** datasets.

| Model                         |  Accuracy  |    AUC   | F1-Score |
| :---------------------------- | :--------: | :------: | :------: |
| MesoInception-4 + Transformer |   86.8 %   |   0.90   |   0.86   |
| MobileNetV2 + GRU             |   88.4 %   |   0.91   |   0.87   |
| **ResNeXt-LSTM (Ours)**       | **92.7 %** | **0.96** | **0.93** |

### Example Outputs

| Figure                                            | Description                                      |
| ------------------------------------------------- | ------------------------------------------------ |
| ![Confusion Matrix](figures/confusion_matrix.png) | Balanced detection between real and fake classes |
| ![ROC Curve](figures/roc_curve.png)               | AUC = 0.9538                                     |
| ![Attention Map](figures/qualitative.png)         | Model focus on facial regions like eyes and lips |

---

## 🧩 Model Insights

* **ResNeXt-LSTM** shows superior class separability and stability in predictions.
* The **Transformer variant** demonstrated the ability to capture long-range frame dependencies.
* The **MobileNetV2 + GRU** offered a practical balance between efficiency and accuracy for edge devices.

---

## 📦 Datasets

* [FaceForensics++](https://github.com/ondyari/FaceForensics)
* [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics)
* [DFDC Dataset](https://ai.facebook.com/datasets/dfdc)

> ⚠️ Dataset files are not included in this repository due to size and licensing constraints.

---

## 👨‍💻 Contributors

| Name                   | Email                                                     |
| ---------------------- | --------------------------------------------------------- | 
| **Sabeeh Muhammed MC** | [saheebm24@iitk.ac.in](mailto:saheebm24@iitk.ac.in)       | 
| **Aabha Jalan**        | [aabhajalan24@iitk.ac.in](mailto:aabhajalan24@iitk.ac.in) | 
| **Diyush S**           | [diyushs24@iitk.ac.in](mailto:diyushs24@iitk.ac.in)       | 
| **Nikhil N**           | [nikhiln23@iitk.ac.in](mailto:nikhiln23@iitk.ac.in)       | 
| **Abhinav Ghosh**      | [abhinavg24@iitk.ac.in](mailto:abhinavg24@iitk.ac.in)     | 


---

## 📚 References

Below are all research works and reports referenced or used as baselines during the project:

1. **Piotr Kawa, Piotr Syga.**  
   *A Note on Deepfake Detection with Low-Resources.*  
   *arXiv preprint arXiv:2006.05183 (2020).*  
   [https://arxiv.org/abs/2006.05183](https://arxiv.org/abs/2006.05183)

2. **Abhijit Jadhav, Hitendra Patil, Jay Patel, Abhishek Patange, Manjushri Mahajan.**  
   *Deepfake Video Detection using ResNeXt Convolutional and LSTM Networks.*  
   *B.E. Project Report, GHRCEM Department of Computer Engineering, Pune (2020).*  
   [https://github.com/abhijitjadhav1998/Deefake_detection_Django_app](https://github.com/abhijitjadhav1998/Deefake_detection_Django_app)

3. **Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen.**  
   *MobileNetV2: Inverted Residuals and Linear Bottlenecks.*  
   *arXiv preprint arXiv:1801.04381 (2018).*  
   [https://arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381)

4. **Kyunghyun Cho, Bart van Merriënboer, Çaglar Gülçehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio.**  
   *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation.*  
   *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1724–1734.*  
   [https://arxiv.org/abs/1406.1078](https://arxiv.org/abs/1406.1078)

5. **Md Shohel Rana, Mohammad Nur Nobi, Beddhu Murali, Andrew H. Sung.**  
   *Deepfake Detection: A Systematic Literature Review.*  
   *IEEE Access, Vol. 10, pp. 25494–25513, 2022.*  
   [https://ieeexplore.ieee.org/document/9721302](https://ieeexplore.ieee.org/document/9721302)

---

These papers collectively form the theoretical and architectural foundation for our experiments with **ResNeXt-LSTM**, **MesoInception-4 + Transformer**, and **MobileNetV2 + GRU** models, influencing decisions on spatial–temporal fusion, feature representation, and performance evaluation strategies.

---
