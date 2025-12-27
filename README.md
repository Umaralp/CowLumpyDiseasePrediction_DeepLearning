# 🐮 Lumpy Skin Disease Detection

An efficient Deep Learning system designed to detect **Lumpy Skin Disease (LSD)** in cattle using computer vision. This project leverages the **MobileNetV2** architecture to provide high-accuracy disease diagnosis while remaining lightweight enough for potential deployment on edge devices (smartphones/Raspberry Pi) in low-resource farming environments.

## 📌 Project Overview

Lumpy Skin Disease is a viral infection in cattle that causes significant economic loss through decreased milk production and skin damage. Early diagnosis is critical to preventing spread.
This project automates the detection process using image classification, aiming to assist farmers and veterinarians with a fast, non-invasive diagnostic tool.

**Key Objectives:**
* **Early Detection:** Identify LSD nodules from skin images.
* **Edge Efficiency:** Use a lightweight model architecture suitable for mobile use.
* **High Recall:** Prioritize sensitivity to minimize false negatives (missing a sick cow).

## 🚀 Features

* **Binary Classification:** Distinguishes between `Healthy` vs. `Lumpy Skin Disease` images.
* **Transfer Learning:** Utilizes a pre-trained **MobileNetV2** (ImageNet weights) for robust feature extraction.
* **Data Augmentation:** Implements rotation, zooming, and flipping to handle small dataset sizes and improve generalization.
* **Performance Metrics:** Detailed evaluation using Confusion Matrix, Precision, Recall, and F1-Score.

## 🛠️ Tech Stack

* **Core:** Python
* **Deep Learning:** TensorFlow / Keras
* **Data Processing:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebook / Google Colab

## 📂 Dataset

The dataset consists of images of cattle skin categorized into two classes:
1.  **LSD (Positive):** Images showing characteristic nodules/lumps.
2.  **Healthy (Negative):** Clear skin without lesions.

*Note: The dataset was pre-processed to resize images to `224x224` pixels (MobileNetV2 standard input).*

## 🧠 Model Architecture

I chose **MobileNetV2** over heavier models (like ResNet or VGG) because:
1.  **Inverted Residual Blocks:** Allows for high accuracy with fewer parameters.
2.  **Depthwise Separable Convolutions:** Reduces computation cost, making it ideal for future mobile app integration.

**Training Strategy:**
* **Base Model:** MobileNetV2 (Frozen, without top layers).
* **Head:** Custom GlobalAveragePooling2D -> Dense (ReLU) -> Dropout (0.5) -> Output (Sigmoid).
* **Optimizer:** Adam.
* **Loss Function:** Binary Cross-Entropy.

## 📊 Results

The model achieved promising results on the validation set, with a specific focus on **Recall** to ensure infected cattle are not overlooked.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | ~95% |
| **Recall (Sensitivity)** | **High Priority** |
| **Precision** | Balanced |

*(You can add your specific Confusion Matrix image here if you have one)*

## 💻 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/lumpy-skin-disease-detection.git](https://github.com/yourusername/lumpy-skin-disease-detection.git)
    cd lumpy-skin-disease-detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Training Notebook:**
    Open `Lumpy_Skin_Detection.ipynb` to view the training pipeline, data augmentation steps, and model evaluation.


## 🔮 Future Improvements

* **Mobile App:** Convert the model to **TensorFlow Lite (.tflite)** for Android deployment.
* **Dataset Expansion:** Collect more diverse lighting conditions and skin tones.
* **Explainability:** Implement **Grad-CAM** to visualize *where* the model is looking (i.e., highlighting the specific lumps).

