# Hemorrhage Detection & Segmentation (MATLAB)

This repository contains a medical imaging pipeline designed to process Head CT scans and classify them into two categories: **Hemorrhage** and **No Hemorrhage**. The system uses a combination of spatial filters for noise reduction, CLAHE for contrast enhancement, and HOG (Histogram of Oriented Gradients) for feature extraction.

## 🧠 Core Methodology

The project follows a standard supervised machine learning workflow:

### **1. Image Preprocessing**

To handle the low-contrast nature of CT scans, the script applies a multi-stage filtering process:

* **Wiener Filtering:** Adaptive pixel-wise restoration to remove noise.
* **Anisotropic Diffusion:** Smooths the background while preserving critical edges (like the boundary of a bleed).
* **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Enhances local contrast to make the hemorrhage regions more distinct.

### **2. Feature Extraction**

* **HOG Features:** Extracts local shape and texture information by calculating the distribution of intensity gradients. This helps the model distinguish between healthy brain tissue and the dense, irregular shapes of a hemorrhage.

### **3. Classification**

* **Algorithm:** k-Nearest Neighbors ().
* **Data Split:** 60% Training, 20% Validation, and 20% Testing.
* **Evaluation:** The system outputs a Confusion Matrix and calculates precision, recall (sensitivity), and F1-score to measure diagnostic performance.

---

## 🛠️ Toolboxes Required

* Image Processing Toolbox
* Statistics and Machine Learning Toolbox
* Computer Vision Toolbox

---

## 📊 Pipeline Visualization

The script generates a multi-plot figure for every image processed, allowing you to visualize the transformation:

| Step | Function | Purpose |
| --- | --- | --- |
| **Filtered** | `wiener2` | Denoising |
| **Smoothed** | `imdiffusefilt` | Edge-preserving smoothing |
| **Enhanced** | `adapthisteq` | Contrast improvement |
| **Features** | `extractHOGFeatures` | Characterizing the image for the model |

---

## 🚀 How to Use

1. **Dataset Setup:** Place your CT scans in a folder named `Patients_CT`. Create two subfolders: `no` (for healthy scans) and `yes` (for scans showing hemorrhage).
2. **Run the Script:** Open `hemorrhage_detection.m` in MATLAB and press **Run**.
3. **View Results:**
* The console will display the **Accuracy**, **Precision**, and **F1-Score**.
* A **Confusion Matrix** window will open to show true positives vs. false positives.
