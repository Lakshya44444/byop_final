# Protean Geo-Analytics Platform: The Geologist's Co-Pilot 🌍⛏️

> **A Neuro-Symbolic AI Architecture for Automated Seismic Interpretation & Risk Assessment.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![PyG](https://img.shields.io/badge/PyG-Graph%20Neural%20Networks-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-orange)

## Executive Summary
The **Protean Geo-Analytics Platform** aims to re-engineer the time-consuming process of Seismic Interpretation in the Oil & Gas industry. Conventional manual fault picking is subjective and takes months. While standard Deep Learning (Computer Vision) offers speed, it acts as a "black box," lacking the logical reasoning required for safety-critical operations.

This project closes the gap by deploying a **Neuro-Symbolic Architecture**:
1.  **Perception ("The Artist"):** A Stabilized Attention U-Net that segments seismic faults from raw amplitude images.
2.  **Reasoning ("The Engineer"):** A Graph Neural Network (GNN) that translates pixel predictions into topological representations to determine physical connectivity and leakage risk.

The system moves beyond simple edge detection to assess dip angles, connection to reservoir seals, and fluid transfer potential, calibrated against real-world Petrel well data.

---

## System Architecture

### 1. The Data Factory (Input Layer)
* **Seismic Volume:** Gullfaks Field (North Sea).
* **Coordinate Translator:** Custom algorithms to map Real-World Coordinates (Meters X/Y) to Image Coordinates (Trace/Pixel), solving severe domain mismatches.
* **Preprocessing:** Robust IQR Normalization and Mask Dilation (3x3 kernel) to prevent "vanishing signals" on thin faults.

### 2. "The Artist" (Visual Engine)
* **Model:** Stabilized U-Net with `nn.BatchNorm2d` to handle high-variance amplitude data.
* **Optimization:** Weighted Binary Cross Entropy (BCE) Loss (pos_weight=10.0) to solve the 99% background vs. 1% fault class imbalance.
* **Output:** Probability segmentation masks.

### 3. "The Engineer" (Physics Engine)
* **Pipeline:** Skeletonization (scikit-image) $\rightarrow$ Pruning $\rightarrow$ KDTree Edge Creation $\rightarrow$ Graph Construction.
* **Model:** Graph Attention Network (FaultGAT) using Multi-Head Attention.
* **Physics Logic:** Implements **Andersonian Fault Theory**.
    * *Nodes:* Carry Depth, Dip, and Amplitude features.
    * *Edges:* Represent physical connectivity.
    * *Risk Assessment:* Flags nodes as **Critical** if they are Vertical (Dip > 0.8) AND puncture the Reservoir Seal (-1783.84m Base Cretaceous).

### 4. Web Deployment
* **Interface:** Streamlit-based "Reality Ensemble Engine."
* **Features:** Real-time visualization of Input, Segmentation, and Risk Map.
* **Offline Reporting:** Automated generation of operational logs without external API dependencies.

---

##  Key Engineering Challenges & Solutions

### The "Vanishing Signal"
* **Problem:** 1-pixel wide faults disappeared when resizing 1000px images to 128px for the U-Net.
* **Solution:** Implemented **Mask Dilation** on training labels before downsampling to artificially thicken faults, ensuring features survived the resize operation.

### The "Empty Graph" Crash
* **Problem:** When the U-Net correctly predicted "No Faults" (clean rock), the Skeletonization algorithm returned an empty set, causing the GAT data loader to crash (division by zero).
* **Solution:** Implemented **Strict Guard Clauses**. The pipeline detects empty masks and bypasses GAT inference, automatically marking the sector as "Safe."

### The Sobel Filter Crisis (Pipeline Parity)
* **Problem:** The deployed web app performed poorly compared to training notebooks. Investigation revealed the training loader implicitly used a Sobel filter while production used raw data.
* **Solution:** Abandoned the Sobel filter entirely. Switched to **Robust IQR Normalization** for both training and inference to preserve low-frequency geological textures.

### The Vertical Depth Mismatch
* **Problem:** Fault labels appeared to be 7km away from the seismic image.
* **Solution:** Forensic analysis of SEGY headers revealed a recording delay. We parsed the `DelayRecordingTime` (1398ms) to align the image window (1398-2306ms) with the fault depth range (1688-2085ms).

---

## Validation Strategy

We employ a three-pillared validation strategy:

1.  **Quantitative:** * BCE Loss reduced from 1.07 to **0.35**.
    * Adjusted Recall using a "Tolerance Zone" to account for 1-pixel variations.
2.  **Physics "Sanity Check":**
    * Synthetic probe tests confirmed the GNN correctly identifies Horizontal lines as *Safe* (Class 0) and Vertical lines as *Risk* (Class 1).
3.  **Geological Context:**
    * **Juxtaposition Analysis:** Validated against Petrel Well Tops.
    * **Basin Classification:** Correctly ignores low-angle Thrust faults in an Extensional Basin context.

---

## Installation & Usage

### Prerequisites
* Python 3.8+
* CUDA-enabled GPU (recommended for training)

### 1. Run 
```bash
git clone [https://github.com/yourusername/protean-geo-analytics.git](https://github.com/yourusername/protean-geo-analytics.git)
cd protean-geo-analytics
pip install -r requirements.txt
streamlit run app.py


