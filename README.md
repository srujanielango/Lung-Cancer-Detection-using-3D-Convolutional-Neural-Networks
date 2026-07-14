# Lung Cancer Detection Using 3D Convolutional Neural Networks

A deep-learning project that processes chest CT scans as three-dimensional volumes and uses a **3D Convolutional Neural Network (3D CNN)** to classify patients as having lung cancer or not having lung cancer.

The project was developed using data from the **2017 Kaggle Data Science Bowl**, which challenged participants to build automated methods for detecting lung cancer from high-resolution CT scans.

> **Important:** This repository is an educational and experimental machine-learning project. It is not a clinically validated diagnostic system and must not be used to make medical decisions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Why a 3D CNN?](#why-a-3d-cnn)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Repository Structure](#repository-structure)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Running the Project](#running-the-project)
- [Model Output](#model-output)
- [Evaluation and Observations](#evaluation-and-observations)
- [Known Limitations](#known-limitations)
- [Potential Improvements](#potential-improvements)
- [Medical and Ethical Disclaimer](#medical-and-ethical-disclaimer)
- [References](#references)

---

## Project Overview

Lung cancer screening commonly relies on radiologists reviewing computed tomography, or CT, scans for suspicious lesions and nodules. A single CT study can contain many two-dimensional DICOM image slices. Reviewing these scans manually is time-consuming, and subtle findings may be difficult to identify consistently.

This project explores whether a neural network can learn three-dimensional patterns from a patient's CT scan and predict one of two classes:

- **Cancer**
- **No cancer**

The workflow:

1. Reads DICOM slices for each patient.
2. Sorts the slices into their anatomical order.
3. Resizes each slice to a consistent image size.
4. reduces or groups each scan into a fixed number of slices.
5. combines the processed slices into a 3D volume.
6. associates each volume with its patient-level cancer label.
7. trains a 3D CNN to perform binary classification.
8. generates predictions for unseen patient scans.

The notebook uses a processed input shape of:

```text
50 × 50 × 20
```

where:

- `50 × 50` is the resized height and width of each CT slice.
- `20` is the standardized number of slices used for each patient.

---

## Problem Statement

The goal is to build an automated classification model that can analyze a patient's chest CT scan and estimate whether the patient is likely to have lung cancer.

The main technical challenge is that CT studies vary between patients:

- scans contain different numbers of slices;
- image dimensions may vary;
- DICOM metadata must be interpreted correctly;
- slices must be placed in anatomical order;
- the complete 3D study is too large to pass directly into a simple model;
- cancer and non-cancer classes may not be evenly distributed.

The project addresses part of this variability by converting each patient scan into a standardized 3D representation before training the CNN.

---

## Why a 3D CNN?

A conventional 2D CNN analyzes individual images independently. That approach can identify patterns within one CT slice but does not directly model how a feature continues across adjacent slices.

A **3D CNN** applies filters across:

- image height;
- image width;
- scan depth.

This allows the network to learn volumetric features, including how structures and possible lesions appear across neighboring CT slices.

For medical imaging, this is important because a pulmonary nodule is a three-dimensional structure rather than an isolated feature on one image.

---

## Dataset

The project is based on the **Data Science Bowl 2017** lung cancer dataset hosted on Kaggle.

Competition page:

```text
https://www.kaggle.com/c/data-science-bowl-2017
```

The dataset includes:

- patient folders identified by anonymized patient IDs;
- chest CT scans stored as DICOM files;
- multiple slices for each patient;
- a CSV file containing patient-level cancer labels.

The label file follows the general structure:

| Patient ID | Cancer |
|---|---:|
| anonymized_patient_id | 0 or 1 |

Label meaning:

- `0`: no cancer
- `1`: cancer

### Data Access

The CT images are not included in this GitHub repository because of their size and the dataset's access requirements. Download the data directly from Kaggle after accepting the competition rules.

---

## Project Workflow

```text
Kaggle CT Scan Data
        |
        v
Read Patient DICOM Files
        |
        v
Sort Slices by Anatomical Position
        |
        v
Resize Each Slice to 50 × 50
        |
        v
Standardize Scan Depth to 20 Slices
        |
        v
Create Patient-Level 3D NumPy Volumes
        |
        v
Attach Binary Cancer Labels
        |
        v
Train 3D Convolutional Neural Network
        |
        v
Evaluate Predictions
        |
        v
Predict Cancer / No Cancer
```

---

## Data Preprocessing

Medical image preprocessing is a major part of this project because every patient's scan can contain a different number of DICOM slices.

### 1. Reading DICOM Files

The notebook reads all DICOM files contained in a patient's directory.

Each DICOM file represents one two-dimensional CT slice and contains both:

- pixel data;
- metadata describing the image and its position.

### 2. Sorting the Slices

Slices must be ordered correctly before constructing a 3D scan. The project reorganizes the images based on DICOM information so they represent the patient's anatomy in sequence.

Incorrect ordering would produce a distorted 3D volume and reduce the model's ability to learn meaningful spatial features.

### 3. Resizing

Each CT slice is resized to:

```python
size = 50
```

This produces a consistent two-dimensional shape of:

```text
50 × 50
```

Reducing the image resolution also lowers memory and GPU requirements, although it removes some fine-grained image detail.

### 4. Standardizing the Number of Slices

Different CT scans contain different numbers of slices. Neural networks require samples within a batch to have consistent dimensions.

The notebook sets:

```python
NoSlices = 20
```

The preprocessing logic divides or groups the available slices into approximately 20 chunks and calculates representative images for those chunks. This creates a fixed-depth volume for every patient.

### 5. Creating the 3D Patient Volume

After resizing and slice standardization, each patient's data is represented as:

```text
50 × 50 × 20
```

The processed volume, label, and patient identifier are stored together.

### 6. Encoding Labels

The binary labels are converted to one-hot encoded values suitable for the two-node output layer:

```text
No cancer -> [1, 0]
Cancer    -> [0, 1]
```

### 7. Saving Processed Data

The processed records are saved as a NumPy file:

```text
imageDataNew-50-50-20.npy
```

Saving the transformed data avoids repeating the computationally expensive DICOM preprocessing every time the model is trained.

---

## Model Architecture

The repository implements a custom TensorFlow 1.x 3D CNN.

### Input

```text
Batch × 50 × 50 × 20 × 1
```

The final dimension represents the single grayscale CT channel.

### Convolutional Blocks

The network includes five 3D convolution stages with increasing channel depth:

| Layer | Kernel | Input Channels | Output Channels | Activation |
|---|---|---:|---:|---|
| Conv3D 1 | 3 × 3 × 3 | 1 | 32 | ReLU |
| Conv3D 2 | 3 × 3 × 3 | 32 | 64 | ReLU |
| Conv3D 3 | 3 × 3 × 3 | 64 | 128 | ReLU |
| Conv3D 4 | 3 × 3 × 3 | 128 | 256 | ReLU |
| Conv3D 5 | 3 × 3 × 3 | 256 | 512 | ReLU |

Each convolutional stage is followed by 3D max pooling using a `2 × 2 × 2` window and stride.

### Fully Connected Layer

The convolutional output is flattened and passed to a dense layer with:

```text
1,024 units
```

A ReLU activation is applied, followed by dropout.

### Dropout

The notebook defines:

```python
keep_rate = 0.8
```

This retains 80% of the dense-layer activations during training and helps reduce overfitting.

### Output Layer

The final layer contains two output units:

```text
[No cancer, Cancer]
```

The class with the highest output score becomes the model's prediction.

### Optimization

The model uses:

- **loss:** softmax cross-entropy;
- **optimizer:** Adam;
- **learning rate:** `0.001`;
- **training epochs:** `10`.

### Architecture Summary

```text
Input Volume: 50 × 50 × 20 × 1
        |
        v
Conv3D, 32 filters, ReLU
        |
        v
3D Max Pooling
        |
        v
Conv3D, 64 filters, ReLU
        |
        v
3D Max Pooling
        |
        v
Conv3D, 128 filters, ReLU
        |
        v
3D Max Pooling
        |
        v
Conv3D, 256 filters, ReLU
        |
        v
3D Max Pooling
        |
        v
Conv3D, 512 filters, ReLU
        |
        v
3D Max Pooling
        |
        v
Flatten
        |
        v
Dense, 1,024 units, ReLU
        |
        v
Dropout
        |
        v
Dense, 2 outputs
        |
        v
Cancer / No Cancer
```

---

## Repository Structure

```text
Lung-Cancer-Detection-using-3D-Convolutional-Neural-Networks/
├── Lung Cancer 3D CNN.ipynb
├── Project Report.docx
├── README.md
├── .gitignore
└── .gitattributes
```

### File Descriptions

#### `Lung Cancer 3D CNN.ipynb`

The primary implementation notebook. It contains:

- project background;
- package imports;
- DICOM loading;
- CT slice visualization;
- image preprocessing;
- patient-volume generation;
- NumPy dataset creation;
- 3D CNN construction;
- model training;
- evaluation logic;
- patient-level prediction.

#### `Project Report.docx`

The accompanying project report containing additional background, implementation discussion, and results.

#### `README.md`

Project documentation and setup instructions.

---

## Technology Stack

The original notebook was created with an older Python and TensorFlow environment.

### Core Technologies

- Python 3.5
- Jupyter Notebook
- TensorFlow 1.x
- TFLearn
- NumPy
- pandas
- OpenCV
- pydicom, imported in the original notebook as `dicom`
- Matplotlib
- scikit-learn
- CUDA and cuDNN for optional GPU acceleration

### Important Compatibility Note

The source code uses TensorFlow 1.x constructs such as:

```python
tf.placeholder
tf.Session
tf.random_normal
tf.nn.max_pool3d
```

These APIs are not directly compatible with standard TensorFlow 2.x execution.

The original notebook also imports:

```python
import dicom
```

Modern pydicom versions normally use:

```python
import pydicom
```

To reproduce the notebook with minimal code changes, use a compatible legacy environment. Alternatively, modernize the implementation to TensorFlow 2.x/Keras or PyTorch.

---

## Installation

### Option 1: Reproduce the Original Legacy Environment

This option is closest to the original implementation.

```bash
git clone https://github.com/srujanielango/Lung-Cancer-Detection-using-3D-Convolutional-Neural-Networks.git

cd Lung-Cancer-Detection-using-3D-Convolutional-Neural-Networks

conda create -n lung-cancer-3dcnn python=3.5
conda activate lung-cancer-3dcnn
```

Install compatible versions of the required packages:

```bash
pip install numpy pandas matplotlib opencv-python scikit-learn jupyter pydicom
pip install tensorflow-gpu==1.4.0
pip install tflearn
```

Because Python 3.5 and TensorFlow 1.x are no longer actively supported, package installation may fail on modern operating systems. A container or older Conda environment may be necessary.

### Option 2: CPU-Only Legacy Setup

For environments without a compatible NVIDIA GPU:

```bash
pip install tensorflow==1.4.0
```

Training a 3D CNN on CPU can be very slow.

### GPU Requirements

The original project recommends an NVIDIA GPU with:

- CUDA;
- cuDNN;
- a CUDA version compatible with the installed TensorFlow release.

TensorFlow, CUDA, cuDNN, the GPU driver, and Python versions must be mutually compatible.

---

## Dataset Setup

After downloading the Data Science Bowl 2017 files, organize the data so the paths match the notebook.

The notebook expects a layout similar to:

```text
Lung_Cancer/
├── stage1/
│   └── stage1/
│       ├── patient_id_1/
│       │   ├── slice_1.dcm
│       │   ├── slice_2.dcm
│       │   └── ...
│       ├── patient_id_2/
│       └── ...
└── stage1_labels/
    └── stage1_labels.csv
```

The original path variables are:

```python
dataDirectory = "Lung_Cancer/stage1/stage1/"
labels = pd.read_csv(
    "Lung_Cancer/stage1_labels/stage1_labels.csv",
    index_col=0
)
```

Update these paths in the notebook when using a different directory structure.

Do not commit the original CT data to GitHub. The files are large and must be handled according to Kaggle's dataset terms.

---

## Running the Project

### 1. Activate the Environment

```bash
conda activate lung-cancer-3dcnn
```

### 2. Start Jupyter Notebook

```bash
jupyter notebook
```

### 3. Open the Notebook

Open:

```text
Lung Cancer 3D CNN.ipynb
```

### 4. Configure Dataset Paths

Verify that the following values point to the downloaded data:

```python
dataDirectory = "Lung_Cancer/stage1/stage1/"
labels = pd.read_csv(
    "Lung_Cancer/stage1_labels/stage1_labels.csv",
    index_col=0
)
```

### 5. Run the Preprocessing Cells

The preprocessing cells:

- load patient scans;
- sort DICOM slices;
- resize images;
- standardize scan depth;
- assign labels;
- save the processed NumPy data.

Expected processed-data filename:

```text
imageDataNew-50-50-20.npy
```

### 6. Train the Model

Run the model-definition and training cells.

The notebook loads the processed data using:

```python
imageData = np.load("imageDataNew-50-50-20.npy")
```

It then creates training and testing subsets and trains the network for 10 epochs.

### 7. Generate a Prediction

The final section outputs a patient ID and its predicted class.

Example format:

```text
Patient: <patient_id>
Predicted: No Cancer
```

---

## Model Output

The model produces one of two patient-level classifications:

```text
Cancer
```

or:

```text
No Cancer
```

The notebook includes an example prediction for one anonymized patient and reports a predicted class of `No Cancer`.

A model prediction is not a diagnosis. Reliable medical systems require external clinical validation, calibrated probabilities, explainability, and review by qualified clinicians.

---

## Evaluation and Observations

The notebook discusses performance using classification outcomes, including false negatives.

One reported observation is:

```text
18 false negatives out of 100 evaluated cases
```

A false negative occurs when the model predicts **No cancer** for a patient whose true label is **Cancer**.

For a screening task, false negatives are particularly serious because they may delay additional testing or treatment. Therefore, raw accuracy alone is not an adequate metric for this use case.

More appropriate evaluation measures include:

- sensitivity or recall for the cancer class;
- specificity;
- precision;
- F1 score;
- ROC-AUC;
- PR-AUC;
- confusion matrix;
- false-negative rate;
- probability calibration.

The reported result should be interpreted as an experimental notebook output, not as evidence of clinical performance.

---

## Known Limitations

### 1. Legacy Software

The notebook depends on Python 3.5 and TensorFlow 1.x APIs that are obsolete and difficult to install on modern systems.

### 2. Limited Image Resolution

Each CT slice is reduced to `50 × 50` pixels. This significantly lowers memory usage but may remove small nodules and subtle radiological detail.

### 3. Fixed Scan Depth

All patients are represented using only 20 aggregated slices. This simplifies training but can discard information from scans containing hundreds of original slices.

### 4. Simplified Preprocessing

A modern lung CT pipeline would typically include:

- conversion from raw pixel values to Hounsfield Units;
- lung-window normalization;
- segmentation of the lung fields;
- resampling to uniform voxel spacing;
- clipping extreme intensity values;
- consistent spatial orientation;
- artifact handling.

These steps are not comprehensively implemented in the original notebook.

### 5. Training and Test Methodology

The code uses fixed array slices for training and testing rather than a documented stratified patient-level split. This may produce unstable results and makes exact reproducibility difficult.

### 6. Small Effective Evaluation Set

Some notebook sections use a small subset of the processed data. Results from a small test set may not generalize to the full population.

### 7. Class Imbalance

Lung cancer datasets frequently contain more negative than positive cases. Without class weighting, balanced sampling, or appropriate metrics, a model may favor the majority class.

### 8. Broad Exception Handling

The training loop contains broad exception handling that can silently skip problematic records. Silent failures make debugging and result validation difficult.

### 9. No Saved Model Artifact

The notebook trains and evaluates the model within a TensorFlow session but does not provide a versioned model checkpoint for direct reuse.

### 10. No Clinical Validation

The model has not been validated prospectively, reviewed as a medical device, or tested across hospitals, scanner manufacturers, demographic groups, and clinical environments.

---

## Potential Improvements

### Data Engineering

- Convert DICOM pixel arrays to Hounsfield Units.
- Resample scans to a standard voxel spacing.
- Segment lungs before classification.
- preserve more spatial resolution.
- cache preprocessing outputs with metadata and versioning.
- validate malformed or incomplete DICOM series.
- replace silent exception handling with structured error reporting.

### Modeling

- migrate the network to TensorFlow 2.x/Keras or PyTorch;
- add batch normalization;
- use modern weight initialization;
- tune dropout and learning rate;
- use weighted loss or focal loss;
- evaluate 3D ResNet or DenseNet architectures;
- use transfer learning from medical-imaging models;
- apply volumetric data augmentation;
- save checkpoints and training histories.

### Evaluation

- create reproducible train, validation, and test splits;
- stratify splits by cancer label;
- use cross-validation where computationally feasible;
- report sensitivity and specificity;
- plot ROC and precision-recall curves;
- calculate confidence intervals;
- choose thresholds based on clinical priorities;
- compare performance with simpler baseline models.

### Explainability

- generate 3D or slice-level saliency maps;
- use Grad-CAM variants for volumetric networks;
- highlight regions contributing to a prediction;
- have visual explanations reviewed by medical experts.

### Software Engineering

- move preprocessing, training, and inference into separate modules;
- add a `requirements.txt` or Conda environment file;
- introduce configuration files instead of hard-coded paths;
- add automated tests;
- log experiments using MLflow, TensorBoard, or Weights & Biases;
- package inference behind a command-line interface or API;
- add continuous integration.

---

## Suggested Modern Project Structure

A modernized version of the project could use:

```text
lung-cancer-3d-cnn/
├── README.md
├── requirements.txt
├── environment.yml
├── configs/
│   └── training.yaml
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── data/
│   │   ├── dicom_loader.py
│   │   ├── preprocessing.py
│   │   └── dataset.py
│   ├── models/
│   │   └── cnn3d.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
├── artifacts/
└── reports/
```

This would improve maintainability, testing, reproducibility, and separation of concerns.

---

## Medical and Ethical Disclaimer

This project is provided only for educational, research, and portfolio purposes.

It is **not**:

- a medical device;
- a clinically approved diagnostic model;
- a substitute for a radiologist;
- suitable for treatment decisions;
- validated for use with real patients.

Medical imaging models can produce false positives and false negatives. Any real-world clinical implementation would require rigorous validation, privacy controls, security reviews, regulatory assessment, bias testing, monitoring, and physician oversight.

---

## References

1. Data Science Bowl 2017, Kaggle  
   `https://www.kaggle.com/c/data-science-bowl-2017`

2. TensorFlow  
   `https://www.tensorflow.org/`

3. pydicom  
   `https://pydicom.github.io/`

4. OpenCV  
   `https://opencv.org/`

---

## Acknowledgments

- Kaggle and Booz Allen Hamilton for organizing the 2017 Data Science Bowl.
- The National Cancer Institute for supporting access to the imaging data used in the competition.
- The open-source Python, TensorFlow, pydicom, NumPy, pandas, OpenCV, Matplotlib, and scikit-learn communities.

---

## License

No explicit license is currently included in this repository.

Before reusing or distributing the source code, add an appropriate open-source license and ensure that all use of the Kaggle dataset complies with the competition's terms.
