# Lightweight Crop Disease Classification for Drones

A lightweight deep learning system for **crop disease classification on resource-constrained edge devices**. The project focuses on building a compact convolutional neural network that can process crop images efficiently enough for deployment on hardware used in agricultural drones and other edge-AI applications.

The model is developed using **PyTorch**, converted to **ONNX**, and prepared for deployment using **Edge Impulse** on the **STM32N6** microcontroller platform.

The main idea is simple:

> **Capture a crop image → process the image → classify the disease using a lightweight neural network → run the model on edge hardware.**

This project combines **computer vision, deep learning, model optimization, and embedded AI** into a single pipeline.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Why This Project?](#why-this-project)
* [Problem Statement](#problem-statement)
* [Project Objectives](#project-objectives)
* [How the System Works](#how-the-system-works)
* [Technology Stack](#technology-stack)
* [Model Architecture](#model-architecture)
* [Why a Lightweight Model?](#why-a-lightweight-model)
* [Dataset](#dataset)
* [Dataset Split](#dataset-split)
* [Training Configuration](#training-configuration)
* [Training Process](#training-process)
* [Model Evaluation](#model-evaluation)
* [Confusion Matrix](#confusion-matrix)
* [ONNX Conversion](#onnx-conversion)
* [Edge Impulse Deployment](#edge-impulse-deployment)
* [STM32N6 Deployment](#stm32n6-deployment)
* [Inference Performance](#inference-performance)
* [Project Pipeline](#project-pipeline)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Running the Project](#running-the-project)
* [Example Workflow](#example-workflow)
* [Important Concepts for Beginners](#important-concepts-for-beginners)
* [Challenges](#challenges)
* [Limitations](#limitations)
* [Future Improvements](#future-improvements)
* [Applications](#applications)
* [Conclusion](#conclusion)
* [Author](#author)

---

# Project Overview

Agricultural monitoring increasingly uses drones and cameras to collect images of crops. These images can contain useful information about plant health and disease.

However, collecting images is only the first part of the problem.

A conventional approach would send the captured images to a powerful computer or cloud server, where a deep learning model performs the classification.

That approach can introduce several problems:

* Internet connectivity may not always be available in agricultural environments.
* Sending every image to the cloud introduces communication overhead.
* Cloud-based inference can increase latency.
* Continuous data transmission can consume additional power.
* Large neural networks are difficult to run on small embedded devices.

This project explores an alternative approach:

**perform the disease classification directly on edge hardware.**

For this purpose, the project uses a lightweight CNN architecture called **FusionNet**.

The model is designed using lightweight neural network components such as:

* Ghost Modules
* Squeeze-and-Excitation (SE) Blocks
* Depthwise Separable Convolutions
* Swish activation

The trained model is then converted to **ONNX** and prepared for deployment on an **STM32N6** microcontroller using Edge Impulse.

---

# Why This Project?

Traditional deep learning models can achieve high accuracy, but they often require significant:

* RAM
* storage
* computational power
* processing time
* energy

These requirements can make them unsuitable for small edge devices.

A drone, for example, has limited computational resources and battery capacity. Adding a large neural network can increase both hardware requirements and energy consumption.

Therefore, the project focuses on the following question:

> **Can a lightweight deep learning model perform crop disease classification while being suitable for resource-constrained edge hardware?**

The project investigates this through model development, training, evaluation, ONNX conversion, and embedded deployment.

---

# Problem Statement

Crop diseases can significantly affect agricultural productivity.

Early identification of diseases can help farmers take appropriate action before the disease spreads across a larger area.

Manual inspection of large agricultural fields can be:

* time-consuming
* expensive
* difficult to scale
* dependent on human expertise

Drones provide a way to capture large numbers of crop images quickly.

However, simply collecting images does not automatically provide a diagnosis.

An automated computer vision system can analyze these images and classify the visible disease.

The challenge is that agricultural drones and embedded devices have considerably fewer computational resources than desktop GPUs or cloud servers.

Therefore, the objective is to develop a **lightweight image-classification model that can eventually run directly on edge hardware.**

---

# Project Objectives

The major objectives of this project are:

1. Develop a deep learning model for crop disease classification.
2. Use lightweight neural network components to reduce computational requirements.
3. Train and validate the model using a dedicated dataset.
4. Evaluate classification performance using multiple metrics.
5. Convert the trained PyTorch model into ONNX format.
6. Prepare the model for embedded deployment.
7. Deploy/test the model using Edge Impulse.
8. Target the STM32N6 edge-AI platform.
9. Investigate inference performance on CPU and hardware acceleration.
10. Explore the feasibility of real-time crop disease classification on resource-constrained devices.

---

# How the System Works

The complete workflow can be divided into several stages.

```text
Crop Image
    ↓
Image Preprocessing
    ↓
FusionNet
    ↓
Disease Classification
    ↓
Model Evaluation
    ↓
PyTorch Model
    ↓
ONNX Conversion
    ↓
Edge Impulse
    ↓
STM32N6
    ↓
Edge Inference
```

Each stage has a specific purpose.

### 1. Image Collection

Images containing crops and disease symptoms are used as input to the system.

### 2. Preprocessing

Before an image is passed to the neural network, it needs to be converted into a format that the model can understand.

Typical preprocessing operations include:

* resizing
* normalization
* tensor conversion

### 3. Model Inference

The processed image is passed through the FusionNet architecture.

The network extracts visual features from the image and uses them to determine the corresponding disease class.

### 4. Evaluation

The trained model is evaluated on data that was not used for training.

This helps measure how well the model generalizes to unseen images.

### 5. Model Conversion

The trained PyTorch model is converted into **ONNX format**.

ONNX provides a standardized representation that can be used across different machine-learning and deployment environments.

### 6. Embedded Deployment

The ONNX model is integrated into an Edge Impulse workflow and targeted toward the STM32N6 platform.

---

# Technology Stack

## Programming Language

### Python

Python is used for:

* dataset processing
* model development
* training
* evaluation
* visualization
* model conversion

---

## Deep Learning Framework

### PyTorch

PyTorch is used to:

* define the neural network
* train the model
* calculate loss
* optimize model parameters
* evaluate predictions
* save the trained model

---

## Computer Vision

### OpenCV

OpenCV is used for image-processing operations and computer-vision-related tasks.

---

## Model Deployment

### ONNX

ONNX, or **Open Neural Network Exchange**, provides a standardized model representation.

The trained PyTorch model is converted to ONNX before deployment.

---

## Edge AI Platform

### Edge Impulse

Edge Impulse is used to prepare and evaluate the model for embedded deployment.

It provides tools for:

* model deployment
* embedded inference
* performance analysis
* hardware-specific optimization

---

## Target Hardware

### STM32N6

The target embedded platform is the **STM32N6**.

The STM32N6 is designed for applications involving machine learning and computer vision at the edge.

The platform includes:

* ARM Cortex-M55 processing
* ST Neural-ART Accelerator

The accelerator is particularly useful for neural-network workloads.

---

# Model Architecture

The project uses a custom lightweight architecture called **FusionNet**.

FusionNet combines multiple lightweight neural network techniques to reduce computational requirements while retaining useful feature-extraction capabilities.

The main components are:

```text
Input Image
     ↓
Convolution / Feature Extraction
     ↓
Ghost Modules
     ↓
Depthwise Separable Convolutions
     ↓
SE Blocks
     ↓
Feature Representation
     ↓
Classification Layer
     ↓
Disease Class
```

The exact implementation of the architecture is contained in the project code.

---

# Ghost Modules

A **Ghost Module** is a lightweight neural-network building block designed to generate feature maps more efficiently.

A conventional convolution can require a large number of calculations to produce many feature maps.

Ghost modules attempt to reduce this cost by:

1. Generating a smaller number of primary feature maps.
2. Producing additional feature maps using inexpensive operations.

This can reduce:

* computational cost
* number of parameters
* memory requirements

while still producing useful feature representations.

This is particularly useful for edge devices.

---

# Squeeze-and-Excitation Blocks

The project also uses **Squeeze-and-Excitation (SE) blocks**.

SE blocks provide a mechanism for the network to learn which feature channels are more important.

Conceptually, the process is:

```text
Feature Maps
     ↓
Squeeze
     ↓
Channel Information
     ↓
Excitation
     ↓
Channel Weights
     ↓
Recalibrated Features
```

The network learns different weights for different channels.

Important feature channels can therefore receive more emphasis during classification.

---

# Depthwise Separable Convolutions

Depthwise separable convolution is another technique used to reduce computation.

A traditional convolution performs spatial filtering and channel mixing together.

Depthwise separable convolution separates these operations into:

1. **Depthwise convolution**
2. **Pointwise convolution**

This significantly reduces the number of calculations compared with a standard convolution.

This makes the technique useful for:

* mobile devices
* microcontrollers
* embedded systems
* edge-AI applications

---

# Swish Activation

The architecture also uses the **Swish activation function**.

Swish is a smooth nonlinear activation function that can be represented as:

```text
Swish(x) = x × sigmoid(x)
```

where:

```text
sigmoid(x) = 1 / (1 + e^-x)
```

The activation introduces non-linearity into the network, allowing the model to learn more complex relationships in the image data.

---

# Why a Lightweight Model?

A large neural network may perform well on a powerful computer but still be impractical for an embedded device.

For example, an embedded device may have significantly less:

* RAM
* flash storage
* processing power
* energy availability

than a desktop or cloud server.

Therefore, this project does not simply focus on classification accuracy.

It also considers:

* model size
* inference latency
* memory requirements
* hardware acceleration
* deployment feasibility

This makes the project an **edge-AI problem**, rather than only a conventional image-classification problem.

---

# Dataset

The project uses a crop-disease image dataset for training and evaluation.

The dataset is divided into three subsets:

```text
Dataset
│
├── Training Set
├── Validation Set
└── Test Set
```

Each subset serves a different purpose.

---

## Training Set

The training set is used to teach the neural network.

During training, the model:

1. receives an image
2. produces a prediction
3. compares the prediction with the actual label
4. calculates the loss
5. updates its parameters
6. repeats the process

---

## Validation Set

The validation set is used during development to monitor how the model performs on data that was not directly used to update its weights.

It helps with:

* monitoring generalization
* selecting training configurations
* detecting overfitting
* determining when training should stop

---

## Test Set

The test set is kept separate from training.

It provides a final evaluation of the trained model on unseen data.

The test set should not be used to repeatedly tune the model because doing so can make the reported test performance less representative of genuinely unseen data.

---

# Dataset Split

The current dataset configuration contains:

| Dataset    |    Images |
| ---------- | --------: |
| Training   |     2,721 |
| Validation |       525 |
| Test       |       178 |
| **Total**  | **3,424** |

The dataset therefore contains **3,424 images** across the three subsets.

---

# Training Configuration

The model was trained using the following configuration:

| Parameter               |     Value |
| ----------------------- | --------: |
| Framework               |   PyTorch |
| Batch Size              |        16 |
| Maximum Epochs          |        35 |
| Early Stopping Patience |         5 |
| Data Loader Workers     |         0 |
| Mixed Precision         |   Enabled |
| Architecture            | FusionNet |

---

## Batch Size

The batch size determines how many images are processed before the model updates its weights.

The project uses:

```text
Batch Size = 16
```

Therefore, the model processes up to 16 images in a training batch before performing the corresponding optimization step.

---

## Epochs

An epoch represents one complete pass through the training dataset.

The maximum number of epochs is:

```text
35
```

However, training may finish earlier because early stopping is enabled.

---

# Early Stopping

Training for too many epochs can cause **overfitting**.

Overfitting occurs when a model becomes very good at recognizing the training data but performs poorly on previously unseen data.

The project uses an early-stopping patience of:

```text
5 epochs
```

This means training can stop when validation performance does not improve for the specified patience period.

This helps prevent unnecessary training and can reduce overfitting.

---

# Mixed Precision Training

Mixed precision training is enabled in the training pipeline.

Instead of performing every calculation using the same numerical precision, compatible operations can use lower precision where appropriate.

This can provide:

* lower memory usage
* faster training
* improved computational efficiency

while maintaining appropriate numerical stability through the framework's mixed-precision mechanisms.

---

# Training Process

The general training process is:

```text
Load Dataset
      ↓
Split into Train / Validation / Test
      ↓
Preprocess Images
      ↓
Initialize FusionNet
      ↓
Start Training
      ↓
Forward Pass
      ↓
Calculate Loss
      ↓
Backpropagation
      ↓
Update Model Weights
      ↓
Validate Model
      ↓
Early Stopping Check
      ↓
Save Best Model
```

---

# Forward Pass

During a forward pass, an image is passed through the neural network.

For example:

```text
Image
 ↓
Convolution
 ↓
Feature Extraction
 ↓
Ghost Modules
 ↓
SE Blocks
 ↓
Depthwise Separable Convolutions
 ↓
Classification
 ↓
Prediction
```

The output represents the model's predicted class probabilities or class scores.

---

# Loss Function

The model compares its prediction with the correct label.

The difference between the prediction and the actual target is represented by a **loss value**.

During training, the objective is to reduce this loss.

Conceptually:

```text
Prediction
     ↓
Compare with Ground Truth
     ↓
Calculate Loss
     ↓
Backpropagation
     ↓
Update Weights
```

This process is repeated across the training dataset.

---

# Backpropagation

Backpropagation calculates how much each model parameter contributed to the prediction error.

The optimizer then updates the model's parameters.

Repeated over many batches and epochs, the model gradually learns useful patterns from the training data.

---

# Model Evaluation

Model evaluation is not limited to a single accuracy value.

The project can analyze:

* training loss
* validation loss
* training accuracy
* validation accuracy
* test performance
* confusion matrix
* inference latency
* model size

These measurements provide different information about the model.

---

# Accuracy

Accuracy represents the proportion of predictions that are correct.

For a classification problem:

```text
Accuracy =
Correct Predictions / Total Predictions
```

For example, if a model correctly classifies 90 out of 100 test images:

```text
Accuracy = 90 / 100
         = 90%
```

However, accuracy alone may not fully describe model performance, especially if the classes are imbalanced.

---

# Loss Curves

Training and validation loss can be plotted across epochs.

A typical plot contains:

```text
Epoch
  ↓
Loss
```

The curves can help identify:

* whether the model is learning
* whether training has stabilized
* whether overfitting may be occurring
* whether additional training is useful

---

# Accuracy Curves

Training and validation accuracy can similarly be monitored over time.

Comparing the two curves can provide insight into the model's generalization.

For example:

```text
Training Accuracy ↑
Validation Accuracy ↑
```

generally indicates that the model is learning useful patterns.

However, a very large difference between training and validation performance can indicate potential overfitting.

---

# Confusion Matrix

A confusion matrix provides a more detailed view of classification performance.

For multiple disease classes, the matrix can show:

* which classes are correctly classified
* which classes are confused with one another
* which disease classes are harder for the model to distinguish

Conceptually:

```text
                 Predicted
              A      B      C
Actual A      ✓      .      .
Actual B      .      ✓      .
Actual C      .      .      ✓
```

The diagonal represents correct classifications.

Off-diagonal values represent misclassifications.

---

# ONNX Conversion

After training, the PyTorch model can be converted into **ONNX**.

The conversion pipeline is:

```text
PyTorch Model
      ↓
ONNX Export
      ↓
ONNX Model
      ↓
Edge Deployment
```

ONNX is useful because it provides a standardized representation of neural-network models.

This makes it easier to move a model from a training environment into another deployment environment.

---

# Why ONNX?

The training environment and deployment environment are often different.

For example:

```text
Training
PyTorch + Python + GPU/CPU
```

while deployment might involve:

```text
Embedded Device
C/C++ Runtime
Microcontroller
Hardware Accelerator
```

ONNX provides an intermediate representation between these environments.

---

# Edge Impulse Deployment

The converted model is prepared for embedded deployment using **Edge Impulse**.

The general process is:

```text
Train Model
     ↓
Export PyTorch Model
     ↓
Convert to ONNX
     ↓
Import/Prepare Model
     ↓
Configure Embedded Deployment
     ↓
Build Inference
     ↓
Test on Target Hardware
```

Edge Impulse provides tools for evaluating the model in an embedded context.

---

# STM32N6 Deployment

The target hardware for this project is the **STM32N6**.

The platform contains:

* **Cortex-M55 CPU**
* **ST Neural-ART Accelerator**

The Cortex-M55 provides the general-purpose processing capability, while the Neural-ART Accelerator is designed to accelerate supported neural-network workloads.

This makes the platform suitable for investigating AI inference directly at the edge.

---

# Inference Performance

One of the important parts of this project is measuring inference performance.

The current deployment measurements include:

| Metric                        | Measured Value |
| ----------------------------- | -------------: |
| CPU Inference Latency         |     ~12,892 ms |
| Accelerator Inference Latency |      ~2,149 ms |

The accelerator measurement is substantially lower than the CPU-only measurement for the tested deployment configuration.

These measurements demonstrate why hardware acceleration can be important when running neural networks on embedded devices.

> **Important:** Inference latency depends on the exact deployment configuration, input dimensions, runtime, model implementation, and hardware settings. These values should therefore be treated as measurements from this project's tested configuration rather than universal STM32N6 performance numbers.

---

# Model Size vs Runtime Memory

A common beginner mistake is to assume that:

```text
Model Size = Total Memory Required
```

This is not necessarily true.

A neural network can have a relatively small serialized model while requiring significantly more memory during execution.

Runtime memory can include:

* model weights
* intermediate feature maps
* activation buffers
* tensor buffers
* runtime structures
* memory used by the inference engine

Therefore, embedded deployment needs to consider both:

```text
Storage Requirements
+
Runtime Memory Requirements
```

rather than only the size of the model file.

---

# Project Pipeline

The complete project can be summarized as:

```text
                ┌───────────────────┐
                │   Crop Images     │
                └─────────┬─────────┘
                          ↓
                ┌───────────────────┐
                │   Preprocessing   │
                └─────────┬─────────┘
                          ↓
                ┌───────────────────┐
                │     FusionNet     │
                │  Lightweight CNN  │
                └─────────┬─────────┘
                          ↓
                ┌───────────────────┐
                │ Disease Prediction│
                └─────────┬─────────┘
                          ↓
                ┌───────────────────┐
                │    Evaluation     │
                └─────────┬─────────┘
                          ↓
                ┌───────────────────┐
                │  PyTorch → ONNX   │
                └─────────┬─────────┘
                          ↓
                ┌───────────────────┐
                │   Edge Impulse    │
                └─────────┬─────────┘
                          ↓
                ┌───────────────────┐
                │     STM32N6       │
                └─────────┬─────────┘
                          ↓
                ┌───────────────────┐
                │ Edge AI Inference │
                └───────────────────┘
```

---

# Project Structure

A recommended repository structure is:

```text
Crop-Disease-Classification/
│
├── dataset/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── models/
│   ├── fusionnet.py
│   └── ...
│
├── notebooks/
│   └── training.ipynb
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── export_onnx.py
│
├── results/
│   ├── confusion_matrix/
│   ├── accuracy/
│   └── loss/
│
├── requirements.txt
├── README.md
└── .gitignore
```

The exact directory names can be changed to match the actual repository.

---

# Installation

## 1. Clone the Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd Crop-Disease-Classification
```

Replace `<YOUR-GITHUB-REPOSITORY-URL>` with the actual repository URL.

---

## 2. Create a Virtual Environment

Creating a virtual environment prevents project dependencies from interfering with other Python projects.

On Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Typical dependencies for this type of project include:

```text
torch
torchvision
opencv-python
numpy
matplotlib
scikit-learn
onnx
onnxruntime
```

The exact versions should be taken from the project's actual dependency file.

---

# Running the Project

The general workflow is:

### Train the model

```bash
python train.py
```

### Evaluate the model

```bash
python evaluate.py
```

### Run inference

```bash
python inference.py
```

### Export to ONNX

```bash
python export_onnx.py
```

The exact commands depend on the files included in the repository.

---

# Example Workflow

Suppose a user has a crop image:

```text
crop_image.jpg
```

The system processes it as follows:

```text
crop_image.jpg
       ↓
Resize / Normalize
       ↓
Convert to Tensor
       ↓
FusionNet
       ↓
Feature Extraction
       ↓
Classification Layer
       ↓
Predicted Disease
```

The final output is the disease class predicted by the model.

---

# Important Concepts for Beginners

## What is Machine Learning?

Machine learning is a method where a computer learns patterns from data instead of being explicitly programmed with every possible rule.

For example, instead of manually writing:

```text
IF leaf has pattern X:
    disease = A
```

we provide many labeled examples and allow the model to learn the relationship.

---

# What is Deep Learning?

Deep learning is a branch of machine learning based on neural networks with multiple computational layers.

Deep learning is particularly useful for complex data such as:

* images
* audio
* video
* text

This project uses deep learning for image classification.

---

# What is a CNN?

CNN stands for **Convolutional Neural Network**.

CNNs are commonly used for image-related tasks because they can learn visual patterns such as:

* edges
* textures
* shapes
* colors
* more complex structures

A CNN gradually combines these low-level features into higher-level representations.

---

# What is Edge AI?

**Edge AI** means running artificial-intelligence models close to where the data is generated.

Instead of:

```text
Camera
  ↓
Internet
  ↓
Cloud Server
  ↓
Prediction
```

an edge-AI system can use:

```text
Camera
  ↓
Embedded Device
  ↓
Prediction
```

This can reduce dependency on network connectivity and can reduce communication latency.

---

# What is Model Optimization?

Model optimization involves modifying or preparing a model so that it can operate more efficiently.

Possible optimization techniques include:

* reducing parameters
* reducing computation
* quantization
* pruning
* lightweight architectures
* hardware acceleration

This project primarily addresses efficiency through the use of lightweight architectural components.

---

# What is Inference?

Inference is the process of using a trained model to make a prediction.

Training:

```text
Images + Labels
      ↓
Learn Parameters
```

Inference:

```text
New Image
    ↓
Trained Model
    ↓
Prediction
```

The STM32N6 deployment focuses on inference rather than training.

---

# Challenges

Several challenges are involved in deploying deep learning models on microcontrollers.

## 1. Limited Memory

Microcontrollers have considerably less memory than desktop computers.

The model therefore needs to fit within the available memory constraints.

---

## 2. Computational Cost

Convolutional neural networks can require a large number of mathematical operations.

This makes lightweight convolution techniques particularly useful.

---

## 3. Model Conversion

A model that works correctly in PyTorch does not automatically guarantee that every operation will work identically after conversion to another deployment format.

The conversion process therefore needs to be tested carefully.

---

## 4. Runtime Memory

Even when the serialized model appears small, intermediate tensors and activation buffers can require additional memory.

This can become an important constraint on embedded hardware.

---

## 5. Accuracy vs Efficiency

Making a model smaller can sometimes affect its predictive performance.

Therefore, embedded AI involves balancing multiple objectives:

```text
Accuracy
    ↕
Model Size
    ↕
Latency
    ↕
Memory
    ↕
Power Consumption
```

A useful embedded model needs to satisfy the requirements of the target application rather than optimizing only one metric.

---

# Limitations

The current project has several limitations.

### Dataset Size

The dataset contains 3,424 images across the training, validation, and test sets. A larger and more diverse dataset could provide better coverage of real-world conditions.

### Environmental Variation

Real drone images can vary significantly because of:

* lighting
* camera angle
* distance
* shadows
* background vegetation
* weather
* leaf orientation

A model trained primarily on controlled or limited datasets may not perform identically under every real-world condition.

### Embedded Memory

Model deployment needs to account for both model storage and runtime memory.

### Hardware-Specific Performance

The reported latency values correspond to the tested deployment configuration and should not automatically be interpreted as universal performance for every STM32N6 configuration.

---

# Future Improvements

Potential future improvements include:

## 1. Larger Dataset

Increase the number and diversity of training images.

This could help the model handle more real-world conditions.

---

## 2. Real Drone Images

Collect images directly from agricultural drones.

This would allow the model to be evaluated under realistic aerial conditions.

---

## 3. Quantization

Investigate lower-precision representations such as INT8 where supported.

Potential benefits include:

* smaller model size
* lower memory requirements
* faster inference
* lower power consumption

---

## 4. Further Model Optimization

Additional optimization techniques could be investigated, including:

* pruning
* knowledge distillation
* operator optimization
* architecture search
* hardware-specific optimization

---

## 5. Real-Time Drone Integration

A future version could connect the classification system directly to a drone camera.

The intended workflow would be:

```text
Drone Camera
     ↓
Crop Image
     ↓
STM32N6
     ↓
Disease Classification
     ↓
Disease Location / Result
```

---

## 6. Field-Level Disease Mapping

If GPS information is combined with model predictions, detected disease locations could potentially be mapped across an agricultural field.

This could help create a field-level disease monitoring system.

---

# Applications

The technology developed in this project can potentially be used in:

* agricultural drones
* smart farming systems
* crop monitoring
* precision agriculture
* automated disease detection
* edge-AI cameras
* agricultural robotics
* embedded computer vision systems

The broader concept is to move AI inference closer to the physical environment where the data is generated.

---

# Conclusion

This project explores the development of a **lightweight crop disease classification system designed with edge deployment in mind**.

A custom **FusionNet** architecture combines:

* Ghost Modules
* Squeeze-and-Excitation Blocks
* Depthwise Separable Convolutions
* Swish activation

to create a neural network intended to reduce computational requirements compared with heavier conventional architectures.

The model is trained using **PyTorch**, evaluated using training/validation/test data, converted into **ONNX**, and prepared for deployment using **Edge Impulse**.

The project then targets the **STM32N6**, combining the Cortex-M55 processor with the ST Neural-ART Accelerator for embedded neural-network inference.

The measured deployment configuration achieved approximately:

```text
CPU latency:         ~12,892 ms
Accelerator latency: ~2,149 ms
```

The project therefore demonstrates the complete workflow from:

```text
Dataset
   ↓
Deep Learning
   ↓
Lightweight CNN
   ↓
Model Evaluation
   ↓
ONNX
   ↓
Edge Impulse
   ↓
Embedded Hardware
   ↓
Edge AI Inference
```

Rather than treating crop disease detection purely as a software classification problem, this project investigates the additional challenges involved in taking a deep learning model from a development environment and bringing it closer to **real-world edge deployment**.

---

# Author

**Sarang Palsutkar**

B.Tech — Computer Science Engineering
VIT Bhopal University

---

## Project Focus

```text
Computer Vision
Deep Learning
PyTorch
CNN
Edge AI
Model Optimization
ONNX
Embedded Machine Learning
STM32N6
Agricultural AI
```
