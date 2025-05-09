# Traffic Monitoring in Bangladesh: Custom Object Detection with YOLO
![](https://github.com/SawsanYusuf/Traffic-Monitoring-in-Bangladesh/blob/main/Traffic.jpg)

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Data Transformation for YOLO](#data-transformation-for-yolo)
    * [Annotation Parsing and Conversion](#annotation-parsing-and-conversion)
    * [YOLO Annotation Format](#yolo-annotation-format)
3.  [Preparing the Directory Structure](#preparing-the-directory-structure)
    * [Image Format Conversion](#image-format-conversion)
    * [Directory Organization](#directory-organization)
    * [Data Splitting](#data-splitting)
4.  [Training the Model](#training-the-model)
    * [Configuration with YAML](#configuration-with-yaml)
    * [Model Selection and Training](#model-selection-and-training)
    * [Training Output](#training-output)
5.  [Evaluating the Model](#evaluating-the-model)
    * [Exploring Training Results](#exploring-training-results)
        * [Precision-Recall Curve](#precision-recall-curve)
        * [Confusion Matrix](#confusion-matrix)
        * [Analyzing Training Loss](#analyzing-training-loss)

---

## 1. Project Overview <a name="project-overview"></a>

This project aims to develop a custom object detection model for traffic analysis in Bangladesh using the [Dhaka-AI traffic video dataset](https://www.kaggle.com/datasets/rifat963/dhakaai-dhaka-based-traffic-detection-dataset?select=train). While pre-trained YOLO models are effective for general object detection, they lack the ability to identify specific, custom traffic-related classes relevant to our needs. To address this, we fine-tune a YOLO model on the Dhaka-AI dataset, enabling it to accurately detect and classify targeted traffic objects.

## 2. Data Transformation for YOLO <a name="data-transformation-for-yolo"></a>

### 2.1 Annotation Parsing and Conversion <a name="annotation-parsing-and-conversion"></a>

Our process involves retraining a YOLO model to recognize specific traffic objects within the Dhaka-AI dataset. Initially, a pre-trained YOLO model was used, but it could only detect objects from its original training set. To achieve finer-grained classification, we retrain the YOLO model with a custom dataset containing labeled annotations.

This section covers the essential steps for preparing the data:

* Formatting data for YOLO training.
* Parsing annotation files.
* Converting annotation data to the YOLO format.
* Saving processed annotations for model training.

Before processing, we define paths to the training images and annotation directories. We use the `tree` command (or equivalent file system navigation) to understand the dataset's organization.

### 2.2 YOLO Annotation Format <a name="yolo-annotation-format"></a>

Each image has a corresponding annotation file in XML format. These files contain:

* Object class names.
* Bounding box coordinates (xmin, ymin, xmax, ymax).
* Image dimensions (width, height).

YOLO requires annotations in a different format:

* Class index (numerical).
* Center (x, y) of the bounding box (normalized).
* Width and height of the bounding box (normalized).

We create a mapping to convert object names to numerical class indices. The XML annotations provide pixel-based bounding box coordinates, which are then converted to normalized center coordinates and dimensions.

We implement functions to:

* Convert bounding box coordinates to the YOLO format.
* Save the parsed annotations in the required text file format, where each line represents an object: `class_idx x_center y_center width height`.

These functions are applied to all XML annotation files in the dataset.

**Outcome:**

* Dataset paths are organized.
* XML annotations are parsed.
* Bounding box coordinates are converted to the YOLO format.
* Annotations are saved in a YOLO-compatible format.

## 3. Preparing the Directory Structure <a name="preparing-the-directory-structure"></a>

After parsing the annotations, we construct the specific directory structure that YOLO expects for training.

### 3.1 Image Format Conversion <a name="image-format-conversion"></a>

We inspect the training images to identify their file types. The dataset may contain images in various formats (e.g., `.jpg`, `.png`). We use Python's `pathlib` library and a set to determine unique file extensions.

To standardize the dataset and avoid potential issues, we convert all images to JPEG format. PNG files are generally less efficient for storing photographs, and inconsistencies or minor corruption in files can lead to training warnings.

### 3.2 Directory Organization <a name="directory-organization"></a>

YOLO requires a specific directory structure:

* Top-level directories: `images/` and `labels/`
* Each top-level directory contains subdirectories for training and validation data (e.g., `images/train/`, `images/val/`, `labels/train/`, `labels/val/`).

We use Python's `pathlib` to create this structure.

### 3.3 Data Splitting <a name="data-splitting"></a>

We distribute the images and annotations into training and validation sets, typically with an 80/20 split. Images are randomly assigned to either set.

**Outcome:**

* Images are converted to a consistent JPEG format.
* Directories are structured according to YOLO's requirements.
* Images and annotations are split into training and validation sets.

## 4. Training the Model <a name="training-the-model"></a>

We use a YAML file to manage the numerous arguments required for training the YOLO model.

### 4.1 Configuration with YAML <a name="configuration-with-yaml"></a>

We create a Python dictionary containing the training configuration:

* Base path to the dataset.
* Paths to training and validation images.
* Class names.
* Number of classes.

We then save this dictionary as a YAML file using the `PyYAML` library.

### 4.2 Model Selection and Training <a name="model-selection-and-training"></a>

We load the YOLO model. In this case, we use YOLOv8n (Nano model) for its smaller size and faster training time compared to larger models, while still maintaining reasonable performance.

We initiate training using the `model.train()` method, specifying:

* The `data.yaml` file (dataset configuration).
* The number of epochs.
* Additional parameters (e.g., batch size, workers, early stopping).

### 4.3 Training Output <a name="training-output"></a>

During training, YOLO displays detailed output:

* Dataset verification (checking image and label availability).
* Training loss values (for different loss functions).
* Validation results at the end of each epoch.

We monitor the loss values to ensure the model is learning and to detect potential overfitting.

## 5. Evaluating the Model <a name="evaluating-the-model"></a>

### 5.1 Exploring Training Results <a name="exploring-training-results"></a>

We analyze various metrics to evaluate the trained YOLO model.

#### 5.1.1 Precision-Recall Curve <a name="precision-recall-curve"></a>

The Precision-Recall (PR) curve visualizes the trade-off between precision and recall at different thresholds. The `pr_curve.png` file in the results directory displays this curve. We aim for a high area under the curve.

#### 5.1.2 Confusion Matrix <a name="confusion-matrix"></a>

The confusion matrix helps identify classification errors. A good model has high values along the diagonal, indicating correct predictions. This matrix reveals which classes the model confuses, providing insights for further improvement (e.g., adding more training data for specific classes).

#### 5.1.3 Analyzing Training Loss <a name="analyzing-training-loss"></a>

Monitoring training and validation loss is crucial for detecting overfitting. If validation loss increases while training loss decreases, the model is likely overfitting.
