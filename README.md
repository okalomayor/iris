# Project Title: Iris Dataset Analysis with K-Nearest Neighbors (KNN)

## Overview
This project demonstrates a machine learning analysis of the classic Iris dataset, implementing the K-Nearest Neighbors (KNN) algorithm. The Iris dataset contains measurements of sepal length, sepal width, petal length, and petal width for three species of iris flowers: Setosa, Versicolor, and Virginica. The goal is to classify each flower species based on these features.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Setup Instructions](#setup-instructions)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Contributions](#contributions)

---

## Project Structure
- `iris_analysis.py`: Contains the script for loading data, training the model, and generating predictions.
- `README.md`: Provides an overview of the project and usage instructions.

## Setup Instructions
### Requirements
- Python 3.6+
- Libraries: 
  - `pandas`
  - `scikit-learn`

### Installation
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd iris-dataset-analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
The Iris dataset is sourced from the sklearn library and includes:
- 150 samples
- 4 features: `sepal length`, `sepal width`, `petal length`, and `petal width`
- 3 classes (50 samples each): Setosa, Versicolor, Virginica

## Methodology
1. **Data Loading**: The Iris dataset is loaded and converted into a DataFrame.
2. **Data Splitting**: The dataset is split into training and testing sets (usually 70% training and 30% testing).
3. **Model Selection**: K-Nearest Neighbors (KNN) classifier is used with `n_neighbors=1`.

## Model Training
After splitting the dataset, we train the KNN classifier using the training set. The classifier is then used to predict the classes in the test set.

## Evaluation
Evaluation metrics include:
- **Precision**: Proportion of true positive predictions among the positive predictions.
- **Recall**: Proportion of true positive instances correctly identified.
- **F1-Score**: Harmonic mean of precision and recall, providing a balanced evaluation metric.
- **Accuracy**: The overall percentage of correctly classified samples.

The model achieved an accuracy of **97%**, with high precision, recall, and F1-scores across all classes.

## Results
The classification report provided the following summary for each class:
- Setosa: Perfect accuracy with a precision, recall, and F1-score of 1.00.
- Versicolor: High performance with a recall of 0.94 and F1-score of 0.97.
- Virginica: Strong results with a recall of 1.00 and F1-score of 0.95.

The model's macro-average metrics (precision: 0.97, recall: 0.98, F1-score: 0.97) indicate its robustness in classifying all three flower species.

## Contributions
Contributions are welcome. Feel free to submit issues or pull requests to enhance this project.

---


