# Banknote ML Pipeline with GitHub Actions

## 1. Project Overview

This project implements an automated machine learning pipeline using GitHub Actions.

The pipeline automatically:

1. Downloads and validates the dataset.
2. Splits the data into training and validation sets.
3. Trains a baseline model.
4. Trains a candidate machine learning model.
5. Evaluates both models using F1-score.
6. Applies a minimum improvement quality gate.
7. Saves the trained model only when the quality gate passes.
8. Runs application tests.
9. Packages the trained model and supporting files.
10. Uploads the model package as a GitHub Actions artifact.

The pipeline is designed so that a model with insufficient quality or a broken application cannot produce a successful model artifact.

---

## 2. Dataset

### Dataset Name

Banknote Authentication Dataset

### Source

UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/267/banknote+authentication

The dataset is also downloaded automatically from the following stable public URL:

https://raw.githubusercontent.com/jbrownlee/Datasets/master/banknote_authentication.csv

The dataset contains 1,372 instances and four input features.

### Features

The input features are:

- `variance`
- `skewness`
- `curtosis`
- `entropy`

### Target

The target column is:

- `class`

The target contains two classes:

- `0`
- `1`

### Machine Learning Task

This is a binary classification problem.

The objective is to classify whether a banknote belongs to one of the two target classes using the four numerical banknote features.

---

## 3. Data Validation

The pipeline validates the dataset before model training.

The validation checks:

- Required feature columns exist.
- Target column exists.
- No missing values are present.
- Target values are restricted to 0 and 1.
- Feature columns are numeric.

The pipeline fails with a non-zero exit code if the required dataset structure or values are invalid.

---

## 4. Train/Validation Split

The dataset is divided into:

- 80% training data
- 20% validation data

The split uses:

```text
random_state = 42
stratify = y