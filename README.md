# Liver Disease Prediction Using Logistic Regression and Random Forest

## Description
This project aims to predict liver disease using logistic regression and random forest models on the liver dataset. The dataset contains various medical parameters that are indicators of liver function.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Results](#results)
- [Model Performance](#model-performance)
- [Conclusion](#conclusion)

## Introduction
Liver disease is a significant health issue, and early detection is crucial. This project uses machine learning models to predict the presence of liver disease based on medical data.

## Dataset
The liver dataset contains 583 samples with 11 features, including age, gender, and various blood test results. The dataset is sourced from the UCI Machine Learning Repository.

### Features
- Age
- Gender
- Total_Bilirubin
- Direct_Bilirubin
- Alkaline_Phosphotase
- Alamine_Aminotransferase
- Aspartate_Aminotransferase
- Total_Protiens
- Albumin
- Albumin_and_Globulin_Ratio
- Dataset (target variable: 1 for liver disease, 2 for no liver disease)

### Result

Logistic Regression:
Mean Squared Error (MSE): 0.2857142857142857
Cross-Validation Accuracy Scores: [0.76923077, 0.73504274, 0.72413793, 0.72413793, 0.63793103]
Mean Cross-Validation Accuracy: 0.72
Model Accuracy: 0.71
Overall Accuracy: 0.71
Macro Average: Precision: 0.63, Recall: 0.60, F1-Score: 0.61
Weighted Average: Precision: 0.69, Recall: 0.71, F1-Score: 0.70

Random Forest:
Training Score: 100.0
Test Score: 68.0
Accuracy: 0.68
Overall Accuracy: 0.68
Macro Average: Precision: 0.59, Recall: 0.58, F1-Score: 0.59
Weighted Average: Precision: 0.66, Recall: 0.68, F1-Score: 0.67

### Model Performance
Logistic Regression:
Class	Precision	Recall	F1-Score	Support
1	0.77	0.86	0.81	125
2	0.50	0.34	0.40	50

Random Forest:
Class	Precision	Recall	F1-Score	Support
1	0.76	0.81	0.78	125
2	0.43	0.36	0.39	50

### Conclusion

The logistic regression model achieved a slightly higher accuracy (0.71) compared to the random forest model (0.68). Both models show better performance for the majority class (class 1) than the minority class (class 2). Future work includes addressing class imbalance and hyperparameter tuning.


