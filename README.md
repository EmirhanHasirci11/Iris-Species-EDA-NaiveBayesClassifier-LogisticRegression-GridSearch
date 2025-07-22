# Iris-Species-EDA-NaiveBayesClassifier-LogisticRegression-GridSearch

## Overview

This project focuses on the classification of the famous Iris dataset. The goal is to build and evaluate machine learning models that can accurately predict the species of an Iris flower based on its sepal and petal measurements.

Two different classification models are implemented and compared:
1.  **Gaussian Naive Bayes Classifier**
2.  **Logistic Regression** with hyperparameter tuning using `GridSearchCV`.

The project covers the entire data science pipeline, from data loading and exploratory data analysis (EDA) to preprocessing, model training, tuning, and evaluation.

## Dataset

The dataset used is the classic "Iris" dataset, which contains 150 samples from three species of Iris flowers (Iris setosa, Iris versicolor, and Iris virginica).

*   [Dataset Link](https://www.kaggle.com/datasets/uciml/iris)

### Features

The dataset consists of four features (predictor variables) and one target variable (species).

| Feature         | Description                                     | Data Type |
|-----------------|-------------------------------------------------|-----------|
| SepalLengthCm   | The length of the sepal in centimeters.         | float64   |
| SepalWidthCm    | The width of the sepal in centimeters.          | float64   |
| PetalLengthCm   | The length of the petal in centimeters.         | float64   |
| PetalWidthCm    | The width of the petal in centimeters.          | float64   |
| **Species**     | **(Target)** The species of the Iris flower.    | object    |

## Project Pipeline

### 1. Exploratory Data Analysis (EDA)

The first step was to explore the dataset to understand its structure and the relationships between features. The following EDA techniques were used:
*   **Descriptive Statistics**: `df.describe()` provided a summary of the central tendency, dispersion, and shape of the dataset's distribution.
*   **Class Distribution**: `df['Species'].value_counts()` confirmed that the dataset is balanced, with 50 samples for each of the three species.
*   **Pairplot**: A `seaborn.pairplot` was generated to visualize the pairwise relationships between all features. This plot showed clear clusters, especially when using petal measurements, indicating that the species are well-separated.
*   **Scatter Plots**: Individual scatter plots for Sepal Length vs. Width and Petal Length vs. Width (colored by species) were created to get a closer look at the separability of the classes.
*   **Correlation Matrix**: A heatmap of the correlation matrix was generated to quantify the linear relationships between the features. It revealed a very strong positive correlation between `PetalLengthCm` and `PetalWidthCm`.

### 2. Data Preprocessing

Before model training, the data was preprocessed with the following steps:
1.  **Dropping Unnecessary Columns**: The `Id` column was removed as it serves as an index and provides no predictive value.
2.  **Label Encoding**: The categorical target variable `Species` was converted into numerical labels using `sklearn.preprocessing.LabelEncoder`. The mapping was:
    *   `0`: Iris-setosa
    *   `1`: Iris-versicolor
    *   `2`: Iris-virginica
3.  **Data Splitting**: The dataset was split into training (75%) and testing (25%) sets using `train_test_split` with a `random_state` for reproducibility.
4.  **Feature Scaling**: `StandardScaler` was used to scale the features (`X_train` and `X_test`). This step standardizes the features by removing the mean and scaling to unit variance, which is crucial for the optimal performance of many machine learning algorithms like Logistic Regression.

### 3. Model Training and Evaluation

#### a. Gaussian Naive Bayes Classifier

A Gaussian Naive Bayes (GNB) classifier was the first model trained. GNB is a probabilistic classifier based on Bayes' theorem with the "naive" assumption of conditional independence between features.

The model was trained on the scaled training data and evaluated on the scaled test data. It achieved a **perfect accuracy of 1.0** on the test set, correctly classifying all 38 samples.

#### b. Logistic Regression with GridSearchCV

The second model was a Logistic Regression classifier. To find the optimal hyperparameters for this model, `GridSearchCV` was employed. This technique performs an exhaustive search over a specified parameter grid to find the combination that yields the best cross-validation performance.

The hyperparameters tuned were:
*   `C`: The regularization strength (`[0.1, 1, 10, 100]`).
*   `penalty`: The regularization norm (`['l1', 'l2']`).
*   `solver`: The optimization algorithm (`['liblinear']`).

The grid search was performed with 5-fold cross-validation. The best parameters found were:
*   **C**: 10
*   **penalty**: 'l1'
*   **solver**: 'liblinear'

This optimized Logistic Regression model also achieved a very high accuracy on the test set.

## Results Summary

Both models performed exceptionally well on this dataset, which is expected as the Iris species are known to be linearly separable.

| Model                       | Test Accuracy |
|-----------------------------|---------------|
| Gaussian Naive Bayes        | 1.0           |
| Logistic Regression (Tuned) | 0.95          |

Both the classification reports and confusion matrices confirm that the models are highly effective at distinguishing between the three Iris species.
