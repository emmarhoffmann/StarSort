"""
Star Classification System using Multiple Machine Learning Models

This program implements various machine learning algorithms to classify stars based on their physical characteristics.
It uses five different ML models: Naive Bayes, Random Forest, SVM, Neural Network, and Decision Tree to provide 
comprehensive analysis and comparison of classification performance across all star types in the dataset.

Features:
- Multi-model classification using 5 different algorithms
- Data preprocessing and feature scaling
- Model performance comparison and visualization
- Cross-validation analysis
- Feature importance analysis (Random Forest)
- Confusion matrix visualization for each model
- Detailed evaluation of each star type's classification performance

Input:
- Dataset (Stars.csv) with features: Temperature, Luminosity, Radius, Absolute magnitude
- Target variable: Star type (0-5 representing different star types)

Output:
- Classification reports for each model, including precision, recall, and F1-score for all star types
- Confusion matrices visualizing classification performance across all star types
- Model accuracy comparisons
- Cross-validation score comparisons
- Feature importance visualization (for Random Forest)

Author: Emma Hoffmann
Date: October 2024

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv(r'C:\Users\Emma Hoffmann\OneDrive\PROFESSIONAL\StarSort\Stars.csv')

# Separate features and target
X = df[['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)']]
y = df['Star type']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Dictionary to store results
results = {}
cross_val_results = {}

# Train and evaluate each model on all star types
print("Model Performance Comparison:\n")
for name, model in models.items():
    print(f"\n{name} Results:")
    print("-" * 50)
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on the entire test set
    y_pred = model.predict(X_test_scaled)
    
    # Store results
    results[name] = {
        'predictions': y_pred,
        'accuracy': model.score(X_test_scaled, y_test)
    }
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    cross_val_results[name] = cv_scores.mean()
    
    # Print classification report for all star types
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create and display confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Compare model accuracies
plt.figure(figsize=(10, 6))
accuracies = {name: results[name]['accuracy'] for name in models.keys()}
plt.bar(accuracies.keys(), accuracies.values())
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Compare cross-validation scores
plt.figure(figsize=(10, 6))
plt.bar(cross_val_results.keys(), cross_val_results.values())
plt.title('Cross-Validation Scores Comparison')
plt.ylabel('Mean CV Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Feature importance analysis (for Random Forest)
rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.show()
