# Heart Disease Prediction - Playground Series S6E2

## Project Overview

This project implements a machine learning solution for predicting heart disease presence using ensemble methods. The solution achieves high accuracy through extensive feature engineering and model stacking.

## Dataset

- **Source**: Kaggle Playground Series S6E2 - Heart Disease
- **Training Samples**: ~270,000
- **Test Samples**: 270,000
- **Features**: 13 original medical features including age, blood pressure, cholesterol, etc.
- **Target**: Binary classification (Presence/Absence of heart disease)

## Methodology

### Data Preprocessing
- Label encoding of categorical target variable
- Feature selection and data splitting

### Feature Engineering
Created **55 engineered features** from 13 original features:

- **Polynomial Features**: Degree-2 interactions of continuous variables
- **Domain Knowledge Features**:
  - Cholesterol/BP ratio
  - Age/Heart Rate ratio
  - Risk score (Age × BP × Cholesterol ÷ Max HR)
  - Stress index (ST depression × Max HR)
- **Statistical Features**:
  - Relative differences (normalized by mean)
  - Ranking features
- **Nonlinear Transformations**:
  - Logarithmic and square root transformations
  - Inverse transformations
- **Categorical Encoding**:
  - Frequency encoding for some variables
  - One-hot encoding for others
- **Binning**: Age and cholesterol discretization
- **Composite Flags**: Boolean combinations for risk indicators

### Model Architecture

**Ensemble of Three Gradient Boosting Models:**
1. **XGBoost** - Fast and accurate tree-based learner
2. **LightGBM** - Efficient with large datasets
3. **CatBoost** - Robust to categorical features

**Hyperparameter Optimization:**
Three parameter configurations tested via 5-fold cross-validation:
- Case 1: learning_rate=0.1, n_estimators=300, max_depth=4
- Case 2: learning_rate=0.05, n_estimators=500, max_depth=6
- Case 3: learning_rate=0.03, n_estimators=800, max_depth=8

### Training Process
- **Cross-Validation**: Stratified 5-fold CV to maintain class balance
- **Evaluation Metric**: ROC-AUC (Area Under ROC Curve)
- **Final Training**: Best models trained on full dataset
- **Prediction**: Ensemble averaging of probability predictions
- **Thresholding**: 0.6 threshold for binary classification

## Results

### Cross-Validation Performance
The notebook evaluates three hyperparameter configurations:

```
Case 1 Results:
- XGBoost OOF ROC-AUC: [See notebook output]
- LightGBM OOF ROC-AUC: [See notebook output]
- CatBoost OOF ROC-AUC: [See notebook output]
- Ensemble OOF ROC-AUC: [See notebook output]

Case 2 Results:
- XGBoost OOF ROC-AUC: [See notebook output]
- LightGBM OOF ROC-AUC: [See notebook output]
- CatBoost OOF ROC-AUC: [See notebook output]
- Ensemble OOF ROC-AUC: [See notebook output]

Case 3 Results:
- XGBoost OOF ROC-AUC: [See notebook output]
- LightGBM OOF ROC-AUC: [See notebook output]
- CatBoost OOF ROC-AUC: [See notebook output]
- Ensemble OOF ROC-AUC: [See notebook output]
```

*Note: Exact scores are available in the notebook output after running the cross-validation cell.*

### Final Submission
- **File**: `submission_2.csv`
- **Format**: 270,000 predictions (0/1) with IDs
- **Threshold**: 0.6 applied to ensemble probabilities

## Key Features

- **Comprehensive Feature Engineering**: 55 features from 13 originals
- **Robust Evaluation**: Stratified cross-validation
- **Ensemble Learning**: Three complementary models
- **Scalable Pipeline**: MinMax scaling for neural network compatibility
- **Detailed Documentation**: Each notebook cell explained

## File Structure

```
├── HeartDisease_detailed.ipynb    # Main notebook with explanations
├── HeartDisease.ipynb             # Original notebook
├── train.csv                      # Training data
├── test.csv                       # Test data
├── sample_submission.csv          # Submission format
├── submission_2.csv               # Final predictions
└── README.md                      # This file
```

## Dependencies

- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- catboost
- matplotlib (optional)

## Usage

1. Ensure all dependencies are installed
2. Run the notebook cells in order
3. Cross-validation results will be printed
4. Final submission saved as `submission_2.csv`

## Model Interpretability

The ensemble approach provides robust predictions, though individual model contributions can be analyzed through feature importance plots available in XGBoost and LightGBM.

## Future Improvements

- Feature selection to reduce dimensionality
- Hyperparameter tuning with grid/random search
- Neural network integration
- Additional ensemble techniques
- Model stacking with meta-learner

---

*This project demonstrates advanced machine learning techniques for medical prediction tasks, emphasizing feature engineering and ensemble methods for improved performance.*