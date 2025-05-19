# Alzheimer's Disease Prediction: Machine Learning Approach

## Overview
This repository contains the final project for CSCA 5622 - Supervised Machine Learning course. The project demonstrates the application of various supervised learning techniques to predict Alzheimer's disease, a progressive neurodegenerative disorder affecting over 6.7 million Americans. Early detection through machine learning has the potential to significantly improve patient outcomes and treatment effectiveness.

## Project Structure
- `notebooks/`:
    - `alzheimers-disease-prediction.ipynb`: Main analysis notebook with data exploration and model building
    - `model_evaluation.ipynb`: Detailed performance metrics and visualizations
- `results/`: Visualization outputs, performance metrics, and summary findings
- `docs/`: Additional documentation and resources

## Technical Approach
- **Data Preprocessing**: Handling of missing values, feature engineering, and data normalization
- **Feature Selection**: Methods used to identify the most predictive factors for Alzheimer's
- **Model Development**: Implementation details for SVM, Random Forest, Neural Networks, and ensemble methods
- **Evaluation Strategy**: Cross-validation approach, metrics selection, and performance assessment

## Dataset
This project utilizes a synthetic Alzheimer's disease dataset containing 2,149 records with 35 features including:
- Demographic information (age, gender, ethnicity)
- Health metrics (BMI, blood pressure, cholesterol)
- Lifestyle factors (smoking, alcohol consumption, physical activity)
- Medical history (family history, cardiovascular disease, diabetes)
- Cognitive assessments (MMSE scores, memory complaints)

The target variable is a binary classification indicating Alzheimer's diagnosis.

## Key Results

- Analyzed comprehensive dataset including demographics, medical history, lifestyle factors and cognitive assessments
- Key Preprocessing Techniques
  - Feature Selection: SelectKBest with f_classif
  - Resampling: SMOTE (Synthetic Minority Over-sampling Technique)
  - Scaling: StandardScaler
- Implemented multiple ML models:
    - Support Vector Machines (SVM)
    - Multi-Layer Perceptron (Neural Network)
    - Ensemble Methods (Voting and Stacking Classifiers)
- Achieved strong predictive performance:
    - ROC-AUC scores significantly above baseline
    - Stacking classifier with logistic regression meta-learner showed particularly promising results
- Demonstrated potential for early disease detection and risk assessment

## Model Performance

| Model                                   | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-----------------------------------------|----------|-----------|--------|----------|---------|
| Support Vector Machines (SVM)           | 0.86     | 0.85      | 0.84   | 0.84     | 0.92    |
| Multi-Layer Perceptron (Neural Network) | 0.87     | 0.86      | 0.85   | 0.85     | 0.93    |
| Voting Ensemble                         | 0.91     | 0.90      | 0.89   | 0.89     | 0.96    |
| Stacking Ensemble                       | 0.91     | 0.90      | 0.89   | 0.89     | 0.96    |

## Impact & Future Work

The models show promise as valuable clinical decision support tools while highlighting important predictive factors that align with medical knowledge. Key impacts include:

- Early detection of Alzheimer's disease risk factors
- Support for clinical decision making
- Enablement of preventive interventions
- Improved patient outcomes through timely treatment

Future work will focus on:
- Gathering more diverse training data
- Additional feature engineering
- External validation studies

## Getting Started

### Requirements
- Python 3.8.12
- NumPy (1.20.3)
- Pandas (1.3.4)
- Scikit-learn (1.0.1)
- Matplotlib (3.4.3)
- Seaborn (0.11.2)
- TensorFlow (2.7.0) or PyTorch (1.10.0)
- Imbalanced-learn (0.8.1)

### Installation

1. Install pyenv (Python version manager):
   ```bash
   # On macOS
   brew install pyenv
   
   # On Ubuntu/Debian
   curl https://pyenv.run | bash
   ```

2. Add pyenv to your shell configuration (~/.bashrc, ~/.zshrc, etc.):
   ```bash
   export PYENV_ROOT="$HOME/.pyenv"
   export PATH="$PYENV_ROOT/bin:$PATH"
   eval "$(pyenv init --path)"
   eval "$(pyenv init -)"
   ```

3. Install Python 3.8.x:
   ```bash
   pyenv install 3.8.12
   ```

4. Set up project environment:
   ```bash
   # Create project directory and navigate to it
   git clone <repository-url>
   cd supervised-ml-project
   
   # Set local Python version
   pyenv local 3.8.12
   
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

5. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

6. Run the notebooks to reproduce results:
   ```bash
   jupyter notebook notebooks/alzheimers-disease-prediction.ipynb
   ```

## How to Cite
If you use this project in your research or work, please cite it as:

Mohammed, A. (2025). Machine Learning for Alzheimer's Disease Prediction. GitHub Repository. https://github.com/cu-mscso/csca5622-supervised-ml-final

## Contributors
- [Abdul Mohammed](https://github.com/am368a)