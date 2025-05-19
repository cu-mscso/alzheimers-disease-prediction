# Alzheimer's Disease Prediction: Machine Learning Approach

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cu-mscso/csca5622-supervised-ml-final)

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Results](#key-results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
This repository contains the final project for CSCA 5622 - Supervised Machine Learning course. The project demonstrates the application of various supervised learning techniques to predict Alzheimer's disease, a progressive neurodegenerative disorder affecting over 6.7 million Americans. Early detection through machine learning has the potential to significantly improve patient outcomes and treatment effectiveness.

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/cu-mscso/csca5622-supervised-ml-final.git
   cd csca5622-supervised-ml-final
   ```

2. Set up the environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/alzheimers-disease-prediction.ipynb
   ```

## Project Structure

```
.
├── data/                   # Raw and processed data
│   ├── raw/                # Original data files
│   └── processed/          # Cleaned and processed data
├── notebooks/              # Jupyter notebooks
│   └── alzheimers-disease-prediction.ipynb
├── results/                # Output files
│   ├── models/            # Saved models
│   └── figures/           # Generated visualizations
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## Dependencies

- Python 3.8+

### Core Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

### Development Dependencies
- black
- flake8
- pytest

## Dataset

This project utilizes a synthetic Alzheimer's disease dataset containing 2,149 records with 35 features including:
- **Demographic Information**: Age, gender, ethnicity
- **Health Metrics**: BMI, blood pressure, cholesterol levels
- **Lifestyle Factors**: Smoking status, alcohol consumption, physical activity
- **Medical History**: Family history, cardiovascular disease, diabetes
- **Cognitive Assessments**: MMSE scores, memory complaints

### Dataset Preview

| Age | Gender | MMSE_Score | Systolic_BP | Diagnosis |
|-----|--------|------------|-------------|-----------|
| 72  | F      | 28         | 128         | 0         |
| 68  | M      | 24         | 135         | 1         |
| 75  | F      | 22         | 142         | 1         |

*Table 1: Sample data from the Alzheimer's disease dataset*

### Data Source
The dataset is available on [Kaggle](https://www.kaggle.com/). Please check the dataset's license for usage restrictions.

The target variable is a binary classification indicating Alzheimer's diagnosis (1 = Positive, 0 = Negative).

## Key Results

### Data Analysis
- Analyzed comprehensive dataset including demographics, medical history, lifestyle factors and cognitive assessments

### Preprocessing Pipeline
- **Feature Selection**: SelectKBest with f_classif
- **Class Imbalance**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Feature Scaling**: StandardScaler
- **Cross-Validation**: 5-fold stratified cross-validation

### Model Implementation
- **Support Vector Machines (SVM)** with RBF kernel
- **Multi-Layer Perceptron (Neural Network)** with 2 hidden layers
- **Ensemble Methods**:
  - Hard Voting Classifier
  - Stacking Classifier with Logistic Regression meta-learner

### Performance Highlights
- ROC-AUC scores significantly above baseline (0.96 for best model)
- Stacking classifier showed best overall performance
- Demonstrated strong potential for early disease detection and risk assessment

## Model Performance

### Performance Metrics

| Model                                   | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-----------------------------------------|----------|-----------|--------|----------|---------|
| Base SVM Classifier (SVC)               | 0.81     | 0.85      | 0.81   | 0.80     | 0.92    |
| Tuned SVC                               | 0.87     | 0.87      | 0.87   | 0.87     | 0.93    |
| Voting Classifier                       | 0.94     | 0.94      | 0.94   | 0.94     | 0.96    |
| Stacking Classifier                     | 0.95     | 0.95      | 0.95   | 0.95     | 0.96    |
| MLP Classifier                          | 0.35     | 0.77      | 0.35   | 0.18     | -       |

*Table 2: Performance comparison of different machine learning models*

### Performance Visualization

![Model Comparison](results/figures/model_performance.png)
*Figure 1: Comparison of model performances across different metrics*

### Key Findings
- Ensemble methods (Voting and Stacking) outperformed individual models
- Neural Network showed the best performance among base models
- All models demonstrated strong predictive power with ROC-AUC > 0.9

## Usage

### Training a New Model
```python
from sklearn.model_selection import train_test_split
from src.models.train import train_model

# Load and preprocess data
X, y = load_data('data/processed/final_dataset.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = train_model(X_train, y_train)
```

### Making Predictions
```python
from src.models.predict import load_model, predict

# Load trained model
model = load_model('results/models/best_model.pkl')

# Make predictions
predictions = predict(model, X_test)
```

## Impact & Future Work

### Clinical Impact
- **Early Detection**: Identify high-risk individuals before clinical symptoms manifest
- **Personalized Medicine**: Enable targeted interventions based on individual risk profiles
- **Resource Optimization**: Help healthcare providers allocate resources more efficiently
- **Research Insights**: Contribute to understanding of Alzheimer's disease risk factors

### Future Directions
1. **Data Enhancement**
   - Collect more diverse demographic data
   - Include additional biomarkers and genetic factors
   - Longitudinal studies for disease progression tracking

2. **Model Improvements**
   - Experiment with deep learning architectures
   - Incorporate time-series data from patient histories
   - Develop interpretable AI models for clinical settings

3. **Clinical Integration**
   - External validation with independent datasets
   - Development of clinical decision support system
   - Integration with electronic health records (EHR) systems

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/cu-mscso/csca5622-supervised-ml-final.git
   cd csca5622-supervised-ml-final
   ```

2. **Set up a virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Jupyter Notebook**
   ```bash
   jupyter notebook notebooks/alzheimers-disease-prediction.ipynb
   ```

### Development Setup

1. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Run tests**
   ```bash
   pytest tests/
   ```

3. **Code formatting**
   ```bash
   black .
   flake8
   ```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please make sure to update tests as appropriate.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## How to Cite

If you use this project in your research or work, please cite it as:

```
@misc{mohammed2025alzheimer,
  author = {Mohammed, Abdul},
  title = {Machine Learning for Alzheimer's Disease Prediction},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cu-mscso/csca5622-supervised-ml-final}}
}
```

## Contact

For questions or feedback, please reach out to [Abdul Mohammed](https://github.com/am368a) or open an issue on GitHub.

## Acknowledgments

- [Kaggle](https://www.kaggle.com/) for the dataset
- [scikit-learn](https://scikit-learn.org/) for machine learning tools
- [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) for visualizations
- The open-source community for their valuable contributions

---

<div align="center">
  <sub>Built with ❤️ by <a href="https://github.com/am368a">Abdul Mohammed</a></sub>
</div>