# Alzheimer's Disease Prediction: Machine Learning Approach

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cu-mscso/csca5622-supervised-ml-final)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
  - [Results](#results)
  - [Visualizations](#visualizations)
- [Impact & Future Work](#impact--future-work)
- [Contributing](#contributing)
- [Contact](#contact)

## Overview

A machine learning project for early detection of Alzheimer's disease using supervised learning techniques. This project demonstrates the application of various ML models to predict Alzheimer's disease risk based on demographic, health, and lifestyle factors.

### Key Features
- Multiple model implementations (SVM, MLP, Ensemble methods)
- Comprehensive data preprocessing pipeline
- Detailed performance analysis and visualizations
- Jupyter notebook with complete analysis

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/cu-mscso/csca5622-supervised-ml-final.git
   cd csca5622-supervised-ml-final
   ```

2. [Set up the environment](#installation)

3. Launch the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/alzheimers-disease-prediction.ipynb
   ```

## Installation

### Prerequisites
- Python 3.10+
- Git
- pip (Python package manager)
- pyenv (Python version manager)

### Setup

1. **Using pyenv (recommended)**:
   ```bash
   # Install Python 3.10.13
   pyenv install 3.10.13
   pyenv local 3.10.13
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
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

## Dataset

This project utilizes a synthetic Alzheimer's disease dataset containing 2,149 records with 35 features including:

### Features
- **Demographic**: Age, gender, ethnicity
- **Health Metrics**: BMI, blood pressure, cholesterol
- **Lifestyle**: Smoking status, alcohol consumption, physical activity
- **Medical History**: Family history, cardiovascular disease, diabetes
- **Cognitive**: MMSE scores, memory complaints

### Sample Data
| Age | Gender | MMSE_Score | Systolic_BP | Diagnosis |
|-----|--------|------------|-------------|-----------|
| 72  | F      | 28         | 128         | 0         |
| 68  | M      | 24         | 135         | 1         |
| 75  | F      | 22         | 142         | 1         |

*Table 1: Sample data from the Alzheimer's disease dataset*

## Model Performance

### Results

| Model                | Accuracy | Precision (0/1) | Recall (0/1) | F1-Score (0/1) | ROC-AUC |
|----------------------|----------|-----------------|--------------|----------------|---------|
| Base SVM (SVC)       | 0.81     | 0.79 / 0.88     | 0.96 / 0.54  | 0.87 / 0.67    | 0.92    |
| Tuned SVC            | 0.87     | 0.89 / 0.84     | 0.92 / 0.79  | 0.90 / 0.81    | 0.93    |
| Tuned SVC (Resampled)| 0.65     | 0.65 / 1.00     | 1.00 / 0.00  | 0.79 / 0.00    | -       |
| MLP Classifier       | 0.35     | 1.00 / 0.35     | 0.00 / 1.00  | 0.00 / 0.52    | -       |
| Voting Classifier    | 0.94     | 0.96 / 0.92     | 0.95 / 0.93  | 0.96 / 0.92    | 0.96    |
| Stacking Classifier  | 0.95     | 0.96 / 0.95     | 0.97 / 0.92  | 0.96 / 0.93    | 0.96    |

*Table 2: Performance comparison of machine learning models (Class 0: No Alzheimer's, Class 1: Alzheimer's)*

### Key Observations
- The **Stacking Classifier** achieved the highest accuracy (95%) and balanced performance across both classes
- **Ensemble methods** (Voting and Stacking) consistently outperformed individual models
- The MLP Classifier showed poor performance (35% accuracy) with complete failure to predict Class 0
- Class imbalance is evident, particularly in the resampled SVC model which predicted all samples as Class 0

### Visualizations

#### ROC Curves
![ROC Curves](results/figures/alzheimers_model_performance_roc_curves.png)
*Figure 1: ROC curves showing model performance (higher AUC = better performance)*

#### Confusion Matrices
![Confusion Matrices](results/figures/alzheimers_model_performance_confusion_matrices.png)
*Figure 2: Confusion matrices for model evaluation*

#### Feature Correlation
![Feature Correlation](results/figures/correlation_heatmap.png)
*Figure 3: Correlation between dataset features*

## Impact & Future Work

### Key Findings
- Stacking Classifier achieved 95% accuracy (0.96 ROC-AUC)
- Ensemble methods (Voting/Stacking) outperformed single models by 8-15%
- Class imbalance identified as a key challenge
- Top predictive features: MMSE scores, age, and specific biomarkers

### Future Directions

#### Data & Models
- Expand dataset diversity and add neuroimaging data
- Test transformer architectures and attention mechanisms
- Address class imbalance with advanced techniques

#### Clinical Application
- Validate across multiple clinical sites
- Develop EHR integration
- Ensure regulatory compliance (HIPAA/GDPR)

#### Research
- Explore personalized medicine approaches
- Partner with medical institutions
- Publish peer-reviewed research

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## How to Cite

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please reach out to [Abdul Mohammed](https://github.com/am368a) or open an issue on GitHub.

## Video Resources

This project was inspired by and makes use of insights from the following video resources:

- [Title of Video 1](https://youtube.com/example1) - Brief description of how this video contributed to the project
- [Title of Video 2](https://youtube.com/example2) - Brief description of key concepts or techniques learned
- [Title of Video 3](https://youtube.com/example3) - Any specific implementation details or algorithms referenced

## Acknowledgments

- [Kaggle](https://www.kaggle.com/) for the dataset
- [scikit-learn](https://scikit-learn.org/) for machine learning tools
- [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) for visualizations