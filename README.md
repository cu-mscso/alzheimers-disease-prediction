# Alzheimer's Disease Prediction

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cu-mscso/csca5622-supervised-ml-final)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Machine learning project for early Alzheimer's disease prediction using demographic, health, and lifestyle factors.

## Quick Start

1. Clone and set up:
   ```bash
   git clone https://github.com/cu-mscso/csca5622-supervised-ml-final.git
   cd csca5622-supervised-ml-final
   python -m venv venv
   source venv/bin/activate  # Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   jupyter notebook notebooks/alzheimers-disease-prediction.ipynb
   ```

## Model Performance

| Model                | Accuracy | Precision (0/1) | F1-Score (0/1) | ROC-AUC |
|----------------------|----------|-----------------|----------------|---------|
| Stacking Classifier  | 0.95     | 0.96 / 0.95     | 0.96 / 0.93    | 0.96    |
| Voting Classifier    | 0.94     | 0.96 / 0.92     | 0.96 / 0.92    | 0.96    |
| Tuned SVC           | 0.87     | 0.89 / 0.84     | 0.90 / 0.81    | 0.93    |
| Base SVC            | 0.81     | 0.79 / 0.88     | 0.87 / 0.67    | 0.92    |
| MLP Classifier      | 0.35     | 1.00 / 0.35     | 0.00 / 0.52    | -       |

*Class 0: No Alzheimer's, Class 1: Alzheimer's*

### Key Findings
- Stacking Classifier achieved 95% accuracy (0.96 ROC-AUC)
- Ensemble methods (Voting/Stacking) outperformed single models
- Class imbalance identified as a key challenge

## Dataset

2,149 records with 35 features including:
- **Demographics**: Age, gender, ethnicity
- **Health**: MMSE scores, blood pressure, BMI
- **Lifestyle**: Smoking, alcohol, activity levels

## Future Work
- Expand dataset with neuroimaging data
- Test transformer architectures
- Clinical validation and EHR integration

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

[Abdul Mohammed](https://github.com/am368a)  
*or*  
[Open an issue](https://github.com/cu-mscso/csca5622-supervised-ml-final/issues/new)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
- The open-source community for their valuable contributions