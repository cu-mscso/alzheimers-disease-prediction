# Supervised Machine Learning Final Project

## Overview
This repository contains the final project for CSCA 5622 - Supervised Machine Learning course. The project demonstrates the application of various supervised learning techniques to solve real-world problems.

## Project Structure
- `notebooks/`: Jupyter notebooks with analysis and model implementations
- `results/`: Model outputs and evaluation metrics

## Getting Started
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

6. Follow instructions in notebooks to reproduce results

# Project Overview

This project implements machine learning models for Alzheimer's disease prediction using patient data. Our goal is to develop accurate predictive models that can assist in early detection and risk assessment.

## Key Results

- Analyzed comprehensive dataset including demographics, medical history, lifestyle factors and cognitive assessments
- Implemented multiple ML models:
  - Support Vector Machines (SVM)
  - Random Forest
  - Neural Networks 
  - Ensemble methods
- Achieved strong predictive performance:
  - ROC-AUC scores significantly above baseline
  - Stacking classifier with logistic regression meta-learner showed particularly promising results
- Demonstrated potential for early disease detection and risk assessment

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



## Contributors
- Abdul Mohammed
