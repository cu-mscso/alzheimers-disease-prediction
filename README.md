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

# Project Summary
This project implements machine learning models for Alzheimer's disease prediction using patient data. The goal is to develop accurate predictive models that can assist in early detection and risk assessment.

## Key Features
- Comprehensive analysis of patient data including:
  - Demographics
  - Medical history 
  - Lifestyle factors
  - Cognitive assessments
- Rigorous model development and evaluation:
  - Support Vector Machines (SVM)
  - Random Forest
  - Neural Networks
  - Ensemble methods
- Advanced model architecture:
  - Stacking Classifier with logistic regression meta-learner
  - Achieved ROC-AUC scores significantly above baseline

## Impact
The models demonstrate strong potential for:
- Early detection of Alzheimer's disease risk factors
- Supporting clinical decision making
- Enabling preventive interventions
- Improving patient outcomes through timely treatment



## Contributors
- Abdul Mohammed
