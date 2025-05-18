# Supervised Machine Learning Final Project

## Overview
This repository contains the final project for CSCA 5622 - Supervised Machine Learning course. The project demonstrates the application of various supervised learning techniques to solve real-world problems.

## Project Structure
- `data/`: Contains datasets used in the project
- `notebooks/`: Jupyter notebooks with analysis and model implementations
- `src/`: Source code for model implementations
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

## Models Implemented
- Model descriptions and implementations will be added here

## Results
- Key findings and results will be documented here

## Contributors
- Add contributor names here

## License
This project is licensed under the MIT License - see the LICENSE file for details