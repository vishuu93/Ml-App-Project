

# ML Data Exploration, Model Training, and Shapley Value Application

## Overview
This web application allows users to:
1. **Explore datasets**: View dataset statistics, histograms, and correlation heatmaps.
2. **Train machine learning models**: Train classification or regression models using the uploaded or preloaded datasets.
3. **Use Evidential AI**: Evaluate trained models, display feature importance, and visualize Shapley values to interpret model predictions.

The application leverages Streamlit for an interactive UI, Scikit-learn for machine learning tasks, and SHAP for model interpretation.

## Features
- **Data Exploration**: Load and analyze datasets, visualize distributions and correlations.
- **Model Training**: Train models (Random Forest) on the dataset and evaluate their performance.
- **Evidential AI**: Interpret the trained models using SHAP values, plot feature importance, and generate dependency plots.
- **Shapley Value Page**: Display SHAP value visualizations, including summary plots and dependence plots, for better model explainability.

## Installation

### Clone the repository
Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/ml-data-exploration-app.git
cd ml-data-exploration-app
```

### Install the required packages
To install the necessary Python dependencies, create a virtual environment and install the dependencies:

```bash
# Create a virtual environment (optional but recommended)
python -m venv ml_env

# Activate the virtual environment
# On Windows:
ml_env\Scripts\activate


# Install dependencies
pip install -r requirements.txt
```

### Required Libraries
- **Streamlit**: For the web interface
- **Pandas**: For data handling
- **Matplotlib & Seaborn**: For data visualization
- **Scikit-learn**: For machine learning models
- **SHAP**: For model interpretability using Shapley values



```bash
pip install streamlit pandas matplotlib seaborn scikit-learn shap
```

## Usage

1. **Run the Streamlit app**:
   To start the web application, run:

   ```bash
   streamlit run .\frontend.py
   ```

2. **Data Exploration Page**:
   - Choose or upload a dataset.
   - View basic statistics, histograms, and correlation heatmaps.

3. **Model Training Page**:
   - After selecting a dataset, train a machine learning model (Random Forest).
   - The app automatically detects whether the target variable is categorical (classification) or continuous (regression).
   - View a summary of the model performance, and download the trained model as a `.pkl` file.

4. **Evidential AI Page**:
   - Select a trained model to load.
   - View the feature importance of the model.
   - Use SHAP for model interpretation and visualizations.

5. **Shapley Value Page**:
   - Visualize Shapley values with summary plots and dependence plots for detailed insights into the model’s decision-making process.

## Screenshots

### Data Exploration
![Data Exploration Screenshot](https://github.com/Tanmay-hue/Streamlit_ml_app/blob/main/image/1.png)

### Model Training
![Model Training Screenshot](https://github.com/Tanmay-hue/Streamlit_ml_app/blob/main/image/3.png)

### Shapley Value Visualizations
![Shapley Value Screenshot](https://github.com/Tanmay-hue/Streamlit_ml_app/blob/main/image/4.png)

## Troubleshooting
- Ensure you have the correct Python version (>= 3.7).
- If you run into issues with SHAP, make sure the `shap` library is installed correctly.

## Contributing
If you want to contribute to this project, feel free to open an issue or submit a pull request. 

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Commit and push your changes.
5. Create a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




---
Author: Tanmay Singh

