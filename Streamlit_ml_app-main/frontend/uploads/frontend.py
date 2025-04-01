import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import shap  # Import the shap library

# Preloaded dataset options
dataset_options = {
    "Titanic Dataset": "titanic.csv",
    "House-Price Dataset": "house-price.csv"
}

st.title("ML Data Exploration & Model Training App")

# Sidebar Navigation
page = st.sidebar.selectbox("Select Page", ["Data Exploration", "Model Training", "Evidential AI", "Shapley Values"])

# Function to load dataset
def load_dataset(selected_dataset, uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["uploaded_dataset"] = df  # Store in session state
        return df
    elif selected_dataset in dataset_options:
        return pd.read_csv(dataset_options[selected_dataset])
    return None

if page == "Data Exploration":
    st.header("Data Exploration")

    # Force user to select dataset before proceeding
    selected_dataset = st.selectbox("Choose a dataset or upload your own:", list(dataset_options.keys()) + ["Upload your own"], key="dataset_select")
    
    uploaded_file = None
    if selected_dataset == "Upload your own":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="file_uploader")

    df = load_dataset(selected_dataset, uploaded_file)

    if df is not None:
        st.session_state["selected_dataset"] = selected_dataset  # Store dataset name
        st.session_state["current_df"] = df  # Store dataset in session

        st.write("### Dataset Preview")
        st.write(df.head())

        st.write("### Dataset Statistics")
        st.write(df.describe())

        # Histograms
        st.write("### Histograms")
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            numeric_df.hist(ax=ax, bins=20)
            st.pyplot(fig)
        else:
            st.warning("No numerical columns available for histograms.")

        # Correlation Heatmap
        st.write("### Correlation Heatmap")
        if not numeric_df.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No numerical columns available for correlation heatmap.")

    else:
        st.warning("Please select a dataset or upload your own.")

elif page == "Model Training":
    st.header("Model Training")

    # Retrieve dataset from session state
    if "current_df" in st.session_state:
        df = st.session_state["current_df"]
        selected_dataset = st.session_state["selected_dataset"]

        st.write(f"### Training Model on: {selected_dataset}")

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder

        # Data Preprocessing
        df = df.select_dtypes(include=['number']).dropna()
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Display cleaned dataset
        st.write("### Processed Dataset Used for Training")
        st.write(df.head())

        # Check if classification or regression
        if y.dtype == 'object' or y.nunique() <= 10:  
            y = LabelEncoder().fit_transform(y)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:  
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Display Training Data
        st.write("### Sample of Training Data (X_train)")
        st.write(pd.DataFrame(X_train).head())

        st.write("### Sample of Training Labels (y_train)")
        st.write(pd.DataFrame(y_train, columns=["Target"]).head())

        # Add a train button
        if st.button("Train Model"):
            # Train Model
            model.fit(X_train, y_train)

            # Save Model
            model_filename = f"{selected_dataset.replace(' ', '_').lower()}.pkl"
            with open(model_filename, "wb") as f:
                pickle.dump(model, f)

            st.session_state["trained_model"] = model_filename  # Store trained model name

            st.success(f"Model trained successfully and saved as {model_filename}!")

            # Provide option to download the model
            with open(model_filename, "rb") as f:
                st.download_button("Download Model", f, file_name=model_filename)

    else:
        st.warning("Please go to the Data Exploration page and select a dataset first.")

elif page == "Evidential AI":
    st.header("Evidential AI")

    # Allow user to select a trained dataset model
    trained_model_options = [f for f in os.listdir() if f.endswith(".pkl")]
    
    if trained_model_options:
        selected_model_file = st.selectbox("Choose a trained model", trained_model_options, key="evidential_model_select")
        selected_dataset = selected_model_file.replace("_", " ").replace(".pkl", "").title()

        # Check if the selected model file exists
        if os.path.exists(selected_model_file):
            st.write(f"### Loading Model for: {selected_dataset}")

            # Load the trained model
            with open(selected_model_file, "rb") as f:
                model = pickle.load(f)

            st.write("### Model Details")
            st.write(model)

            # Load dataset used for training
            df = None
            if selected_dataset in dataset_options:
                df = pd.read_csv(dataset_options[selected_dataset])
            elif "uploaded_dataset" in st.session_state:
                df = st.session_state["uploaded_dataset"]

            if df is not None:
                df = df.select_dtypes(include=['number']).dropna()
                X = df.iloc[:, :-1]

                # Display dataset preview
                st.write("### Processed Dataset (Used in Evidential AI)")
                st.write(df.head())

                # Feature Importance Visualization
                st.write("### Feature Importance")
                if hasattr(model, "feature_importances_"):
                    fig, ax = plt.subplots()
                    sns.barplot(x=model.feature_importances_, y=X.columns, ax=ax)
                    plt.xlabel("Importance")
                    plt.ylabel("Features")
                    st.pyplot(fig)
                else:
                    st.warning("Feature importance not available for this model.")
            else:
                st.warning("Could not load the dataset. Please check if the dataset exists.")

        else:
            st.warning("No trained model found. Please train a model first in the Model Training page.")
    else:
        st.warning("No trained models available. Train a model first.")

elif page == "Shapley Values":
    # A new page for displaying detailed Shapley value plots
    st.header("Shapley Value Analysis")

    if "trained_model" in st.session_state:
        # Retrieve the model and data
        model_filename = st.session_state["trained_model"]
        with open(model_filename, "rb") as f:
            model = pickle.load(f)

        # Load dataset
        df = st.session_state.get("current_df", None)
        if df is not None:
            X = df.select_dtypes(include=['number']).dropna().iloc[:, :-1]  # Features for SHAP

            # SHAP Analysis
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            # Show SHAP Summary Plot
            st.write("### SHAP Summary Plot")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X, plot_type="dot", show=False)
            st.pyplot(fig)

            # Show Histogram of SHAP Values
            st.write("### Histogram of SHAP Values")
            shap_values_flattened = shap_values[0].flatten()  # Flatten the SHAP values for class 0 (binary classification)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(shap_values_flattened, bins=50, color='skyblue', edgecolor='black')
            ax.set_title("Distribution of SHAP Values")
            ax.set_xlabel("Shapley Value")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

            # SHAP Dependence Plot for a specific feature
            feature = st.selectbox("Select feature for SHAP dependence plot:", X.columns)
            st.write(f"### SHAP Dependence Plot for {feature}")
            
            # For multi-class models, shap_values might be a list of arrays, where each array corresponds to a class.
            # We use shap_values[0] for binary classification and shap_values[index] for multi-class classification.

            if isinstance(shap_values, list):
                # If the model is multi-class, ask the user to select the class
                class_index = st.selectbox("Select class for dependence plot:", range(len(shap_values)))
                shap.dependence_plot(feature, shap_values[class_index], X, show=False)
            else:
                shap.dependence_plot(feature, shap_values, X, show=False)

            st.pyplot(fig)

            # SHAP Feature Importance Plot (Bar chart)
            st.write("### Feature Importance Plot")
            feature_importance = pd.DataFrame(list(zip(X.columns, model.feature_importances_)), columns=['Feature', 'Importance'])
            feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x="Importance", y="Feature", data=feature_importance, ax=ax)
            ax.set_title("Feature Importance from SHAP")
            st.pyplot(fig)

        else:
            st.warning("Dataset not found.")
    else:
        st.warning("No model available. Please train a model first.")