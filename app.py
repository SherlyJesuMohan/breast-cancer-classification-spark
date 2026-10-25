import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# PROJECT CONFIGURATION
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR, "new_dataset", "dataR2.csv"
)

COMPARISON_PATH = os.path.join(
    BASE_DIR, "model_comparison.csv"
)

TUNING_PATH = os.path.join(
    BASE_DIR, "tuned_model_results.csv"
)

MODEL_PARAMETERS_PATH = os.path.join(
    BASE_DIR, "model_parameters.json"
)

FEATURES = [
    "Age",
    "BMI",
    "Glucose",
    "Insulin",
    "HOMA",
    "Leptin",
    "Adiponectin",
    "Resistin",
    "MCP_1"
]

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

h1 {
    color: #60a5fa !important;
    font-weight: 800 !important;
}

h2 {
    color: #93c5fd !important;
    font-weight: 700 !important;
}

h3 {
    color: #c4b5fd !important;
}

p, li {
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #263449;
}

[data-testid="stMetric"] {
    background-color: #172338;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #334155;
}

[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}

[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-size: 28px !important;
    font-weight: bold !important;
}

.stButton > button {
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1.2rem;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
    border: 1px solid #60a5fa;
}

hr {
    border-color: #334155;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_dataset(path):
    data = pd.read_csv(path)

    if "MCP.1" in data.columns:
        data = data.rename(columns={"MCP.1": "MCP_1"})

    return data


@st.cache_data
def load_results(path):
    return pd.read_csv(path)


df = None
comparison_df = None
tuning_df = None

if os.path.exists(DATASET_PATH):
    try:
        df = load_dataset(DATASET_PATH)
    except Exception as error:
        st.error(f"Could not load the dataset: {error}")

if os.path.exists(COMPARISON_PATH):
    try:
        comparison_df = load_results(COMPARISON_PATH)
    except Exception:
        comparison_df = None

if os.path.exists(TUNING_PATH):
    try:
        tuning_df = load_results(TUNING_PATH)
    except Exception:
        tuning_df = None

# =====================================================
# PREPARE TARGET COUNTS
# =====================================================

if df is not None and "Classification" in df.columns:

    df["Classification"] = pd.to_numeric(
        df["Classification"],
        errors="coerce"
    )

    class_counts = df["Classification"].value_counts()

    healthy_count = int(class_counts.get(1, 0))
    patient_count = int(class_counts.get(2, 0))

else:
    healthy_count = 0
    patient_count = 0

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("🎗️ Navigation")

page = st.sidebar.radio(
    "Select a section",
    [
        "Home Dashboard",
        "Dataset Overview",
        "Model Performance",
        "Model Comparison",
        "Hyperparameter Tuning",
        "Confusion Matrix",
        "ROC Curve",
        "Live Prediction",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Machine Learning Project\n\n"
    "Built using Python, Apache Spark MLlib, "
    "Pandas, Plotly and Streamlit."
)

st.sidebar.caption(
    "Dataset: Breast Cancer Coimbra (UCI)"
)

# =====================================================
# HOME DASHBOARD
# =====================================================

if page == "Home Dashboard":

    st.title("🎗️ Breast Cancer Classification")

    st.markdown(
        "### Apache Spark MLlib | Machine Learning | Model Evaluation"
    )

    st.success(
        "Welcome to the Breast Cancer Classification dashboard!"
    )

    st.markdown("## Project Overview")

    st.write(
        "This project demonstrates machine learning classification "
        "using the Breast Cancer Coimbra dataset. The dataset contains "
        "numerical measurements associated with healthy controls "
        "and breast cancer patients."
    )

    st.write(
        "Logistic Regression and Decision Tree models are evaluated "
        "using accuracy, precision, recall, F1 score and ROC-AUC. "
        "Logistic Regression is also tuned using cross-validation."
    )

    st.warning(
        "Educational demonstration only. This application is not "
        "a validated medical diagnostic tool and must not be used "
        "to make medical decisions."
    )

    st.markdown("## Project Statistics")

    if df is not None:

        feature_count = sum(
            feature in df.columns for feature in FEATURES
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Records", len(df))
        col2.metric("Numerical Features", feature_count)
        col3.metric("Models Evaluated", 2)

        st.markdown("## Dataset Class Distribution")

        distribution = pd.DataFrame({
            "Class": [
                "Healthy Control (1)",
                "Breast Cancer Patient (2)"
            ],
            "Records": [
                healthy_count,
                patient_count
            ]
        })

        col1, col2 = st.columns(2)

        with col1:
            st.bar_chart(
                distribution.set_index("Class")
            )

        with col2:

            fig = px.pie(
                distribution,
                names="Class",
                values="Records",
                title="Distribution of Dataset Classes",
                hole=0.45,
                color_discrete_sequence=[
                    "#38bdf8",
                    "#a78bfa"
                ]
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1120",
                plot_bgcolor="#0b1120",
                font_color="#f1f5f9",
                title_x=0.5
            )

            fig.update_traces(
                textinfo="label+percent"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.error(
            "Dataset not found. Expected location: "
            "new_dataset/dataR2.csv"
        )

    st.markdown("## Machine Learning Algorithms")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Logistic Regression")
        st.write(
            "A classification algorithm that estimates the "
            "probability of a binary outcome."
        )

    with col2:
        st.subheader("Decision Tree")
        st.write(
            "A tree-based algorithm that predicts a class "
            "using feature-based decisions."
        )

    with col3:
        st.subheader("Hyperparameter Tuning")
        st.write(
            "Cross-validation compares parameter combinations "
            "to select a suitable model configuration."
        )

# =====================================================
# DATASET OVERVIEW
# =====================================================

elif page == "Dataset Overview":

    st.title("📊 Dataset Overview")

    if df is not None:

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Records", len(df))
        col2.metric("Total Columns", len(df.columns))

        col3.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

        st.markdown("### Dataset Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "📥 Download Dataset (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="coimbra_breast_cancer_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.markdown("### Statistical Summary")

        st.dataframe(
            df.describe().T,
            use_container_width=True
        )

        if "Classification" in df.columns:

            counts = pd.DataFrame({
                "Classification": [
                    "Healthy Control (1)",
                    "Breast Cancer Patient (2)"
                ],
                "Records": [
                    healthy_count,
                    patient_count
                ]
            })

            st.markdown("### Classification Counts")

            st.dataframe(
                counts,
                use_container_width=True,
                hide_index=True
            )

        st.markdown("### Feature Information")

        feature_info = pd.DataFrame({
            "Feature": FEATURES,
            "Description": [
                "Age of the individual",
                "Body Mass Index",
                "Glucose measurement",
                "Insulin measurement",
                "HOMA measurement",
                "Leptin measurement",
                "Adiponectin measurement",
                "Resistin measurement",
                "MCP-1 measurement"
            ]
        })

        st.dataframe(
            feature_info,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.error(
            "Dataset not found. Check "
            "new_dataset/dataR2.csv."
        )

# =====================================================
# MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.title("🤖 Model Performance")

    if comparison_df is not None:

        required_columns = [
            "Model",
            "Accuracy (%)",
            "Precision (%)",
            "Recall (%)",
            "F1 Score (%)",
            "ROC-AUC (%)"
        ]

        if all(
            column in comparison_df.columns
            for column in required_columns
        ):

            selected_model = st.selectbox(
                "Choose a model",
                comparison_df["Model"].tolist()
            )

            row = comparison_df[
                comparison_df["Model"] == selected_model
            ].iloc[0]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Accuracy",
                f'{row["Accuracy (%)"]:.2f}%'
            )

            col2.metric(
                "Precision",
                f'{row["Precision (%)"]:.2f}%'
            )

            col3.metric(
                "Recall",
                f'{row["Recall (%)"]:.2f}%'
            )

            col4, col5 = st.columns(2)

            col4.metric(
                "F1 Score",
                f'{row["F1 Score (%)"]:.2f}%'
            )

            col5.metric(
                "ROC-AUC",
                f'{row["ROC-AUC (%)"]:.2f}%'
            )

            st.markdown("### Complete Performance Table")

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.error(
                "The comparison CSV has unexpected columns. "
                "We need to regenerate model_comparison.csv "
                "for the new dataset."
            )

    else:

        st.error(
            "model_comparison.csv was not found. "
            "We need to create it using the new model results."
        )

# =====================================================
# MODEL COMPARISON
# =====================================================

elif page == "Model Comparison":

    st.title("📈 Model Comparison")

    if comparison_df is not None:

        metrics = [
            "Accuracy (%)",
            "Precision (%)",
            "Recall (%)",
            "F1 Score (%)",
            "ROC-AUC (%)"
        ]

        if all(
            column in comparison_df.columns
            for column in ["Model"] + metrics
        ):

            st.write(
                "Compare the evaluation metrics of the models "
                "trained on the Coimbra dataset."
            )

            st.bar_chart(
                comparison_df.set_index("Model")[metrics]
            )

            chart_data = comparison_df.melt(
                id_vars=["Model"],
                value_vars=metrics,
                var_name="Metric",
                value_name="Score (%)"
            )

            fig = px.bar(
                chart_data,
                x="Metric",
                y="Score (%)",
                color="Model",
                barmode="group",
                title="Model Performance Comparison",
                text_auto=".2f"
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1120",
                plot_bgcolor="#0b1120",
                font_color="#f1f5f9",
                yaxis_range=[0, 100]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "📥 Download Comparison Results",
                data=comparison_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="model_comparison_results.csv",
                mime="text/csv",
                use_container_width=True
            )

        else:

            st.error(
                "The model comparison file has unexpected columns."
            )

    else:

        st.error(
            "Model comparison results are missing. "
            "We will generate them next."
        )

# =====================================================
# HYPERPARAMETER TUNING
# =====================================================

elif page == "Hyperparameter Tuning":

    st.title("🎯 Hyperparameter Tuning")

    st.write(
        "Logistic Regression was tuned using three-fold "
        "cross-validation and ROC-AUC evaluation."
    )

    st.markdown("### Parameters Evaluated")

    st.dataframe(
        pd.DataFrame({
            "Parameter": [
                "regParam",
                "elasticNetParam",
                "maxIter",
                "Cross-validation folds",
                "Parameter combinations"
            ],
            "Configuration": [
                "0.001, 0.01, 0.1",
                "0.0, 0.5, 1.0",
                "100",
                "3",
                "9"
            ]
        }),
        use_container_width=True,
        hide_index=True
    )

    if tuning_df is not None:

        st.markdown("### Saved Tuning Results")

        st.dataframe(
            tuning_df,
            use_container_width=True,
            hide_index=True
        )

        if len(tuning_df) > 0:

            row = tuning_df.iloc[0]

            if "Best regParam" in tuning_df.columns:
                reg_param = row["Best regParam"]
            else:
                reg_param = None

            if "Best elasticNetParam" in tuning_df.columns:
                elastic_param = row["Best elasticNetParam"]
            else:
                elastic_param = None

            if (
                reg_param is not None
                and elastic_param is not None
            ):

                col1, col2 = st.columns(2)

                col1.metric(
                    "Best regParam",
                    f"{float(reg_param):g}"
                )

                col2.metric(
                    "Best elasticNetParam",
                    f"{float(elastic_param):g}"
                )

            auc_column = (
                "Best Cross-Validation ROC-AUC (%)"
            )

            if auc_column in tuning_df.columns:

                st.metric(
                    "Best Cross-Validation ROC-AUC",
                    f'{float(row[auc_column]):.2f}%'
                )

        st.download_button(
            "📥 Download Tuning Results",
            data=tuning_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="hyperparameter_tuning_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:

        st.error(
            "Tuning results were not found. Run "
            "hyperparameter_tuning.py first."
        )

    st.info(
        "Your latest tuning run selected regParam = 0.1 "
        "and elasticNetParam = 0.5, with a cross-validation "
        "ROC-AUC of 79.80%."
    )

# =====================================================
# CONFUSION MATRIX
# =====================================================

# Confusion Matrix Page
elif page == "Confusion Matrix":

    st.title("🔲 Confusion Matrix")

    confusion_path = os.path.join(
        BASE_DIR,
        "confusion_matrix.png"
    )

    if os.path.exists(confusion_path):
        st.image(
            confusion_path,
            caption="Confusion Matrix - Logistic Regression",
            use_container_width=True
        )
    else:
        st.warning(
            "Confusion matrix image not found. "
            "Run logistic_regression_model.py first."
        )

    st.markdown("""
    ### Interpretation

    - **Class 0:** Healthy control
    - **Class 1:** Breast cancer patient

    The confusion matrix compares the actual classifications
    with the classifications predicted by the Logistic Regression model.
    """)

# =====================================================
# ROC CURVE
# =====================================================

# ROC Curve Page
elif page == "ROC Curve":

    st.title("📈 ROC Curve")

    roc_path = os.path.join(
        BASE_DIR,
        "roc_curve.png"
    )

    if os.path.exists(roc_path):
        st.image(
            roc_path,
            caption="ROC Curve - Logistic Regression",
            use_container_width=True
        )
    else:
        st.warning(
            "ROC curve image not found. "
            "Run logistic_regression_model.py first."
        )

    st.markdown("""
    ### Interpretation

    The ROC curve evaluates the model's ability to distinguish
    between healthy controls and breast cancer patients.

    **ROC-AUC: 81.11%**

    A value closer to 1 indicates better discrimination.
    This result is based on the current test split and does not
    establish clinical diagnostic performance.
    """)

# =====================================================
# LIVE PREDICTION
# =====================================================

elif page == "Live Prediction":

    st.title("🔬 Live Classification")
    st.markdown("### Logistic Regression | Apache Spark MLlib")

    st.warning(
        "Educational prototype only. This model has not been "
        "clinically validated and must not be used to diagnose "
        "patients or make medical decisions."
    )

    if not os.path.exists(MODEL_PARAMETERS_PATH):

        st.error(
            "Model parameters were not found. Run "
            "logistic_regression_model.py first."
        )

    else:

        try:

            with open(
                MODEL_PARAMETERS_PATH,
                "r",
                encoding="utf-8"
            ) as file:
                params = json.load(file)

            feature_names = params["feature_names"]
            means = np.asarray(params["scaler_mean"], dtype=float)
            stds = np.asarray(params["scaler_std"], dtype=float)
            coefficients = np.asarray(
                params["coefficients"], dtype=float
            )
            intercept = float(params["intercept"])
            threshold = float(params["threshold"])

            lengths = [
                len(feature_names),
                len(means),
                len(stds),
                len(coefficients)
            ]

            if len(set(lengths)) != 1:
                st.error(
                    "Model parameter dimensions do not match. "
                    "Run logistic_regression_model.py again."
                )
                st.stop()

            if feature_names != FEATURES:
                st.error(
                    "The saved model does not match the Coimbra "
                    "dataset features. Run the updated "
                    "logistic_regression_model.py first."
                )
                st.stop()

            if (
                not np.all(np.isfinite(means))
                or not np.all(np.isfinite(stds))
                or not np.all(np.isfinite(coefficients))
                or not np.isfinite(intercept)
                or not np.isfinite(threshold)
                or np.any(stds <= 0)
                or not 0 < threshold < 1
            ):
                st.error(
                    "The saved model parameters contain invalid values. "
                    "Regenerate the model parameters."
                )
                st.stop()

            st.success("Coimbra Logistic Regression model loaded.")

            st.markdown("## Enter the Nine Measurements")

            st.caption(
                "Enter values in the same measurement units and "
                "ranges as the Coimbra dataset. Zero is used only "
                "as an initial input value, not as a recommended "
                "measurement."
            )

            input_mode = st.radio(
                "Input method",
                [
                    "Enter values manually",
                    "Select a dataset record"
                ],
                horizontal=True
            )

            sample_row = None

            if (
                input_mode == "Select a dataset record"
                and df is not None
                and all(name in df.columns for name in FEATURES)
            ):

                index = st.number_input(
                    "Dataset row (starting from 0)",
                    min_value=0,
                    max_value=len(df) - 1,
                    value=0,
                    step=1
                )

                sample_row = df.iloc[int(index)]

                st.caption(
                    "The record's class label is not used as a model input."
                )

            with st.form("coimbra_prediction_form"):

                values = {}

                for start in range(0, len(feature_names), 3):

                    group = feature_names[start:start + 3]
                    boxes = st.columns(len(group))

                    for box, feature in zip(boxes, group):

                        default = (
                            float(sample_row[feature])
                            if sample_row is not None
                            else float(means[feature_names.index(feature)])
                        )

                        values[feature] = box.number_input(
                            feature,
                            value=default,
                            format="%.6f",
                            key=f"coimbra_{feature}"
                        )

                submitted = st.form_submit_button(
                    "🔍 Classify Measurements",
                    use_container_width=True
                )

            if submitted:

                raw_values = np.array(
                    [values[name] for name in feature_names],
                    dtype=float
                )

                if not np.all(np.isfinite(raw_values)):

                    st.error("Please enter valid numerical values.")

                else:

                    scaled = (raw_values - means) / stds
                    logit = float(np.dot(scaled, coefficients) + intercept)

                    # Numerically stable sigmoid
                    if logit >= 0:
                        probability = 1.0 / (
                            1.0 + np.exp(-min(logit, 700))
                        )
                    else:
                        exp_value = np.exp(max(logit, -700))
                        probability = exp_value / (1.0 + exp_value)

                    predicted_class = (
                        1 if probability > threshold else 0
                    )

                    healthy_probability = 1.0 - probability

                    st.markdown("---")
                    st.markdown("## Model Output")

                    result1, result2 = st.columns(2)

                    with result1:
                        if predicted_class == 1:
                            st.metric(
                                "Predicted Dataset Class",
                                "Class 2"
                            )
                            st.write("Breast cancer patient class")
                        else:
                            st.metric(
                                "Predicted Dataset Class",
                                "Class 1"
                            )
                            st.write("Healthy control class")

                    with result2:
                        st.metric(
                            "Model Score for Class 2",
                            f"{probability * 100:.2f}%"
                        )

                    st.caption(
                        "These are model scores for the dataset classes, "
                        "not a clinically validated probability of cancer."
                    )

                    probability_df = pd.DataFrame({
                        "Dataset Class": [
                            "Class 1: Healthy control",
                            "Class 2: Breast cancer patient"
                        ],
                        "Model Score (%)": [
                            healthy_probability * 100,
                            probability * 100
                        ]
                    })

                    fig = px.bar(
                        probability_df,
                        x="Dataset Class",
                        y="Model Score (%)",
                        title="Model Score Distribution",
                        text_auto=".2f"
                    )

                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0b1120",
                        plot_bgcolor="#0b1120",
                        font_color="#f1f5f9",
                        yaxis_range=[0, 100]
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.dataframe(
                        probability_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.warning(
                        "This output is for education and research only. "
                        "It is not a medical diagnosis, and a low model "
                        "score does not rule out breast cancer. Consult "
                        "a qualified healthcare professional for medical "
                        "assessment."
                    )

        except (
            KeyError,
            ValueError,
            TypeError,
            OSError,
            OverflowError
        ) as error:

            st.error(
                f"Could not load or use model parameters: {error}"
            )

# =====================================================
# ABOUT PROJECT
# =====================================================

elif page == "About Project":

    st.title("ℹ️ About Project")

    st.markdown("### Project Title")

    st.write(
        "Breast Cancer Classification Using Apache Spark MLlib "
        "with Hyperparameter Tuning and Performance Evaluation"
    )

    st.markdown("### Dataset")

    st.write(
        "The project uses the Breast Cancer Coimbra dataset, "
        "which contains 116 records and nine numerical features. "
        "The target column is Classification."
    )

    st.write(
        "Dataset labels: 1 = healthy control; "
        "2 = breast cancer patient."
    )

    st.markdown("### Project Objective")

    st.write(
        "To demonstrate binary classification, model evaluation "
        "and hyperparameter tuning using Apache Spark MLlib."
    )

    st.markdown("### Technologies Used")

    st.dataframe(
        pd.DataFrame({
            "Technology": [
                "Python",
                "Apache Spark MLlib",
                "Pandas",
                "NumPy",
                "Streamlit",
                "Plotly",
                "CSV and JSON"
            ],
            "Purpose": [
                "Programming language",
                "Machine learning model training",
                "Dataset processing",
                "Numerical calculations",
                "Interactive dashboard",
                "Interactive visualizations",
                "Dataset and model parameter storage"
            ]
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Algorithms and Methods")

    st.write("1. Logistic Regression")
    st.write("2. Decision Tree Classification")
    st.write("3. Three-fold cross-validation")
    st.write("4. Hyperparameter tuning")
    st.write("5. Confusion matrix and ROC-AUC evaluation")

    st.markdown("### Project Workflow")

    workflow = [
        "1. Load the Coimbra dataset.",
        "2. Prepare nine numerical features and the binary target.",
        "3. Split the data into training and testing sets.",
        "4. Train Logistic Regression and Decision Tree models.",
        "5. Tune Logistic Regression using cross-validation.",
        "6. Evaluate models using performance metrics.",
        "7. Display the results in the Streamlit dashboard."
    ]

    for step in workflow:
        st.write(step)

    st.warning(
        "This project is for educational and research purposes only. "
        "It must not be used to diagnose cancer or make medical decisions."
    )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Breast Cancer Classification | "
    "Python • Apache Spark MLlib • Streamlit • Plotly"
)