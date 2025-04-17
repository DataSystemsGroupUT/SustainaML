import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="SustainaML AutoML", layout="wide")
st.title("SustainaML AutoML")

# Sidebar: Dataset upload
st.sidebar.header("Dataset Configuration")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
# Add a button to show dataset insights
if "show_dataset_insights" not in st.session_state:
    st.session_state["show_dataset_insights"] = False

if st.sidebar.button(" Dataset Insights"):
    st.session_state["show_dataset_insights"] = not st.session_state["show_dataset_insights"]

# Display Dataset Insights only if the button is clicked
if st.session_state["show_dataset_insights"] and uploaded_file:
    df = pd.read_csv(uploaded_file)

    #st.subheader("Dataset Insights")

    # Show dataset preview
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Show dataset summary statistics
    summary_stats = df.describe().drop('count').rename(index={
        "25%": "Q1 (25th percentile)",
        "50%": "Median (50th percentile)",
        "75%": "Q3 (75th percentile)"
    })
    
    st.write("### Summary Statistics")

    # Create a styled table for summary statistics
    styled_stats = summary_stats.style.set_table_styles([
        # Style header cells in the table head
        {'selector': 'thead th', 'props': [('color', 'black'), ('background-color', 'white')]},
        # Style the row index cells (often rendered as th in tbody)
        {'selector': 'tbody th', 'props': [('color', 'black'), ('background-color', 'white')]},
        # Style the first data cell (if needed; for non-index first column cells)
        {'selector': 'tbody td:first-child', 'props': [('color', 'black'), ('background-color', 'white')]}
    ])
    # Render the styled table as HTML
    st.markdown(styled_stats.to_html(), unsafe_allow_html=True)

    # Show missing values
    st.write("### Missing Values")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        st.write(missing_values[missing_values > 0])
    else:
        st.write("No missing values in the dataset.")
    # # Get numeric columns
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

    # Show correlation heatmap
    if len(numeric_columns) > 1:
        st.write("### Correlation Heatmap")
        plt.figure(figsize=(6, 4))  # Make the plot smaller
        corr = df[numeric_columns].corr()
        # Use smaller annotation text and rotate x-axis labels
        sns.heatmap(
            corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5,
            annot_kws={"size": 8}   # reduce annotation font size
        )
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        st.pyplot(plt)
    else:
        st.write("Not enough numeric features for a correlation heatmap.")
# Sidebar: Framework and algorithm selection
with st.sidebar.expander("Select Frameworks and Algorithms"):
    frameworks = ["FLAML", "H2O", "MLJAR"]
    selected_frameworks = st.multiselect("Choose AutoML Frameworks:", frameworks)

    algorithm_selection = {}
    default_hyperparams = { # Default hyperparameters for all algorithms
        "FLAML": {
            "Random Forest": {"n_estimators": 100, "max_depth": 6, "min_samples_split": 2, "min_samples_leaf": 1},
            "XGBoost": {"learning_rate": 0.1, "n_estimators": 100, "max_depth": 6, "subsample": 1.0, "colsample_bytree": 1.0},
            "LightGBM": {"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 100, "min_data_in_leaf": 20},
            "Extra Trees": {"n_estimators": 100, "max_depth": 6, "min_samples_split": 2, "min_samples_leaf": 1},
           # "CatBoost": {"iterations": 500, "learning_rate": 0.03, "depth": 6},
            "K-Nearest Neighbors": {"n_neighbors": 5, "weights": 'uniform'}, #"algorithm": 'auto'
            "Logistic Regression": {"penalty": 'l2', "C": 1.0}, # , "solver": 'lbfgs'
        },
        "H2O": {
            "GLM": {"alpha": 0.5, "lambda": 0.1},
            "GBM": {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 6},
            "Naive Bayes": {"var_smoothing": 1e-9},
            "Distributed Random Forest": {"n_estimators": 200, "max_depth": 6, "min_samples_split": 2, "min_samples_leaf": 1},
            "XGBoost": {"learning_rate": 0.1, "n_estimators": 100, "max_depth": 6, "subsample": 1.0,
                        "colsample_bytree": 1.0},
        },
        "MLJAR": {
            "Baseline": {"penalty": 'l2', "C": 1.0}, #, "solver": 'lbfgs'
            "Decision Tree": {"max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1},
            "Random Forest": {"n_estimators": 100, "max_depth": 6, "min_samples_split": 2, "min_samples_leaf": 1},
            "XGBoost": {"learning_rate": 0.1, "n_estimators": 100, "max_depth": 6, "subsample": 1.0, "colsample_bytree": 1.0},
            "Neural Network": {"kernel": 'linear', "C": 1.0}, #
            "Extra Trees": {"n_estimators": 100, "max_depth": 6, "min_samples_split": 2, "min_samples_leaf": 1},
            "LightGBM": {"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 100, "min_data_in_leaf": 20},
            "Support Vector Machines": { "kernel": 'rbf', "C": 1.0}, #
            "K-Nearest Neighbors": {"n_neighbors": 5, "weights": 'uniform'}, #, "algorithm": 'auto'
        }
    }
    # Algorithm Selection UI
    for framework in selected_frameworks:
        st.subheader(f"{framework} Algorithms")
        algorithm_selection[framework] = {}

        for algo in default_hyperparams[framework].keys():
            algorithm_selection[framework][algo] = st.checkbox(algo, value=True, key=f"{framework}_{algo}")

# Step 1: Add a button to modify hyperparameters
if st.sidebar.button("Modify Hyperparameters"):
    if not selected_frameworks or not any(any(val for val in algo.values()) for algo in algorithm_selection.values()):
        st.sidebar.warning("Please select at least one framework and algorithm.")
    else:
        st.session_state["show_dataset_insights"] = False
        st.session_state["show_hyperparam_ui"] = True
####################
# Time Budget Dropdown in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("⏱️ Choose Time Budget")
time_options = {
     "10 seconds": 10,
    "30 seconds": 30,
    "60 seconds": 60,
    "120 seconds": 120,
}
default_time = "10 seconds"
                # Initialize only once
if "selected_time_budget" not in st.session_state:
    st.session_state.selected_time_budget = default_time
    st.session_state.time_budget = time_options[default_time]
# Dropdown selector
selected_label = st.sidebar.selectbox(
    "Select a time budget for AutoML:",
    list(time_options.keys()),
    index=list(time_options.keys()).index(st.session_state.selected_time_budget)
)
st.session_state.selected_time_budget = selected_label
st.session_state.time_budget = time_options[selected_label]
####################   
    # Ensure "Dataset Insights" is hidden when modifying hyperparameters
if st.session_state.get("show_hyperparam_ui", False):
    st.session_state["show_dataset_insights"] = False

# Step 2: Show Hyperparameter Modification UI
if "show_hyperparam_ui" in st.session_state and st.session_state["show_hyperparam_ui"]:
    st.subheader("Modify Hyperparameters")

    modified_hyperparams = {}
    for framework in selected_frameworks:
        st.write(f"**{framework} Algorithms**")
        modified_hyperparams[framework] = {}

        for algo, selected in algorithm_selection[framework].items():
            if selected:
                st.write(f"**{algo} Hyperparameters:**")
                hyperparams = default_hyperparams[framework].get(algo, {})

                modified_hyperparams[framework][algo] = {}
                for param, default_value in hyperparams.items():
                    unique_key = f"{framework}_{algo}_{param}"  # Unique key for each UI element
                    if isinstance(default_value, (int, float)):  # Handle numeric parameters
                        modified_hyperparams[framework][algo][param] = st.number_input(
                            f"{algo} - {param}",
                            value=default_value,
                            key=unique_key,  # Assign unique key
                        )
                    elif isinstance(default_value, str):  # Handle string parameters
                        options = {"weights": ["uniform", "distance"], "kernel": ["linear", "rbf", "poly", "sigmoid"]}
                        modified_hyperparams[framework][algo][param] = st.selectbox(
                            f"{algo} - {param}", #options=["uniform", "distance"],
                            options=options.get(param, [default_value]),
                            key=unique_key,  # Assign unique key
                        )
                    else:
                        st.warning(f"Skipping unsupported parameter: {param} (type: {type(default_value)})")

    if st.button("Confirm Hyperparameters"):
        st.session_state["modified_hyperparams"] = modified_hyperparams
        st.session_state["show_hyperparam_ui"] = False
        st.success("Hyperparameters updated! You can now run AutoML.")

# Persistent state for DataFrame and valid metrics
if "df_metrics" not in st.session_state:
    st.session_state.df_metrics = None
if "valid_metrics" not in st.session_state:
    st.session_state.valid_metrics = []
if "show_feature_importance" not in st.session_state:
    st.session_state["show_feature_importance"] = False
if "show_pipeline_analysis" not in st.session_state:
    st.session_state["show_pipeline_analysis"] = False
if "automl_results" not in st.session_state:
    st.session_state["automl_results"] = {}
    # Sidebar button to toggle feature importance
if 'show_dataset_insights' in st.session_state:
    st.session_state['show_dataset_insights'] = False
if 'show_hyperparam_ui' in st.session_state:
    st.session_state['show_hyperparam_ui'] = False


# Run AutoML Button / step 3
if st.sidebar.button("Run AutoML"):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.info("Running AutoML with selected frameworks and metrics...")

        # Send data to backend
        dataset_json = df.to_json()
        payload = {
            "frameworks": selected_frameworks,
            "algorithms": algorithm_selection,
            "hyperparams": st.session_state.get("modified_hyperparams", {}),
            "metric": "Accuracy",
            "data": dataset_json,
            "time_budget": st.session_state.get("time_budget", 30),
        }

        response = requests.post("http://127.0.0.1:5000/run_automl", json=payload)

        if response.status_code == 200:

            results = response.json().get("results", {})
            st.session_state["automl_results"] = results  # Store results for feature importance
            # Store comparison by time budget
            time_label = f"{st.session_state.selected_time_budget}"
            comparison_entry = {
                "time_budget": time_label,
                "results": results
            }

            if "time_budget_comparisons" not in st.session_state:
                st.session_state["time_budget_comparisons"] = []

            st.session_state["time_budget_comparisons"].append(comparison_entry)
            if not results:
                st.error("No results were generated. Please check the backend logs for details.")
            else:
                metrics_data = []
                for algo, metrics in results.items():
                    if "error" in metrics:
                        st.warning(f"Algorithm {algo} failed: {metrics['error']}")
                    else:
                        row = {"Algorithm": algo}
                        # Ensure CO2 emissions are rounded to 4 decimal places
                        if "CO2 Emission" in metrics:
                            metrics["CO2 Emission"] = round(metrics["CO2 Emission"], 10)
                        row.update(metrics)
                        metrics_data.append(row)

                # Update session state
                if metrics_data:
                    st.session_state.df_metrics = pd.DataFrame(metrics_data)
                    st.session_state.valid_metrics = [
                        col for col in st.session_state.df_metrics.columns[1:]
                        if pd.api.types.is_numeric_dtype(st.session_state.df_metrics[col])
                    ]

                    st.success("AutoML run completed successfully!")

        else:
            st.error(f"Error running AutoML: {response.json().get('message', 'Unknown error')}")

    #  Ensure session state is initialized
    if "show_feature_importance" not in st.session_state:
        st.session_state["show_feature_importance"] = False
    if "automl_results" not in st.session_state:
        st.session_state["automl_results"] = {}

#  Sidebar: Feature Importance Button (Always Visible)
st.sidebar.markdown("---")
st.sidebar.subheader("Analysis Tool")
# Sidebar: Show AutoML Results Button
if st.sidebar.button("Model Leaderboard"):
    st.session_state["show_automl_results"] = True
    st.session_state["show_feature_importance"] = False
    st.session_state["show_pipeline_analysis"] = False
    st.session_state["show_comparison"] = False

    if "automl_results" in st.session_state and st.session_state["automl_results"]:
        st.session_state["show_automl_results"] = True
    else:
        st.warning("Please run AutoML first.")
        st.rerun()

if st.sidebar.button("Feature Importance"):
    if not st.session_state.get("automl_results"):
        st.sidebar.warning("⚠️ Please run AutoML first before viewing feature importance.")
    else:
        st.session_state["show_feature_importance"] = True
        st.session_state["show_automl_results"] = False
        st.session_state["show_pipeline_analysis"] = False  # Ensure only one view is active
        st.rerun()  #  Force UI refresh to apply changes immediately
#  Pipeline Analysis Toggle
#if "show_pipeline_analysis" not in st.session_state:
#    st.session_state["show_pipeline_analysis"] = False
if st.sidebar.button("Hyperparameter Analysis"):
    st.session_state["show_hyperimpact_analysis"] = True
    st.session_state["show_pipeline_analysis"] = False
    st.session_state["show_feature_importance"] = False
    st.session_state["show_automl_results"] = False
    st.rerun()
if st.sidebar.button("Time Budgets Comparison"):
    st.session_state["show_comparison"] = True
    st.session_state["show_feature_importance"] = False
    st.session_state["show_pipeline_analysis"] = False
    st.session_state["show_automl_results"] = False

if st.sidebar.button("Process Overview"):
    if not st.session_state.get("automl_results"):
        st.sidebar.warning("⚠️ Please run AutoML first before viewing pipeline analysis.")
    else:
        st.session_state["show_pipeline_analysis"] = True
        st.session_state["show_automl_results"] = False
        st.session_state["show_feature_importance"] = False  # Ensure only one view is active
        st.rerun()
####################################################
# 🚀 Show either the Main UI or Feature Importance (NEVER both)
# ✅ Always initialize df_metrics at the start to prevent NameError
df_metrics = st.session_state.df_metrics if "df_metrics" in st.session_state else None

if st.session_state.get("show_automl_results", False):
    valid_metrics = st.session_state.valid_metrics if st.session_state.valid_metrics else ["No Metrics Available"]
    # Algorithm Metrics Section
    st.subheader("Algorithm Metrics")

    if df_metrics is not None and not df_metrics.empty:
        available_frameworks = list(set(algo.split("_")[0] for algo in df_metrics["Algorithm"]))
        selected_frameworks = st.multiselect(
            "Select Framework(s) to Display:", available_frameworks, default=available_frameworks
        )

        max_algorithms = len(df_metrics)
        top_n = st.number_input(
            f"Select Number of Top Algorithms (Max: {max_algorithms}):",
            min_value=1, max_value=max_algorithms, value=max_algorithms, step=1
        )

        df_filtered = df_metrics[df_metrics["Algorithm"].str.startswith(tuple(selected_frameworks))]
        df_filtered = df_filtered.sort_values(by="Accuracy", ascending=False).head(top_n)

        if "feature_importance" in df_filtered.columns:
            df_filtered = df_filtered.drop(columns=["feature_importance"])
        if "hyperparameters" in df_filtered.columns:
            df_filtered = df_filtered.drop(columns=["hyperparameters"])

        selected_metric = st.selectbox(
            "Select which single metric to display:",
            ["Accuracy", "F1 Score", "CO2 Emission"]
        )

        columns_to_keep = ["Algorithm", selected_metric]
        df_filtered = df_filtered[columns_to_keep]

        st.dataframe(df_filtered, hide_index=True)

    # Algorithm Performance Section (Bar Chart)
    st.subheader("Algorithm Performance")

    if df_metrics is not None and not df_metrics.empty:
        available_frameworks = list(set(algo.split("_")[0] for algo in df_metrics["Algorithm"]))
        selected_frameworks = st.multiselect(
            "Select Framework(s) to Display:", available_frameworks, default=available_frameworks,
            key="framework_selection_performance"
        )

        metric_to_plot = st.selectbox("Select Metric to Visualize", valid_metrics, index=0, key="bar_metric")
        max_algorithms = len(df_metrics)
        top_n = st.number_input(
            f"Select Number of Top Algorithms (Max: {max_algorithms}):",
            min_value=1, max_value=max_algorithms, value=max_algorithms, step=1, key="top_n_performance"
        )

        df_filtered = df_metrics[df_metrics["Algorithm"].str.startswith(tuple(selected_frameworks))]
        df_filtered = df_filtered.sort_values(by=metric_to_plot, ascending=False).head(top_n)
        df_filtered["Framework"] = df_filtered["Algorithm"].apply(lambda x: x.split("_")[0])

        bar_fig = px.bar(
            df_filtered,
            x="Algorithm",
            y=metric_to_plot,
            color="Framework",
            barmode="group",
            title=f"Algorithm Comparison by {metric_to_plot}",
        )
        bar_fig.update_layout(
                    xaxis=dict(
                        title=dict(font=dict(size=14, color="black", family="Arial Black")),
                        tickfont=dict(size=15, color="black", family="Arial")
                    ),
                    yaxis=dict(
                        title=dict(font=dict(size=14, color="black", family="Arial Black")),
                        tickfont=dict(size=15, color="black", family="Times New Roman")
                    ),
                    legend=dict(
                        font=dict(
                            size=13,  # Set the font size for the legend
                            color="black"  # Set the color of the legend text to black
                        ),
                    ),
                )

        st.plotly_chart(bar_fig)

    # Performance Trends Section (Scatter Plot)
    st.subheader("Performance Trends")

    if valid_metrics and valid_metrics != ["No Metrics Available"]:
        selected_frameworks_scatter = st.multiselect(
            "Select Framework(s) for Scatter Plot:", available_frameworks, default=available_frameworks,
            key="scatter_framework"
        )

        x_axis = st.selectbox("Select X-Axis", valid_metrics, index=0, key="scatter_x")
        y_axis = st.selectbox("Select Y-Axis", valid_metrics, index=1, key="scatter_y")

        df_scatter_filtered = df_metrics[df_metrics["Algorithm"].str.startswith(tuple(selected_frameworks_scatter))]
        if "Framework" not in df_scatter_filtered.columns:
            df_scatter_filtered["Framework"] = df_scatter_filtered["Algorithm"].apply(lambda x: x.split("_")[0])

        if not df_scatter_filtered.empty:
            bubble_metric = st.selectbox(
                "Select Metric for Bubble Size:", valid_metrics, index=valid_metrics.index("CO2 Emission"),
                key="bubble_size"
            )

            scatter_fig = px.scatter(
                df_scatter_filtered,
                x=x_axis,
                y=y_axis,
                color="Framework",
                size=bubble_metric,
                hover_name="Algorithm",
                hover_data=["Accuracy", "F1 Score", "CO2 Emission"],
                title=f"{y_axis} vs {x_axis} by Framework",
                size_max=12
            )
            scatter_fig.update_layout(
                    xaxis=dict(
                        title=dict(font=dict(size=14, color="black", family="Arial Black")),
                        tickfont=dict(size=15, color="black", family="Arial")
                    ),
                    yaxis=dict(
                        title=dict(font=dict(size=14, color="black", family="Arial Black")),
                        tickfont=dict(size=15, color="black", family="Times New Roman")
                    ),
                    legend=dict(
                        font=dict(
                            size=13,  # Set the font size for the legend
                            color="black"  # Set the color of the legend text to black
                        ),
                    ),
                )
            st.plotly_chart(scatter_fig)

        else:
            st.warning("No data available for the selected frameworks.")

# ✅ If no data is available, show a warning
if df_metrics is None or (isinstance(df_metrics, pd.DataFrame) and df_metrics.empty):
    st.warning("No data available for visualization. Run AutoML first.")
elif st.session_state.get("show_pipeline_analysis", False):
    st.subheader("Pipeline Analysis")

    # Define pipeline steps per framework-algorithm combo
    pipeline_steps = {
        "FLAML_Random Forest": [
            "Input", "Random Forest Initialization", "Hyperparameter Initialization", "Model Fit",
            "Random Subsampling", "Majority Voting", "Model Evaluation", "Prediction"
        ],
        "FLAML_Extra Trees": [
            "Input", "Extra Trees Initialization", "Hyperparameter Initialization", "Model Fit",
            "Random Subsampling", "Tree Construction", "Majority Voting", "Model Evaluation", "Prediction"
        ],
        "FLAML_Logistic Regression": [
            "Input", "Logistic Regression Function: Sigmoid Function", "Hyperparameter Initialization",
            "Model Fit", "Optimization and Regularization", "Prediction"
        ],
        "FLAML_XGBoost": [
            "Input", "XGBoost Initialization", "Hyperparameter Initialization", "Model Fit",
            "Optimization", "Regularization", "Model Evaluation", "Prediction"
        ],
        "FLAML_CatBoost": [
            "Input", "CatBoost Initialization", "Hyperparameter Initialization", "Model Fit",
            "Optimization", "Regularization", "Model Evaluation", "Prediction"
        ],
        "FLAML_LightGBM": [
            "Input", "LightGBM Initialization", "Hyperparameter Initialization", "Model Fit",
            "Optimization", "Regularization", "Model Evaluation", "Prediction"
        ],
        "FLAML_K-Nearest Neighbors": [
            "Input", "KNN Initialization", "Model Fit", "Distance Calculation", "Finding K Neighbors",
            "Majority Voting", "Model Evaluation", "Prediction"
        ],

        "H2O_GLM": [
            "Input", "Logistic Regression Function: Sigmoid Function", "Hyperparameter Initialization",
            "Model Fit", "Optimization and Regularization", "Prediction"
        ],
        "H2O_GBM": [
            "Input", "LightGBM Initialization", "Hyperparameter Initialization", "Model Fit",
            "Optimization", "Regularization", "Model Evaluation", "Prediction"
        ],
        "H2O_Naive Bayes": [
            "Input", "Gaussian NB Initialization", "Hyperparameter Initialization", "Model Fit",
            "Parameter Estimation", "Model Evaluation", "Prediction"
        ],
        "H2O_Distributed Random Forest": [
            "Input", "Random Forest Initialization", "Hyperparameter Initialization", "Model Fit",
            "Random Subsampling", "Majority Voting", "Model Evaluation", "Prediction"
        ],
        "H2O_XGBoost": [
            "Input", "XGBoost Initialization", "Hyperparameter Initialization", "Model Fit",
            "Optimization", "Regularization", "Model Evaluation", "Prediction"
        ],

        "MLJAR_Baseline": [
            "Input", "Logistic Regression Function: Sigmoid Function", "Hyperparameter Initialization",
            "Model Fit", "Optimization and Regularization", "Prediction"
        ],
        "MLJAR_Decision Tree": [
            "Input", "Extra Trees Initialization", "Hyperparameter Initialization", "Model Fit",
            "Random Subsampling", "Tree Construction", "Majority Voting", "Model Evaluation", "Prediction"
        ],
        "MLJAR_Random Forest": [
            "Input", "Random Forest Initialization", "Hyperparameter Initialization", "Model Fit",
            "Random Subsampling", "Majority Voting", "Model Evaluation", "Prediction"
        ],
        "MLJAR_XGBoost": [
            "Input", "XGBoost Initialization", "Hyperparameter Initialization", "Model Fit",
            "Optimization", "Regularization", "Model Evaluation", "Prediction"
        ],
        "MLJAR_Neural Network": [
            "Input", "SVC Initialization", "Hyperparameter Initialization", "Model Fit",
            "Kernel Transformation", "Optimization", "Regularization", "Model Evaluation", "Prediction"
        ],
        "MLJAR_Extra Trees": [
            "Input", "Extra Trees Initialization", "Hyperparameter Initialization", "Model Fit",
            "Random Subsampling", "Tree Construction", "Majority Voting", "Model Evaluation", "Prediction"
        ],
        "MLJAR_LightGBM": [
            "Input", "LightGBM Initialization", "Hyperparameter Initialization", "Model Fit",
            "Optimization", "Regularization", "Model Evaluation", "Prediction"
        ],
        "MLJAR_Support Vector Machines": [
            "Input", "SVC Initialization", "Hyperparameter Initialization", "Model Fit",
            "Kernel Transformation", "Optimization", "Regularization", "Model Evaluation", "Prediction"
        ],
        "MLJAR_K-Nearest Neighbors": [
            "Input", "KNN Initialization", "Model Fit", "Distance Calculation", "Finding K Neighbors",
            "Majority Voting", "Model Evaluation", "Prediction"
        ]
    }

    selected_algos = list(st.session_state["automl_results"].keys())

    for algo_key in selected_algos:
        st.markdown(f"### {algo_key}")
        steps = pipeline_steps.get(algo_key)

        if steps:
            # Inline rendering
            flow_str = ""
            for i, step in enumerate(steps):
                if "Hyperparameter Initialization" in step:
                    # Add an expandable tooltip-style container
                    with st.expander(f"🔧 {step} (click to view hyperparameters)"):
                        hyperparams = st.session_state["automl_results"].get(algo_key, {}).get("hyperparameters", {})
                        if hyperparams:
                            st.json(hyperparams)
                        else:
                            st.write("No hyperparameters found.")
                    flow_str += f"`{step}`"
                else:
                    flow_str += f"`{step}`"
                if i < len(steps) - 1:
                    flow_str += " ➡️ "
            st.markdown(flow_str)
        else:
            st.warning(f"No pipeline steps defined for {algo_key}")

elif st.session_state.get("show_feature_importance", False):
    # ✅ Ensure Main UI is hidden and Feature Importance is fully displayed
    st.subheader(" Feature Importance Analysis")

    results = st.session_state.automl_results
    feature_data = []

    # Extract feature importance data from results
    for algo, metrics in results.items():
        if "feature_importance" in metrics and metrics["feature_importance"]:
            for feature, importance in metrics["feature_importance"].items():
                feature_data.append({
                    "Algorithm": algo,
                    "Feature": feature,
                    "Importance": float(importance)
                })

    if feature_data:
        df_feature_importance = pd.DataFrame(feature_data).sort_values(by="Importance", ascending=False)

        # ✅ Add Framework Selection
        available_frameworks = list(set(algo.split("_")[0] for algo in df_feature_importance["Algorithm"]))
        selected_frameworks = st.multiselect(
            "Select Framework(s) to Display:", available_frameworks, default=available_frameworks,
            key="feature_framework"
        )

        # ✅ Add Top N Selection
        max_features = len(df_feature_importance)
        top_n = st.number_input(
            f"Select Number of Top Features (Max: {max_features}):",
            min_value=1, max_value=max_features, value=max_features, step=1, key="top_n_features"
        )

        # ✅ Filter Data by Selected Frameworks
        df_filtered = df_feature_importance[
            df_feature_importance["Algorithm"].str.startswith(tuple(selected_frameworks))]

        # ✅ Show only Top N most important features
        df_filtered = df_filtered.sort_values(by="Importance", ascending=False).head(top_n)

        # ✅ Remove Index Column
        df_filtered = df_filtered.reset_index(drop=True)

        # ✅ Display the Filtered Table
        st.dataframe(df_filtered, hide_index=True)

        # ✅ Feature Importance Bar Chart
        fig = px.bar(
            df_filtered, x="Feature", y="Importance", color="Algorithm",
            barmode="group", title="Feature Importance Across Models"
        )
        fig.update_layout(
                    xaxis=dict(
                        title=dict(font=dict(size=14, color="black", family="Arial Black")),
                        tickfont=dict(size=14, color="black", family="Arial")
                    ),
                    yaxis=dict(
                        title=dict(font=dict(size=14, color="black", family="Arial Black")),
                        tickfont=dict(size=14, color="black", family="Times New Roman")
                    )
                )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No feature importance data found for the selected models.")
#################################################################
# ✅ Comparison View: Show metrics over time budget history
elif st.session_state.get("show_comparison", False):
    st.subheader("Metric comparison")

    comparisons = st.session_state.get("time_budget_comparisons", [])
    if not comparisons:
        st.warning("No time budget comparisons available yet.")
    else:
        records = []
        for entry in comparisons:
            tb = entry["time_budget"]
            for algo, metrics in entry["results"].items():
                if "error" not in metrics:
                    records.append({
                        "Time Budget": tb,
                        "Algorithm": algo,
                        #"Accuracy": round(metrics.get("accuracy", 0), 4),
                        #"F1 Score": round(metrics.get("f1_score", 0), 4),
                        "CO2 Emission": round(metrics.get("CO2 Emission", 0), 4),
                        "Energy Consumption": round(metrics.get("Energy Consumption", 0), 6),
                         "Cost (µ¢)": round(metrics.get("cost_micro_cents", 0), 4),
                    })

        if records:
            df_comparison = pd.DataFrame(records)
            st.dataframe(df_comparison)
                # Refined Energy vs Accuracy plot with connected lines per framework
            st.subheader(" Multi-Budget Evaluation by time")

            if not df_comparison.empty:
                # Extract framework from algorithm name (e.g., FLAML from FLAML_XGBoost)
                df_comparison["Framework"] = df_comparison["Algorithm"].apply(lambda x: x.split("_")[0])
                df_comparison["Time Label"] = df_comparison["Time Budget"].apply(lambda x: x.split()[0] + "s")
                df_comparison["Time Budget Value"] = df_comparison["Time Budget"].str.extract(r'(\d+)').astype(int)
                #df_comparison = df_comparison.sort_values(by=["Framework", "Energy (µWh)", "CO2 Emissions (µg)"])
                df_comparison = df_comparison.sort_values(by=["Framework", "CO2 Emission", "Energy Consumption"])
                fig_line = px.line(
                    df_comparison,
                    x="CO2 Emission",
                    y="Energy Consumption",
                    color="Time Label",
                    text="Time Label",
                    hover_name="Algorithm",
                    hover_data={
                        "Algorithm": True,
                        "Time Label": True,
                        "CO2 Emission": True,
                        "Energy Consumption": True,
                        "Framework": False   },
                    markers=True,  # Ensures points show up with lines
                    line_group="Framework",
                    log_y=True,
                    title="Energy vs CO2 for Each Framework (log scale)",
                    labels={"Energy Consumption": "Energy Consumption"}, )
                #fig_line.update_traces(mode="lines+markers+text", textposition="top center")
                fig_line.update_traces(
                mode="lines+markers+text",
                textposition="middle center",
                textfont=dict(
                    size=11,
                    family="Arial Black",
                    color=None  ),
                marker=dict(
                    size=22,
                    color="white",  # White inner circle
                    line=dict(width=2, color=None)    )      )
                fig_line.update_layout(
                yaxis=dict(
                    type="log",
                    tickvals=[0.1, 0.3, 1, 3, 10],
                    showgrid=False,        # ❌ remove grey horizontal grid lines
                    zeroline=False,        # ❌ remove baseline if any
                    showline=True,         # ✅ show Y-axis line
                    linecolor='black',     # axis color
                    ticks="outside",
                    tickfont=dict(size=12) ),
                xaxis=dict(
                    showgrid=False,        # ❌ remove grey vertical grid lines
                    zeroline=False,
                    showline=True,         # ✅ show X-axis line
                    linecolor='black',
                    ticks="outside",
                    tickfont=dict(size=12) ),
                plot_bgcolor="white",      # Set background to white
            )
                 # Update x axis and y axis font properties:
                fig_line.update_layout(
                    xaxis=dict(
                        title=dict(font=dict(size=14, color="black", family="Arial Black")),
                        tickfont=dict(size=14, color="black", family="Arial")),
                    yaxis=dict(
                        title=dict(font=dict(size=14, color="black", family="Arial Black")),
                        tickfont=dict(size=14, color="black", family="Times New Roman")) )   
                st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.warning("No comparison data available to plot.")
            #####------------------------------------
        st.subheader("Single-Budget Comparison by Framework")
    if df_comparison is not None and not df_comparison.empty:
        # Set X and Y axes to CO2 Emission and Energy Consumption
        x_axis = "CO2 Emission"
        y_axis = "Energy Consumption"

        # Create a scatter plot
        fig = go.Figure()

        # Define framework colors
        framework_colors = {
            "FLAML": "red",
            "MLJAR": "blue",
            "H2O": "green",
        }

        # Loop through each framework
        for framework in df_comparison["Framework"].unique():
            df_framework = df_comparison[df_comparison["Framework"] == framework]
            framework_color = framework_colors.get(framework, "gray")  # Default color

            # Loop through each algorithm in the framework and add the scatter plot for each algorithm
            for algorithm in df_framework["Algorithm"].unique():
                df_algo = df_framework[df_framework["Algorithm"] == algorithm]

                # Plot each algorithm individually
                fig.add_trace(go.Scatter(
                    x=df_algo[x_axis], 
                    y=df_algo[y_axis],
                    mode="markers+lines",
                    line=dict(color=framework_color),  # Line color for the framework
                    marker=dict(size=10, color=framework_color, line=dict(width=2, color='black')),  # Circle properties
                    name=algorithm,  # Name each algorithm for plotting
                    text=df_algo["Time Budget"],  # Time budget for hover
                    hoverinfo='text',  # Show time budget on hover
                    showlegend=False  # Hide individual algorithm names in the legend
                ))

            # Add a dummy trace for the framework to show in the legend (only once per framework)
            fig.add_trace(go.Scatter(
                x=[None],  # Empty x-data to avoid plotting
                y=[None],  # Empty y-data to avoid plotting
                mode="markers",  # Just show a marker in the legend
                marker=dict(size=10, color=framework_color, line=dict(width=2, color='black')),  # Circle properties
                name=framework,  # Name for the framework to show in the legend
                showlegend=True  # Only show the framework name in the legend
            ))

        # Update the layout with labels and title
        fig.update_layout(
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            template='plotly_white',
            xaxis_title_font=dict(
                size=16,  # Increase the font size for the x-axis title
                color="black",  # Set the color to black
                family="Arial Black",  # Set the font to Arial Black for better visibility
            ),
            yaxis_title_font=dict(
                size=16,  # Increase the font size for the y-axis title
                color="black",  # Set the color to black
                family="Arial Black",  # Set the font to Arial Black for better visibility
            ),
            xaxis=dict(
                showgrid=False,  # Remove the grey grid lines
                showline=True,  # Show the X-axis line
                linecolor='black',  # Set the color of the axis line to black
                ticks="outside",  # Move the ticks outside the plot area
                tickfont=dict(size=15, color="black")  # Set tick font size and color
            ),
            yaxis=dict(
                showgrid=False,  # Remove the grey grid lines
                showline=True,  # Show the Y-axis line
                linecolor='black',  # Set the color of the axis line to black
                ticks="outside",  # Move the ticks outside the plot area
                tickfont=dict(size=15, color="black")  # Set tick font size and color
            )
        )

        # Display the figure
        st.plotly_chart(fig)
    else:
        st.warning("No comparison data available for plotting.")        

            #____________________________      
elif st.session_state.get("show_hyperimpact_analysis", False):
    st.subheader("Hyperparameter Impact Analysis")
    # Ask the user to select a performance metric for analysis.
    selected_metric = st.selectbox(
        "Select performance metric for analysis:",
        options=["Accuracy", "F1 Score", "CO2 Emission"] )
    # Let the user choose which algorithm's hyperparameters to display.
    available_algorithms = list(st.session_state.get("automl_results", {}).keys())
    if not available_algorithms:
        st.warning("No AutoML results available. Run AutoML first.")
    else:
        selected_algo = st.selectbox(
            "Select an algorithm for hyperparameter impact analysis:",
            options=available_algorithms
        )
        
        # Retrieve the hyperparameters and the selected metric value for the chosen algorithm.
        algo_result = st.session_state["automl_results"].get(selected_algo, {})
        hyperparams = algo_result.get("hyperparameters", {})
        metric_value = algo_result.get(selected_metric, None)
        
        if not hyperparams or metric_value is None:
            st.warning("Hyperparameter data or the selected performance metric is missing for the chosen algorithm.")
        else:
            # Filter to keep only numeric hyperparameters.
            numeric_hyperparams = {k: v for k, v in hyperparams.items() if isinstance(v, (int, float))}
            
            if not numeric_hyperparams:
                st.warning("No numeric hyperparameters found for impact analysis.")
            else:
                # Normalize the hyperparameter values by dividing by the maximum value.
                max_val = max(numeric_hyperparams.values())
                impact_data = []
                for param, value in numeric_hyperparams.items():
                    normalized_value = value / max_val if max_val != 0 else 0
                    # Compute an "impact" score by scaling the normalized value with the performance metric.
                    impact_score = normalized_value * metric_value
                    impact_data.append({
                        "Hyperparameter": param,
                        "Value": value,
                        "Impact": impact_score
                    })
                df_impact = pd.DataFrame(impact_data)
                
                # Create a grouped bar chart using Plotly Express.
                fig = px.bar(
                    df_impact,
                    x="Hyperparameter",
                    y="Impact",
                    #text="Value",
                     hover_data=["Value"],
                    title=f"Impact of Hyperparameters on {selected_metric.capitalize()} ({selected_algo})"
                )
                # Update x axis and y axis font properties:
                fig.update_layout(
                    xaxis=dict(
                        title=dict(font=dict(size=14, color="black", family="Arial black")),
                        tickfont=dict(size=16, color="black", family="Arial")
                    ),
                    yaxis=dict(
                        title=dict(font=dict(size=14, color="black", family="Arial black")),
                        tickfont=dict(size=16, color="black", family="Times New Roman"), 
                         title_text=f"Impact ({selected_metric})"
                    )
                    
                )
                st.plotly_chart(fig, use_container_width=True)
