from catboost import CatBoostClassifier
from flaml.default import XGBClassifier, LGBMClassifier
from flask import Flask, request, jsonify
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from codecarbon import OfflineEmissionsTracker
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
import time

def calculate_feature_importance(model, X, y):
    """Calculate feature importance for a given model."""
    try:
        if hasattr(model, "feature_importances_"):  # Tree-based models
            return dict(zip(X.columns, model.feature_importances_))
        elif hasattr(model, "coef_") and model.coef_.ndim == 1:  # Linear models
            return dict(zip(X.columns, model.coef_))
        else:  # Use permutation importance
            try:
                result = permutation_importance(model, X, y, scoring="accuracy", n_repeats=5, random_state=42)
                return dict(zip(X.columns, result.importances_mean))
            except Exception:
                return {}  # Return empty dictionary if permutation importance fails
    except Exception as e:
        print(f"Feature importance calculation failed: {e}")
        return {}  # Return empty dictionary if feature importance can't be calculated


app = Flask(__name__)

# Full algorithm mapping
framework_algorithms = {
    "FLAML": {
        "Random Forest": RandomForestClassifier(n_estimators=100, warm_start=True, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100,warm_start=True, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42),
        #"CatBoost": CatBoostClassifier(iterations=500, learning_rate=0.03, depth=6, verbose=0),
        "LightGBM": LGBMClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
    },
    "H2O": {
        "GLM": LogisticRegression(random_state=42),
        "GBM": LGBMClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "Distributed Random Forest": RandomForestClassifier(n_estimators=200,warm_start=True, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42),
    },
    "MLJAR": {
        "Baseline": LogisticRegression(random_state=42),
        "Decision Tree": ExtraTreesClassifier(max_depth=3, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, warm_start=True, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42),
        "Neural Network": SVC(kernel="linear", probability=True, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100,warm_start=True, random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "Support Vector Machines": SVC(kernel="rbf", probability=True, random_state=42),
        "K-Nearest Neighbors": SVC(kernel="linear", probability=True, random_state=42),
    },
}

@app.route('/run_automl', methods=['POST'])
def run_automl():
    try:
        # Parse incoming request
        data = request.json
        dataset_json = data.get("data")
        selected_frameworks = data.get("frameworks", [])
        selected_algorithms = data.get("algorithms", {})
        custom_hyperparams = data.get("hyperparams", {})  # New: Receive hyperparameters

        # Log the received input for debugging
        print("Received frameworks:", selected_frameworks)
        print("Received algorithms:", selected_algorithms)
        print("Custom hyperparameters:", custom_hyperparams)

        # Convert JSON dataset to DataFrame
        from io import StringIO
        df = pd.read_json(StringIO(dataset_json))
        print("Dataset shape:", df.shape)  # Log dataset shape for debugging
        X = df.iloc[:, :-1]  # Features
        y = df.iloc[:, -1]   # Target variable

        results = {}

        # Process each selected framework and algorithm
        for framework in selected_frameworks:
            framework_algos = framework_algorithms.get(framework, {})
            algorithms = selected_algorithms.get(framework, {})
            import time
            time_budget = data.get("time_budget", 60)
            for algo_name, is_selected in algorithms.items():

                if is_selected and algo_name in framework_algos:

                    model = framework_algos[algo_name]  # Get existing model
                    default_params = model.get_params()  # Get default parameters
                    # algo_info = framework_algos[algo_name]
                    # model_class = algo_info["class"]
                    # default_params = algo_info["default_params"]
                    # model = model_class(**default_params)  # 💥 New model instance every time
                    


                    # Apply custom hyperparameters if provided
                    user_params = custom_hyperparams.get(framework, {}).get(algo_name, {})
                    final_params = {**default_params, **user_params}  # Merge default and custom params
                    # Log the final parameters before applying
                    print(f"Final parameters for {framework} - {algo_name}: {final_params}")
                    # Update the existing model parameters
                    try:
                        model.set_params(**final_params)
                        # Log the successfully applied parameters
                        print(f"Successfully applied parameters to {framework} - {algo_name}: {model.get_params()}")
                    except Exception as e:
                        print(f"Error applying parameters to {framework} - {algo_name}: {e}")
                        continue
                    # Validate custom hyperparameters
                    valid_params = model.get_params().keys()
                    filtered_params = {k: v for k, v in final_params.items() if k in valid_params}
                    print(f"Filtered parameters for {framework} - {algo_name}: {filtered_params}")

                    # Apply only valid parameters
                    model.set_params(**filtered_params)
                    # Update the existing model parameters
                    model.set_params(**final_params)

                    # Initialize CodeCarbon tracker

                    tracker=OfflineEmissionsTracker(country_iso_code="EST",log_level="critical",allow_multiple_runs=True)
                    tracker.start()

                    tracker.start_task()

                    try:
                        accuracy, f1 = evaluate_model(model, X, y, time_budget)
                        #Model Evaluation is done here
                        # Compute feature importance using helper function
                        feature_importance = calculate_feature_importance(model, X, y)

                        # Stop tracker and get CO2 emissions
                        emissions = tracker.stop_task()

                        # Convert emissions from kg to micro-units for visualization consistency
                        co2_emissions_micro = emissions.emissions * 1_000_000
                        energy_micro_wh = emissions.energy_consumed * 1_000_000
                        # Convert energy to micro-cents (€0.20/kWh)
                        cost_micro_cents = energy_micro_wh * 0.0001  # since €0.20/kWh = 0.0001 € per µWh = 0.01 cents

                        results[f"{framework}_{algo_name}"] = {
                            "Accuracy": accuracy,
                            "F1 Score": f1,
                            "CO2 Emission": co2_emissions_micro,
                            "Energy Consumption": energy_micro_wh ,  # ✅ New field
                            "cost_micro_cents": cost_micro_cents,
                            "feature_importance": {k: float(v) for k, v in feature_importance.items()} if isinstance(feature_importance, dict) else None,
                            "hyperparameters": final_params  # for hovering over parameters

                        }
                    except Exception as e:
                        tracker.stop()  # Ensure tracker stops even if there's an error
                        # Log errors for debugging
                        print(f"Error with {framework}_{algo_name}: {e}")
                        results[f"{framework}_{algo_name}"] = {"error": str(e)}
        # Check if results are empty
        if not results:
            print("No results generated. Check input data or algorithm configurations.")
            return jsonify({"status": "error", "message": "No results generated. Check input data or algorithm configurations."})

        print("Results generated:", results)  # Log results
        return jsonify({"status": "success", "results": results})

    except Exception as e:
        print("Error in run_automl:", str(e))
        return jsonify({"status": "error", "message": str(e)})




def evaluate_model(model, X, y, time_budget):
    import time
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    # Split the dataset once
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    start_time = time.time()

    # Check if model supports incremental training via partial_fit
    if hasattr(model, "partial_fit"):
        classes = np.unique(y_train)
        iteration = 0
        while (time.time() - start_time) < time_budget:
            if iteration == 0:
                # For the first iteration, include the classes parameter
                model.partial_fit(X_train, y_train, classes=classes)
            else:
                model.partial_fit(X_train, y_train)
            iteration += 1
            # Optional: Log iteration info for debugging
            print(f"partial_fit iteration {iteration} -- elapsed time: {time.time() - start_time:.2f}s")
    # If the model supports warm_start (e.g., ensemble models)
    elif hasattr(model, "warm_start") and model.get_params().get("warm_start", False):
        # For iterative training, we can gradually increase n_estimators.
        # Use a lower starting value (e.g., 10) and then increase by a fixed increment.
        current_estimators = model.get_params().get("n_estimators", 10)
        # If you want a warm-start, it must be enabled when the model is instantiated.
        while (time.time() - start_time) < time_budget:
            current_estimators += 5  # Increase in batches (adjust as needed)
            model.set_params(n_estimators=current_estimators)
            model.fit(X_train, y_train)
            # Optional: Log current n_estimators and elapsed time
            print(f"warm_start fit with n_estimators = {current_estimators} -- elapsed time: {time.time() - start_time:.2f}s")
    else:
        # For models that do not support incremental updates, do a single fit.
        print("Model does not support incremental training; performing one fit call.")
        model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    return accuracy, f1

if __name__ == '__main__':
    app.run(debug=True)
