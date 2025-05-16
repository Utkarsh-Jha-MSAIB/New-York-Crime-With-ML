import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm



def train_ann_with_outputs(df, feature_list, target_col, max_iter_val, batch_size_val, solver_val):
    """
    Trains an ANN with specified hyperparameters and returns Accuracy, Confusion Matrix, and Feature Importances.
    """
    inputFields = df[feature_list].values
    enc = OrdinalEncoder()
    enc.fit(inputFields)
    encodedInput = enc.transform(inputFields)

    target = df[target_col]
    xTrain, xTest, yTrain, yTest = train_test_split(encodedInput, target, test_size=0.3, random_state=42)

    ann = MLPClassifier(
        max_iter=max_iter_val,
        batch_size=batch_size_val,
        activation='relu',
        solver=solver_val,
        random_state=42,
        learning_rate='adaptive',
    )
    ann.fit(xTrain, yTrain)
    predictions_ann = ann.predict(xTest)

    accuracy = ann.score(xTest, yTest)
    conf_matrix = confusion_matrix(yTest, predictions_ann)

    return {
        'Accuracy': accuracy,
        'Confusion_Matrix': conf_matrix,
        'Feature_Importance': ann.coefs_[0] if hasattr(ann, 'coefs_') else None,
        'Model_Params': {
            'max_iter': max_iter_val,
            'batch_size': batch_size_val,
            'solver': solver_val
        },
        'feature_list': feature_list
    }


def run_ann_gridsearch_and_save_outputs(
    df,
    features,
    target_col,
    output_dir="output",
    max_iters=[1000, 2000, 3000],
    batch_sizes=[64, 128, 256],
    solvers=['adam', 'lbfgs', 'sgd']
):
    """
    Trains ANN models with various hyperparameter combinations and stores results in 'output/ann'.
    """
    output_path = os.path.join(output_dir, "ann")
    os.makedirs(output_path, exist_ok=True)

    results = []
    models_outputs = []

    param_combinations = [(mi, bs, sv) for mi in max_iters for bs in batch_sizes for sv in solvers]

    for max_iter_val, batch_size_val, solver_val in tqdm(param_combinations, desc="Training ANN Models"):
        output = train_ann_with_outputs(df, features, target_col, max_iter_val, batch_size_val, solver_val)
        models_outputs.append(output)
        results.append({
            'max_iter': max_iter_val,
            'batch_size': batch_size_val,
            'solver': solver_val,
            'accuracy': output['Accuracy']
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_path, "ann_grid_results.csv"), index=False)

    best_model_idx = results_df['accuracy'].idxmax()
    best_model_output = models_outputs[best_model_idx]

    # Confusion Matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=best_model_output['Confusion_Matrix'],
        display_labels=['Safe but Vulnerable (0)', 'Unsafe (1)']
    )
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Best ANN Model')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "confusion_matrix.png"))
    plt.close()

    # Feature Importance
    try:
        importance = np.mean(np.abs(best_model_output['Feature_Importance']), axis=1)
        feature_importance = pd.Series(importance, index=best_model_output['feature_list']).sort_values(ascending=False)

        plt.figure(figsize=(10, 8))
        bars = plt.barh(feature_importance.index, feature_importance.values)
        plt.xlabel('Approximate Importance')
        plt.title(f"Feature Importance - Best ANN Model ({best_model_output['Model_Params']})")

        for bar in bars:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                     f'{bar.get_width():.3f}', va='center', ha='left')

        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "feature_importance.png"))
        plt.close()

    except Exception as e:
        print("Feature Importance could not be computed:", e)

    return results_df, best_model_output


def train_rf_with_outputs(df, feature_list, target_col, n_estimators_val, max_depth_val, min_samples_split_val):
    """
    Trains a Random Forest with specified hyperparameters and returns Accuracy, Confusion Matrix, and Feature Importances.
    """
    inputFields = df[feature_list].values
    enc = OrdinalEncoder()
    enc.fit(inputFields)
    encodedInput = enc.transform(inputFields)

    target = df[target_col]
    xTrain, xTest, yTrain, yTest = train_test_split(encodedInput, target, test_size=0.3, random_state=42)

    rft = RandomForestClassifier(
        n_estimators=n_estimators_val,
        max_depth=max_depth_val,
        min_samples_split=min_samples_split_val,
        bootstrap=True,
        random_state=42
    )

    rft.fit(xTrain, yTrain)
    predictions_rf = rft.predict(xTest)
    accuracy = rft.score(xTest, yTest)
    conf_matrix = confusion_matrix(yTest, predictions_rf)

    return {
        'Accuracy': accuracy,
        'Confusion_Matrix': conf_matrix,
        'Feature_Importance': rft.feature_importances_,
        'Model_Params': {
            'n_estimators': n_estimators_val,
            'max_depth': max_depth_val,
            'min_samples_split': min_samples_split_val
        },
        'feature_list': feature_list
    }


def run_rf_gridsearch_and_save_outputs(
    df,
    features,
    target_col,
    output_dir="output",
    n_estimators_list=[100, 200, 300],
    max_depth_list=[10, 15, 20],
    min_samples_split_list=[2, 5, 10]
):
    """
    Trains Random Forest models and saves outputs in 'output/rf'.
    """
    output_path = os.path.join(output_dir, "rf")
    os.makedirs(output_path, exist_ok=True)

    results = []
    models_outputs = []

    param_combinations = [
        (n_val, d_val, s_val)
        for n_val in n_estimators_list
        for d_val in max_depth_list
        for s_val in min_samples_split_list
    ]

    for n_val, depth_val, split_val in tqdm(param_combinations, desc="Training Random Forest Models"):
        output = train_rf_with_outputs(df, features, target_col, n_val, depth_val, split_val)
        models_outputs.append(output)
        results.append({
            'n_estimators': n_val,
            'max_depth': depth_val,
            'min_samples_split': split_val,
            'accuracy': output['Accuracy']
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_path, "rf_grid_results.csv"), index=False)

    best_model_idx = results_df['accuracy'].idxmax()
    best_model_output = models_outputs[best_model_idx]

    # Confusion Matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=best_model_output['Confusion_Matrix'],
        display_labels=['Safe but Vulnerable(0)', 'Unsafe (1)']
    )
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Best Random Forest Model')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "confusion_matrix.png"))
    plt.close()

    # Feature Importance
    try:
        importance = pd.Series(
            best_model_output['Feature_Importance'],
            index=best_model_output['feature_list']
        ).sort_values(ascending=False)

        plt.figure(figsize=(10, 8))
        bars = plt.barh(importance.index, importance.values)
        plt.xlabel('Importance')
        plt.title(f"Feature Importance - Best Random Forest Model ({best_model_output['Model_Params']})")

        for bar in bars:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                     f'{bar.get_width():.3f}', va='center', ha='left')

        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "feature_importance.png"))
        plt.close()

    except Exception as e:
        print("Feature importance could not be computed:", e)

    return results_df, best_model_output

