import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.xgboost
from sklearn.metrics import confusion_matrix
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay


data_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

def main():

    n_estimators = 100
    learning_rate = 0.1
    max_depth = 3
    subsample = 0.9
    colsample_bytree = 0.9

    df = pd.read_csv(data_path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    X = df.drop(columns=["customerID", "Churn"]) # Avoid data leakage
    y = df["Churn"]

    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        eval_metric="logloss",
                        random_state=42
                    ))
    ])

    mlflow.set_experiment("customer_churn_project")

    with mlflow.start_run(run_name="xgboost_baseline_best_threshold"):

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        fp_cost = 20
        fn_cost = 60

        cost_results = []
        thresholds = np.arange(0.1, 0.9, 0.05)

        for threshold in thresholds:
            y_pred_threshold = (y_proba >= threshold).astype(int)

            accuracy = accuracy_score(y_test, y_pred_threshold)
            precision = precision_score(y_test, y_pred_threshold)
            recall = recall_score(y_test, y_pred_threshold)
            f1 = f1_score(y_test, y_pred_threshold)
            roc_auc = roc_auc_score(y_test, y_proba)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()

            total_cost = fp * fp_cost + fn * fn_cost

            cost_results.append({
                "threshold": threshold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "false_positives": fp,
                "false_negatives": fn,
                "total_cost": total_cost
            })

        cost_df = pd.DataFrame(cost_results)

        print(cost_df.sort_values("total_cost").head(10))

        best_row = cost_df.sort_values("total_cost").iloc[0]

        best_threshold = best_row["threshold"]
        y_pred_best = (y_proba >= best_threshold).astype(int)

        disp = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred_best,
            cmap="Blues"
        )

        plt.title(f"Confusion Matrix (threshold={best_threshold:.2f})")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")

        mlflow.log_artifact("confusion_matrix.png")

        plt.close()

        RocCurveDisplay.from_predictions(y_test, y_proba)

        plt.title("ROC Curve")
        plt.tight_layout()
        plt.savefig("roc_curve.png")

        mlflow.log_artifact("roc_curve.png")

        plt.close()

        PrecisionRecallDisplay.from_predictions(
            y_test,
            y_proba
        )

        plt.title("Precision-Recall Curve")
        plt.tight_layout()
        plt.savefig("precision_recall_curve.png")

        mlflow.log_artifact("precision_recall_curve.png")

        plt.close()

        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("subsample", subsample)
        mlflow.log_param("colsample_bytree", colsample_bytree)
        mlflow.log_param("fp_cost", fp_cost)
        mlflow.log_param("fn_cost", fn_cost)
        mlflow.log_param("threshold", best_row["threshold"])

        mlflow.log_metric("accuracy", best_row["accuracy"])
        mlflow.log_metric("precision", best_row["precision"])
        mlflow.log_metric("recall", best_row["recall"])
        mlflow.log_metric("f1_score", best_row["f1_score"])
        mlflow.log_metric("roc_auc", best_row["roc_auc"])

        print("Best business threshold")
        print("-----------------------")
        print(f"Threshold : {best_row['threshold']:.2f}")
        print(f"Min cost  : {best_row['total_cost']:.0f}")

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=[
                "numpy.dtype",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )


    print("XGBoost - best business threshold")
    print("---------------------------------")
    print(f"Threshold : {best_row['threshold']:.2f}")
    print(f"Accuracy  : {best_row['accuracy']:.4f}")
    print(f"Precision : {best_row['precision']:.4f}")
    print(f"Recall    : {best_row['recall']:.4f}")
    print(f"F1-score  : {best_row['f1_score']:.4f}")
    print(f"Min cost  : {best_row['total_cost']:.0f}")


if __name__ == "__main__":
    main()
