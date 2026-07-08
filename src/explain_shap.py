import os
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def main():
    os.makedirs("artifacts", exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    X = df.drop(columns=["customerID", "Churn"])
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
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    X_test_transformed = model.named_steps["preprocessor"].transform(X_test)

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    X_test_transformed = pd.DataFrame(
        X_test_transformed,
        columns=feature_names
    )

    xgb_model = model.named_steps["classifier"]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test_transformed)

    # Global explanation
    shap.summary_plot(
        shap_values,
        X_test_transformed,
        show=False
    )
    plt.tight_layout()
    plt.savefig("artifacts/shap_summary.png")
    plt.close()

    # Local explanation: client with highest churn probability
    y_proba = model.predict_proba(X_test)[:, 1]
    client_index = y_proba.argmax()

    shap.plots.waterfall(
        shap_values[client_index],
        show=False
    )
    plt.tight_layout()
    plt.savefig("artifacts/shap_waterfall.png")
    plt.close()

    print("SHAP plots saved in artifacts/")


if __name__ == "__main__":
    main()