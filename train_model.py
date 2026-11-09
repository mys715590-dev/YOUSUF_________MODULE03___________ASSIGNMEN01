"""
ANN Salary Regression
Train an artificial neural network to predict EstimatedSalary from Churn_Modelling.csv.
This is a classroom exercise, not a real banking or salary-estimation product.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Churn_Modelling.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print("\nDATA OVERVIEW")
    print("Shape:", df.shape)
    print(df.head())
    print("\nMissing values:\n", df.isna().sum())
    print("\nData types:\n", df.dtypes)
    print("\nTarget summary:\n", df["EstimatedSalary"].describe())
    return df


def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols)
    ], remainder="drop")

    return preprocessor, numeric_cols, categorical_cols


def main():
    tf.keras.utils.set_random_seed(SEED)
    ARTIFACT_DIR.mkdir(exist_ok=True)

    df = load_data()

    # RowNumber and CustomerId are identifiers. Surname is a name, not a useful
    # numeric signal. EstimatedSalary is the value we are trying to predict.
    drop_cols = ["RowNumber", "CustomerId", "Surname", "EstimatedSalary"]
    X = df.drop(columns=drop_cols)
    y = df["EstimatedSalary"].astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )

    # Fit preprocessing only on the training partition to avoid data leakage.
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)
    X_train_ready = preprocessor.fit_transform(X_train).astype("float32")
    X_test_ready = preprocessor.transform(X_test).astype("float32")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_ready.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=12, restore_best_weights=True
    )
    history = model.fit(
        X_train_ready, y_train,
        validation_split=0.20,
        epochs=150,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    predictions = model.predict(X_test_ready, verbose=0).reshape(-1)
    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))

    print("\nTEST SET REGRESSION RESULTS")
    print(f"MAE : {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²  : {r2:.4f}")

    # Save the exact fitted preprocessing object used during training.
    joblib.dump(preprocessor, ARTIFACT_DIR / "preprocessor.joblib")
    model.save(ARTIFACT_DIR / "salary_ann.keras")

    metadata = {
        "feature_columns": X.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
        "target": "EstimatedSalary",
        "random_state": SEED
    }
    with open(ARTIFACT_DIR / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    # Reload check: confirms the saved model and preprocessor can be used.
    loaded_preprocessor = joblib.load(ARTIFACT_DIR / "preprocessor.joblib")
    loaded_model = tf.keras.models.load_model(ARTIFACT_DIR / "salary_ann.keras")
    sample = loaded_preprocessor.transform(X_test.iloc[:3]).astype("float32")
    print("\nReload check predictions:", loaded_model.predict(sample, verbose=0).reshape(-1))
    print("\nArtifacts saved in:", ARTIFACT_DIR)


if __name__ == "__main__":
    main()