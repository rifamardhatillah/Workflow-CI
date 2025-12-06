import mlflow
import mlflow.sklearn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import argparse

# -----------------------------
# PARSE ARGUMEN DARI MLflow
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=42)
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

# Load dataset
df = pd.read_csv(args.data_path)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)

# AUTLOG (boleh)
mlflow.sklearn.autolog()


model = RandomForestClassifier(
    n_estimators=100,
    random_state=args.random_state
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
