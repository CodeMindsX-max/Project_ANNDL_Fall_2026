"""Simple ML pipeline example: generates synthetic data, builds a scikit-learn
pipeline (StandardScaler + LogisticRegression), trains, evaluates, and saves the model.
"""
from pathlib import Path
import joblib
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def build_pipeline() -> Pipeline:
	return Pipeline([
		("scaler", StandardScaler()),
		("clf", LogisticRegression(solver="liblinear")),
	])


def main(output_path: str = "model.joblib"):
	# generate a small synthetic dataset
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, n_redundant=2, random_state=42)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	pipe = build_pipeline()
	pipe.fit(X_train, y_train)

	preds = pipe.predict(X_test)
	acc = accuracy_score(y_test, preds)
	print(f"Test accuracy: {acc:.4f}")
	print(classification_report(y_test, preds))

	# save model
	out = Path(output_path)
	joblib.dump(pipe, out)
	print(f"Saved model to {out.resolve()}")


if __name__ == "__main__":
	main()

