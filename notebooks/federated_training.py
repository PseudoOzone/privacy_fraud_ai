"""
Federated learning pipeline with differential privacy.

Trains local models per bank, aggregates globally, and applies DP noise.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class PrivacyWrapper:
    """Adds differential privacy noise to predictions."""

    def __init__(self, noise_sigma=1.0):
        self.noise_sigma = noise_sigma

    def add_noise(self, predictions):
        """Add Laplace noise to maintain ε-differential privacy."""
        noise = np.random.laplace(0, self.noise_sigma, len(predictions))
        return np.clip(predictions + noise, 0, 1)


class DPRandomForest:
    """Wraps RandomForest with differential privacy noise injection."""

    def __init__(self, base_model, noise_sigma=1.0):
        self.base_model = base_model
        self.privacy_wrapper = PrivacyWrapper(noise_sigma)

    def predict_proba(self, X):
        """Predict probability with DP noise added to positive class."""
        proba = self.base_model.predict_proba(X)
        noisy = proba.copy()
        noisy[:, 1] = self.privacy_wrapper.add_noise(proba[:, 1])
        return np.clip(noisy, 0, 1)

    def predict(self, X):
        """Predict class with DP-noised probabilities."""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


class FederatedFraudDetector:
    """Train separate models per bank, federate, and apply DP."""

    def __init__(self, data_dir="../data", models_dir="../models", noise_sigma=1.0):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.noise_sigma = noise_sigma
        self.encoders = {}
        self.global_model = None
        self.dp_model = None

    def remove_pii(self, df):
        """Drop PII, keep only safe features."""
        # Drop if exists: name, email, phone, account_number, address
        pii_cols = ["name", "email", "phone", "account_number", "address"]
        df = df.drop(columns=[c for c in pii_cols if c in df.columns], errors="ignore")
        return df

    def encode_categorical(self, df, fit=False):
        """Encode categorical features."""
        categorical = ["merchant_category", "device_type"]

        for col in categorical:
            if col not in df.columns:
                continue

            if fit:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.encoders:
                    df[col] = self.encoders[col].transform(df[col].astype(str))

        return df

    def train_local_model(self, df, bank_name):
        """Train RandomForest on single bank's data."""
        print(f"\n[{bank_name}] Training local model...")

        df = self.remove_pii(df)
        df = self.encode_categorical(df, fit=True)

        # Prepare features
        X = df[
            [c for c in df.columns if c not in ["is_fraud", "timestamp", "transaction_id"]]
        ]
        y = df["is_fraud"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

        return model, X.columns.tolist()

    def aggregate_models(self, models_dict):
        """Aggregate via union of training data."""
        print("\n[FEDERATION] Aggregating models...")

        # Load all training data
        all_dataframes = []
        for bank_name in ["bankA", "bankB"]:
            csv_path = self.data_dir / f"{bank_name}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                all_dataframes.append(df)

        if not all_dataframes:
            print("  No data found to aggregate!")
            return

        df_aggregate = pd.concat(all_dataframes, ignore_index=True)
        df_aggregate = self.remove_pii(df_aggregate)
        df_aggregate = self.encode_categorical(df_aggregate, fit=False)

        X = df_aggregate[
            [
                c
                for c in df_aggregate.columns
                if c not in ["is_fraud", "timestamp", "transaction_id"]
            ]
        ]
        y = df_aggregate["is_fraud"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train global model
        global_model = RandomForestClassifier(
            n_estimators=150, max_depth=12, random_state=42, n_jobs=-1
        )
        global_model.fit(X_train, y_train)

        y_pred = global_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  Global Model Accuracy: {acc:.4f}")

        return global_model

    def apply_differential_privacy(self, model):
        """Wrap model with DP noise injection."""
        print(f"\n[DP] Applying differential privacy (sigma={self.noise_sigma})...")
        return DPRandomForest(model, self.noise_sigma)

    def aggregate_models_custom(self, df):
        """Aggregate models from custom dataframe."""
        df = self.remove_pii(df)
        df = self.encode_categorical(df, fit=True)

        X = df[
            [
                c
                for c in df.columns
                if c not in ["is_fraud", "timestamp", "transaction_id"]
            ]
        ]
        y = df.get("is_fraud", [0] * len(df))

        if len(X) == 0:
            return RandomForestClassifier(n_estimators=10, random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

        return model

    def run_pipeline(self):
        """Execute full federated learning pipeline."""
        print("=" * 60)
        print("FEDERATED FRAUD DETECTION PIPELINE")
        print("=" * 60)

        # Load data
        banks = {}
        for bank_name in ["bankA", "bankB"]:
            csv_path = self.data_dir / f"{bank_name}.csv"
            if csv_path.exists():
                banks[bank_name] = pd.read_csv(csv_path)
                print(f"\n[OK] Loaded {bank_name}: {len(banks[bank_name])} rows")

        if not banks:
            print("\n[ERROR] No bank data found. Run dataset_generator.py first.")
            return

        # Train local models
        models = {}
        for bank_name, df in banks.items():
            model, features = self.train_local_model(df, bank_name)
            models[bank_name] = model

        # Aggregate
        self.global_model = self.aggregate_models(models)

        # Save global model
        with open(self.models_dir / "global_model.pkl", "wb") as f:
            pickle.dump(self.global_model, f)
        print(f"\n[OK] Saved: {self.models_dir / 'global_model.pkl'}")

        # Apply DP
        self.dp_model = self.apply_differential_privacy(self.global_model)

        # Save DP model
        with open(self.models_dir / "global_model_dp.pkl", "wb") as f:
            pickle.dump(self.dp_model, f)
        print(f"[OK] Saved: {self.models_dir / 'global_model_dp.pkl'}")

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)

        return self.global_model, self.dp_model


if __name__ == "__main__":
    federated = FederatedFraudDetector()
    federated.run_pipeline()
