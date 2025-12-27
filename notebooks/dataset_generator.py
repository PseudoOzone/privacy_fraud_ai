"""
Privacy-safe synthetic dataset generator for federated fraud detection.

Generates:
- bankA.csv, bankB.csv: transactional data (PII-free)
- fraud_patterns.txt: template fraud descriptions for Fraud-GPT
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


class SyntheticBankDataGenerator:
    """Generate privacy-safe synthetic banking datasets."""

    def __init__(self, seed=42, n_samples_per_bank=10000):
        np.random.seed(seed)
        self.n_samples = n_samples_per_bank
        self.data_dir = Path(__file__).parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)

    def generate_transaction_data(self, bank_name: str, fraud_rate: float = 0.05):
        """
        Generate synthetic transaction data with minimal PII.

        Args:
            bank_name: "bankA" or "bankB"
            fraud_rate: proportion of fraudulent transactions

        Returns:
            DataFrame with columns: transaction_id, amount, merchant_category,
                                   timestamp, device_type, location_risk, is_fraud
        """
        n_fraud = int(self.n_samples * fraud_rate)
        n_legit = self.n_samples - n_fraud

        # Legitimate transactions
        legit_data = {
            "transaction_id": np.arange(n_legit),
            "amount": np.random.lognormal(mean=4.5, sigma=1.5, size=n_legit),
            "merchant_category": np.random.choice(
                ["grocery", "gas", "dining", "retail", "online", "utility"],
                size=n_legit,
            ),
            "timestamp": [
                datetime.now() - timedelta(days=int(d))
                for d in np.random.randint(0, 365, n_legit)
            ],
            "device_type": np.random.choice(["mobile", "web", "atm"], size=n_legit),
            "location_risk": np.random.uniform(0, 0.3, n_legit),
            "is_fraud": 0,
        }

        # Fraudulent transactions (higher amounts, unusual patterns)
        fraud_data = {
            "transaction_id": np.arange(n_legit, self.n_samples),
            "amount": np.random.lognormal(mean=6.5, sigma=1.8, size=n_fraud),
            "merchant_category": np.random.choice(
                ["online", "cash_advance", "wire_transfer"], size=n_fraud
            ),
            "timestamp": [
                datetime.now() - timedelta(days=int(d))
                for d in np.random.randint(0, 30, n_fraud)
            ],
            "device_type": np.random.choice(["vpn", "proxy", "unknown"], size=n_fraud),
            "location_risk": np.random.uniform(0.7, 1.0, n_fraud),
            "is_fraud": 1,
        }

        # Combine and shuffle
        df = pd.concat(
            [pd.DataFrame(legit_data), pd.DataFrame(fraud_data)], ignore_index=True
        )
        df = df.sample(frac=1).reset_index(drop=True)

        # Amount to integer cents
        df["amount"] = (df["amount"] * 100).astype(int)

        return df

    def generate_fraud_patterns(self):
        """Generate fraud pattern descriptions (no PII)."""
        patterns = [
            "Rapid sequence of small transactions followed by large withdrawal",
            "Transaction from unusual geographic location within 2 hours",
            "Card used for category switch: fuel to jewelry in 30 minutes",
            "Multiple failed authentication attempts then successful login",
            "Account balance check followed by maximum withdrawal",
            "Device fingerprint mismatch with historical pattern",
            "Transaction during known offline hours for merchant",
            "Velocity spike: 10 transactions in 5 minutes",
            "Peer-to-peer transfer to new recipient then immediate cash-out",
            "Transaction amount exactly matches available credit limit",
            "Unusual merchant category for account behavior",
            "Login from Tor exit node IP address",
            "Cross-border transaction without prior travel history",
            "ATM withdrawal at high-fraud-risk location",
            "Velocity anomaly: $5k transferred in one hour",
        ]

        return patterns * (self.n_samples // len(patterns) + 1)

    def save_datasets(self):
        """Generate and save all datasets."""
        print(f"[INFO] Generating synthetic datasets in {self.data_dir}...")

        # Bank A
        df_a = self.generate_transaction_data("bankA", fraud_rate=0.03)
        df_a.to_csv(self.data_dir / "bankA.csv", index=False)
        print(f"[OK] bankA.csv: {len(df_a)} transactions ({df_a['is_fraud'].sum()} fraud)")

        # Bank B
        df_b = self.generate_transaction_data("bankB", fraud_rate=0.07)
        df_b.to_csv(self.data_dir / "bankB.csv", index=False)
        print(f"[OK] bankB.csv: {len(df_b)} transactions ({df_b['is_fraud'].sum()} fraud)")

        # Merged
        df_merged = pd.concat([df_a, df_b], ignore_index=True)
        df_merged.to_csv(self.data_dir / "full_merged.csv", index=False)
        print(f"[OK] full_merged.csv: {len(df_merged)} total transactions")

        # Fraud patterns for GPT training
        patterns = self.generate_fraud_patterns()
        with open(self.data_dir / "fraud_patterns.txt", "w") as f:
            for pattern in patterns:
                f.write(pattern + "\n")
        print(f"[OK] fraud_patterns.txt: {len(patterns)} pattern descriptions")

        return df_a, df_b, df_merged


if __name__ == "__main__":
    generator = SyntheticBankDataGenerator(n_samples_per_bank=10000)
    generator.save_datasets()
    print("\n[SUCCESS] All datasets generated and saved.")
