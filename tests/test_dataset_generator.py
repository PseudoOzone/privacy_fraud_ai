"""Unit tests for dataset_generator.py"""

import unittest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))
from dataset_generator import SyntheticBankDataGenerator


class TestDatasetGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = SyntheticBankDataGenerator(seed=42, n_samples_per_bank=1000)

    def test_transaction_generation_bankA(self):
        """Test synthetic Bank A data generation."""
        df = self.generator.generate_transaction_data("bankA", fraud_rate=0.03)

        # Check shape
        self.assertEqual(len(df), 1000)

        # Check fraud rate
        fraud_count = df["is_fraud"].sum()
        expected_fraud = int(1000 * 0.03)
        self.assertAlmostEqual(fraud_count, expected_fraud, delta=10)

        # Check required columns
        required_cols = [
            "transaction_id",
            "amount",
            "merchant_category",
            "device_type",
            "location_risk",
            "is_fraud",
        ]
        for col in required_cols:
            self.assertIn(col, df.columns)

        # Check no NaNs
        self.assertEqual(df.isnull().sum().sum(), 0)

    def test_transaction_generation_bankB(self):
        """Test synthetic Bank B data generation (higher fraud)."""
        df = self.generator.generate_transaction_data("bankB", fraud_rate=0.07)

        # Check fraud rate is higher
        fraud_rate = df["is_fraud"].mean()
        self.assertGreater(fraud_rate, 0.05)

    def test_fraud_patterns_generation(self):
        """Test fraud pattern descriptions."""
        patterns = self.generator.generate_fraud_patterns()

        # Check we have patterns
        self.assertGreater(len(patterns), 0)

        # Check no PII in patterns
        for pattern in patterns:
            self.assertNotIn("@", pattern)  # No emails
            self.assertNotIn("555-", pattern)  # No phone patterns

    def test_save_datasets(self):
        """Test saving all datasets."""
        import tempfile
        import shutil

        # Create temp dir
        temp_dir = tempfile.mkdtemp()
        try:
            self.generator.data_dir = Path(temp_dir)
            df_a, df_b, df_merged = self.generator.save_datasets()

            # Check files exist
            self.assertTrue((Path(temp_dir) / "bankA.csv").exists())
            self.assertTrue((Path(temp_dir) / "bankB.csv").exists())
            self.assertTrue((Path(temp_dir) / "full_merged.csv").exists())
            self.assertTrue((Path(temp_dir) / "fraud_patterns.txt").exists())

            # Check merged is union
            self.assertEqual(len(df_merged), len(df_a) + len(df_b))

        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
