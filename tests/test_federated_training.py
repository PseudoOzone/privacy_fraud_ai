"""Unit tests for federated_training.py"""

import unittest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))
from federated_training import FederatedFraudDetector
from dataset_generator import SyntheticBankDataGenerator


class TestFederatedTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create temporary test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.data_dir = Path(cls.temp_dir) / "data"
        cls.models_dir = Path(cls.temp_dir) / "models"

        cls.data_dir.mkdir(exist_ok=True)
        cls.models_dir.mkdir(exist_ok=True)

        # Generate test data
        generator = SyntheticBankDataGenerator(seed=42, n_samples_per_bank=500)
        generator.data_dir = cls.data_dir
        generator.save_datasets()

    @classmethod
    def tearDownClass(cls):
        """Clean up temp dir."""
        shutil.rmtree(cls.temp_dir)

    def setUp(self):
        self.federated = FederatedFraudDetector(
            data_dir=str(self.data_dir),
            models_dir=str(self.models_dir),
            noise_sigma=1.0,
        )

    def test_remove_pii(self):
        """Test PII removal."""
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob"],
                "amount": [100, 200],
                "email": ["a@b.com", "c@d.com"],
            }
        )
        df_clean = self.federated.remove_pii(df)

        # Check PII cols removed
        self.assertNotIn("name", df_clean.columns)
        self.assertNotIn("email", df_clean.columns)

        # Check safe cols remain
        self.assertIn("amount", df_clean.columns)

    def test_encode_categorical(self):
        """Test categorical encoding."""
        df = pd.DataFrame(
            {
                "merchant_category": ["grocery", "gas", "grocery"],
                "amount": [100, 200, 300],
            }
        )

        df_encoded = self.federated.encode_categorical(df, fit=True)

        # Check encoded
        self.assertTrue(pd.api.types.is_numeric_dtype(df_encoded["merchant_category"]))

    def test_train_local_model(self):
        """Test local model training."""
        df = pd.read_csv(self.data_dir / "bankA.csv")
        model, features = self.federated.train_local_model(df, "bankA")

        # Check model exists and has predict
        self.assertTrue(hasattr(model, "predict"))

        # Check features list
        self.assertGreater(len(features), 0)

    def test_differential_privacy_wrapper(self):
        """Test DP noise injection."""
        from federated_training import PrivacyWrapper

        wrapper = PrivacyWrapper(noise_sigma=1.0)
        predictions = [0.2, 0.5, 0.8]

        noisy = wrapper.add_noise(predictions)

        # Check output is clipped [0, 1]
        self.assertTrue(all(0 <= p <= 1 for p in noisy))

        # Check noise was applied (shouldn't be identical)
        self.assertNotEqual(list(noisy), predictions)


if __name__ == "__main__":
    unittest.main()
