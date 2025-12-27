"""Unit tests for fraud_gpt.py"""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))


class TestFraudGPT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize Fraud-GPT model (slow)."""
        try:
            from fraud_gpt import FraudGPT

            cls.fraud_gpt = FraudGPT(base_model="gpt2")
            cls.skip_gpu_tests = False
        except Exception as e:
            print(f"Skipping GPU tests: {e}")
            cls.fraud_gpt = None
            cls.skip_gpu_tests = True

    def test_sanitize_output(self):
        """Test PII removal from generated text."""
        if self.skip_gpu_tests:
            self.skipTest("Model not available")

        # Test email removal
        text = "Contact admin@example.com for details"
        clean = self.fraud_gpt._sanitize_output(text)
        self.assertNotIn("admin@", clean)
        self.assertIn("[EMAIL]", clean)

        # Test phone removal
        text = "Call +1 5551234567 now"
        clean = self.fraud_gpt._sanitize_output(text)
        self.assertNotIn("555", clean)

    def test_generate_fraud_summary_format(self):
        """Test fraud summary generation format."""
        if self.skip_gpu_tests:
            self.skipTest("Model not available")

        stats = {
            "avg_amount": 150.0,
            "fraud_count": 45,
            "velocity": 12,
            "risk_devices": "VPN, Mobile",
        }

        summary = self.fraud_gpt.generate_fraud_summary(stats)

        # Check output is string
        self.assertIsInstance(summary, str)

        # Check non-empty
        self.assertGreater(len(summary), 0)

        # Check no raw PII patterns
        self.assertNotIn("@", summary)  # No email

    def test_generate_attack_simulation_returns_list(self):
        """Test attack simulation returns proper list."""
        if self.skip_gpu_tests:
            self.skipTest("Model not available")

        attacks = self.fraud_gpt.generate_attack_simulation(n=3)

        # Check list of strings
        self.assertIsInstance(attacks, list)
        self.assertGreaterEqual(len(attacks), 1)

        for attack in attacks:
            self.assertIsInstance(attack, str)
            self.assertGreater(len(attack), 0)

    def test_no_pii_in_outputs(self):
        """Test all outputs are PII-free."""
        if self.skip_gpu_tests:
            self.skipTest("Model not available")

        # Generate multiple summaries
        for i in range(2):
            stats = {
                "avg_amount": 200.0 + i * 50,
                "fraud_count": 30 + i * 10,
                "velocity": 10 + i,
                "risk_devices": "Unknown",
            }

            summary = self.fraud_gpt.generate_fraud_summary(stats)

            # No email pattern
            self.assertNotIn("@", summary)

            # No phone pattern (rough check)
            import re

            emails = re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", summary)
            self.assertEqual(len(emails), 0)


if __name__ == "__main__":
    unittest.main()
