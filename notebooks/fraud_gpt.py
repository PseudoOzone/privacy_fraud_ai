"""
Fraud-GPT inference module.

Generates privacy-safe fraud summaries and synthetic attack scenarios.
"""

import pickle
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class FraudGPT:
    """Load fine-tuned fraud model and generate descriptions."""

    def __init__(self, base_model="gpt2", models_dir="../models"):
        self.base_model_name = base_model
        self.models_dir = Path(models_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[FraudGPT] Loading on device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model).to(
            self.device
        )

        # Try to load fine-tuned LoRA weights
        try:
            lora_path = self.models_dir / "fraud_gpt.pt"
            if lora_path.exists():
                print(f"[FraudGPT] Loading LoRA weights from {lora_path}")
                self.model = PeftModel.from_pretrained(self.base_model, str(lora_path))
            else:
                print("[FraudGPT] Using base model (no fine-tuned weights found)")
                self.model = self.base_model
        except Exception as e:
            print(f"[WARNING] Could not load LoRA weights: {e}. Using base model.")
            self.model = self.base_model

    def _sanitize_output(self, text):
        """Remove any PII-like patterns from generated text."""
        # Remove email patterns
        import re

        text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL]", text)
        # Remove phone patterns
        text = re.sub(r"\+?1?\d{9,15}", "[PHONE]", text)
        # Remove account numbers
        text = re.sub(r"\b\d{16}\b", "[ACCOUNT]", text)

        return text

    def generate_fraud_summary(self, feature_stats):
        """
        Generate fraud analysis summary from transaction features.

        Args:
            feature_stats (dict): Aggregated transaction statistics
                e.g., {"avg_amount": 150.5, "fraud_count": 45, "velocity": 12}

        Returns:
            str: Fraud summary without PII
        """
        # Extract stats with defaults
        avg_amount = feature_stats.get("avg_amount", 150.0)
        fraud_count = feature_stats.get("fraud_count", 50)
        velocity = feature_stats.get("velocity", 10)
        risk_devices = feature_stats.get("risk_devices", "mobile")
        fraud_rate = feature_stats.get("fraud_rate", 5.0)
        
        prompt = f"""As a fraud analyst, analyze the following fraud incident data and provide detailed insights:

FRAUD INCIDENT DATA:
- Transaction Volume: {fraud_count} fraud cases detected
- Average Fraudulent Amount: ${avg_amount:.2f}
- Fraud Rate: {fraud_rate}%
- Transaction Velocity: {velocity} transactions per hour
- Primary Device Types: {risk_devices}
- Detection Method: Federated Learning Model

ANALYSIS REQUIRED:
1. Fraud Pattern: Describe the specific fraud pattern observed in these transactions
2. Risk Assessment: Explain the financial and operational risk level
3. Root Cause: What underlying behavior indicates this is fraud?
4. Mitigation: What specific actions should be taken to prevent similar incidents?

Be specific about transaction characteristics, amounts, timing, and behavioral anomalies that indicate fraud. Provide actionable recommendations."""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.75,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.2
            )

        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = self._sanitize_output(summary)

        return summary

    def generate_attack_simulation(self, n=5):
        """
        Generate synthetic fraud attack descriptions.

        Args:
            n (int): Number of attack scenarios to generate

        Returns:
            list[str]: List of attack descriptions without PII
        """
        attack_scenarios = [
            {
                "name": "Rapid Card Testing",
                "desc": "Multiple small transactions ($1-$10) in sequence within minutes to validate stolen card"
            },
            {
                "name": "Geolocation Anomaly",
                "desc": "Transactions originating from geographically impossible locations (NYC to LA in 30 minutes)"
            },
            {
                "name": "Account Takeover via Phishing",
                "desc": "Attacker gains login credentials, performs bulk transfers with velocity spike to new beneficiaries"
            },
            {
                "name": "Synthetic Identity Fraud",
                "desc": "Multiple accounts with similar characteristics opened simultaneously, low transaction history, high-risk merchants"
            },
            {
                "name": "Velocity-Based Fraud Ring",
                "desc": "Coordinated transactions from multiple accounts to single beneficiary within short timeframe"
            },
            {
                "name": "Mobile Device Compromise",
                "desc": "Legitimate account with sudden device type change, different location, unusual merchant categories"
            },
            {
                "name": "Money Mule Network",
                "desc": "Deposits followed by immediate withdrawals through multiple intermediate accounts"
            },
            {
                "name": "Collusion Fraud",
                "desc": "Insider + external party: employee processes fraudulent refunds to unknown accounts"
            },
        ]

        selected = attack_scenarios[:n]
        
        prompt = f"""As a fraud prevention expert, simulate and describe {n} realistic fraud attack scenarios that could evade standard detection:

ATTACK SCENARIOS TO DETAIL:

"""
        for i, scenario in enumerate(selected, 1):
            prompt += f"{i}. {scenario['name']}: {scenario['desc']}\n"

        prompt += f"""
For each scenario, provide:
- Attack Flow: Step-by-step execution of the attack
- Detection Challenge: Why this attack is hard to detect
- Red Flags: Specific indicators to watch for
- Defense Strategy: Technical/operational controls to prevent

Be specific with amounts, timing, device types, merchant categories, and behavioral patterns. 
Focus on how attackers evade traditional fraud rules."""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.3
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        full_text = self._sanitize_output(full_text)

        # Split into individual attacks
        attacks = []
        lines = full_text.split("\n")
        current_attack = []
        
        for line in lines:
            if any(char.isdigit() and "." in line[:3] for char in line[:3]):
                if current_attack:
                    attacks.append("\n".join(current_attack).strip())
                current_attack = [line]
            else:
                if current_attack:
                    current_attack.append(line)
        
        if current_attack:
            attacks.append("\n".join(current_attack).strip())

        return [a.strip() for a in attacks if a.strip()][:n]

        return attacks if attacks else ["Generic fraud attack - contact fraud team"]


if __name__ == "__main__":
    fraud_gpt = FraudGPT(base_model="gpt2")

    # Test summary generation
    test_stats = {
        "avg_amount": 250.50,
        "fraud_count": 45,
        "velocity": 12,
        "risk_devices": "VPN, Mobile",
    }

    print("\n[TEST] Fraud Summary:")
    print(fraud_gpt.generate_fraud_summary(test_stats))

    print("\n[TEST] Attack Simulations:")
    for i, attack in enumerate(fraud_gpt.generate_attack_simulation(n=3), 1):
        print(f"\n{i}. {attack[:100]}...")

