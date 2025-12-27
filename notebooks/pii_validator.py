"""
PII (Personally Identifiable Information) Detection & Removal Validator

Detects and removes sensitive data with proof report.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class PIIValidator:
    """Detects and removes PII from datasets with detailed reporting."""
    
    # PII patterns
    PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "phone": r'\b(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
        "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        "ipv4": r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
        "name_patterns": r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # FirstName LastName
    }
    
    # Sensitive column names
    SENSITIVE_COLS = {
        "name", "full_name", "firstname", "lastname", "email", "phone", "phone_number",
        "ssn", "social_security", "account_number", "card_number", "credit_card",
        "address", "street", "city", "zip", "zipcode", "dob", "date_of_birth",
        "password", "pin", "secret", "token", "api_key"
    }
    
    def __init__(self):
        self.removed_columns = []
        self.detected_pii = {}
        self.safe_columns = []
    
    def detect_pii_in_cell(self, value: str) -> List[str]:
        """Detect PII patterns in a single cell value."""
        if not isinstance(value, str):
            return []
        
        found_pii = []
        for pii_type, pattern in self.PATTERNS.items():
            if re.search(pattern, value):
                found_pii.append(pii_type)
        
        return found_pii
    
    def detect_pii_in_column(self, series: pd.Series, col_name: str) -> Tuple[bool, int, List[str]]:
        """Detect PII in an entire column."""
        col_lower = col_name.lower()
        
        # Check if column name is sensitive
        if col_lower in self.SENSITIVE_COLS:
            return True, len(series), ["sensitive_column_name"]
        
        # Check column values for patterns
        pii_found = set()
        count = 0
        
        for value in series.dropna():
            detected = self.detect_pii_in_cell(str(value))
            if detected:
                pii_found.update(detected)
                count += 1
        
        return bool(pii_found), count, list(pii_found)
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict:
        """Validate entire dataset for PII and return detailed report."""
        report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns_removed": [],
            "columns_safe": [],
            "pii_detected": {},
            "cells_with_pii": 0,
            "risk_level": "SAFE",
            "details": []
        }
        
        self.removed_columns = []
        self.detected_pii = {}
        self.safe_columns = []
        
        for col in df.columns:
            has_pii, count, pii_types = self.detect_pii_in_column(df[col], col)
            
            if has_pii:
                self.removed_columns.append(col)
                self.detected_pii[col] = {
                    "pii_types": pii_types,
                    "cells_affected": count,
                    "percentage": round(100 * count / len(df), 2)
                }
                report["pii_detected"][col] = self.detected_pii[col]
                report["columns_removed"].append(col)
                report["cells_with_pii"] += count
                report["details"].append(
                    f"🚨 Column '{col}': {count}/{len(df)} cells have PII ({pii_types})"
                )
            else:
                self.safe_columns.append(col)
                report["columns_safe"].append(col)
        
        # Determine risk level
        if self.removed_columns:
            if len(self.removed_columns) >= len(df.columns) * 0.5:
                report["risk_level"] = "CRITICAL"
            elif len(self.removed_columns) >= len(df.columns) * 0.25:
                report["risk_level"] = "HIGH"
            else:
                report["risk_level"] = "MEDIUM"
        else:
            report["risk_level"] = "SAFE"
        
        return report
    
    def remove_pii(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Remove PII columns and return cleaned dataframe + report."""
        report = self.validate_dataset(df)
        
        # Remove columns with PII
        df_clean = df.drop(columns=self.removed_columns, errors='ignore')
        
        report["action"] = "REMOVED" if self.removed_columns else "NO_ACTION"
        report["columns_remaining"] = list(df_clean.columns)
        report["data_retention"] = round(100 * len(df_clean.columns) / len(df.columns), 2)
        
        return df_clean, report
    
    def generate_pii_report(self, report: Dict) -> str:
        """Generate human-readable PII report."""
        lines = [
            "=" * 70,
            "PII DETECTION & REMOVAL REPORT",
            "=" * 70,
            "",
            f"📊 Dataset Overview:",
            f"   • Total Rows: {report['total_rows']:,}",
            f"   • Total Columns: {report['total_columns']}",
            f"   • Risk Level: {report['risk_level']}",
            "",
        ]
        
        if report['pii_detected']:
            lines.append("🚨 PII DETECTED & REMOVED:")
            for col, info in report['pii_detected'].items():
                lines.append(f"   • {col}")
                lines.append(f"     - Type: {', '.join(info['pii_types'])}")
                lines.append(f"     - Cells affected: {info['cells_affected']}/{report['total_rows']} ({info['percentage']}%)")
            lines.append("")
        else:
            lines.append("✅ NO PII DETECTED - Dataset is SAFE")
            lines.append("")
        
        lines.extend([
            "✅ SAFE COLUMNS (Kept):",
            f"   {len(report['columns_safe'])} out of {report['total_columns']} columns retained",
            "",
            "Columns retained:",
        ])
        
        for i, col in enumerate(report['columns_safe'], 1):
            lines.append(f"   {i}. {col}")
        
        lines.extend([
            "",
            f"📈 Data Retention: {report['data_retention']}%",
            f"✅ Status: {report['action']}",
            "=" * 70,
        ])
        
        return "\n".join(lines)


# Example usage & test
if __name__ == "__main__":
    # Test with sample data containing PII
    test_data = {
        "transaction_id": [1, 2, 3, 4, 5],
        "name": ["John Doe", "Jane Smith", "Bob Johnson", "Alice Brown", "Charlie Davis"],
        "email": ["john@example.com", "jane@example.com", "bob@example.com", "alice@example.com", "charlie@example.com"],
        "amount": [150.50, 1500.00, 50.25, 200.75, 300.00],
        "merchant_category": ["Grocery", "Electronics", "Gas Station", "Restaurant", "Hotel"],
        "device_type": ["Mobile", "Desktop", "Mobile", "Tablet", "Desktop"],
        "is_fraud": [0, 1, 0, 0, 1]
    }
    
    df_test = pd.DataFrame(test_data)
    
    print("BEFORE CLEANING:")
    print(df_test)
    print("\n")
    
    validator = PIIValidator()
    df_clean, report = validator.remove_pii(df_test)
    
    print(validator.generate_pii_report(report))
    print("\n")
    print("AFTER CLEANING:")
    print(df_clean)
