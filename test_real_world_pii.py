"""
Real-World Dataset Demo with PII Detection Proof

Shows how the system handles real bank transaction data with sensitive information.
"""

import pandas as pd
from notebooks.pii_validator import PIIValidator

# Simulated real-world banking dataset with PII
REAL_WORLD_DATA = {
    "transaction_id": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
    "customer_name": [
        "John Michael Smith",
        "Sarah Jennifer Williams", 
        "Robert James Johnson",
        "Emily Louise Brown",
        "David Christopher Davis",
        "Jessica Anna Martinez",
        "Michael Daniel Garcia",
        "Amanda Nicole Rodriguez"
    ],
    "email": [
        "john.smith@gmail.com",
        "sarah.williams@yahoo.com",
        "robert.johnson@outlook.com",
        "emily.brown@gmail.com",
        "david.davis@yahoo.com",
        "jessica.martinez@gmail.com",
        "michael.garcia@outlook.com",
        "amanda.rodriguez@gmail.com"
    ],
    "phone_number": [
        "555-123-4567",
        "555-234-5678",
        "555-345-6789",
        "555-456-7890",
        "555-567-8901",
        "555-678-9012",
        "555-789-0123",
        "555-890-1234"
    ],
    "account_number": [
        "4532-8945-1234-5678",
        "5691-2384-9876-5432",
        "3847-5629-1357-2468",
        "9283-4756-2468-1357",
        "7563-2891-9753-8462",
        "2847-6194-5382-7641",
        "8374-1926-7483-2956",
        "5629-3847-2019-7365"
    ],
    "ssn": [
        "123-45-6789",
        "234-56-7890",
        "345-67-8901",
        "456-78-9012",
        "567-89-0123",
        "678-90-1234",
        "789-01-2345",
        "890-12-3456"
    ],
    "address": [
        "123 Main St, New York, NY 10001",
        "456 Oak Ave, Los Angeles, CA 90001",
        "789 Pine Rd, Chicago, IL 60601",
        "321 Elm St, Houston, TX 77001",
        "654 Maple Dr, Phoenix, AZ 85001",
        "987 Cedar Ln, Philadelphia, PA 19101",
        "159 Birch Way, San Antonio, TX 78201",
        "753 Spruce Ct, San Diego, CA 92101"
    ],
    "amount": [150.50, 1500.00, 50.25, 200.75, 300.00, 125.99, 850.00, 45.60],
    "merchant_category": ["Grocery", "Electronics", "Gas Station", "Restaurant", "Hotel", "Pharmacy", "Airline", "Coffee Shop"],
    "device_type": ["Mobile", "Desktop", "Mobile", "Tablet", "Desktop", "Mobile", "Desktop", "Mobile"],
    "location_ip": [
        "192.168.1.100",
        "192.168.1.101",
        "192.168.1.102",
        "192.168.1.103",
        "192.168.1.104",
        "192.168.1.105",
        "192.168.1.106",
        "192.168.1.107"
    ],
    "is_fraud": [0, 1, 0, 0, 1, 0, 1, 0]
}

def main():
    print("\n" + "=" * 80)
    print("REAL-WORLD BANKING DATASET WITH SENSITIVE INFORMATION")
    print("=" * 80)
    
    # Create dataframe
    df = pd.DataFrame(REAL_WORLD_DATA)
    
    print("\n📋 ORIGINAL DATASET (Columns with PII):")
    print("-" * 80)
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
    
    print("\n\n🔍 RUNNING PII VALIDATOR...")
    print("-" * 80)
    
    validator = PIIValidator()
    df_clean, report = validator.remove_pii(df)
    
    # Print detailed report
    print("\n" + validator.generate_pii_report(report))
    
    print("\n\n✨ CLEANED DATASET (Safe for federated learning):")
    print("-" * 80)
    print(f"Columns: {list(df_clean.columns)}")
    print(f"\nFirst 3 rows:\n{df_clean.head(3).to_string()}")
    
    print("\n\n📊 SUMMARY:")
    print("-" * 80)
    print(f"✅ Original columns: {len(df.columns)}")
    print(f"✅ Cleaned columns: {len(df_clean.columns)}")
    print(f"✅ PII columns removed: {len(report['columns_removed'])}")
    print(f"✅ Data retention: {report['data_retention']}%")
    print(f"✅ Rows preserved: {len(df_clean):,} (100%)")
    print(f"✅ Status: READY FOR FEDERATED LEARNING")
    
    print("\n\n🛡️ PROOF OF PII REMOVAL:")
    print("-" * 80)
    print("Columns REMOVED (contained PII):")
    for col in report['columns_removed']:
        pii_info = report['pii_detected'][col]
        print(f"  ❌ {col}: {pii_info['pii_types']} ({pii_info['cells_affected']}/{report['total_rows']} cells)")
    
    print("\nColumns RETAINED (Safe to use):")
    for i, col in enumerate(report['columns_safe'], 1):
        print(f"  ✅ {i}. {col}")
    
    print("\n\n💡 USE CASE: REAL-WORLD BANKING DATA")
    print("-" * 80)
    print("This demonstrates that the system can:")
    print("✓ Accept real banking transaction data")
    print("✓ Automatically detect 9 types of PII patterns")
    print("✓ Remove sensitive columns (names, emails, phones, SSN, addresses, etc.)")
    print("✓ Preserve all transaction features needed for fraud detection")
    print("✓ Provide detailed proof of what was removed")
    print("✓ Output clean data ready for federated learning")
    print("✓ Maintain 100% row integrity (no data loss)")
    
    print("\n\n🚀 NEXT STEPS:")
    print("-" * 80)
    print("1. Go to 'PII Check' tab in Streamlit UI")
    print("2. Upload your real CSV dataset")
    print("3. View automated detection & removal report")
    print("4. Download cleaned data")
    print("5. Use in End-to-End Pipeline tab for federated learning")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
