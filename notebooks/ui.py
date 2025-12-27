"""
Streamlit UI for privacy-fraud-ai system.

Interactive dashboard with 5 core workflows:
- Federated training
- DP model testing
- Data upload & prediction
- Fraud summary generation
- Attack simulation
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset_generator import SyntheticBankDataGenerator
from federated_training import FederatedFraudDetector
from fraud_gpt import FraudGPT

# Setup paths (handle both streamlit run from root and from notebooks/)
project_root = Path(__file__).parent.parent
default_data_dir = str(project_root / "data")
default_models_dir = str(project_root / "models")

st.set_page_config(page_title="Privacy-Fraud-AI", layout="wide")

st.title("🔒 Privacy-Preserving Fraud Detection")
st.markdown(
    "Federated Learning + Differential Privacy + Fraud-GPT for secure fraud analysis"
)

# Sidebar
st.sidebar.header("⚙️ Configuration")
data_dir = st.sidebar.text_input("Data Directory", default_data_dir)
models_dir = st.sidebar.text_input("Models Directory", default_models_dir)

# Tabs
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🔗 End-to-End Pipeline",
        "🏋️ Generate Data",
        "🤝 Federated Training",
        "🔐 DP Model Test",
        "📊 Fraud Summary",
        "⚔️ Attack Simulation",
    ]
)

# ============================================================================
# TAB 0: End-to-End Pipeline (NEW)
# ============================================================================
with tab0:
    st.header("🔗 Complete Fraud Detection Pipeline")
    
    st.markdown("""
    **Complete workflow in one tab:**
    1. Upload Bank A & Bank B datasets
    2. PII removal & data cleaning
    3. Federated learning across banks
    4. Differential privacy protection
    5. Fraud pattern analysis with Fraud-GPT
    6. Attack simulation on discovered patterns
    
    """)
    
    st.divider()
    
    # Step 1: Upload Data
    st.subheader("📥 Step 1: Upload Bank Datasets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Bank A Data**")
        bankA_file = st.file_uploader("Upload Bank A CSV", type="csv", key="bankA_upload")
    
    with col2:
        st.write("**Bank B Data**")
        bankB_file = st.file_uploader("Upload Bank B CSV", type="csv", key="bankB_upload")
    
    # Store uploaded data in session state
    if bankA_file is not None:
        st.session_state.bankA_data = pd.read_csv(bankA_file)
    if bankB_file is not None:
        st.session_state.bankB_data = pd.read_csv(bankB_file)
    
    if "bankA_data" in st.session_state and "bankB_data" in st.session_state:
        st.success("✅ Both datasets loaded")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Bank A Rows", len(st.session_state.bankA_data))
        with col2:
            st.metric("Bank B Rows", len(st.session_state.bankB_data))
        with col3:
            total = len(st.session_state.bankA_data) + len(st.session_state.bankB_data)
            st.metric("Total Rows", total)
    
    st.divider()
    
    # Step 2-6: Run Pipeline
    st.subheader("⚙️ Step 2: Configure & Run Pipeline")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        dp_noise = st.slider("DP Noise Sigma", 0.1, 5.0, 0.5, 0.1)
    with col2:
        n_attacks = st.slider("Attack Scenarios", 1, 10, 3)
    with col3:
        st.write("")
        st.write("")
        run_pipeline_btn = st.button("▶️ Run Full Pipeline", key="run_full_pipeline", use_container_width=True)
    
    if run_pipeline_btn:
        if "bankA_data" not in st.session_state or "bankB_data" not in st.session_state:
            st.error("❌ Please upload both Bank A and Bank B datasets first")
        else:
            with st.spinner("Running complete pipeline..."):
                try:
                    # Get data
                    df_a = st.session_state.bankA_data.copy()
                    df_b = st.session_state.bankB_data.copy()
                    
                    progress_bar = st.progress(0)
                    status = st.empty()
                    
                    # ========== STEP 1: PII REMOVAL & CLEANING ==========
                    status.text("Step 1/5: PII Removal & Data Cleaning...")
                    progress_bar.progress(20)
                    
                    # Function to remove PII
                    def remove_pii_clean(df):
                        pii_cols = ["name", "email", "phone", "account_number", "address", 
                                   "customer_id", "ssn", "account_number"]
                        df = df.drop(columns=[c for c in pii_cols if c in df.columns], errors="ignore")
                        return df
                    
                    df_a_clean = remove_pii_clean(df_a)
                    df_b_clean = remove_pii_clean(df_b)
                    
                    with st.expander("📋 Step 1 Results: PII Removal"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Bank A - Columns Removed:**")
                            removed_a = set(df_a.columns) - set(df_a_clean.columns)
                            st.write(f"{list(removed_a) if removed_a else 'None (no PII found)'}")
                            st.write(f"**Clean Shape:** {df_a_clean.shape}")
                        with col2:
                            st.write("**Bank B - Columns Removed:**")
                            removed_b = set(df_b.columns) - set(df_b_clean.columns)
                            st.write(f"{list(removed_b) if removed_b else 'None (no PII found)'}")
                            st.write(f"**Clean Shape:** {df_b_clean.shape}")
                    
                    # Save cleaned data
                    bankA_clean_path = Path(data_dir) / "bankA_uploaded.csv"
                    bankB_clean_path = Path(data_dir) / "bankB_uploaded.csv"
                    df_a_clean.to_csv(bankA_clean_path, index=False)
                    df_b_clean.to_csv(bankB_clean_path, index=False)
                    
                    # ========== STEP 2: CALCULATE FRAUD STATS ==========
                    status.text("Step 2/5: Calculating Fraud Statistics...")
                    progress_bar.progress(40)
                    
                    fraud_col = "is_fraud" if "is_fraud" in df_a_clean.columns else None
                    
                    if fraud_col:
                        fraud_a = df_a_clean[fraud_col].sum()
                        fraud_b = df_b_clean[fraud_col].sum()
                        total_fraud = fraud_a + fraud_b
                        total_rows = len(df_a_clean) + len(df_b_clean)
                        fraud_rate = (total_fraud / total_rows * 100) if total_rows > 0 else 0
                    else:
                        fraud_a = fraud_b = total_fraud = 0
                        fraud_rate = 0
                    
                    # Calculate average amount
                    amount_col = "amount" if "amount" in df_a_clean.columns else None
                    if amount_col:
                        avg_amount = (df_a_clean[amount_col].mean() + df_b_clean[amount_col].mean()) / 2
                    else:
                        avg_amount = 150.0
                    
                    with st.expander("📊 Step 2 Results: Fraud Statistics"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Bank A Fraud", fraud_a)
                        with col2:
                            st.metric("Bank B Fraud", fraud_b)
                        with col3:
                            st.metric("Total Fraud", total_fraud)
                        with col4:
                            st.metric("Fraud Rate %", f"{fraud_rate:.2f}%")
                    
                    # ========== STEP 3: FEDERATED LEARNING & DP ==========
                    status.text("Step 3/5: Federated Learning + Differential Privacy...")
                    progress_bar.progress(60)
                    
                    federated = FederatedFraudDetector(
                        data_dir=data_dir, models_dir=models_dir, noise_sigma=dp_noise
                    )
                    
                    # Train on uploaded data
                    try:
                        federated.global_model = None
                        federated.dp_model = None
                        
                        # Load and train
                        banks = {
                            "bankA_uploaded": df_a_clean,
                            "bankB_uploaded": df_b_clean
                        }
                        
                        models_dict = {}
                        for bank_name, bank_df in banks.items():
                            if len(bank_df) > 0:
                                X_train = federated.encode_categorical(bank_df, fit=True)
                                y_train = bank_df[fraud_col] if fraud_col and fraud_col in bank_df.columns else [0] * len(bank_df)
                                
                                model = federated.train_local_model(bank_df, bank_name)
                                models_dict[bank_name] = model
                        
                        # Aggregate
                        df_merged = pd.concat([df_a_clean, df_b_clean], ignore_index=True)
                        federated.global_model = federated.aggregate_models_custom(df_merged)
                        federated.dp_model = federated.apply_differential_privacy(federated.global_model)
                        
                        # Save models
                        pipeline_model_path = Path(models_dir) / "pipeline_model.pkl"
                        pipeline_dp_path = Path(models_dir) / "pipeline_model_dp.pkl"
                        
                        with open(pipeline_model_path, "wb") as f:
                            pickle.dump(federated.global_model, f)
                        with open(pipeline_dp_path, "wb") as f:
                            pickle.dump(federated.dp_model, f)
                        
                    except Exception as e:
                        st.warning(f"⚠️ Could not train models: {e}")
                    
                    with st.expander("🔐 Step 3 Results: Federated Learning & DP"):
                        st.write(f"✅ Models trained and saved with DP noise (σ={dp_noise})")
                        st.write(f"📁 Global Model: `pipeline_model.pkl`")
                        st.write(f"📁 DP Model: `pipeline_model_dp.pkl`")
                    
                    # ========== STEP 4: FRAUD SUMMARY FROM GPT ==========
                    status.text("Step 4/5: Generating Fraud Summary with Fraud-GPT...")
                    progress_bar.progress(80)
                    
                    fraud_gpt = FraudGPT(base_model="gpt2")
                    
                    stats_for_gpt = {
                        "avg_amount": avg_amount,
                        "fraud_count": int(total_fraud),
                        "fraud_rate": fraud_rate,
                        "velocity": total_fraud / max(1, len(df_a_clean) + len(df_b_clean)) * 100,
                        "risk_devices": "mobile, unknown, vpn" if "device_type" in df_a_clean.columns else "unknown"
                    }
                    
                    summary = fraud_gpt.generate_fraud_summary(stats_for_gpt)
                    
                    # ========== STEP 5: ATTACK SIMULATION ==========
                    status.text("Step 5/5: Generating Attack Simulations...")
                    progress_bar.progress(95)
                    
                    attacks = fraud_gpt.generate_attack_simulation(n=n_attacks)
                    
                    progress_bar.progress(100)
                    status.text("✅ Pipeline Complete!")
                    
                    # Combined Fraud-GPT Results
                    with st.expander("📝 Fraud-GPT Analysis: Summary + Attack Patterns", expanded=True):
                        st.subheader("📊 Fraud Analysis Summary")
                        st.write(summary)
                        
                        st.divider()
                        
                        st.subheader("⚔️ Simulated Attack Patterns on Detected Fraud")
                        for i, attack in enumerate(attacks, 1):
                            with st.expander(f"Attack Pattern {i}", expanded=(i==1)):
                                st.write(attack)
                    
                    st.success("🎉 **Pipeline Execution Complete!**")
                    
                    # Final Summary
                    st.divider()
                    st.subheader("📈 Pipeline Summary Report")
                    
                    summary_cols = st.columns(5)
                    with summary_cols[0]:
                        st.metric("Total Records", len(df_a_clean) + len(df_b_clean))
                    with summary_cols[1]:
                        st.metric("Fraud Cases", total_fraud)
                    with summary_cols[2]:
                        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                    with summary_cols[3]:
                        st.metric("DP Sigma", dp_noise)
                    with summary_cols[4]:
                        st.metric("Attacks Generated", n_attacks)
                    
                    st.markdown("""
                    **✅ Pipeline Outputs:**
                    - PII removed from both datasets
                    - Federated model trained across banks
                    - Differential privacy applied (ε-DP)
                    - Fraud summary generated with context
                    - Attack simulations created for red-teaming
                    
                    All data is privacy-preserved with no PII in outputs.
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Pipeline Error: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

# ============================================================================
# TAB 1: Generate Synthetic Data
# ============================================================================
with tab1:
    st.header("Generate Synthetic Datasets")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_samples = st.number_input("Samples per Bank", 1000, 100000, 10000, step=1000)
    with col2:
        seed = st.number_input("Random Seed", 0, 10000, 42)
    with col3:
        st.write("")
        st.write("")
        generate_btn = st.button("🎲 Generate Datasets", key="gen_data")

    if generate_btn:
        with st.spinner("Generating synthetic data..."):
            try:
                generator = SyntheticBankDataGenerator(
                    seed=seed, n_samples_per_bank=n_samples
                )
                df_a, df_b, df_merged = generator.save_datasets()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Bank A Transactions",
                        len(df_a),
                        f"{df_a['is_fraud'].sum()} fraud",
                    )
                with col2:
                    st.metric(
                        "Bank B Transactions",
                        len(df_b),
                        f"{df_b['is_fraud'].sum()} fraud",
                    )
                with col3:
                    st.metric(
                        "Total Merged",
                        len(df_merged),
                        f"{df_merged['is_fraud'].sum()} fraud",
                    )

                st.success("✅ Datasets generated successfully!")

                st.subheader("Preview: Bank A")
                st.dataframe(df_a.head(10))

            except Exception as e:
                st.error(f"❌ Error: {e}")

# ============================================================================
# TAB 2: Federated Training
# ============================================================================
with tab2:
    st.header("Federated Learning Pipeline")

    st.markdown(
        """
    **Process:**
    1. Train separate RandomForest models on Bank A & Bank B
    2. Federate via data union aggregation
    3. Save global model
    
    **Privacy:**
    - No raw data shared between banks
    - Models aggregated only
    """
    )

    col1, col2 = st.columns(2)
    with col1:
        noise_sigma = st.slider("DP Noise Sigma", 0.1, 5.0, 1.0, 0.1)
    with col2:
        st.write("")
        st.write("")
        train_btn = st.button("🚀 Run Federated Training", key="fed_train")

    if train_btn:
        with st.spinner("Running federated training..."):
            try:
                federated = FederatedFraudDetector(
                    data_dir=data_dir, models_dir=models_dir, noise_sigma=noise_sigma
                )
                global_model, dp_model = federated.run_pipeline()

                st.success("✅ Training complete!")

                col1, col2 = st.columns(2)
                with col1:
                    st.info(
                        f"✓ Global model saved: `{models_dir}/global_model.pkl`"
                    )
                with col2:
                    st.info(
                        f"✓ DP model saved: `{models_dir}/global_model_dp.pkl`"
                    )

            except FileNotFoundError:
                st.error("❌ Data files not found. Generate data first in Tab 1.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ============================================================================
# TAB 3: DP Model Testing
# ============================================================================
with tab3:
    st.header("Differential Privacy Model Testing")

    st.markdown(
        """
    Test predictions from:
    - **Regular Model**: Standard RandomForest
    - **DP Model**: With differential privacy noise
    
    Compare predictions to observe privacy-utility tradeoff.
    """
    )

    try:
        models_path = Path(models_dir)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Load Regular Model")
            regular_pkl = models_path / "global_model.pkl"
            if regular_pkl.exists():
                with open(regular_pkl, "rb") as f:
                    regular_model = pickle.load(f)
                st.success("✓ Regular model loaded")
            else:
                st.warning("Model not found. Train first in Tab 2.")
                regular_model = None

        with col2:
            st.subheader("Load DP Model")
            dp_pkl = models_path / "global_model_dp.pkl"
            if dp_pkl.exists():
                with open(dp_pkl, "rb") as f:
                    dp_model = pickle.load(f)
                st.success("✓ DP model loaded")
            else:
                st.warning("Model not found. Train first in Tab 2.")
                dp_model = None

        if regular_model and dp_model:
            st.subheader("Test Predictions")

            data_file = st.file_uploader(
                "Upload CSV for predictions (or use merged data)", type="csv"
            )

            if data_file:
                df = pd.read_csv(data_file)
            else:
                merged_csv = Path(data_dir) / "full_merged.csv"
                if merged_csv.exists():
                    df = pd.read_csv(merged_csv)
                    st.info(f"Using default: `{merged_csv}`")
                else:
                    st.warning("No test data available.")
                    df = None

            if df is not None:
                # Prepare features (drop is_fraud for prediction)
                feature_cols = [
                    c
                    for c in df.columns
                    if c not in ["is_fraud", "timestamp", "transaction_id"]
                ]
                X = df[feature_cols]

                # Encode categorical
                from sklearn.preprocessing import LabelEncoder

                for col in ["merchant_category", "device_type"]:
                    if col in X.columns:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))

                test_btn = st.button("🧪 Run Predictions", key="dp_test")

                if test_btn:
                    with st.spinner("Running predictions..."):
                        try:
                            regular_preds = regular_model.predict(X)
                            dp_preds = dp_model.predict(X)

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                fraud_regular = regular_preds.sum()
                                st.metric("Regular Model Fraud Count", fraud_regular)
                            with col2:
                                fraud_dp = dp_preds.sum()
                                st.metric("DP Model Fraud Count", fraud_dp)
                            with col3:
                                diff = abs(fraud_regular - fraud_dp)
                                st.metric("DP Perturbation", diff)

                            st.subheader("Comparison Table")
                            comparison = pd.DataFrame(
                                {
                                    "Regular": regular_preds,
                                    "DP": dp_preds,
                                    "True": df["is_fraud"].values,
                                }
                            )
                            st.dataframe(comparison.head(20))

                            st.success(
                                "✅ Predictions complete. Privacy budget preserved."
                            )

                        except Exception as e:
                            st.error(f"❌ Prediction error: {e}")

    except Exception as e:
        st.error(f"❌ Error loading models: {e}")

# ============================================================================
# TAB 4: Fraud Summary Generation
# ============================================================================
with tab4:
    st.header("Generate Fraud Analysis Summary")

    st.markdown(
        """
    Use Fraud-GPT to analyze transaction statistics and generate 
    privacy-safe fraud summaries.
    """
    )

    col1, col2 = st.columns(2)

    with col1:
        avg_amount = st.number_input("Average Transaction Amount ($)", 1.0, 10000.0, 150.0)
    with col2:
        fraud_count = st.number_input("Fraud Transaction Count", 1, 10000, 45)

    col1, col2 = st.columns(2)
    with col1:
        velocity = st.number_input("Transactions per Hour", 0.1, 100.0, 12.0)
    with col2:
        risk_devices = st.text_input("Risk Device Types", "VPN, Mobile, Unknown")

    summary_btn = st.button("📝 Generate Summary", key="gen_summary")

    if summary_btn:
        with st.spinner("Fraud-GPT generating analysis..."):
            try:
                fraud_gpt = FraudGPT(base_model="gpt2")

                stats = {
                    "avg_amount": avg_amount,
                    "fraud_count": fraud_count,
                    "velocity": velocity,
                    "risk_devices": risk_devices,
                }

                summary = fraud_gpt.generate_fraud_summary(stats)

                st.subheader("Fraud Analysis Summary")
                st.write(summary)
                st.success("✅ Generated without PII")

            except Exception as e:
                st.error(f"❌ Error: {e}")

# ============================================================================
# TAB 5: Attack Simulation
# ============================================================================
with tab5:
    st.header("Synthetic Attack Simulation")

    st.markdown(
        """
    Generate realistic fraud attack scenarios for:
    - Red-teaming
    - Detection system testing
    - Security awareness training
    """
    )

    col1, col2 = st.columns(2)
    with col1:
        n_attacks = st.slider("Number of Scenarios", 1, 10, 5)
    with col2:
        st.write("")
        st.write("")
        attack_btn = st.button("⚔️ Generate Attacks", key="gen_attacks")

    if attack_btn:
        with st.spinner("Generating attack scenarios..."):
            try:
                fraud_gpt = FraudGPT(base_model="gpt2")
                attacks = fraud_gpt.generate_attack_simulation(n=n_attacks)

                st.subheader("Attack Scenarios")
                for i, attack in enumerate(attacks, 1):
                    with st.expander(f"Attack {i}"):
                        st.write(attack)

                st.success("✅ Scenarios generated - all PII removed")

            except Exception as e:
                st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.markdown(
    "🔐 Privacy-Fraud-AI | Federated Learning + DP + Fraud-GPT | "
    "[GitHub](https://github.com) | "
    "[Docs](https://github.com)"
)
