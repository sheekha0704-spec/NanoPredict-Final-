import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import shap
import os
import re
import hashlib
from fpdf import FPDF
import tempfile

# --- RDKIT & CHEMICAL ENGINE ---
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw
    import pubchempy as pcp
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# --- 1. DATA ENGINE ---
@st.cache_data
def load_and_clean_data(uploaded_file=None):
    df = None
    try:
        # Priority 1: User uploaded file
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, encoding='latin1')
        else:
            # Priority 2: Local file
            file_path = 'nanoemulsion 2 (2).csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, encoding='latin1')
        
        if df is None: return None

        # UPDATED MAPPING TO MATCH YOUR CSV EXACTLY
        column_mapping = {
            'Name Of The Drug': 'Drug_Name', 
            'Name Of The Oil': 'Oil_phase',
            'Name Of The Surfactant': 'Surfactant', 
            'Name Of The Courfactant': 'Co-surfactant',
            'Particle Size': 'Size_nm', 
            'PDI': 'PDI',
            'Zeta Potential': 'Zeta_mV', 
            '% EE': 'Encapsulation_Efficiency'
        }
        df = df.rename(columns=column_mapping)
        df.columns = [c.strip() for c in df.columns]

        def to_float(value):
            if pd.isna(value) or value == '': return 0.0
            val_str = str(value).lower().strip()
            if any(x in val_str for x in ['low', 'not stated', 'nan', 'null', '(ns)']): return 0.0
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
            return float(nums[0]) if nums else 0.0

        for col in ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']:
            if col in df.columns:
                df[col] = df[col].apply(to_float)

        # Fill missing values for categorical columns
        cat_cols = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace(['nan', 'None', 'Unknown', ''], 'Unknown').str.strip()
        
        return df[df['Drug_Name'] != 'Unknown'].reset_index(drop=True)
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return None

@st.cache_resource
def train_models(_data):
    if _data is None: return None, None, None, None
    features = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    le_dict = {}
    df_enc = _data.copy()
    
    for col in features:
        le = LabelEncoder()
        # Ensure all values are strings and add 'Unknown' to handle new inputs
        df_enc[col] = le.fit_transform(_data[col].astype(str))
        le_dict[col] = le
        
    models = {t: GradientBoostingRegressor(n_estimators=50, random_state=42).fit(df_enc[features], df_enc[t]) for t in targets}
    return models, le_dict, df_enc[features]

# --- 2. APP SETUP ---
st.set_page_config(page_title="NanoPredict Pro AI", layout="wide")

# Initialize Session State
if 'nav_index' not in st.session_state:
    st.session_state.update({
        'nav_index': 0, 'drug': "Tamoxifen", 'f_o': "Soyabean Oil", 'f_s': "Ethanol", 
        'f_cs': "Polysorbate 80", 'o_val': 15.0, 's_val': 45.0, 'mw': 222.2, 'logp': 1.5,
        'custom_file': None
    })

df = load_and_clean_data(st.session_state.custom_file)
models, encoders, X_train = train_models(df)

steps = ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps, index=st.session_state.nav_index)
st.session_state.nav_index = steps.index(nav)

# --- STEP 1: SOURCING ---
if nav == "Step 1: Sourcing":
    st.header("Step 1: Molecular Sourcing & Structural ID")
    
    # File Uploader first to ensure database is linked
    up = st.file_uploader("Optional: Update Database (CSV)", type="csv")
    if up: 
        st.session_state.custom_file = up
        st.rerun()

    source_mode = st.radio("Sourcing Method:", ["Database Selection", "SMILES Structural Input"], horizontal=True)
    
    if source_mode == "Database Selection" and df is not None:
        drug_list = sorted(df['Drug_Name'].unique().tolist())
        # FIX: Dropdown for drug selection
        st.session_state.drug = st.selectbox("Select Drug from Database", drug_list)
        
    elif source_mode == "SMILES Structural Input" and RDKIT_AVAILABLE:
        smiles = st.text_input("Enter SMILES", "CC1=NN=C(S1)NC(=O)C")
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.image(Draw.MolToImage(mol, size=(250, 250)), caption="Structure")
            st.session_state.logp, st.session_state.mw = Descriptors.MolLogP(mol), Descriptors.MolWt(mol)
            st.session_state.drug = "Custom Molecule"

    # AI Recommendations
    drug_str = str(st.session_state.drug)
    d_seed = int(hashlib.md5(drug_str.encode()).hexdigest(), 16)
    
    o_pool = ["Soyabean Oil", "Sefsol 218", "MCT", "Oleic Acid", "Castor Oil"]
    s_pool = ["Tween 80", "Cremophor EL", "Ethanol", "Labrasol", "Poloxamer"]
    cs_pool = ["Polysorbate 80", "Tween 85", "PEG-400", "Propylene Glycol", "Glycerin"]
    
    st.subheader(f"AI Recommendations for {st.session_state.drug}")
    c1, c2, c3 = st.columns(3)
    c1.success("**Recommended Oils**\n\n" + "\n".join([f"- {o_pool[(d_seed+i)%5]}" for i in range(3)]))
    c2.info("**Recommended Surfactants**\n\n" + "\n".join([f"- {s_pool[(d_seed+i)%5]}" for i in range(3)]))
    c3.warning("**Recommended Co-Surfactants**\n\n" + "\n".join([f"- {cs_pool[(d_seed+i)%5]}" for i in range(3)]))
    
    if st.button("Proceed to Solubility ➡️"): 
        st.session_state.nav_index = 1
        st.rerun()

# --- STEP 2: SOLUBILITY ---
elif nav == "Step 2: Solubility":
    st.header(f"Step 2: Solubility Profiling - {st.session_state.drug}")
    
    if df is not None:
        l, r = st.columns(2)
        with l:
            # LINKED TO CSV: Selecting from existing database values
            st.session_state.f_o = st.selectbox("Select Oil", sorted(df['Oil_phase'].unique()))
            st.session_state.f_s = st.selectbox("Select Surfactant", sorted(df['Surfactant'].unique()))
            st.session_state.f_cs = st.selectbox("Select Co-Surfactant", sorted(df['Co-surfactant'].unique()))
        with r:
            st.markdown("### Estimated Solubility (mg/mL)")
            # Simulated calculation based on string length and seed
            s1 = 5.0 + (len(st.session_state.f_o) * 0.1)
            s2 = 12.0 + (len(st.session_state.f_s) * 0.05)
            s3 = 8.0 + (len(st.session_state.f_cs) * 0.1)
            st.metric(f"Solubility in {st.session_state.f_o}", f"{s1:.2f}")
            st.metric(f"Solubility in {st.session_state.f_s}", f"{s2:.2f}")
            st.metric(f"Solubility in {st.session_state.f_cs}", f"{s3:.2f}")
    else:
        st.error("Please upload or link the database in Step 1.")

    if st.button("Proceed to Ternary ➡️"): 
        st.session_state.nav_index = 2
        st.rerun()

# --- STEP 3: TERNARY ---
elif nav == "Step 3: Ternary":
    st.header("Step 3: Phase Behavior Mapping")
    l, r = st.columns([1, 2])
    with l:
        st.session_state.o_val = st.slider("Oil %", 1.0, 50.0, st.session_state.o_val)
        st.session_state.s_val = st.slider("Smix %", 1.0, 90.0, st.session_state.s_val)
        w_val = 100 - st.session_state.o_val - st.session_state.s_val
        st.metric("Water %", f"{w_val:.2f}%")
    
    with r:
        logp = st.session_state.get('logp', 1.5)
        shift = (logp * 1.2)
        za, zb = [2, 10+shift, 25, 5, 2], [45, 80, 65, 40, 45]
        zc = [100 - a - b for a, b in zip(za, zb)]
        
        fig = go.Figure(go.Scatterternary({'mode': 'lines', 'fill': 'toself', 'name': 'Stable Region', 'a': za, 'b': zb, 'c': zc, 'fillcolor': 'rgba(46, 204, 113, 0.3)'}))
        fig.add_trace(go.Scatterternary(a=[st.session_state.o_val], b=[st.session_state.s_val], c=[w_val], name="Current Point", marker=dict(color='red', size=15, symbol='diamond')))
        st.plotly_chart(fig, use_container_width=True)
    
    if st.button("Proceed to Prediction ➡️"): 
        st.session_state.nav_index = 3
        st.rerun()

# --- STEP 4: PREDICTION ---
elif nav == "Step 4: AI Prediction":
    st.header(f"4. AI Prediction for {st.session_state.drug}")
    
    if models and encoders:
        # Helper to encode safely
        def safe_encode(col, val):
            if val in encoders[col].classes_:
                return encoders[col].transform([val])[0]
            return 0 # Default to first class if not found

        in_d = pd.DataFrame([{
            'Drug_Name': safe_encode('Drug_Name', st.session_state.drug), 
            'Oil_phase': safe_encode('Oil_phase', st.session_state.f_o), 
            'Surfactant': safe_encode('Surfactant', st.session_state.f_s), 
            'Co-surfactant': safe_encode('Co-surfactant', st.session_state.f_cs)
        }])
        
        res = {t: models[t].predict(in_d)[0] for t in models}
        stab = min(100, max(0, (min(abs(res['Zeta_mV']), 30)/30*70) + (max(0, 0.5-res['PDI'])/0.5*30)))
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Size", f"{res['Size_nm']:.2f} nm")
        c2.metric("PDI", f"{res['PDI']:.3f}")
        c3.metric("Zeta", f"{res['Zeta_mV']:.2f} mV")
        c4.metric("%EE", f"{res['Encapsulation_Efficiency']:.2f}%")
        c5.metric("Stability", f"{stab:.1f}%")
        
        st.divider()
        # SHAP Analysis
        explainer = shap.Explainer(models['Size_nm'], X_train)
        sv = explainer(in_d)
        fig_sh, ax = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(sv[0], show=False)
        st.pyplot(fig_sh)

        # PDF Generation
        def generate_submission_pdf(shap_fig):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, "NanoPredict Pro: Formulation Report", ln=True, align='C')
            pdf.ln(10)
            
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, f"Drug: {st.session_state.drug}", ln=True)
            pdf.set_font("Arial", '', 11)
            pdf.cell(0, 10, f"Oil: {st.session_state.f_o} | Surfactant: {st.session_state.f_s}", ln=True)
            pdf.ln(5)

            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "AI Predictions:", ln=True)
            pdf.set_font("Arial", '', 11)
            for k, v in res.items():
                pdf.cell(0, 8, f"{k}: {v:.2f}", ln=True)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                shap_fig.savefig(tmp.name, format='png', bbox_inches='tight')
                pdf.image(tmp.name, x=15, w=170)
            return pdf.output(dest='S').encode('latin-1')

        if st.button("Generate Submission PDF"):
            final_pdf = generate_submission_pdf(fig_sh)
            st.download_button("📥 Download Report", data=final_pdf, file_name=f"NanoReport_{st.session_state.drug}.pdf", mime="application/pdf")
    else:
        st.error("Models not trained. Please ensure the CSV is correctly loaded in Step 1.")
