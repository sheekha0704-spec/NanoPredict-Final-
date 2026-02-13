import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
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
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# --- 1. DATA ENGINE ---
@st.cache_data
def load_and_clean_data(uploaded_file=None):
    df = None
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, encoding='latin1')
        else:
            file_path = 'nanoemulsion 2 (2).csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, encoding='latin1')
        
        if df is None: return None

        # COLUMN MAPPING (Synchronized with your CSV)
        column_mapping = {
            'Name Of The Drug ': 'Drug_Name', 
            'Name Of The Oil ': 'Oil_phase',
            'Name Of The Surfactant ': 'Surfactant', 
            'Name Of The Courfactant ': 'Co-surfactant',
            'Solubility Of Drug In Oil ': 'Sol_Oil',
            'Solubility Of Drug In Surfactant ': 'Sol_Surf',
            'Solubility Of Drug In Cosurfactant ': 'Sol_CoSurf',
            'Particle Size ': 'Size_nm', 
            'PDI': 'PDI',
            'Zeta Potential ': 'Zeta_mV', 
            '% EE': 'Encapsulation_Efficiency'
        }
        df = df.rename(columns=column_mapping)
        df.columns = [c.strip() for c in df.columns]

        def to_float(value):
            if pd.isna(value) or str(value).strip() == "": return 0.0
            val_str = str(value).lower().strip()
            if any(x in val_str for x in ['low', 'not stated', '(ns)']): return 0.0
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
            return float(nums[0]) if nums else 0.0

        numeric_cols = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency', 'Sol_Oil', 'Sol_Surf', 'Sol_CoSurf']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(to_float)

        cat_cols = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().replace(['nan', ''], 'Unknown')
        
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Load Error: {e}")
        return None

@st.cache_resource
def train_models(_data):
    if _data is None: return None, None, None
    features = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    le_dict = {}
    df_enc = _data.copy()
    for col in features:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(_data[col].astype(str))
        le_dict[col] = le
    models = {t: GradientBoostingRegressor(n_estimators=100).fit(df_enc[features], df_enc[t]) for t in targets}
    return models, le_dict, df_enc[features]

# --- 2. APP SETUP ---
st.set_page_config(page_title="NanoPredict Pro AI", layout="wide")

if 'nav_index' not in st.session_state:
    st.session_state.update({
        'nav_index': 0, 'drug': "Tamoxifen", 'f_o': "Soyabean Oil", 'f_s': "Ethanol", 
        'f_cs': "Polysorbate 80", 'o_val': 15.0, 's_val': 45.0, 'mw': 371.5, 'logp': 6.3,
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
    mode = st.radio("Input Mode:", ["Database Selection", "SMILES String", "Browse Custom Lab Data"], horizontal=True)
    
    if mode == "Database Selection" and df is not None:
        drug_list = sorted(df['Drug_Name'].unique().tolist())
        st.session_state.drug = st.selectbox("Select Drug", drug_list)
        
    elif mode == "SMILES String" and RDKIT_AVAILABLE:
        smiles = st.text_input("Enter SMILES", "CN(C)CCOC1=CC=C(C=C1)C(=C(C2=CC=CC=C2)CC)C3=CC=CC=C3")
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.image(Draw.MolToImage(mol, size=(300, 300)), caption="Molecular Structure")
            st.session_state.mw = Descriptors.MolWt(mol)
            st.session_state.logp = Descriptors.MolLogP(mol)
            st.session_state.drug = "SMILES Defined Molecule"
            
    elif mode == "Browse Custom Lab Data":
        up = st.file_uploader("Upload CSV", type="csv")
        if up:
            st.session_state.custom_file = up
            st.rerun()

    st.subheader("Formulation Components (Unique to Database)")
    c1, c2, c3 = st.columns(3)
    with c1: st.session_state.f_o = st.selectbox("Oil Phase", sorted(df['Oil_phase'].unique()))
    with c2: st.session_state.f_s = st.selectbox("Surfactant", sorted(df['Surfactant'].unique()))
    with c3: st.session_state.f_cs = st.selectbox("Co-Surfactant", sorted(df['Co-surfactant'].unique()))

    if st.button("Proceed to Solubility ➡️"): 
        st.session_state.nav_index = 1
        st.rerun()

# --- STEP 2: SOLUBILITY ---
elif nav == "Step 2: Solubility":
    st.header(f"Step 2: Solubility Profiling - {st.session_state.drug}")
    
    # Check if drug/excipient combo exists in database
    db_match = df[(df['Drug_Name'] == st.session_state.drug) & (df['Oil_phase'] == st.session_state.f_o)]
    
    if not db_match.empty:
        s_oil = db_match['Sol_Oil'].values[0]
        s_surf = db_match['Sol_Surf'].values[0]
        s_cos = db_match['Sol_CoSurf'].values[0]
        st.info("Values retrieved from Experimental Database.")
    else:
        # Prediction logic from Chemical Library (MW/LogP based)
        s_oil = 15.0 + (st.session_state.get('mw', 300)/100)
        s_surf = 25.0 + (st.session_state.get('logp', 2.0)*2)
        s_cos = 20.0
        st.warning("Values predicted via Molecular Descriptors (Not in DB).")

    st.session_state.update({'s_oil': s_oil, 's_surf': s_surf, 's_cos': s_cos})
    
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Solubility in {st.session_state.f_o}", f"{s_oil:.2f} mg/mL")
    c2.metric(f"Solubility in {st.session_state.f_s}", f"{s_surf:.2f} mg/mL")
    c3.metric(f"Solubility in {st.session_state.f_cs}", f"{s_cos:.2f} mg/mL")

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
        za, zb = [2, 10, 25, 5, 2], [45, 80, 65, 40, 45]
        zc = [100 - a - b for a, b in zip(za, zb)]
        fig = go.Figure(go.Scatterternary({'mode': 'lines', 'fill': 'toself', 'name': 'Stable', 'a': za, 'b': zb, 'c': zc, 'fillcolor': 'rgba(46, 204, 113, 0.3)'}))
        fig.add_trace(go.Scatterternary(a=[st.session_state.o_val], b=[st.session_state.s_val], c=[w_val], name="Current", marker=dict(color='red', size=12)))
        st.plotly_chart(fig, use_container_width=True)
    
    if st.button("Proceed to Prediction ➡️"): 
        st.session_state.nav_index = 3
        st.rerun()

# --- STEP 4: PREDICTION & PDF ---
elif nav == "Step 4: AI Prediction":
    st.header(f"Step 4: AI Forecast & Report")
    
    def safe_enc(col, val):
        return encoders[col].transform([val])[0] if val in encoders[col].classes_ else 0

    in_d = pd.DataFrame([{
        'Drug_Name': safe_enc('Drug_Name', st.session_state.drug), 
        'Oil_phase': safe_enc('Oil_phase', st.session_state.f_o), 
        'Surfactant': safe_enc('Surfactant', st.session_state.f_s), 
        'Co-surfactant': safe_enc('Co-surfactant', st.session_state.f_cs)
    }])
    
    res = {t: models[t].predict(in_d)[0] for t in models}
    stability = 100 - (res['PDI'] * 100) # Simple stability metric logic

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Size", f"{res['Size_nm']:.1f} nm")
    c2.metric("Zeta", f"{res['Zeta_mV']:.1f} mV")
    c3.metric("% EE", f"{res['Encapsulation_Efficiency']:.1f}%")
    c4.metric("PDI", f"{res['PDI']:.3f}")
    c5.metric("Stability", f"{stability:.1f}%")

    def generate_full_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "NanoPredict Pro: Comprehensive Lab Report", ln=True, align='C')
        
        sections = [
            ("1. Sourcing", [f"Drug: {st.session_state.drug}", f"Oil: {st.session_state.f_o}", f"Surfactant: {st.session_state.f_s}"]),
            ("2. Solubility", [f"Oil Solubility: {st.session_state.get('s_oil', 0):.2f} mg/mL", f"Surf Solubility: {st.session_state.get('s_surf', 0):.2f} mg/mL"]),
            ("3. Composition", [f"Oil: {st.session_state.o_val}%", f"Smix: {st.session_state.s_val}%", f"Water: {100-st.session_state.o_val-st.session_state.s_val:.2f}%"]),
            ("4. AI Predictions", [f"Size: {res['Size_nm']:.2f} nm", f"Zeta: {res['Zeta_mV']:.2f} mV", f"PDI: {res['PDI']:.3f}", f"Stability: {stability:.1f}%"])
        ]
        
        for title, lines in sections:
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, title, ln=True)
            pdf.set_font("Arial", '', 11)
            for line in lines: pdf.cell(0, 8, line, ln=True)
            
        return pdf.output(dest='S').encode('latin-1')

    if st.button("Generate Comprehensive Report"):
        st.download_button("Download Report", data=generate_full_pdf(), file_name="Nano_Report.pdf")
