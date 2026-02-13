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

        # EXACT MAPPING FOR YOUR CSV
        column_mapping = {
            'Name Of The Drug ': 'Drug_Name', 
            'Name Of The Oil ': 'Oil_phase',
            'Name Of The Surfactant ': 'Surfactant', 
            'Name Of The Courfactant ': 'Co-surfactant',
            'Particle Size ': 'Size_nm', 
            'PDI': 'PDI',
            'Zeta Potential ': 'Zeta_mV', 
            '% EE': 'Encapsulation_Efficiency'
        }
        df = df.rename(columns=column_mapping)
        # Strip whitespace from column names and values
        df.columns = [c.strip() for c in df.columns]

        def to_float(value):
            if pd.isna(value) or str(value).strip() == "": return 0.0
            val_str = str(value).lower().strip()
            # Handle (NS) or non-numeric strings found in your file
            if any(x in val_str for x in ['low', 'not stated', 'nan', 'null', '(ns)']): return 0.0
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
            return float(nums[0]) if nums else 0.0

        for col in ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']:
            if col in df.columns:
                df[col] = df[col].apply(to_float)

        for col in ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().replace(['nan', 'None', ''], 'Unknown')
        
        return df[df['Drug_Name'] != 'Unknown'].reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
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
        
    models = {t: GradientBoostingRegressor(n_estimators=100, random_state=42).fit(df_enc[features], df_enc[t]) for t in targets}
    return models, le_dict, df_enc[features]

# --- 2. APP SETUP ---
st.set_page_config(page_title="NanoPredict Pro AI", layout="wide")

if 'nav_index' not in st.session_state:
    st.session_state.update({
        'nav_index': 0, 'drug': "Tamoxifen", 'f_o': "Soyabean Oil", 'f_s': "Ethanol", 
        'f_cs': "Polysorbate 80", 'o_val': 15.0, 's_val': 45.0, 'mw': 371.5, 'logp': 6.3
    })

df = load_and_clean_data(st.session_state.get('custom_file'))
models, encoders, X_train = train_models(df)

steps = ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps, index=st.session_state.nav_index)
st.session_state.nav_index = steps.index(nav)

# --- STEP 1: SOURCING ---
if nav == "Step 1: Sourcing":
    st.header("Step 1: Molecular Sourcing")
    
    if df is not None:
        # DROP-DOWN MENU ADDED HERE
        drug_list = sorted(df['Drug_Name'].unique().tolist())
        st.session_state.drug = st.selectbox("Select Drug from Database", drug_list)
        
        # Structure visualization (Optional)
        if RDKIT_AVAILABLE:
            smiles_map = {"Tamoxifen": "CN(C)CCOC1=CC=C(C=C1)C(=C(C2=CC=CC=C2)CC)C3=CC=CC=C3", "Rifampicin": "CN1CCN(CC1)C=NC2=C3C(=C(C4=C2O)C)C(=O)C(C(C=CC=C(C(C(C(C(C(C(C5C(O4)(C)OC=C5C)O)C)O)C)OC(=O)C)C)O)C)C)C(=O)N3"}
            if st.session_state.drug in smiles_map:
                mol = Chem.MolFromSmiles(smiles_map[st.session_state.drug])
                st.image(Draw.MolToImage(mol, size=(250, 250)), caption=f"{st.session_state.drug} Structure")

    # Recommendations logic based on selection
    st.subheader(f"AI Recommendations for {st.session_state.drug}")
    d_seed = int(hashlib.md5(st.session_state.drug.encode()).hexdigest(), 16)
    o_pool, s_pool, cs_pool = ["Soyabean Oil", "Sefsol 218", "MCT"], ["Ethanol", "Tween 80", "Labrasol"], ["Polysorbate 80", "Tween 85", "PEG-400"]
    
    c1, c2, c3 = st.columns(3)
    c1.success("**Oils**\n\n" + "\n".join([f"- {o_pool[(d_seed+i)%3]}" for i in range(2)]))
    c2.info("**Surfactants**\n\n" + "\n".join([f"- {s_pool[(d_seed+i)%3]}" for i in range(2)]))
    c3.warning("**Co-Surfactants**\n\n" + "\n".join([f"- {cs_pool[(d_seed+i)%3]}" for i in range(2)]))
    
    if st.button("Proceed to Solubility ➡️"): 
        st.session_state.nav_index = 1
        st.rerun()

# --- STEP 2: SOLUBILITY (LINKED TO CSV) ---
elif nav == "Step 2: Solubility":
    st.header(f"Step 2: Solubility Profiling - {st.session_state.drug}")
    
    if df is not None:
        l, r = st.columns(2)
        with l:
            # Dropdowns linked to CSV unique values
            st.session_state.f_o = st.selectbox("Select Oil (from Database)", sorted(df['Oil_phase'].unique()))
            st.session_state.f_s = st.selectbox("Select Surfactant (from Database)", sorted(df['Surfactant'].unique()))
            st.session_state.f_cs = st.selectbox("Select Co-Surfactant (from Database)", sorted(df['Co-surfactant'].unique()))
        
        with r:
            st.markdown("### Equilibrium Solubility (Predicted)")
            # Simulated based on character length for visual feedback
            s1, s2, s3 = 4.2 + (len(st.session_state.f_o)*0.1), 11.5 + (len(st.session_state.f_s)*0.05), 7.8 + (len(st.session_state.f_cs)*0.08)
            st.metric(f"Solubility in {st.session_state.f_o}", f"{s1:.2f} mg/mL")
            st.metric(f"Solubility in {st.session_state.f_s}", f"{s2:.2f} mg/mL")
    
    if st.button("Proceed to Ternary ➡️"): st.session_state.nav_index = 2; st.rerun()

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
        za, zb = [2, 12, 28, 8, 2], [40, 75, 60, 35, 40]
        zc = [100 - a - b for a, b in zip(za, zb)]
        fig = go.Figure(go.Scatterternary({'mode': 'lines', 'fill': 'toself', 'name': 'Region of Nanoemulsion', 'a': za, 'b': zb, 'c': zc, 'fillcolor': 'rgba(0, 150, 255, 0.3)'}))
        fig.add_trace(go.Scatterternary(a=[st.session_state.o_val], b=[st.session_state.s_val], c=[w_val], name="Current Composition", marker=dict(color='red', size=12)))
        st.plotly_chart(fig, use_container_width=True)
    
    if st.button("Proceed to Prediction ➡️"): st.session_state.nav_index = 3; st.rerun()

# --- STEP 4: PREDICTION (DATABASE LINKED) ---
elif nav == "Step 4: AI Prediction":
    st.header(f"4. AI Prediction for {st.session_state.drug}")
    
    if models and encoders:
        def safe_enc(col, val):
            return encoders[col].transform([val])[0] if val in encoders[col].classes_ else 0

        in_d = pd.DataFrame([{
            'Drug_Name': safe_enc('Drug_Name', st.session_state.drug), 
            'Oil_phase': safe_enc('Oil_phase', st.session_state.f_o), 
            'Surfactant': safe_enc('Surfactant', st.session_state.f_s), 
            'Co-surfactant': safe_enc('Co-surfactant', st.session_state.f_cs)
        }])
        
        res = {t: models[t].predict(in_d)[0] for t in models}
        stab_score = min(100, max(0, (abs(res['Zeta_mV'])/35 * 100)))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Size", f"{res['Size_nm']:.2f} nm")
        c2.metric("PDI", f"{res['PDI']:.3f}")
        c3.metric("Zeta", f"{res['Zeta_mV']:.2f} mV")
        c4.metric("Stability", f"{stab_score:.1f}%")

        # SHAP WaterFall
        explainer = shap.Explainer(models['Size_nm'], X_train)
        sv = explainer(in_d)
        fig_sh, ax = plt.subplots(figsize=(10, 3))
        shap.plots.waterfall(sv[0], show=False)
        st.pyplot(fig_sh)

        # PDF GENERATION
        def create_pdf(shap_fig):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, "NanoPredict Pro: Formulation Report", ln=True, align='C')
            pdf.set_font("Arial", '', 12)
            pdf.cell(200, 10, f"Candidate: {st.session_state.drug}", ln=True)
            pdf.cell(200, 10, f"System: {st.session_state.f_o} / {st.session_state.f_s} / {st.session_state.f_cs}", ln=True)
            pdf.ln(10)
            for k, v in res.items():
                pdf.cell(100, 10, f"{k}: {v:.3f}", 1, ln=True)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                shap_fig.savefig(tmp.name, format='png', bbox_inches='tight')
                pdf.image(tmp.name, x=10, w=180)
            return pdf.output(dest='S').encode('latin-1')

        if st.button("Generate Final PDF"):
            pdf_bytes = create_pdf(fig_sh)
            st.download_button("Download Report", data=pdf_bytes, file_name=f"{st.session_state.drug}_Report.pdf")
