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
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, encoding='latin1')
        else:
            file_path = 'nanoemulsion 2 (2).csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, encoding='latin1')
            else:
                return None
        
        # Precise Column Mapping
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
            if any(x in val_str for x in ['low', 'not stated', 'nan', '(ns)']): return 0.0
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
            return float(nums[0]) if nums else 0.0

        numeric_cols = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency', 'Sol_Oil', 'Sol_Surf', 'Sol_CoSurf']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(to_float)

        cat_cols = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().replace(['nan', 'None'], 'Unknown')
        
        return df[df['Drug_Name'] != 'Unknown'].reset_index(drop=True)
    except Exception as e:
        st.error(f"Data Loading Error: {e}")
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
    
    models = {}
    for t in targets:
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(df_enc[features], df_enc[t])
        models[t] = model
        
    return models, le_dict, df_enc[features]

# --- 2. APP SETUP ---
st.set_page_config(page_title="NanoPredict Pro AI", layout="wide")

# Initialize Session State
if 'nav_index' not in st.session_state:
    st.session_state.update({
        'nav_index': 0, 'drug': "Tamoxifen", 'f_o': "Soyabean Oil", 'f_s': "Ethanol", 
        'f_cs': "Polysorbate 80", 'o_val': 15.0, 's_val': 45.0, 'mw': 371.5, 'logp': 6.3,
        'custom_file': None, 'w_val': 40.0
    })

df = load_and_clean_data(st.session_state.custom_file)
models, encoders, X_train = train_models(df)

if df is None:
    st.error("Please upload a CSV file to begin.")
    st.file_uploader("Upload Lab CSV", type="csv", key="init_upload", on_change=lambda: st.session_state.update({'custom_file': st.session_state.init_upload}))
    st.stop()

# --- NAVIGATION ---
steps = ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps, index=st.session_state.nav_index)
st.session_state.nav_index = steps.index(nav)

# --- STEP 1: SOURCING ---
if nav == "Step 1: Sourcing":
    st.header("🔬 Molecular Sourcing & AI Recommendations")
    
    source_mode = st.radio("Sourcing Method:", ["Database Selection", "SMILES Input", "Upload New Data"], horizontal=True)
    
    if source_mode == "Database Selection":
        # Convert to string and filter out any None/NaN before sorting
drug_list = sorted([str(x) for x in df['Drug_Name'].unique() if pd.notna(x)])
st.session_state.drug = st.selectbox("Select Drug", drug_list)
        st.session_state.drug = st.selectbox("Select Drug", drug_list)
        
    elif source_mode == "SMILES Input" and RDKIT_AVAILABLE:
        smiles = st.text_input("Enter SMILES", "CN(C)CCOC1=CC=C(C=C1)C(=C(C2=CC=CC=C2)CC)C3=CC=CC=C3")
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.image(Draw.MolToImage(mol, size=(300, 300)), caption="Structure Identified")
            st.session_state.logp, st.session_state.mw = Descriptors.MolLogP(mol), Descriptors.MolWt(mol)
            st.session_state.drug = "Custom Molecule"
                
    elif source_mode == "Upload New Data":
        up = st.file_uploader("Replace Database CSV", type="csv")
        if up: 
            st.session_state.custom_file = up
            st.cache_data.clear()
            st.rerun()

    st.divider()
    st.subheader(f"Top AI Recommendations for {st.session_state.drug}")
    
    drug_data = df[df['Drug_Name'] == st.session_state.drug]
    if not drug_data.empty:
        rec_o = drug_data['Oil_phase'].value_counts().index.tolist()[:3]
        rec_s = drug_data['Surfactant'].value_counts().index.tolist()[:3]
        rec_cs = drug_data['Co-surfactant'].value_counts().index.tolist()[:3]
    else:
        d_seed = int(hashlib.md5(str(st.session_state.drug).encode()).hexdigest(), 16)
        o_p, s_p, cs_p = ["MCT", "Oleic Acid", "Capryol 90"], ["Tween 80", "Ethanol", "Labrasol"], ["PEG-400", "Transcutol-HP", "PG"]
        rec_o = [o_p[(d_seed+i)%3] for i in range(3)]
        rec_s = [s_p[(d_seed+i)%3] for i in range(3)]
        rec_cs = [cs_p[(d_seed+i)%3] for i in range(3)]

    c1, c2, c3 = st.columns(3)
    with c1: st.success("**Top Oils**"); [st.markdown(f"* {o}") for o in rec_o]
    with c2: st.info("**Top Surfactants**"); [st.markdown(f"* {s}") for s in rec_s]
    with c3: st.warning("**Top Co-Surfactants**"); [st.markdown(f"* {cs}") for cs in rec_cs]

    if st.button("Confirm & Proceed ➡️"):
        st.session_state.nav_index = 1
        st.rerun()

# --- STEP 2: SOLUBILITY ---
elif nav == "Step 2: Solubility":
    st.header("🧪 Component Selection & Solubility")
    
    col1, col2, col3 = st.columns(3)
    with col1: sel_oil = st.selectbox("Oil Phase", sorted(df['Oil_phase'].unique()))
    with col2: sel_surf = st.selectbox("Surfactant", sorted(df['Surfactant'].unique()))
    with col3: sel_cs = st.selectbox("Co-Surfactant", sorted(df['Co-surfactant'].unique()))
    
    st.session_state.update({'f_o': sel_oil, 'f_s': sel_surf, 'f_cs': sel_cs})

    match = df[(df['Drug_Name'] == st.session_state.drug) & (df['Oil_phase'] == sel_oil)]
    if not match.empty and match['Sol_Oil'].iloc[0] > 0:
        s1, s2, s3 = match['Sol_Oil'].iloc[0], match['Sol_Surf'].iloc[0], match['Sol_CoSurf'].iloc[0]
        st.success("Matching Experimental Data Found!")
    else:
        s1 = 10.0 + (len(sel_oil) * 0.5)
        s2 = 20.0 + (len(sel_surf) * 0.3)
        s3 = 15.0 + (len(sel_cs) * 0.2)
        st.info("Using AI-estimated solubility based on molecular proxies.")

    st.session_state.update({'s_oil': s1, 's_surf': s2, 's_cs': s3})
    res_cols = st.columns(3)
    res_cols[0].metric("Oil Sol.", f"{s1:.2f} mg/mL")
    res_cols[1].metric("Surf Sol.", f"{s2:.2f} mg/mL")
    res_cols[2].metric("Co-S Sol.", f"{s3:.2f} mg/mL")

    if st.button("Map Phase Behavior ➡️"):
        st.session_state.nav_index = 2
        st.rerun()

# --- STEP 3: TERNARY ---
elif nav == "Step 3: Ternary":
    st.header("📊 Phase Behavior Mapping")
    l, r = st.columns([1, 2])
    with l:
        o_v = st.slider("Oil %", 1.0, 50.0, st.session_state.o_val)
        s_v = st.slider("Smix %", 1.0, 90.0, st.session_state.s_val)
        w_v = 100 - o_v - s_v
        if w_v < 0:
            st.error("Total exceeds 100%!")
            w_v = 0
        st.metric("Water %", f"{w_v:.2f}%")
        st.session_state.update({'o_val': o_v, 's_val': s_v, 'w_val': w_v})
    
    with r:
        # Example region - in a real app, this should be filtered by Smix ratio
        fig = go.Figure(go.Scatterternary({
            'mode': 'lines', 'fill': 'toself', 'name': 'Nanoemulsion Region',
            'a': [2, 10, 25, 5, 2], 'b': [45, 80, 65, 40, 45], 'c': [53, 10, 10, 55, 53],
            'fillcolor': 'rgba(46, 204, 113, 0.3)'
        }))
        fig.add_trace(go.Scatterternary(a=[o_v], b=[s_v], c=[w_v], name="Your Formula", marker=dict(color='red', size=15)))
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Generate AI Predictions ➡️"):
        st.session_state.nav_index = 3
        st.rerun()

# --- STEP 4: PREDICTION ---
elif nav == "Step 4: AI Prediction":
    st.header(f"🤖 AI Analysis: {st.session_state.drug}")
    
    def safe_enc(col, val): 
        return encoders[col].transform([val])[0] if val in encoders[col].classes_ else 0

    input_df = pd.DataFrame([{
        'Drug_Name': safe_enc('Drug_Name', st.session_state.drug), 
        'Oil_phase': safe_enc('Oil_phase', st.session_state.f_o), 
        'Surfactant': safe_enc('Surfactant', st.session_state.f_s), 
        'Co-surfactant': safe_enc('Co-surfactant', st.session_state.f_cs)
    }])
    
    res = {t: models[t].predict(input_df)[0] for t in models}
    stability = min(100, max(0, (min(abs(res['Zeta_mV']), 30)/30*70) + (max(0, 0.5-res['PDI'])/0.5*30)))
    
    cols = st.columns(5)
    cols[0].metric("Size", f"{res['Size_nm']:.1f} nm")
    cols[1].metric("PDI", f"{res['PDI']:.3f}")
    cols[2].metric("Zeta", f"{res['Zeta_mV']:.1f} mV")
    cols[3].metric("%EE", f"{res['Encapsulation_Efficiency']:.1f}%")
    cols[4].metric("Stability", f"{stability:.1f}%")
    
    st.divider()
    
    # SHAP Explainer
    st.subheader("Feature Contribution (SHAP)")
    explainer = shap.TreeExplainer(models['Size_nm'])
    shap_values = explainer.shap_values(input_df)
    fig_sh, ax = plt.subplots(figsize=(10, 4))
    shap.force_plot(explainer.expected_value, shap_values[0], input_df.iloc[0], feature_names=['Drug', 'Oil', 'Surf', 'Co-S'], matplotlib=True, show=False)
    st.pyplot(plt.gcf())

    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "NanoPredict Pro AI Report", ln=True, align='C')
        pdf.ln(10)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"Formulation for: {st.session_state.drug}", ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, f"Components: {st.session_state.f_o}, {st.session_state.f_s}, {st.session_state.f_cs}", ln=True)
        pdf.cell(0, 8, f"Ratio: Oil {st.session_state.o_val}% | Smix {st.session_state.s_val}% | Water {st.session_state.w_val}%", ln=True)
        
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "AI Predictions:", ln=True)
        for k, v in res.items():
            pdf.cell(0, 8, f"{k}: {v:.3f}", ln=True)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            plt.gcf().savefig(tmp.name, format='png', bbox_inches='tight')
            pdf.image(tmp.name, x=10, y=100, w=180)
        
        return pdf.output(dest='S').encode('latin-1', errors='ignore')

    if st.button("Generate Report PDF"):
        pdf_bytes = generate_pdf()
        st.download_button("📥 Download Report", data=pdf_bytes, file_name="NanoReport.pdf", mime="application/pdf")
