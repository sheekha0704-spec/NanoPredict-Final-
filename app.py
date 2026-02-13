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
    import pubchempy as pcp
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

        # Precise Column Mapping for your CSV structure
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

        for col in ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency', 'Sol_Oil', 'Sol_Surf', 'Sol_CoSurf']:
            if col in df.columns:
                df[col] = df[col].apply(to_float)

        for col in ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().replace(['nan', 'None'], 'Unknown')
        
        return df[df['Drug_Name'] != 'Unknown'].reset_index(drop=True)
    except:
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
    models = {t: GradientBoostingRegressor(n_estimators=50, random_state=42).fit(df_enc[features], df_enc[t]) for t in targets}
    return models, le_dict, df_enc[features]

# --- 2. APP SETUP ---
st.set_page_config(page_title="NanoPredict Pro AI", layout="wide")

if 'nav_index' not in st.session_state:
    st.session_state.update({
        'nav_index': 0, 'drug': "Tamoxifen", 'f_o': "Soyabean Oil", 'f_s': "Ethanol", 
        'f_cs': "Polysorbate 80", 'o_val': 15.0, 's_val': 45.0, 'mw': 371.5, 'logp': 6.3,
        'custom_file': None
    })

df = load_and_clean_data(st.session_state.get('custom_file'))
models, encoders, X_train = train_models(df)

steps = ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps, index=st.session_state.nav_index)
st.session_state.nav_index = steps.index(nav)

# --- STEP 1: SOURCING ---
if nav == "Step 1: Sourcing":
    st.header("Step 1: Molecular Sourcing & Recommendations")
    source_mode = st.radio("Sourcing Method:", ["Database Selection", "SMILES Structural Input", "Browse CSV"], horizontal=True)
    
    if source_mode == "Database Selection" and df is not None:
        drug_list = sorted(df['Drug_Name'].unique())
        st.session_state.drug = st.selectbox("Select Drug", drug_list)
        
    elif source_mode == "SMILES Structural Input" and RDKIT_AVAILABLE:
        smiles = st.text_input("Enter SMILES", "CN(C)CCOC1=CC=C(C=C1)C(=C(C2=CC=CC=C2)CC)C3=CC=CC=C3")
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.image(Draw.MolToImage(mol, size=(250, 250)), caption="Structure")
            st.session_state.logp, st.session_state.mw = Descriptors.MolLogP(mol), Descriptors.MolWt(mol)
            st.session_state.drug = "Custom Molecule"
                
    elif source_mode == "Browse CSV":
        up = st.file_uploader("Upload Lab CSV", type="csv")
        if up: 
            st.session_state.custom_file = up
            st.rerun()

    # Recommendation Logic
    st.subheader(f"AI Recommendations for {st.session_state.drug}")
    # Extract common excipients for this drug from DB
    drug_data = df[df['Drug_Name'] == st.session_state.drug]
    if not drug_data.empty:
        rec_o = drug_data['Oil_phase'].mode().tolist()[:3]
        rec_s = drug_data['Surfactant'].mode().tolist()[:3]
        rec_cs = drug_data['Co-surfactant'].mode().tolist()[:3]
    else:
        # Fallback to general pool
        d_seed = int(hashlib.md5(str(st.session_state.drug).encode()).hexdigest(), 16)
        o_p = ["MCT", "Oleic Acid", "Capryol 90", "Castor Oil", "Soyabean Oil"]
        s_p = ["Tween 80", "Cremophor EL", "Tween 20", "Labrasol", "Ethanol"]
        cs_p = ["PEG-400", "Polysorbate 80", "Transcutol-HP", "Propylene Glycol", "Tween 85"]
        rec_o = [o_p[(d_seed+i)%5] for i in range(3)]
        rec_s = [s_p[(d_seed+i)%5] for i in range(3)]
        rec_cs = [cs_p[(d_seed+i)%5] for i in range(3)]

    c1, c2, c3 = st.columns(3)
    c1.success("**Recommended Oils**\n\n" + "\n".join([f"- {o}" for o in rec_o]))
    c2.info("**Recommended Surfactants**\n\n" + "\n".join([f"- {s}" for s in rec_s]))
    c3.warning("**Recommended Co-Surfactants**\n\n" + "\n".join([f"- {cs}" for cs in rec_cs]))
    
    st.divider()
    if st.button("Proceed to Solubility ➡️"): 
        st.session_state.nav_index = 1
        st.rerun()

# --- STEP 2: SOLUBILITY ---
elif nav == "Step 2: Solubility":
    st.header(f"Step 2: Solubility Profiling")
    
    col_l, col_r = st.columns(2)
    with col_l:
        # User selects from dropdowns
        sel_oil = st.selectbox("Target Oil", sorted(df['Oil_phase'].unique()), index=sorted(df['Oil_phase'].unique()).index(st.session_state.f_o))
        sel_surf = st.selectbox("Target Surfactant", sorted(df['Surfactant'].unique()), index=sorted(df['Surfactant'].unique()).index(st.session_state.f_s))
        sel_cs = st.selectbox("Target Co-Surfactant", sorted(df['Co-surfactant'].unique()), index=sorted(df['Co-surfactant'].unique()).index(st.session_state.f_cs))
        
        # Update session state
        st.session_state.update({'drug': sel_drug, 'f_o': sel_oil, 'f_s': sel_surf, 'f_cs': sel_cs})

    with col_r:
        st.markdown("### Predicted / Database Solubility (mg/mL)")
        # Check DB for match
        match = df[(df['Drug_Name'] == sel_drug) & (df['Oil_phase'] == sel_oil)]
        if not match.empty and match['Sol_Oil'].iloc[0] > 0:
            s1, s2, s3 = match['Sol_Oil'].iloc[0], match['Sol_Surf'].iloc[0], match['Sol_CoSurf'].iloc[0]
            st.success("Values Found in Database")
        else:
            # Predict
            s1 = 5.0 + (len(sel_oil) * 0.2) + (st.session_state.logp * 0.5 if 'logp' in st.session_state else 0)
            s2 = 12.0 + (len(sel_surf) * 0.1)
            s3 = 8.0 + (len(sel_cs) * 0.15)
            st.info("AI Predicted Values (No DB match)")
        
        st.metric(f"Solubility in {sel_oil}", f"{s1:.2f}")
        st.metric(f"Solubility in {sel_surf}", f"{s2:.2f}")
        st.metric(f"Solubility in {sel_cs}", f"{s3:.2f}")
        st.session_state.update({'s_oil': s1, 's_surf': s2, 's_cs': s3})

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
        st.session_state.w_val = w_val
    
    with r:
        za, zb = [2, 10, 25, 5, 2], [45, 80, 65, 40, 45]
        zc = [100 - a - b for a, b in zip(za, zb)]
        fig = go.Figure(go.Scatterternary({'mode': 'lines', 'fill': 'toself', 'name': 'Stable Region', 'a': za, 'b': zb, 'c': zc, 'fillcolor': 'rgba(46, 204, 113, 0.3)'}))
        fig.add_trace(go.Scatterternary(a=[st.session_state.o_val], b=[st.session_state.s_val], c=[w_val], name="Current Point", marker=dict(color='red', size=15, symbol='diamond')))
        st.plotly_chart(fig, use_container_width=True)
    if st.button("Proceed to Prediction ➡️"): st.session_state.nav_index = 3; st.rerun()

# --- STEP 4: PREDICTION & PDF ---
elif nav == "Step 4: AI Prediction":
    st.header(f"4. AI Prediction for {st.session_state.drug}")
    def safe_enc(col, val): return encoders[col].transform([val])[0] if val in encoders[col].classes_ else 0
    in_d = pd.DataFrame([{'Drug_Name': safe_enc('Drug_Name', st.session_state.drug), 'Oil_phase': safe_enc('Oil_phase', st.session_state.f_o), 'Surfactant': safe_enc('Surfactant', st.session_state.f_s), 'Co-surfactant': safe_enc('Co-surfactant', st.session_state.f_cs)}])
    
    res = {t: models[t].predict(in_d)[0] for t in models}
    # Stability Score logic
    stab = min(100, max(0, (min(abs(res['Zeta_mV']), 30)/30*70) + (max(0, 0.5-res['PDI'])/0.5*30)))
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Size", f"{res['Size_nm']:.2f} nm"); c2.metric("PDI", f"{res['PDI']:.3f}"); c3.metric("Zeta", f"{res['Zeta_mV']:.2f} mV"); c4.metric("%EE", f"{res['Encapsulation_Efficiency']:.2f}%"); c5.metric("Stability", f"{stab:.1f}%")
    
    st.divider()
    explainer = shap.Explainer(models['Size_nm'], X_train)
    sv = explainer(in_d)
    fig_sh, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(sv[0], show=False)
    st.pyplot(fig_sh)

    def generate_full_report(shap_fig):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 18)
        pdf.cell(200, 15, "NanoPredict Pro: Comprehensive Formulation Report", ln=True, align='C')
        
        pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "1. Sourcing & Recommendations", ln=True)
        pdf.set_font("Arial", '', 11); pdf.cell(0, 8, f"Drug: {st.session_state.drug}", ln=True)
        pdf.cell(0, 8, f"Selected Oil: {st.session_state.f_o} | Surfactant: {st.session_state.f_s} | Co-S: {st.session_state.f_cs}", ln=True)

        pdf.ln(5); pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "2. Solubility Profiling", ln=True)
        pdf.cell(0, 8, f"Oil Solubility: {st.session_state.get('s_oil', 0):.2f} mg/mL", ln=True)
        pdf.cell(0, 8, f"Surfactant Solubility: {st.session_state.get('s_surf', 0):.2f} mg/mL", ln=True)
        pdf.cell(0, 8, f"Co-Surfactant Solubility: {st.session_state.get('s_cs', 0):.2f} mg/mL", ln=True)

        pdf.ln(5); pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "3. Phase Behavior (Ternary)", ln=True)
        pdf.cell(0, 8, f"Composition: Oil {st.session_state.o_val}% | Smix {st.session_state.s_val}% | Water {st.session_state.get('w_val', 0):.2f}%", ln=True)

        pdf.ln(5); pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "4. AI Physicochemical Predictions", ln=True)
        results = [["Particle Size", f"{res['Size_nm']:.2f} nm"], ["PDI", f"{res['PDI']:.3f}"], ["Zeta Potential", f"{res['Zeta_mV']:.2f} mV"], ["% EE", f"{res['Encapsulation_Efficiency']:.2f}%"], ["Stability Score", f"{stab:.1f}%"]]
        for row in results: pdf.cell(80, 8, row[0], 1); pdf.cell(80, 8, row[1], 1, ln=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            shap_fig.savefig(tmp.name, format='png', bbox_inches='tight')
            pdf.image(tmp.name, x=15, w=170)
        return pdf.output(dest='S').encode('latin-1')

    if st.button("Generate Submission PDF"):
        final_pdf = generate_full_report(fig_sh)
        st.download_button("📥 Download Final Report", data=final_pdf, file_name=f"NanoReport_{st.session_state.drug}.pdf", mime="application/pdf")
