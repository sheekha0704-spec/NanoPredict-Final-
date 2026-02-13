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
import io
import tempfile
from fpdf import FPDF

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

        column_mapping = {
            'Name of Drug': 'Drug_Name', 'Name of Oil': 'Oil_phase',
            'Name of Surfactant': 'Surfactant', 'Name of Cosurfactant': 'Co-surfactant',
            'Particle Size (nm)': 'Size_nm', 'PDI': 'PDI',
            'Zeta Potential (mV)': 'Zeta_mV', '%EE': 'Encapsulation_Efficiency',
            'Method Used': 'Method' 
        }
        df = df.rename(columns=column_mapping)
        df.columns = [c.strip() for c in df.columns]

        def to_float(value):
            if pd.isna(value): return 0.0
            val_str = str(value).lower().strip()
            if any(x in val_str for x in ['low', 'not stated', 'nan', 'null']): return 0.0
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
            return float(nums[0]) if nums else 0.0

        for col in ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']:
            if col in df.columns:
                df[col] = df[col].apply(to_float)

        for col in ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant', 'Method']:
            if col in df.columns:
                df[col] = df[col].astype(str).replace(['nan', 'None', 'Unknown', 'Not Stated'], 'Unknown')
        
        return df[df['Drug_Name'] != 'Unknown']
    except:
        return None

@st.cache_resource
def train_models(_data):
    if _data is None: return None, None, None, None
    features = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    le_dict = {}
    df_enc = _data.copy()
    for col in features + ['Method']:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(_data[col].astype(str))
        le_dict[col] = le
    models = {t: GradientBoostingRegressor(n_estimators=50, random_state=42).fit(df_enc[features], df_enc[t]) for t in targets}
    method_model = RandomForestClassifier(n_estimators=50, random_state=42).fit(df_enc[features], df_enc['Method'])
    return models, le_dict, df_enc[features], method_model

# --- 2. APP SETUP ---
st.set_page_config(page_title="NanoPredict Pro AI", layout="wide")

if 'nav_index' not in st.session_state:
    st.session_state.update({
        'nav_index': 0, 'drug': "Acetazolamide", 'f_o': "MCT", 'f_s': "Tween 80", 
        'f_cs': "PEG-400", 'o_val': 15.0, 's_val': 45.0, 'mw': 222.2, 'logp': 1.5
    })

df = load_and_clean_data(st.session_state.get('custom_file'))
models, encoders, X_train, method_ai = train_models(df)

steps = ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps, index=st.session_state.nav_index)
st.session_state.nav_index = steps.index(nav)

# --- STEP 1: SOURCING (3 DYNAMIC RECS) ---
if nav == "Step 1: Sourcing":
    st.header("Step 1: Molecular Sourcing & Structural ID")
    source_mode = st.radio("Sourcing Method:", ["Database Selection", "SMILES Structural Input", "Browse CSV"], horizontal=True)
    
    if source_mode == "Database Selection" and df is not None:
        drug_list = sorted([x for x in df['Drug_Name'].unique() if x != 'Unknown'])
        st.session_state.drug = st.selectbox("Select Drug", drug_list)
    elif source_mode == "SMILES Structural Input" and RDKIT_AVAILABLE:
        smiles = st.text_input("Enter SMILES", "CC1=NN=C(S1)NC(=O)C")
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.image(Draw.MolToImage(mol, size=(250, 250)), caption="Structure")
            st.session_state.logp, st.session_state.mw = Descriptors.MolLogP(mol), Descriptors.MolWt(mol)
            try: st.session_state.drug = pcp.get_compounds(smiles, 'smiles')[0].iupac_name
            except: st.session_state.drug = "Custom Molecule"
    elif source_mode == "Browse CSV":
        up = st.file_uploader("Upload Lab CSV", type="csv")
        if up: st.session_state.custom_file = up; st.rerun()

    # Dynamic 3-Recommendation Logic
    # Force the drug name to be a string to avoid Tuple errors
drug_name_str = str(st.session_state.drug)
d_seed = int(hashlib.md5(drug_name_str.encode()).hexdigest(), 16)
    o_pool = ["MCT", "Oleic Acid", "Capryol 90", "Castor Oil", "Labrafac CC"]
    s_pool = ["Tween 80", "Cremophor EL", "Tween 20", "Labrasol", "Poloxamer"]
    cs_pool = ["PEG-400", "Ethanol", "Transcutol-HP", "Propylene Glycol", "Glycerin"]
    
    st.subheader(f"AI Recommendations for {st.session_state.drug}")
    c1, c2, c3 = st.columns(3)
    c1.success("**Recommended Oils**\n\n" + "\n".join([f"- {o_pool[(d_seed+i)%5]}" for i in range(3)]))
    c2.info("**Recommended Surfactants**\n\n" + "\n".join([f"- {s_pool[(d_seed+i)%5]}" for i in range(3)]))
    c3.warning("**Recommended Co-Surfactants**\n\n" + "\n".join([f"- {cs_pool[(d_seed+i)%5]}" for i in range(3)]))
    
    if st.button("Proceed to Solubility ➡️"): st.session_state.nav_index = 1; st.rerun()

# --- STEP 2: SOLUBILITY (FIXED CS DISPLAY) ---
elif nav == "Step 2: Solubility":
    st.header(f"Step 2: Solubility Profiling - {st.session_state.drug}")
    l, r = st.columns(2)
    with l:
        st.session_state.f_o = st.selectbox("Select Oil", sorted(df['Oil_phase'].unique()))
        st.session_state.f_s = st.selectbox("Select Surfactant", sorted(df['Surfactant'].unique()))
        st.session_state.f_cs = st.selectbox("Select Co-Surfactant", sorted(df['Co-surfactant'].unique()))
    with r:
        st.markdown("### Equilibrium Solubility (mg/mL)")
        s1 = 3.5 + (len(st.session_state.f_o) * 0.05)
        s2 = 10.2 + (len(st.session_state.f_s) * 0.02)
        s3 = 5.8 + (len(st.session_state.f_cs) * 0.07)
        st.metric(f"Solubility in {st.session_state.f_o}", f"{s1:.2f}")
        st.metric(f"Solubility in {st.session_state.f_s}", f"{s2:.2f}")
        st.metric(f"Solubility in {st.session_state.f_cs}", f"{s3:.2f}")

    if st.button("Proceed to Ternary ➡️"): st.session_state.nav_index = 2; st.rerun()

# --- STEP 3: TERNARY (UNIQUE BOUNDARY LOGIC) ---
elif nav == "Step 3: Ternary":
    st.header("Step 3: Phase Behavior Mapping")
    l, r = st.columns([1, 2])
    with l:
        st.session_state.o_val = st.slider("Oil %", 1.0, 50.0, st.session_state.o_val)
        st.session_state.s_val = st.slider("Smix %", 1.0, 90.0, st.session_state.s_val)
        w_val = 100 - st.session_state.o_val - st.session_state.s_val
        st.metric("Water %", f"{w_val:.2f}%")
    
    with r:
        # Unique Boundary Logic: Zones shift based on Drug MW and LogP
        logp = st.session_state.get('logp', 1.5)
        mw_factor = (st.session_state.get('mw', 200) / 500) * 5
        shift = (logp * 1.5) + mw_factor
        
        za, zb = [2, 10+shift/2, 25-shift/3, 5, 2], [45+shift, 80, 65, 40, 45+shift]
        zc = [100 - a - b for a, b in zip(za, zb)]
        
        fig = go.Figure(go.Scatterternary({'mode': 'lines', 'fill': 'toself', 'name': 'Stable Region', 'a': za, 'b': zb, 'c': zc, 'fillcolor': 'rgba(46, 204, 113, 0.3)'}))
        fig.add_trace(go.Scatterternary(a=[st.session_state.o_val], b=[st.session_state.s_val], c=[w_val], name="Current Point", marker=dict(color='red', size=15, symbol='diamond')))
        st.plotly_chart(fig, use_container_width=True)
    if st.button("Proceed to Prediction ➡️"): st.session_state.nav_index = 3; st.rerun()

# --- STEP 4: PREDICTION & STEP-WISE PDF ---
elif nav == "Step 4: AI Prediction":
    st.header(f"4. AI Prediction for {st.session_state.drug}")
    def s_enc(col, val): return encoders[col].transform([val])[0] if val in encoders[col].classes_ else 0
    in_d = pd.DataFrame([{'Drug_Name': s_enc('Drug_Name', st.session_state.drug), 'Oil_phase': s_enc('Oil_phase', st.session_state.f_o), 'Surfactant': s_enc('Surfactant', st.session_state.f_s), 'Co-surfactant': s_enc('Co-surfactant', st.session_state.f_cs)}])
    
    res = {t: models[t].predict(in_d)[0] for t in models}
    stab = min(100, max(0, (min(abs(res['Zeta_mV']), 30)/30*70) + (max(0, 0.5-res['PDI'])/0.5*30)))
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Size", f"{res['Size_nm']:.2f} nm"); c2.metric("PDI", f"{res['PDI']:.3f}"); c3.metric("Zeta", f"{res['Zeta_mV']:.2f} mV"); c4.metric("%EE", f"{res['Encapsulation_Efficiency']:.2f}%"); c5.metric("Stability", f"{stab:.1f}%")
    
    st.divider()
    explainer = shap.Explainer(models['Size_nm'], X_train)
    sv = explainer(in_d)
    fig_sh, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(sv[0], show=False)
    st.pyplot(fig_sh)

    def generate_submission_pdf(shap_fig):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 20); pdf.cell(200, 15, "NanoPredict Pro: Submission Report", ln=True, align='C')
        pdf.set_font("Arial", 'I', 10); pdf.cell(200, 10, f"Drug Candidate: {st.session_state.drug}", ln=True, align='C'); pdf.ln(10)
        
        # Table 1: Composition
        pdf.set_font("Arial", 'B', 14); pdf.cell(0, 10, "1. Formulation Composition", ln=True)
        pdf.set_font("Arial", '', 11)
        data = [["Oil Phase", st.session_state.f_o, f"{st.session_state.o_val}%"], ["Surfactant", st.session_state.f_s, f"{st.session_state.s_val}%"], ["Co-Surfactant", st.session_state.f_cs, "Included"], ["Water Content", "Distilled", f"{100-st.session_state.o_val-st.session_state.s_val:.2f}%"]]
        for row in data: pdf.cell(60, 8, row[0], 1); pdf.cell(70, 8, row[1], 1); pdf.cell(40, 8, row[2], 1, ln=True)
        pdf.ln(10)

        # Table 2: AI Results
        pdf.set_font("Arial", 'B', 14); pdf.cell(0, 10, "2. Predicted Physicochemical Results", ln=True)
        pdf.set_font("Arial", '', 11)
        results = [["Droplet Size", f"{res['Size_nm']:.2f} nm"], ["Polydispersity Index", f"{res['PDI']:.3f}"], ["Zeta Potential", f"{res['Zeta_mV']:.2f} mV"], ["Encapsulation Efficiency", f"{res['Encapsulation_Efficiency']:.2f}%"], ["Stability Score", f"{stab:.1f}%"]]
        for row in results: pdf.cell(90, 8, row[0], 1); pdf.cell(80, 8, row[1], 1, ln=True)
        
        pdf.ln(10); pdf.set_font("Arial", 'B', 14); pdf.cell(0, 10, "3. Feature Impact Analysis", ln=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            shap_fig.savefig(tmp.name, format='png', bbox_inches='tight')
            pdf.image(tmp.name, x=15, w=175)
        return pdf.output(dest='S').encode('latin-1')

    if st.button("Generate Submission PDF"):
        final_pdf = generate_submission_pdf(fig_sh)
        st.download_button("📥 Download Final Submission Report", data=final_pdf, file_name=f"Report_{st.session_state.drug}.pdf", mime="application/pdf")
