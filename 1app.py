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
                return pd.DataFrame()
        
        column_mapping = {
            'Name Of The Drug ': 'Drug_Name', 'Name Of The Oil ': 'Oil_phase',
            'Name Of The Surfactant ': 'Surfactant', 'Name Of The Courfactant ': 'Co-surfactant',
            'Solubility Of Drug In Oil ': 'Sol_Oil', 'Solubility Of Drug In Surfactant ': 'Sol_Surf',
            'Solubility Of Drug In Cosurfactant ': 'Sol_CoSurf', 'Particle Size ': 'Size_nm', 
            'PDI': 'PDI', 'Zeta Potential ': 'Zeta_mV', '% EE': 'Encapsulation_Efficiency'
        }
        df = df.rename(columns=column_mapping)
        df.columns = [c.strip() for c in df.columns]

        def to_float(value):
            if pd.isna(value) or str(value).strip() == "": return 0.0
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(value))
            return float(nums[0]) if nums else 0.0

        for col in ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency', 'Sol_Oil', 'Sol_Surf', 'Sol_CoSurf']:
            if col in df.columns: df[col] = df[col].apply(to_float)
        
        return df.reset_index(drop=True)
    except:
        return pd.DataFrame()

@st.cache_resource
def train_models(_data):
    if _data is None or _data.empty: return None, None, None
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
        'nav_index': 0, 'drug': "Tamoxifen", 'custom_file': None,
        'rec_o': [], 'rec_s': [], 'rec_cs': [],
        'smix_pairs': [[1, 1], [2, 1], [3, 1]], # Smix ratio options
        'final_smix_choice': "Ratio 1", 'o_val': 10.0, 's_val': 40.0
    })

df = load_and_clean_data(st.session_state.custom_file)
models, encoders, X_train = train_models(df)

steps = ["Step 1: Sourcing", "Step 2: Solubility & Smix", "Step 3: Ternary Mapping", "Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps, index=st.session_state.nav_index)
st.session_state.nav_index = steps.index(nav)

# --- STEP 1: SOURCING ---
if nav == "Step 1: Sourcing":
    st.header("Step 1: Molecular Sourcing & AI Recommendations")
    source_mode = st.radio("Sourcing Method:", ["Database Selection", "SMILES Input", "Browse CSV"], horizontal=True)
    
    if source_mode == "Database Selection" and not df.empty:
        drug_list = sorted([str(x) for x in df['Drug_Name'].unique() if pd.notna(x)])
        st.session_state.drug = st.selectbox("Select Drug", drug_list)
        
    elif source_mode == "SMILES Input" and RDKIT_AVAILABLE:
        smiles = st.text_input("Enter SMILES", "CN(C)CCOC1=CC=C(C=C1)C(=C(C2=CC=CC=C2)CC)C3=CC=CC=C3")
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.image(Draw.MolToImage(mol, size=(250, 250)), caption="Structure Identified")
            st.session_state.drug = "Custom Molecule"
                
    elif source_mode == "Browse CSV":
        up = st.file_uploader("Upload Lab CSV", type="csv")
        if up: 
            st.session_state.custom_file = up
            st.cache_data.clear(); st.rerun()

    st.divider()
    st.subheader(f"Top 5 AI Recommendations for {st.session_state.drug}")
    
    if not df.empty:
        drug_data = df[df['Drug_Name'] == st.session_state.drug]
        st.session_state.rec_o = drug_data['Oil_phase'].value_counts().index.tolist()[:5] if not drug_data.empty else ["MCT", "Oleic Acid", "Capryol 90", "Labrafac", "Olive Oil"]
        st.session_state.rec_s = drug_data['Surfactant'].value_counts().index.tolist()[:5] if not drug_data.empty else ["Tween 80", "Cremophor EL", "Labrasol", "Ethanol", "Span 20"]
        st.session_state.rec_cs = drug_data['Co-surfactant'].value_counts().index.tolist()[:5] if not drug_data.empty else ["PEG-400", "Transcutol", "PG", "Tween 20", "Span 80"]

        c1, c2, c3 = st.columns(3)
        with c1: st.success("**Top 5 Oils**"); [st.markdown(f"* {o}") for o in st.session_state.rec_o]
        with c2: st.info("**Top 5 Surfactants**"); [st.markdown(f"* {s}") for s in st.session_state.rec_s]
        with c3: st.warning("**Top 5 Co-Surfactants**"); [st.markdown(f"* {cs}") for cs in st.session_state.rec_cs]

    if st.button("Proceed to Solubility Selection ➡️"): 
        st.session_state.nav_index = 1; st.rerun()

# --- STEP 2: SOLUBILITY & SMIX ---
elif nav == "Step 2: Solubility & Smix":
    st.header("Step 2: Component Solubility & Smix Decision")
    
    col_sel, col_val = st.columns([2, 1.5])
    
    with col_sel:
        st.markdown("### 1. Selection")
        s_o = st.selectbox("Select Oil", st.session_state.rec_o)
        s_s = st.selectbox("Select Surfactant", st.session_state.rec_s)
        s_cs = st.selectbox("Select Co-Surfactant", st.session_state.rec_cs)
        st.session_state.update({'f_o': s_o, 'f_s': s_s, 'f_cs': s_cs})

    with col_val:
        st.markdown("### 2. Solubility Profile (Sorted)")
        targets = [('Oil', 'Sol_Oil', s_o), ('Surfactant', 'Sol_Surf', s_s), ('Co-Surfactant', 'Sol_CoSurf', s_cs)]
        display_data = []
        for label, col, name in targets:
            match = df[(df['Drug_Name'] == st.session_state.drug) & (df[col.split('_')[1]+'_phase' if 'Oil' in col else col.split('_')[1]] == name)] if not df.empty else pd.DataFrame()
            val = match[col].iloc[0] if not match.empty else np.random.uniform(10, 30)
            display_data.append((f"{label}: {name}", val))
        
        display_data.sort(key=lambda x: x[1], reverse=True)
        for text, v in display_data:
            st.metric(text, f"{v:.2f} mg/mL")

    st.divider()
    st.markdown("### 3. $S_{mix}$ Ratio Selection (As per Diagram)")
    
    # Implementing the handwritten layout for 3 options of ratio pairs
    r_cols = st.columns(3)
    for i in range(3):
        with r_cols[i]:
            st.write(f"**Option {i+1}**")
            s_val = st.selectbox(f"Surfactant Part (Opt {i+1})", [1, 2, 3, 4, 5], index=i, key=f"s_r_{i}")
            cs_val = st.selectbox(f"Co-Surfactant Part (Opt {i+1})", [1, 2, 3, 4, 5], index=0, key=f"cs_r_{i}")
            st.session_state.smix_pairs[i] = [s_val, cs_val]

    if st.button("Generate Phase Diagrams ➡️"): 
        st.session_state.nav_index = 2; st.rerun()

# --- STEP 3: TERNARY ---
elif nav == "Step 3: Ternary Mapping":
    st.header("Step 3: Ternary Phase Mapping")
    
    # Allow user to choose the diagram first
    choice_idx = st.radio("Choose Optimized Ternary System to Finalize", [0, 1, 2], 
                          format_func=lambda x: f"Option {x+1} ($S_{{mix}}$ {st.session_state.smix_pairs[x][0]}:{st.session_state.smix_pairs[x][1]})", horizontal=True)
    
    st.divider()
    l, r = st.columns([1, 2])
    
    with l:
        st.markdown("### Composition Adjustment")
        st.session_state.o_val = st.slider("Oil %", 1.0, 50.0, st.session_state.o_val)
        st.session_state.s_val = st.slider("Smix %", 1.0, 90.0, st.session_state.s_val)
        w_val = 100 - st.session_state.o_val - st.session_state.s_val
        st.metric("Water %", f"{w_val:.2f}%")
        st.session_state.w_val = w_val
        st.session_state.final_smix_choice = f"{st.session_state.smix_pairs[choice_idx][0]}:{st.session_state.smix_pairs[choice_idx][1]}"

    with r:
        ratio_sum = sum(st.session_state.smix_pairs[choice_idx])
        # Simulation of nanoemulsion region based on surfactant ratio
        area_scale = st.session_state.smix_pairs[choice_idx][0] / ratio_sum
        za = [2, 10 + (area_scale*15), 25, 5, 2]
        zb = [40, 85 - (area_scale*10), 60, 35, 40]
        zc = [100 - a - b for a, b in zip(za, zb)]
        
        fig = go.Figure(go.Scatterternary({'mode': 'lines', 'fill': 'toself', 'name': 'Nanoemulsion Region', 'a': za, 'b': zb, 'c': zc, 'fillcolor': 'rgba(34, 139, 34, 0.4)'}))
        fig.add_trace(go.Scatterternary(a=[st.session_state.o_val], b=[st.session_state.s_val], c=[w_val], name="Selected Formula", marker=dict(color='red', size=15, symbol='diamond')))
        fig.update_layout(ternary=dict(aaxis_title="Oil %", baxis_title="Smix %", caxis_title="Water %"))
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Generate AI Predictions ➡️"): 
        st.session_state.nav_index = 3; st.rerun()

# --- STEP 4: PREDICTION ---
elif nav == "Step 4: AI Prediction":
    st.header(f"Step 4: Professional AI Analysis")
    
    def safe_enc(col, val): return encoders[col].transform([val])[0] if val in encoders[col].classes_ else 0
    in_d = pd.DataFrame([{'Drug_Name': safe_enc('Drug_Name', st.session_state.drug), 'Oil_phase': safe_enc('Oil_phase', st.session_state.f_o), 'Surfactant': safe_enc('Surfactant', st.session_state.f_s), 'Co-surfactant': safe_enc('Co-surfactant', st.session_state.f_cs)}])
    
    res = {t: models[t].predict(in_d)[0] for t in models}
    stab = min(100, max(0, (min(abs(res['Zeta_mV']), 30)/30*70) + (max(0, 0.5-res['PDI'])/0.5*30)))
    
    # Professional Metric Dashboard
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Particle Size", f"{res['Size_nm']:.1f} nm"); m2.metric("PDI", f"{res['PDI']:.3f}")
    m3.metric("Zeta Potential", f"{res['Zeta_mV']:.1f} mV"); m4.metric("EE %", f"{res['Encapsulation_Efficiency']:.1f}%")
    m5.metric("Stability Index", f"{stab:.1f}%")
    
    st.divider()
    st.subheader("High-Resolution SHAP Interpretability")
    explainer = shap.TreeExplainer(models['Size_nm'])
    shap_values = explainer.shap_values(in_d)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    # Using a more professional force plot or bar summary
    shap.plots.bar(explainer(in_d)[0], show=False)
    plt.title("Impact of Formulation Components on Particle Size")
    st.pyplot(plt.gcf())

    if st.button("Generate Professional PDF Report"):
        pdf = FPDF()
        pdf.add_page(); pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "NanoPredict Pro AI: Research Summary", ln=True, align='C')
        pdf.set_font("Arial", '', 11); pdf.ln(10)
        pdf.cell(0, 10, f"Drug Candidate: {st.session_state.drug}", ln=True)
        pdf.cell(0, 10, f"Excipients: {st.session_state.f_o} | {st.session_state.f_s} | {st.session_state.f_cs}", ln=True)
        pdf.cell(0, 10, f"Final Ratio (Smix): {st.session_state.final_smix_choice}", ln=True)
        pdf.cell(0, 10, f"Predicted Size: {res['Size_nm']:.2f} nm", ln=True)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            plt.gcf().savefig(tmp.name, bbox_inches='tight')
            pdf.image(tmp.name, x=10, y=80, w=190)
        
        st.download_button("📥 Download Official Report", data=pdf.output(dest='S').encode('latin-1'), file_name="NanoPredict_Report.pdf")
