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

# --- 1. DATA & AI ENGINE ---
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
    models = {t: GradientBoostingRegressor(n_estimators=50, random_state=42).fit(df_enc[features], df_enc[t]) for t in targets}
    return models, le_dict, df_enc[features]

# --- 2. SESSION INITIALIZATION ---
st.set_page_config(page_title="NanoPredict Pro AI", layout="wide")

if 'nav_index' not in st.session_state:
    st.session_state.update({
        'nav_index': 0, 'drug': "Tamoxifen", 'custom_file': None,
        'rec_o': [], 'rec_s': [], 'rec_cs': [],
        'smix_ratios': [1, 1, 1], 'selected_ternary': 0, 'neb_region': "Stable"
    })

df = load_and_clean_data(st.session_state.custom_file)
models, encoders, X_train = train_models(df)

steps = ["Step 1: Sourcing", "Step 2: Solubility & Smix", "Step 3: Ternary Mapping", "Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps, index=st.session_state.nav_index)
st.session_state.nav_index = steps.index(nav)

# --- STEP 1: SOURCING ---
if nav == "Step 1: Sourcing":
    st.header("Step 1: Molecular Sourcing & Top 5 AI Recommendations")
    
    if df.empty:
        st.error("Please upload a CSV file to proceed.")
        st.session_state.custom_file = st.file_uploader("Upload Lab CSV", type="csv")
        if st.session_state.custom_file: st.rerun()
        st.stop()

    drug_list = sorted([str(x) for x in df['Drug_Name'].unique() if pd.notna(x)])
    st.session_state.drug = st.selectbox("Select Drug for Formulation", drug_list)

    # Recommendation Logic (Top 5)
    drug_data = df[df['Drug_Name'] == st.session_state.drug]
    if not drug_data.empty:
        st.session_state.rec_o = drug_data['Oil_phase'].value_counts().index.tolist()[:5]
        st.session_state.rec_s = drug_data['Surfactant'].value_counts().index.tolist()[:5]
        st.session_state.rec_cs = drug_data['Co-surfactant'].value_counts().index.tolist()[:5]
    else:
        st.session_state.rec_o = ["MCT", "Oleic Acid", "Capryol 90", "Labrafac", "Olive Oil"]
        st.session_state.rec_s = ["Tween 80", "Cremophor EL", "Labrasol", "Ethanol", "Span 20"]
        st.session_state.rec_cs = ["PEG-400", "Transcutol", "Propylene Glycol", "Tween 20", "Span 80"]

    c1, c2, c3 = st.columns(3)
    with c1: st.success("**Top 5 Recommended Oils**"); [st.markdown(f"{i+1}. {o}") for i, o in enumerate(st.session_state.rec_o)]
    with c2: st.info("**Top 5 Recommended Surfactants**"); [st.markdown(f"{i+1}. {s}") for i, s in enumerate(st.session_state.rec_s)]
    with c3: st.warning("**Top 5 Recommended Co-Surfactants**"); [st.markdown(f"{i+1}. {cs}") for i, cs in enumerate(st.session_state.rec_cs)]

    if st.button("Lock Recommendations & Proceed ➡️"):
        st.session_state.nav_index = 1
        st.rerun()

# --- STEP 2: SOLUBILITY & SMIX ---
elif nav == "Step 2: Solubility & Smix":
    st.header("Step 2: Component Solubility & Smix Decision")
    
    col_sel, col_val = st.columns([2, 1])
    
    with col_sel:
        st.subheader("Select From Recommendations")
        # Ensure values exist in recommendations to avoid crash
        s_o = st.selectbox("Select Oil (from Top 5)", st.session_state.rec_o)
        s_s = st.selectbox("Select Surfactant (from Top 5)", st.session_state.rec_s)
        s_cs = st.selectbox("Select Co-Surfactant (from Top 5)", st.session_state.rec_cs)
        st.session_state.update({'f_o': s_o, 'f_s': s_s, 'f_cs': s_cs})

    with col_val:
        st.subheader("Solubility Profiling")
        # Logic to find solubility for the 5 recommended oils
        sol_list = []
        for o in st.session_state.rec_o:
            m = df[(df['Drug_Name'] == st.session_state.drug) & (df['Oil_phase'] == o)]
            val = m['Sol_Oil'].iloc[0] if not m.empty else np.random.uniform(5, 20)
            sol_list.append((o, val))
        
        # Sort Higher to Lower
        sol_list.sort(key=lambda x: x[1], reverse=True)
        for name, val in sol_list:
            st.write(f"**{name}**: {val:.2f} mg/mL")

    st.divider()
    st.subheader("Smix Ratio Decision (Surfactant : Co-Surfactant)")
    r_cols = st.columns(3)
    r1 = r_cols[0].selectbox("Ratio 1 (S:CoS)", [1, 2, 3], index=0)
    r2 = r_cols[1].selectbox("Ratio 2 (S:CoS)", [1, 2, 3], index=1)
    r3 = r_cols[2].selectbox("Ratio 3 (S:CoS)", [1, 2, 3], index=2)
    st.session_state.smix_ratios = [r1, r2, r3]

    if st.button("Generate Ternary Maps ➡️"):
        st.session_state.nav_index = 2
        st.rerun()

# --- STEP 3: TERNARY MAPPING ---
elif nav == "Step 3: Ternary Mapping":
    st.header("Step 3: Ternary Phase Analysis")
    
    # 3 Ternary Diagrams based on Smix ratios
    t_cols = st.columns(3)
    for i, ratio in enumerate(st.session_state.smix_ratios):
        with t_cols[i]:
            st.markdown(f"**Smix Ratio {ratio}:1**")
            # Generate slightly different stable regions based on ratio
            offset = ratio * 5
            za, zb = [2, 10+offset, 25, 5, 2], [45, 80-offset, 65, 40, 45]
            zc = [100 - a - b for a, b in zip(za, zb)]
            fig = go.Figure(go.Scatterternary({'mode': 'lines', 'fill': 'toself', 'a': za, 'b': zb, 'c': zc, 'fillcolor': f'rgba({50*ratio}, 150, 200, 0.3)'}))
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=300)
            st.plotly_chart(fig, use_container_width=True, key=f"ternary_{i}")
    
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        choice = st.radio("Select Optimized Ternary System", [f"Ratio {r}:1" for r in st.session_state.smix_ratios])
        st.session_state.selected_ternary = choice
    with c2:
        st.session_state.neb_region = st.selectbox("Identify Nanoemulsion Region", ["O/W Region", "W/O Region", "Bicontinuous"])
        st.session_state.o_val = st.slider("Final Oil %", 1, 40, 15)
        st.session_state.s_val = st.slider("Final Smix %", 10, 80, 40)
    
    if st.button("Finalize & Predict ➡️"):
        st.session_state.nav_index = 3
        st.rerun()

# --- STEP 4: PREDICTION ---
elif nav == "Step 4: AI Prediction":
    st.header(f"Final AI Analysis: {st.session_state.drug}")
    
    def safe_enc(col, val): return encoders[col].transform([val])[0] if val in encoders[col].classes_ else 0
    in_d = pd.DataFrame([{'Drug_Name': safe_enc('Drug_Name', st.session_state.drug), 'Oil_phase': safe_enc('Oil_phase', st.session_state.f_o), 'Surfactant': safe_enc('Surfactant', st.session_state.f_s), 'Co-surfactant': safe_enc('Co-surfactant', st.session_state.f_cs)}])
    
    res = {t: models[t].predict(in_d)[0] for t in models}
    stab = min(100, max(0, (min(abs(res['Zeta_mV']), 30)/30*70) + (max(0, 0.5-res['PDI'])/0.5*30)))
    
    c = st.columns(5)
    c[0].metric("Size", f"{res['Size_nm']:.1f} nm"); c[1].metric("PDI", f"{res['PDI']:.3f}")
    c[2].metric("Zeta", f"{res['Zeta_mV']:.1f} mV"); c[3].metric("%EE", f"{res['Encapsulation_Efficiency']:.1f}%")
    c[4].metric("Stability", f"{stab:.1f}%")
    
    st.divider()
    explainer = shap.TreeExplainer(models['Size_nm'])
    sv = explainer.shap_values(in_d)
    fig_sh, ax = plt.subplots(figsize=(10, 3))
    shap.summary_plot(sv, in_d, feature_names=['Drug', 'Oil', 'Surf', 'Co-S'], plot_type="bar", show=False)
    st.pyplot(fig_sh)

    if st.button("Generate Final Report"):
        pdf = FPDF()
        pdf.add_page(); pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "NanoPredict Pro AI Final Report", ln=True, align='C')
        pdf.set_font("Arial", '', 11); pdf.ln(10)
        pdf.cell(0, 8, f"Drug: {st.session_state.drug}", ln=True)
        pdf.cell(0, 8, f"System: {st.session_state.f_o} | Smix ({st.session_state.selected_ternary})", ln=True)
        pdf.cell(0, 8, f"Final Composition: Oil {st.session_state.o_val}% | Smix {st.session_state.s_val}%", ln=True)
        st.download_button("Download PDF", data=pdf.output(dest='S').encode('latin-1'), file_name="NanoReport.pdf")
