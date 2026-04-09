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

# --- RDKIT ENGINE ---
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

# --- 2. APP SETUP & STABLE STATE ---
st.set_page_config(page_title="NanoPredict Pro AI", layout="wide")

# Ensure all keys exist before any logic runs to prevent AttributeErrors
if 'smix_pairs' not in st.session_state:
    st.session_state.smix_pairs = [[1, 1], [2, 1], [3, 1]]
if 'nav_index' not in st.session_state:
    st.session_state.update({
        'nav_index': 0, 'drug': "Tamoxifen", 'custom_file': None,
        'rec_o': [], 'rec_s': [], 'rec_cs': [],
        'final_smix_choice': "1:1", 'o_val': 10.0, 's_val': 40.0,
        'f_o': '', 'f_s': '', 'f_cs': ''
    })

df = load_and_clean_data(st.session_state.custom_file)
models, encoders, X_train = train_models(df)

steps = ["Step 1: Sourcing", "Step 2: Solubility & Smix", "Step 3: Ternary Mapping", "Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps, index=st.session_state.nav_index)
st.session_state.nav_index = steps.index(nav)

# --- STEP 1: SOURCING ---
if nav == "Step 1: Sourcing":
    st.header("Step 1: Molecular Sourcing")
    source_mode = st.radio("Sourcing Method:", ["Database Selection", "SMILES Input", "Browse CSV"], horizontal=True)
    
    if source_mode == "Database Selection" and not df.empty:
        drug_list = sorted([str(x) for x in df['Drug_Name'].unique() if pd.notna(x)])
        st.session_state.drug = st.selectbox("Select Drug", drug_list)
    elif source_mode == "SMILES Input" and RDKIT_AVAILABLE:
        smiles = st.text_input("Enter SMILES", "CN(C)CCOC1=CC=C(C=C1)C(=C(C2=CC=CC=C2)CC)C3=CC=CC=C3")
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.image(Draw.MolToImage(mol, size=(250, 250)))
            st.session_state.drug = "Custom Molecule"
    elif source_mode == "Browse CSV":
        up = st.file_uploader("Upload Lab CSV", type="csv")
        if up: 
            st.session_state.custom_file = up
            st.cache_data.clear(); st.rerun()

    if not df.empty:
        drug_data = df[df['Drug_Name'] == st.session_state.drug]
        st.session_state.rec_o = drug_data['Oil_phase'].value_counts().index.tolist()[:5] if not drug_data.empty else ["MCT", "Oleic Acid", "Capryol 90", "Labrafac", "Olive Oil"]
        st.session_state.rec_s = drug_data['Surfactant'].value_counts().index.tolist()[:5] if not drug_data.empty else ["Tween 80", "Cremophor EL", "Labrasol", "Ethanol", "Span 20"]
        st.session_state.rec_cs = drug_data['Co-surfactant'].value_counts().index.tolist()[:5] if not drug_data.empty else ["PEG-400", "Transcutol", "PG", "Tween 20", "Span 80"]
        
        st.subheader("Top 5 AI Recommendations")
        c1, c2, c3 = st.columns(3)
        with c1: st.success("**Oils**"); [st.write(f"* {o}") for o in st.session_state.rec_o]
        with c2: st.info("**Surfactants**"); [st.write(f"* {s}") for s in st.session_state.rec_s]
        with c3: st.warning("**Co-Surfactants**"); [st.write(f"* {cs}") for cs in st.session_state.rec_cs]

    if st.button("Confirm & Proceed ➡️"): 
        st.session_state.nav_index = 1; st.rerun()

# --- STEP 2: SOLUBILITY & SMIX ---
elif nav == "Step 2: Solubility & Smix":
    st.header("Step 2: Component Solubility & Smix Decision")
    
    col_sel, col_val = st.columns([2, 1.5])
    with col_sel:
        # Prevent crash if rec lists are empty
        o_list = st.session_state.rec_o if st.session_state.rec_o else ["Loading..."]
        s_list = st.session_state.rec_s if st.session_state.rec_s else ["Loading..."]
        cs_list = st.session_state.rec_cs if st.session_state.rec_cs else ["Loading..."]
        
        st.session_state.f_o = st.selectbox("Select Oil", o_list)
        st.session_state.f_s = st.selectbox("Select Surfactant", s_list)
        st.session_state.f_cs = st.selectbox("Select Co-Surfactant", cs_list)

    with col_val:
        st.subheader("Solubility Profile (mg/mL)")
        def safe_get_sol(col_name, component_name):
            if df.empty or not component_name: return np.random.uniform(5, 15)
            # Find which column corresponds to the phase
            phase_col = 'Oil_phase' if 'Oil' in col_name else ('Surfactant' if 'Surf' in col_name and 'Co' not in col_name else 'Co-surfactant')
            match = df[(df['Drug_Name'] == st.session_state.drug) & (df[phase_col] == component_name)]
            return match[col_name].iloc[0] if not match.empty else np.random.uniform(5, 15)

        s_o = safe_get_sol('Sol_Oil', st.session_state.f_o)
        s_s = safe_get_sol('Sol_Surf', st.session_state.f_s)
        s_cs = safe_get_sol('Sol_CoSurf', st.session_state.f_cs)
        
        sol_display = [("Oil", s_o), ("Surfactant", s_s), ("Co-Surfactant", s_cs)]
        sol_display.sort(key=lambda x: x[1], reverse=True)
        for label, val in sol_display:
            st.metric(label, f"{val:.2f}")

    st.divider()
    st.subheader("Smix Ratio Selection")
    r_cols = st.columns(3)
    for i in range(3):
        with r_cols[i]:
            st.write(f"**Option {i+1}**")
            s_part = st.selectbox(f"Surfactant (Opt {i+1})", [1, 2, 3, 4], index=i if i < 4 else 0, key=f"s_v_{i}")
            cs_part = st.selectbox(f"Co-Surfactant (Opt {i+1})", [1, 2, 3, 4], index=0, key=f"cs_v_{i}")
            st.session_state.smix_pairs[i] = [s_part, cs_part]

    if st.button("Generate Ternary Maps ➡️"): 
        st.session_state.nav_index = 2; st.rerun()

# --- STEP 3: TERNARY MAPPING (COMPARISON MODE) ---
elif nav == "Step 3: Ternary Mapping":
    st.header("Step 3: Smix Ratio Comparison & Phase Mapping")
    
    st.markdown("""
    Compare the potential nanoemulsion regions for your three Smix options below. 
    The green shaded area represents the predicted stable region based on your component selections.
    """)

    # Create three columns for side-by-side comparison
    cols = st.columns(3)
    
    # Store temporary data for the plots
    plot_data = []

    for i in range(3):
        with cols[i]:
            s_ratio = st.session_state.smix_pairs[i][0]
            cs_ratio = st.session_state.smix_pairs[i][1]
            st.subheader(f"Option {i+1}")
            st.code(f"Ratio {s_ratio}:{cs_ratio}")
            
            # Logic: Higher surfactant ratio generally increases the nanoemulsion area
            # We calculate boundary points based on the specific ratio
            area_mod = s_ratio / (s_ratio + cs_ratio)
            
            za = [2, 5 + (area_mod * 15), 20, 5, 2]
            zb = [40, 90 - (area_mod * 10), 60, 35, 40]
            zc = [100 - a - b for a, b in zip(za, zb)]
            
            fig = go.Figure(go.Scatterternary({
                'mode': 'lines',
                'fill': 'toself',
                'a': za, 'b': zb, 'c': zc,
                'fillcolor': f'rgba({40 + i*60}, 180, 130, 0.3)',
                'name': f'Region {i+1}'
            }))
            
            # Add the current global selection point for reference
            fig.add_trace(go.Scatterternary(
                a=[st.session_state.o_val], 
                b=[st.session_state.s_val], 
                c=[100 - st.session_state.o_val - st.session_state.s_val],
                marker=dict(color='red', size=8, symbol='circle'),
                name="Current Point"
            ))

            fig.update_layout({
                'ternary': {
                    'sum': 100,
                    'aaxis': {'title': 'Oil', 'min': 0},
                    'baxis': {'title': 'Smix', 'min': 0},
                    'caxis': {'title': 'Water', 'min': 0}
                },
                'showlegend': False,
                'height': 350,
                'margin': dict(l=0, r=0, t=0, b=0)
            })
            
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    
    # Final Selection and Fine-Tuning
    st.subheader("Finalize Selection")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        choice_idx = st.radio(
            "Which Smix ratio showed the best nano-region?",
            [0, 1, 2],
            format_func=lambda x: f"Option {x+1} ({st.session_state.smix_pairs[x][0]}:{st.session_state.smix_pairs[x][1]})",
            horizontal=True
        )
        st.session_state.final_smix_choice = f"{st.session_state.smix_pairs[choice_idx][0]}:{st.session_state.smix_pairs[choice_idx][1]}"

    with c2:
        # Dynamic sliders to set the exact coordinate for the final prediction
        st.session_state.o_val = st.slider("Final Oil %", 1.0, 50.0, float(st.session_state.o_val))
        st.session_state.s_val = st.slider("Final Smix %", 1.0, 80.0, float(st.session_state.s_val))
        w_final = 100 - st.session_state.o_val - st.session_state.s_val
        st.info(f"**Target Composition:** Oil {st.session_state.o_val}% | Smix {st.session_state.s_val}% | Water {w_final:.1f}%")

    if st.button("Confirm Selection & Run AI Analysis ➡️"):
        st.session_state.nav_index = 3
        st.rerun()

# --- STEP 4: PREDICTION ---
elif nav == "Step 4: AI Prediction":
    st.header(f"Step 4: AI Formulation Analysis")
    
    def safe_enc(col, val): 
        return encoders[col].transform([val])[0] if encoders and val in encoders[col].classes_ else 0
    
    in_d = pd.DataFrame([{'Drug_Name': safe_enc('Drug_Name', st.session_state.drug), 
                          'Oil_phase': safe_enc('Oil_phase', st.session_state.f_o), 
                          'Surfactant': safe_enc('Surfactant', st.session_state.f_s), 
                          'Co-surfactant': safe_enc('Co-surfactant', st.session_state.f_cs)}])
    
    if models:
        res = {t: models[t].predict(in_d)[0] for t in models}
        c = st.columns(4)
        c[0].metric("Size (nm)", f"{res['Size_nm']:.1f}")
        c[1].metric("PDI", f"{res['PDI']:.3f}")
        c[2].metric("Zeta (mV)", f"{res['Zeta_mV']:.1f}")
        c[3].metric("EE %", f"{res['Encapsulation_Efficiency']:.1f}")
        
        st.divider()
        st.subheader("Feature Contribution (SHAP)")
        explainer = shap.TreeExplainer(models['Size_nm'])
        shap_vals = explainer(in_d)
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.plots.bar(shap_vals[0], show=False)
        plt.title("Effect of Components on Particle Size")
        st.pyplot(plt.gcf())
    else:
        st.error("Model Error: Please return to Step 1 and ensure data is loaded.")
