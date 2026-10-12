import os
import pickle
import traceback
from collections import defaultdict, UserList
import numpy as np
import pandas as pd
import streamlit as st

# Compatibility patch for scikit-learn ColumnTransformer unpickling across versions
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, '_RemainderColsList'):
        class _RemainderColsList(UserList):
            pass
        _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass

# Streamlit Page Configuration
st.set_page_config(
    page_title="CUET UG Cutoff Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        border-radius: 14px;
        padding: 24px;
        color: white;
        text-align: center;
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.25);
    }
    .metric-title {
        font-size: 1.05rem;
        font-weight: 500;
        opacity: 0.9;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
    }
    .info-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_metadata():
    try:
        # Load the unified pipeline file
        with open('model.pkl', 'rb') as f:
            model_pipeline = pickle.load(f)
    except Exception:
        err = traceback.format_exc()
        return None, {}, [], [], [], err

    # Load mapping options if available for rich UI dropdowns
    univ_to_progs = defaultdict(list)
    universities = []
    categories = ['UR', 'OBC-NCL', 'EWS', 'SC', 'ST', 'PwBD', 'KM', 'CW']
    rounds = ['1', '2', '3', '4', '5', 'spot', 'mop-up-1']

    if os.path.exists('mappings.pkl'):
        try:
            with open('mappings.pkl', 'rb') as f:
                maps = pickle.load(f)
            
            for k in maps.get('us_means', {}):
                parts = k.split('_', 1)
                if len(parts) == 2:
                    univ_to_progs[parts[0]].append(parts[1])

            for u in univ_to_progs:
                univ_to_progs[u] = sorted(list(set(univ_to_progs[u])))

            if univ_to_progs:
                universities = sorted(list(univ_to_progs.keys()))

            if 'usc_means' in maps:
                cats = set(k.rsplit('_', 1)[1] for k in maps['usc_means'] if '_' in k)
                if cats:
                    categories = sorted(list(cats))

            if 'uscr_means' in maps:
                rnds = set(k.rsplit('_', 1)[1] for k in maps['uscr_means'] if '_' in k)
                if rnds:
                    def sort_key(r):
                        return (0, int(r)) if r.isdigit() else (1, r)
                    rounds = sorted(list(rnds), key=sort_key)
        except Exception:
            pass

    return model_pipeline, univ_to_progs, universities, categories, rounds, None

# Load model pipeline and metadata safely
model_pipeline, univ_to_progs, universities, categories, rounds, error_msg = load_model_and_metadata()

if error_msg:
    st.error("⚠️ Failed to load ML model pipeline")
    st.code(error_msg, language="text")
    st.info("Ensure all dependencies match the training environment (scikit-learn, lightgbm, numpy).")
    st.stop()

# Header
st.markdown('<div class="main-header">🎓 CUET UG Cutoff Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Instant AI admission cutoff prediction using trained Scikit-learn Pipeline</div>', unsafe_allow_html=True)

col_input, col_result = st.columns([1.1, 0.9], gap="large")

with col_input:
    st.subheader("📋 Candidate & Course Details")
    
    # University selection
    if universities:
        formatted_univs = {u: u.replace('-', ' ').title() for u in universities}
        selected_univ = st.selectbox(
            "University",
            options=universities,
            format_func=lambda x: formatted_univs.get(x, x),
            help="Select the university"
        )
        available_progs = univ_to_progs.get(selected_univ, [])
    else:
        selected_univ = st.text_input("University ID", value="banaras-hindu-university")
        available_progs = []

    # Programme selection
    if available_progs:
        selected_prog = st.selectbox(
            "Programme / Course",
            options=available_progs,
            help="Select the course offered by the university"
        )
    else:
        selected_prog = st.text_input("Programme / Course", value="B.Com. (Honours)")

    col_cat, col_rnd = st.columns(2)
    with col_cat:
        default_cat_idx = categories.index('UR') if 'UR' in categories else 0
        selected_category = st.selectbox(
            "Category / Quota",
            options=categories,
            index=default_cat_idx,
            help="Select reservation category"
        )
    with col_rnd:
        selected_round = st.selectbox(
            "Admission Round",
            options=rounds,
            index=0,
            help="Counselling round number or spot round"
        )

    col_yr, _ = st.columns(2)
    with col_yr:
        selected_year = st.number_input(
            "Year",
            min_value=2020,
            max_value=2030,
            value=2026,
            step=1
        )

    predict_btn = st.button("🔮 Predict Expected Cutoff", type="primary", use_container_width=True)

with col_result:
    st.subheader("📊 Prediction Output")
    
    if predict_btn:
        try:
            with st.spinner("Calculating cutoff with pipeline..."):
                # Prepare raw input matching the pipeline's expected features
                input_data = pd.DataFrame([{
                    'year': int(selected_year),
                    'university_id': str(selected_univ),
                    'programme': str(selected_prog),
                    'category': str(selected_category),
                    'round': str(selected_round)
                }])
                
                # Predict directly using the loaded pipeline
                raw_pred = model_pipeline.predict(input_data)
                predicted_cutoff = float(raw_pred[0])
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">ESTIMATED CUTOFF SCORE</div>
                <div class="metric-value">{predicted_cutoff:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""<div class="info-box">
                <h4>📌 Prediction Parameters</h4>
            """, unsafe_allow_html=True)
            st.write(f"• **University:** {selected_univ.replace('-', ' ').title()}")
            st.write(f"• **Programme:** {selected_prog}")
            st.write(f"• **Category:** {selected_category}")
            st.write(f"• **Round:** {selected_round}")
            st.write(f"• **Year:** {selected_year}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as err:
            st.error(f"Prediction failed: {err}")
    else:
        st.info("👈 Fill in the details on the left and click **Predict Expected Cutoff**.")
        if universities:
            st.markdown(f"""
            <div style="background-color: #F1F5F9; border-radius: 8px; padding: 16px; margin-top: 10px;">
                <p style="margin: 0; color: #475569; font-size: 0.95rem;">
                    <b>Model Pipeline Status:</b><br>
                    ✅ <b>TargetEncoder + Estimator</b> Pipeline Loaded<br>
                    🏛️ <b>{len(universities)}</b> Universities available<br>
                    📚 <b>{sum(len(p) for p in univ_to_progs.values())}</b> Programmes available<br>
                    🏷️ <b>{len(categories)}</b> Reservation categories
                </p>
            </div>
            """, unsafe_allow_html=True)
