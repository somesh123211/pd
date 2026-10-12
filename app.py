import os
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="CUET UG Cutoff Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern look
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
        border-radius: 12px;
        padding: 24px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3);
    }
    .metric-title {
        font-size: 1.1rem;
        font-weight: 500;
        opacity: 0.9;
        margin-bottom: 8px;
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
def load_resources():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scalar.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('mappings.pkl', 'rb') as f:
        maps = pickle.load(f)
        
    # Extract unique universities and corresponding programmes
    univ_to_progs = defaultdict(list)
    for k in maps.get('us_means', {}):
        parts = k.split('_', 1)
        if len(parts) == 2:
            univ_to_progs[parts[0]].append(parts[1])
            
    for u in univ_to_progs:
        univ_to_progs[u] = sorted(list(set(univ_to_progs[u])))
        
    universities = sorted(list(univ_to_progs.keys()))
    
    # Extract categories
    categories = sorted(list(set(k.rsplit('_', 1)[1] for k in maps.get('usc_means', {}) if '_' in k)))
    
    # Extract rounds and sort naturally
    raw_rounds = list(set(k.rsplit('_', 1)[1] for k in maps.get('uscr_means', {}) if '_' in k))
    def sort_round_key(r):
        if r.isdigit():
            return (0, int(r))
        return (1, r)
    rounds = sorted(raw_rounds, key=sort_round_key)
    
    return model, scaler, maps, univ_to_progs, universities, categories, rounds

# Load artifacts
try:
    model, scaler, maps, univ_to_progs, universities, categories, rounds = load_resources()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# Header
st.markdown('<div class="main-header">🎓 CUET UG Cutoff Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Estimate university admission cutoffs powered by machine learning</div>', unsafe_allow_html=True)

col_input, col_result = st.columns([1.1, 0.9], gap="large")

with col_input:
    st.subheader("📋 Candidate & Programme Details")
    
    # University selection
    formatted_univs = {u: u.replace('-', ' ').title() for u in universities}
    selected_univ = st.selectbox(
        "University",
        options=universities,
        format_func=lambda x: formatted_univs.get(x, x),
        help="Select the university you are applying to"
    )
    
    # Programme selection (dynamically filtered by university)
    available_progs = univ_to_progs.get(selected_univ, [])
    if not available_progs:
        available_progs = ["General Programme"]
    selected_prog = st.selectbox(
        "Programme / Course",
        options=available_progs,
        help="Available programmes for the selected university"
    )
    
    col_cat, col_rnd = st.columns(2)
    with col_cat:
        # Default to 'UR' (General) if available
        default_cat_idx = categories.index('UR') if 'UR' in categories else 0
        selected_category = st.selectbox(
            "Category / Quota",
            options=categories,
            index=default_cat_idx,
            help="Your reservation category"
        )
        
    with col_rnd:
        selected_round = st.selectbox(
            "Counselling Round",
            options=rounds,
            index=0,
            help="Admission allotment round"
        )
        
    col_yr, _ = st.columns(2)
    with col_yr:
        selected_year = st.number_input(
            "Admission Year",
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
            with st.spinner("Calculating cutoff prediction..."):
                # Interaction keys matching training logic
                univ_subject = f"{selected_univ}_{selected_prog}"
                univ_subject_cat = f"{univ_subject}_{selected_category}"
                univ_subject_cat_round = f"{univ_subject_cat}_{selected_round}"
                
                # Target encoding lookups with fallback
                g_mean = maps['global_mean']
                univ_subject_encoded = maps['us_means'].get(univ_subject, g_mean)
                univ_subject_cat_encoded = maps['usc_means'].get(univ_subject_cat, g_mean)
                univ_subject_cat_round_encoded = maps['uscr_means'].get(univ_subject_cat_round, g_mean)
                
                features_df = pd.DataFrame([[
                    int(selected_year),
                    univ_subject_encoded,
                    univ_subject_cat_encoded,
                    univ_subject_cat_round_encoded
                ]], columns=['year', 'univ_subject_encoded', 'univ_subject_cat_encoded', 'univ_subject_cat_round_encoded'])
                
                scaled_features = scaler.transform(features_df)
                raw_pred = model.predict(scaled_features)
                predicted_cutoff = float(raw_pred[0])
                
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">ESTIMATED CUTOFF SCORE</div>
                <div class="metric-value">{predicted_cutoff:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""<div class="info-box">
                <h4>📌 Selected Parameters</h4>
            """, unsafe_allow_html=True)
            st.write(f"• **University:** {selected_univ.replace('-', ' ').title()}")
            st.write(f"• **Programme:** {selected_prog}")
            st.write(f"• **Category:** {selected_category}")
            st.write(f"• **Round:** {selected_round}")
            st.write(f"• **Year:** {selected_year}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as err:
            st.error(f"Error generating prediction: {err}")
    else:
        st.info("👈 Select your preferences on the left and click **Predict Expected Cutoff** to view results.")
        
        # Display summary card
        st.markdown(f"""
        <div style="background-color: #F1F5F9; border-radius: 8px; padding: 16px; margin-top: 10px;">
            <p style="margin: 0; color: #475569; font-size: 0.95rem;">
                <b>Dataset coverage:</b><br>
                🏛️ <b>{len(universities)}</b> Universities<br>
                📚 <b>{sum(len(p) for p in univ_to_progs.values())}</b> Programmes<br>
                🏷️ <b>{len(categories)}</b> Reservation categories
            </p>
        </div>
        """, unsafe_allow_html=True)
