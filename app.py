import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os

from climate_engine import get_climate_data
from ask_propertyguard import get_chat_response, SUGGESTED_QUESTIONS

# Set Streamlit Page Config
st.set_page_config(
    page_title="PropertyGuard | Climate & Sustainability House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Custom CSS for Dark Glassmorphism Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0f172a 100%);
        color: #f8fafc;
    }
    
    .stAppHeader {
        background: transparent !important;
    }
    
    /* Glassmorphism Card Style */
    .glass-card {
        background: rgba(30, 41, 59, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .metric-value {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .score-badge {
        display: inline-block;
        padding: 8px 18px;
        border-radius: 20px;
        font-size: 1.6rem;
        font-weight: 700;
        text-align: center;
    }
    
    .score-high {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid #10b981;
    }
    
    .score-medium {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid #f59e0b;
    }
    
    .score-low {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
    }

    .advice-card {
        background: rgba(15, 23, 42, 0.7);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    
    .advice-urgent { border-left-color: #ef4444; }
    .advice-warning { border-left-color: #f59e0b; }
    .advice-good { border-left-color: #10b981; }
    .advice-recommended { border-left-color: #3b82f6; }
</style>
""", unsafe_allow_html=True)

# Load Trained Model and Locality Statistics
@st.cache_resource
def load_ml_artifacts():
    model_obj = joblib.load("model.joblib")
    encoders = joblib.load("encoders.joblib")
    with open("locality_stats.json", "r", encoding="utf-8") as f:
        locality_stats = json.load(f)
    return model_obj, encoders, locality_stats

try:
    model_obj, encoders, locality_stats = load_ml_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    locality_stats = {}
    st.error(f"Error loading trained model artifacts: {e}")

# Sidebar - Property Input Form
st.sidebar.markdown("## 🏠 PropertyGuard Setup")
st.sidebar.markdown("*Model 1: House Price & NASA Climate Awareness*")

cities = ["Bengaluru", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]
selected_city = st.sidebar.selectbox("Select City", cities, index=0)

# Filter localities for selected city
city_localities = sorted([
    stats["location"] for key, stats in locality_stats.items()
    if stats["city"] == selected_city
]) if artifacts_loaded else []

if not city_localities:
    city_localities = ["Central", "North", "South", "East", "West"]

selected_locality = st.sidebar.selectbox("Select Locality", city_localities, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### Property Specifications")

area_sqft = st.sidebar.slider("Property Area (sq.ft.)", min_value=300, max_value=6000, value=1200, step=50)
bedrooms = st.sidebar.slider("Bedrooms (BHK)", min_value=1, max_value=6, value=2, step=1)
is_resale = st.sidebar.radio("Property Status", ["New Construction", "Resale / Pre-owned"], index=1)
resale_val = 1 if is_resale == "Resale / Pre-owned" else 0

st.sidebar.markdown("---")
st.sidebar.markdown("### Amenities & Characteristics")

col_a1, col_a2 = st.sidebar.columns(2)
has_gym = col_a1.checkbox("Gymnasium", value=True)
has_pool = col_a2.checkbox("Swimming Pool", value=False)
has_rwh = col_a1.checkbox("Rainwater Harvest", value=True)
has_power = col_a2.checkbox("Power Backup", value=True)
has_security = col_a1.checkbox("24x7 Security", value=True)
has_lift = col_a2.checkbox("Lift Available", value=True)
has_parking = col_a1.checkbox("Car Parking", value=True)
has_club = col_a2.checkbox("Club House", value=False)
has_gas = col_a1.checkbox("Gas Connection", value=True)
has_ac = col_a2.checkbox("Air Conditioning", value=False)

# Compile selected amenities dictionary
amenities_dict = {
    "MaintenanceStaff": 0, "Gymnasium": 1 if has_gym else 0, "SwimmingPool": 1 if has_pool else 0,
    "LandscapedGardens": 0, "JoggingTrack": 0, "RainWaterHarvesting": 1 if has_rwh else 0,
    "IndoorGames": 0, "ShoppingMall": 0, "Intercom": 0, "SportsFacility": 0,
    "ATM": 0, "ClubHouse": 1 if has_club else 0, "School": 0, "24X7Security": 1 if has_security else 0,
    "PowerBackup": 1 if has_power else 0, "CarParking": 1 if has_parking else 0,
    "StaffQuarter": 0, "Cafeteria": 0, "MultipurposeRoom": 0, "Hospital": 0, "WashingMachine": 0,
    "Gasconnection": 1 if has_gas else 0, "AC": 1 if has_ac else 0, "Wifi": 0,
    "Children'splayarea": 0, "LiftAvailable": 1 if has_lift else 0, "BED": 0,
    "VaastuCompliant": 0, "Microwave": 0, "GolfCourse": 0, "TV": 0, "DiningTable": 0,
    "Sofa": 0, "Wardrobe": 0, "Stadium": 0
}

# Main Application Dashboard
st.markdown("# 🏠 PropertyGuard")
st.markdown("### Climate & Sustainability-Aware House Price Valuation Engine")
st.markdown(f"**Target Property:** {bedrooms} BHK Flat, {area_sqft:,} sq.ft. in **{selected_locality}, {selected_city}**")

# Helper function to predict price
def predict_property_price():
    if not artifacts_loaded:
        return 5000000 # fallback
        
    loc_stats = encoders.get("loc_stats", {})
    city_stats = encoders.get("city_stats", {})
    global_mean = encoders.get("global_mean", 15.5)
    
    l_info = loc_stats.get(selected_locality, None)
    c_info = city_stats.get(selected_city, None)
    smoothing = 10
    
    if l_info:
        loc_encoded = (l_info['count'] * l_info['mean'] + smoothing * global_mean) / (l_info['count'] + smoothing)
    elif c_info:
        loc_encoded = c_info['mean']
    else:
        loc_encoded = global_mean
        
    city_encoded = c_info['mean'] if c_info else global_mean
    
    amenity_count = sum(amenities_dict.values())
    area_per_bhk = area_sqft / bedrooms
    log_area = np.log(area_sqft)
    
    luxury_cols = encoders.get("luxury_cols", [])
    essential_cols = encoders.get("essential_cols", [])
    
    luxury_score = sum([amenities_dict.get(c, 0) for c in luxury_cols])
    essential_score = sum([amenities_dict.get(c, 0) for c in essential_cols])
    
    row_data = {
        "Area": area_sqft,
        "log_area": log_area,
        "No. of Bedrooms": bedrooms,
        "Resale": resale_val,
        "amenity_count": amenity_count,
        "area_per_bhk": area_per_bhk,
        "luxury_score": luxury_score,
        "essential_score": essential_score,
        "loc_encoded": loc_encoded,
        "city_encoded": city_encoded
    }
    row_data.update(amenities_dict)
    
    features = encoders["features"]
    input_df = pd.DataFrame([row_data])[features]
    
    if isinstance(model_obj, dict):
        cb = model_obj['cb']
        hgb = model_obj['hgb']
        xgb = model_obj['xgb']
        w = model_obj.get('weights', [0.45, 0.35, 0.20])
        log_pred = w[0] * cb.predict(input_df)[0] + w[1] * hgb.predict(input_df)[0] + w[2] * xgb.predict(input_df)[0]
    else:
        log_pred = model_obj.predict(input_df)[0]
        
    predicted_price = np.expm1(log_pred)
    return predicted_price

# Calculate predictions & climate lookup
pred_price = predict_property_price()
climate_info = get_climate_data(selected_city, selected_locality)

# Create Dashboard Tabs
tab_val, tab_climate, tab_advice, tab_chat = st.tabs([
    "💵 ML Price Valuation",
    "🌍 NASA Climate & Sustainability",
    "💡 Buyer Property Advice",
    "🤖 Ask PropertyGuard"
])

# TAB 1: ML Price Valuation
with tab_val:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    
    col_p1, col_p2, col_p3 = st.columns(3)
    
    # Format Price (Crores / Lakhs)
    if pred_price >= 10000000:
        price_str = f"₹{pred_price/10000000:.2f} Cr"
    else:
        price_str = f"₹{pred_price/100000:.2f} Lakhs"
        
    col_p1.metric("Predicted Market Value", price_str)
    
    price_per_sqft = pred_price / area_sqft
    col_p2.metric("Price per Sq.Ft.", f"₹{price_per_sqft:,.0f} / sq.ft.")
    
    # Estimated Value Range
    min_est = pred_price * 0.90
    max_est = pred_price * 1.10
    col_p3.metric("Fair Valuation Range", f"₹{min_est/100000:.1f}L - ₹{max_est/100000:.1f}L")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Comparison with locality statistics
    loc_key = f"{selected_city}::{selected_locality}"
    loc_data = locality_stats.get(loc_key, None)
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 📊 Property Characteristic Summary")
        st.write(f"- **City & Locality:** {selected_locality}, {selected_city}")
        st.write(f"- **Super Built-up Area:** {area_sqft:,} sq.ft.")
        st.write(f"- **Configuration:** {bedrooms} BHK ({area_sqft/bedrooms:.0f} sq.ft. per bedroom)")
        st.write(f"- **Construction:** {is_resale}")
        st.write(f"- **Amenities Enabled:** {sum(amenities_dict.values())} selected features")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_info2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 📍 Locality Benchmark Data")
        if loc_data:
            st.write(f"- **Locality Median Price:** ₹{loc_data['median_price']/100000:.2f} Lakhs")
            st.write(f"- **Locality Price/Sq.Ft:** ₹{loc_data['median_price_sqft']:,.0f} / sq.ft.")
            st.write(f"- **Locality Average Area:** {loc_data['mean_area']:.0f} sq.ft.")
            
            diff_pct = ((price_per_sqft - loc_data['median_price_sqft']) / loc_data['median_price_sqft']) * 100
            if diff_pct > 0:
                st.info(f"💡 This property is valued **{diff_pct:.1f}% higher** than locality median due to features/size.")
            else:
                st.success(f"💡 This property offers attractive pricing (**{abs(diff_pct):.1f}% below** locality median).")
        else:
            st.write(f"- City Average Price/Sq.Ft: ₹{price_per_sqft:,.0f}")
            st.write("- Locality sample size: Growing dataset")
        st.markdown("</div>", unsafe_allow_html=True)

# TAB 2: NASA Climate & Sustainability Intelligence
with tab_climate:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    
    col_s1, col_s2 = st.columns([1, 2])
    
    score = climate_info["sustainability_score"]
    badge_class = "score-high" if score >= 75 else ("score-medium" if score >= 55 else "score-low")
    
    with col_s1:
        st.markdown("### Sustainability & Living Score")
        st.markdown(f"<div class='score-badge {badge_class}'>{score} / 100</div>", unsafe_allow_html=True)
        st.caption("Weighted score combining Solar Energy, Thermal Comfort, Air Quality & Water Security.")
        
    with col_s2:
        st.markdown("#### Score Breakdown")
        m = climate_info["metrics"]
        st.progress(m["solar_score"] / 100.0, text=f"☀️ Solar Energy Potential: {m['solar_score']}/100")
        st.progress(m["thermal_score"] / 100.0, text=f"🌡️ Thermal & Heat Comfort: {m['thermal_score']}/100")
        st.progress(m["aqi_score"] / 100.0, text=f"🌿 Air Quality Index: {m['aqi_score']}/100")
        st.progress(m["water_score"] / 100.0, text=f"💧 Water & Flood Resilience: {m['water_score']}/100")
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # NASA Climate Parameter Cards
    st.markdown("### 📡 NASA POWER Microclimate Indicators")
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    
    col_c1.metric("Peak Summer Temp", f"{m['summer_max_temp_c']} °C")
    col_c2.metric("Monsoon Rainfall", f"{m['annual_precipitation_mm']} mm/yr")
    col_c3.metric("Solar Insolation", f"{m['solar_insolation']} kWh/m²/day")
    col_c4.metric("Avg Air Quality (AQI)", f"{m['aqi']} ({m['air_quality_category']})")

# TAB 3: Buyer Property Advice & Living Guidance
with tab_advice:
    st.markdown("### 💡 Actionable Buyer & Property Advice")
    st.markdown("These non-financial recommendations give home buyers a clear picture of climate resilience, energy savings, and living comfort before buying.")
    
    for advice in climate_info["advice"]:
        status_class = f"advice-{advice['status'].lower()}"
        st.markdown(f"""
        <div class='advice-card {status_class}'>
            <h4 style='margin:0;'>{advice['icon']} {advice['title']} ({advice['category']})</h4>
            <p style='margin-top:6px; color:#cbd5e1;'>{advice['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("#### 📋 Homebuyer Pre-Purchase Checklist for " + selected_locality)
    st.checkbox("Check building elevation against 10-year monsoon waterlogging levels.")
    st.checkbox("Verify dual plumbing lines for recycled water and rainwater harvesting tanks.")
    st.checkbox("Confirm rooftop solar PV installation feasibility & net-metering approval.")
    st.checkbox("Inspect window glazing & wall insulation for afternoon heat facing rooms.")

# TAB 4: Ask PropertyGuard — IBM Bob-powered conversational interface
with tab_chat:
    st.markdown("### 🤖 Ask PropertyGuard")
    st.markdown(
        "Ask questions about this property's price estimate, climate risks, "
        "sustainability score, or what to check before buying. "
        "All answers are grounded in the ML predictions and NASA climate data "
        "already calculated for this property."
    )

    # Resolve locality benchmark data (same lookup used in Tab 1)
    _loc_key = f"{selected_city}::{selected_locality}"
    _loc_data = locality_stats.get(_loc_key, None)

    # Initialise session state for chat history
    if "pg_chat_history" not in st.session_state:
        st.session_state.pg_chat_history = []

    # Reset chat when the property changes
    _property_key = f"{selected_city}|{selected_locality}|{area_sqft}|{bedrooms}|{resale_val}"
    if st.session_state.get("pg_property_key") != _property_key:
        st.session_state.pg_chat_history = []
        st.session_state.pg_property_key = _property_key

    # Suggested question buttons
    st.markdown("**Suggested questions:**")
    suggestion_cols = st.columns(2)
    for idx, sq in enumerate(SUGGESTED_QUESTIONS):
        if suggestion_cols[idx % 2].button(sq, key=f"sq_{idx}"):
            st.session_state.pg_chat_history.append({"role": "user", "content": sq})
            with st.spinner("Thinking..."):
                reply = get_chat_response(
                    user_question=sq,
                    chat_history=st.session_state.pg_chat_history[:-1],
                    city=selected_city,
                    locality=selected_locality,
                    bedrooms=bedrooms,
                    area_sqft=area_sqft,
                    is_resale=is_resale,
                    amenities_dict=amenities_dict,
                    pred_price=pred_price,
                    loc_data=_loc_data,
                    climate_info=climate_info,
                )
            st.session_state.pg_chat_history.append({"role": "assistant", "content": reply})

    st.markdown("---")

    # Render existing chat history
    for turn in st.session_state.pg_chat_history:
        if turn["role"] == "user":
            st.markdown(
                f"<div class='advice-card advice-recommended'>"
                f"<strong>You:</strong> {turn['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='advice-card advice-good'>"
                f"<strong>🤖 PropertyGuard Assistant:</strong><br>{turn['content']}</div>",
                unsafe_allow_html=True,
            )

    # Free-text input
    user_input = st.chat_input("Ask about this property's price, climate risks, sustainability...")
    if user_input and user_input.strip():
        st.session_state.pg_chat_history.append({"role": "user", "content": user_input.strip()})
        with st.spinner("Thinking..."):
            reply = get_chat_response(
                user_question=user_input.strip(),
                chat_history=st.session_state.pg_chat_history[:-1],
                city=selected_city,
                locality=selected_locality,
                bedrooms=bedrooms,
                area_sqft=area_sqft,
                is_resale=is_resale,
                amenities_dict=amenities_dict,
                pred_price=pred_price,
                loc_data=_loc_data,
                climate_info=climate_info,
            )
        st.session_state.pg_chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    # Clear chat button
    if st.session_state.pg_chat_history:
        if st.button("🗑️ Clear conversation", key="clear_chat"):
            st.session_state.pg_chat_history = []
            st.rerun()

st.markdown("---")
st.markdown("<p style='text-align:center; color:#64748b;'>PropertyGuard © 2026 — Model 1 Climate & Sustainability Property Valuation Engine</p>", unsafe_allow_html=True)
