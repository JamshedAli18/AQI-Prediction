# ============================================================
# app.py — AQI Prediction Streamlit App (Fixed Version)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌫️",
    layout="wide"
)

# ── Load Saved Artifacts ──────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model        = joblib.load('aqi_model.pkl')
    imputer      = joblib.load('imputer.pkl')
    le           = joblib.load('label_encoder.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    city_list    = joblib.load('city_list.pkl')
    return model, imputer, le, feature_cols, city_list

model, imputer, le, feature_cols, city_list = load_artifacts()

# ── AQI Category Helper ───────────────────────────────────────
def get_aqi_category(aqi):
    """Return category, color, and health message based on AQI value."""
    if aqi <= 50:
        return "Good", "#00e400", "Air quality is satisfactory. Enjoy outdoor activities!"
    elif aqi <= 100:
        return "Satisfactory", "#92d050", "Air quality is acceptable for most people."
    elif aqi <= 200:
        return "Moderate", "#ffff00", "Sensitive groups may experience minor issues."
    elif aqi <= 300:
        return "Poor", "#ff7e00", "Everyone may begin to experience health effects."
    elif aqi <= 400:
        return "Very Poor", "#ff0000", "Health alert: serious effects for everyone."
    else:
        return "Severe", "#8b0000", "Health warning of emergency conditions!"

# ── Gauge Chart ───────────────────────────────────────────────
def draw_aqi_gauge(aqi_value):
    """Draw a semicircular gauge chart showing AQI level."""
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('#0e1117')

    # Clamp AQI between 0 and 500
    aqi_clamped = min(max(aqi_value, 0), 500)
    angle = np.pi * (1 - aqi_clamped / 500)

    # Draw colored arcs for each AQI zone
    zones = [
        (0,   50,  '#00e400'),
        (50,  100, '#92d050'),
        (100, 200, '#ffff00'),
        (200, 300, '#ff7e00'),
        (300, 400, '#ff0000'),
        (400, 500, '#8b0000'),
    ]
    for start, end, color in zones:
        theta_start = np.pi * (1 - start / 500)
        theta_end   = np.pi * (1 - end  / 500)
        theta = np.linspace(theta_start, theta_end, 50)
        ax.plot(theta, [0.8] * 50, lw=15, color=color, solid_capstyle='butt')

    # Needle pointing to predicted AQI
    ax.annotate('', xy=(angle, 0.75), xytext=(angle, 0.0),
                arrowprops=dict(arrowstyle='->', color='white', lw=2.5))

    # AQI number in center
    ax.text(np.pi / 2, -0.3, f"{int(aqi_value)}", ha='center', va='center',
            fontsize=28, fontweight='bold', color='white',
            transform=ax.transData)

    ax.set_ylim(0, 1)
    ax.set_xlim(0, np.pi)
    ax.set_theta_zero_location('W')
    ax.set_theta_direction(-1)
    ax.axis('off')
    plt.tight_layout()
    return fig

# ── App Header ────────────────────────────────────────────────
st.title("🌫️ Air Quality Index (AQI) Predictor")
st.markdown("Predict the Air Quality Index for Indian cities using pollutant levels.")
st.divider()

# ── Sidebar Inputs ────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Input Pollutant Levels")
    st.markdown("Adjust the sliders to match observed pollution readings.")

    # City and Date inputs
    city  = st.selectbox("🏙️ Select City", city_list)
    year  = st.selectbox("📅 Year", list(range(2015, 2026)), index=5)
    month = st.slider("📆 Month", 1, 12, 6)
    day   = st.slider("📅 Day",   1, 31, 15)

    st.divider()
    st.subheader("Particulate Matter")
    pm25 = st.slider("PM2.5 (µg/m³)",  0.0, 500.0,  60.0, step=0.5)
    pm10 = st.slider("PM10  (µg/m³)",  0.0, 700.0, 100.0, step=0.5)

    st.subheader("Nitrogen Compounds")
    no   = st.slider("NO    (µg/m³)",  0.0, 200.0,  10.0, step=0.1)
    no2  = st.slider("NO2   (µg/m³)",  0.0, 200.0,  25.0, step=0.1)
    nox  = st.slider("NOx   (µg/m³)",  0.0, 300.0,  35.0, step=0.1)
    nh3  = st.slider("NH3   (µg/m³)",  0.0, 100.0,  15.0, step=0.1)

    st.subheader("Other Pollutants")
    co      = st.slider("CO    (mg/m³)",   0.0,  50.0,  1.0, step=0.1)
    so2     = st.slider("SO2   (µg/m³)",   0.0, 200.0, 10.0, step=0.1)
    o3      = st.slider("O3    (µg/m³)",   0.0, 300.0, 40.0, step=0.1)
    benzene = st.slider("Benzene (µg/m³)", 0.0,  50.0,  2.0, step=0.1)
    toluene = st.slider("Toluene (µg/m³)", 0.0, 100.0,  5.0, step=0.1)
    xylene  = st.slider("Xylene  (µg/m³)", 0.0,  50.0,  1.0, step=0.1)

    predict_btn = st.button("🔍 Predict AQI", type="primary", use_container_width=True)

# ── Prediction Logic ──────────────────────────────────────────
if predict_btn:

    try:
        # Step 1: Encode the selected city
        city_encoded = le.transform([city])[0]

        # Step 2: Build input DataFrame with correct column order
        input_data = pd.DataFrame([[
            city_encoded, pm25, pm10, no, no2, nox,
            nh3, co, so2, o3, benzene, toluene, xylene,
            year, month, day
        ]], columns=feature_cols)

        # Step 3: Cast to float64 — fixes sklearn version dtype mismatch
        input_data = input_data.astype(np.float64)

        # Step 4: Apply the same imputer used during training
        input_imputed = imputer.transform(input_data)

        # Step 5: Predict AQI
        aqi_pred = model.predict(input_imputed)[0]
        aqi_pred = max(0.0, float(aqi_pred))  # AQI cannot be negative

        # Get category info
        category, color, message = get_aqi_category(aqi_pred)

        # ── Results Layout ──────────────────────────────────────
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📊 Prediction Result")

            # Colored AQI result card
            st.markdown(
                f"""
                <div style="background-color:{color}22;
                            border-left: 6px solid {color};
                            padding: 20px;
                            border-radius: 10px;
                            margin-bottom: 16px;">
                    <h2 style="color:{color}; margin:0;">AQI: {aqi_pred:.1f}</h2>
                    <h3 style="color:{color}; margin:4px 0;">{category}</h3>
                    <p style="color:#ccc; margin:0;">{message}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Quick metric cards
            m1, m2, m3 = st.columns(3)
            m1.metric("City",   city)
            m2.metric("Month",  f"{month}/{year}")
            m3.metric("PM2.5",  f"{pm25} µg/m³")

        with col2:
            st.subheader("🎯 AQI Gauge")
            gauge_fig = draw_aqi_gauge(aqi_pred)
            st.pyplot(gauge_fig, use_container_width=True)

        st.divider()

        # ── AQI Scale Reference Table ───────────────────────────
        st.subheader("📋 AQI Scale Reference")
        scale_df = pd.DataFrame({
            "Category":      ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"],
            "AQI Range":     ["0–50", "51–100", "101–200", "201–300", "301–400", "401–500"],
            "Health Impact": [
                "Minimal impact",
                "Minor breathing issues for sensitive people",
                "Breathing discomfort on prolonged exposure",
                "Breathing discomfort for most people",
                "Respiratory illness on prolonged exposure",
                "Affects healthy people; serious for sensitive groups"
            ]
        })
        st.dataframe(scale_df, use_container_width=True, hide_index=True)

        # ── Pollutant Bar Chart ─────────────────────────────────
        st.subheader("🔬 Pollutant Levels You Entered")
        pollutants = {
            'PM2.5': pm25, 'PM10': pm10, 'NO': no,
            'NO2': no2,   'NOx': nox,  'NH3': nh3,
            'SO2': so2,   'O3': o3,    'CO×10': co * 10
        }
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.bar(pollutants.keys(), pollutants.values(),
                color='steelblue', edgecolor='black', alpha=0.85)
        ax2.set_title('Pollutant Levels (Input)', fontsize=13)
        ax2.set_ylabel('Concentration (µg/m³)')
        ax2.tick_params(axis='x', rotation=30)
        fig2.patch.set_facecolor('#0e1117')
        ax2.set_facecolor('#0e1117')
        ax2.tick_params(colors='white')
        ax2.yaxis.label.set_color('white')
        ax2.title.set_color('white')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#444')
        st.pyplot(fig2, use_container_width=True)

    except Exception as e:
        # Show a clean error message instead of crashing
        st.error(f"⚠️ Prediction failed: {str(e)}")
        st.info("Make sure all .pkl files match the sklearn version in requirements.txt (1.6.1)")

else:
    # Default state before prediction
    st.info("👈 Set the pollutant levels in the sidebar and click **Predict AQI** to get a prediction.")

    st.subheader("📋 AQI Scale Reference")
    scale_df = pd.DataFrame({
        "Category":  ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"],
        "AQI Range": ["0–50", "51–100", "101–200", "201–300", "301–400", "401–500"],
    })
    st.dataframe(scale_df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption("Built with Scikit-learn 1.6.1 & Streamlit · Dataset: Air Quality Data in India (Kaggle)")
