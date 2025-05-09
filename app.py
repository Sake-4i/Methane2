import pandas as pd
import streamlit as st
import joblib
import time

# Parameters
threshold = 2.0  # Same as training threshold
lookahead_steps = 8  # How many steps ahead to check for true exceedance
log_data = []  # For accumulating log entries

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load("methane_exceedance_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

model, scaler = load_model()

# Load test dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("test_data_for_dashboard.csv")
        return df
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        return pd.DataFrame()

df = load_data()

# Check if model, scaler, and data are ready
if model is not None and not df.empty:
    st.title("Methane Monitoring Dashboard")
    st.markdown("This dashboard shows real-time prediction of methane exceedance based on sensor data.")

    feature_cols = ['AN422', 'BA1713_max', 'RH1712', 'TP1711', 'MM263', 'MM264', 'MM256']
    speed = st.slider("Speed multiplier", min_value=1, max_value=20, value=5, step=1)

    placeholder = st.empty()

    for i in range(len(df)):
        row = df.iloc[[i]]
        features = row[feature_cols]

        try:
            scaled = scaler.transform(features)
        except Exception as e:
            st.error(f"Error scaling input: {e}")
            break

        prediction = model.predict(scaled)[0]
        mm_actual = row[['MM263', 'MM264', 'MM256']].mean(axis=1).values[0]

        exceed_future = False
        for j in range(1, lookahead_steps + 1):
            if i + j < len(df):
                future_row = df.iloc[i + j]
                future_mm = future_row[['MM263', 'MM264', 'MM256']].mean()
                if future_mm > threshold:
                    exceed_future = True
                    break

        log_data.append({
            'Index': i,
            'MM263': row['MM263'].values[0],
            'MM264': row['MM264'].values[0],
            'MM256': row['MM256'].values[0],
            'MM_actual': mm_actual,
            'Prediction': prediction,
            'Alert Triggered': '⚠️ YES' if prediction == 1 else '',
            'Future Exceedance?': '✅ Yes' if exceed_future else '❌ No'
        })

        with placeholder.container():
            st.subheader(f"Live Reading #{i+1}")
            st.metric("Model Prediction", "⚠️ ALERT" if prediction == 1 else "✅ Normal")
            st.metric("Future Exceedance?", "✅ Yes" if exceed_future else "❌ No")
            st.write("Sensor Data Snapshot:")
            st.dataframe(row[feature_cols])

            st.markdown("---")
            st.subheader("🔍 Log of Recent Readings")
            log_df = pd.DataFrame(log_data).tail(10)
            st.dataframe(log_df)

            time.sleep(5.0 / speed)
else:
    st.error("Model or data not loaded. Please check your files.")
