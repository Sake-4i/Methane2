import streamlit as st
import pandas as pd
import joblib
import time

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

    # Feature columns
    feature_cols = ['AN422', 'BA1713_max', 'RH1712', 'TP1711', 'MM263', 'MM264', 'MM256']

    # Speedup simulation option
    speed = st.slider("Speed multiplier", min_value=1, max_value=20, value=5, step=1)

    # Real-time simulation
    placeholder = st.empty()
    for i in range(len(df)):
        row = df.iloc[i:i+1]
        features = row[feature_cols]

        # Scale the features
        try:
            scaled_features = scaler.transform(features)
        except Exception as e:
            st.error(f"Error scaling input: {e}")
            break

        # Predict
        try:
            prediction = model.predict(scaled_features)[0]
        except Exception as e:
            st.error(f"Error predicting: {e}")
            break

    #
    # Get actual exceedance label from test data
actual = row['Exceed'].values[0]

# Determine if prediction was correct
correct = "✅ Correct" if prediction == actual else "❌ Incorrect"

# Display
with placeholder.container():
    st.metric("Prediction", "⚠️ ALERT" if prediction == 1 else "✅ Normal")
    st.metric("Actual", "⚠️ EXCEEDED" if actual == 1 else "✅ Safe")
    st.metric("Result", correct)
    st.write("Latest Sensor Readings:")
    st.dataframe(row[feature_cols])

    #
    #

    
        # Display
        with placeholder.container():
            st.metric("Methane Exceedance Prediction", "⚠️ ALERT" if prediction == 1 else "✅ Normal")
            st.write("Latest Sensor Readings:")
            st.dataframe(row[feature_cols])

        # Wait (speed controlled)
        time.sleep(5.0 / speed)
else:
    st.warning("App not ready. Ensure model, scaler, and dataset are uploaded properly.")
