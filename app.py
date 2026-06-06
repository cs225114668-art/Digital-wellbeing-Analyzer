import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("best_model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# Page settings
st.set_page_config(
    page_title="Digital Well-Being Dashboard",
    page_icon="📱",
    layout="wide"
)

# Header
st.markdown("""
<h1 style='text-align:center; color:#4CAF50;'>
📱 Digital Well-Being Dashboard
</h1>
<p style='text-align:center;'>
Track your digital habits and predict your well-being status
</p>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙ User Inputs")

screen_time = st.sidebar.slider(
    "Screen Time (Hours)", 0.0, 15.0, 5.0
)

social_media = st.sidebar.slider(
    "Social Media (Hours)", 0.0, 10.0, 2.0
)

gaming = st.sidebar.slider(
    "Gaming (Hours)", 0.0, 10.0, 1.0
)

sleep = st.sidebar.slider(
    "Sleep (Hours)", 0.0, 12.0, 7.0
)

study = st.sidebar.slider(
    "Study / Work (Hours)", 0.0, 15.0, 6.0
)

notifications = st.sidebar.slider(
    "Notifications", 0, 500, 100
)

pickups = st.sidebar.slider(
    "Phone Pickups", 0, 300, 60
)

# Metrics Dashboard
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📺 Screen Time", f"{screen_time} hrs")

with col2:
    st.metric("📱 Social Media", f"{social_media} hrs")

with col3:
    st.metric("😴 Sleep", f"{sleep} hrs")

st.divider()

# Wellness Score
wellness_score = max(
    0,
    min(
        100,
        int(
            (sleep * 10)
            + (study * 5)
            - (screen_time * 3)
            - (social_media * 2)
        )
    )
)

st.subheader("🎯 Wellness Score")

st.progress(wellness_score)

st.write(f"Score: **{wellness_score}/100**")

# Prediction
if st.button("🔍 Analyze Well-Being"):

    user_data = pd.DataFrame([[
        screen_time,
        social_media,
        gaming,
        sleep,
        study,
        notifications,
        pickups
    ]], columns=[
        "Screen_Time_Hours",
        "Social_Media_Hours",
        "Gaming_Hours",
        "Sleep_Hours",
        "Study_Work_Hours",
        "Notifications_Per_Day",
        "Phone_Pickups"
    ])

    prediction = model.predict(user_data)

    result = le.inverse_transform(prediction)

    status = result[0]

    st.subheader("📊 Prediction Result")

    if status.lower() == "healthy":
        st.success(f"✅ {status}")

    elif status.lower() == "moderate":
        st.warning(f"⚠ {status}")

    else:
        st.error(f"🚨 {status}")

    st.subheader("💡 Recommendations")

    if screen_time > 8:
        st.write("• Reduce daily screen time.")

    if social_media > 4:
        st.write("• Limit social media usage.")

    if sleep < 7:
        st.write("• Increase sleep duration.")

    if pickups > 100:
        st.write("• Avoid checking your phone too frequently.")

    if status.lower() == "healthy":
        st.write("🎉 Great job! Keep maintaining healthy habits.")