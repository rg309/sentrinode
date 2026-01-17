import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

# --- INITIAL STATE ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False



HERO_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syncopate:wght@700&display=swap');
.sentri-hero {
    font-family: 'Syncopate', sans-serif;
    letter-spacing: 0.65rem;
    font-size: 2.75rem;
    text-transform: uppercase;
    text-align: center;
    margin-bottom: 1.5rem;
    background: linear-gradient(120deg, #0ea5e9, #38bdf8, #e0f2fe, #0ea5e9);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    animation: sentriShift 6s ease-in-out infinite;
}
@keyframes sentriShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
</style>
"""


def render_hero(text: str) -> None:
    st.markdown(HERO_STYLE, unsafe_allow_html=True)
    st.markdown(f"<div class='sentri-hero'>{text}</div>", unsafe_allow_html=True)


# --- UI LOGIC ---
def show_login():
    render_hero("SENTRINODE")
    st.title("SentriNode Login")
    with st.form("login_form"):
        st.text_input("Username")
        st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            # We will add DB check here later
            st.session_state.logged_in = True
            st.rerun()
    if st.button("Create an account"):
        st.session_state.show_signup = True
        st.rerun()


def show_signup():
    st.title("Create SentriNode Account")
    with st.form("signup_form"):
        st.text_input("Email")
        st.text_input("New Username")
        st.text_input("New Password", type="password")
        if st.form_submit_button("Register Node"):
            # We will add DB save here later
            st.success("Account created! Please log in.")
            st.session_state.show_signup = False
            st.rerun()
    if st.button("Back to Login"):
        st.session_state.show_signup = False
        st.rerun()


def show_dashboard():
    render_hero("SentriNode Operational Command")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Active Nodes", "1,240", "+12%")
    metric_cols[1].metric("Network Latency", "24ms", "-2ms")
    metric_cols[2].metric("Uptime", "99.9%", "Stable")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Node Distribution")
        chart_data = pd.DataFrame(
            {
                "north": [400, 420, 430, 445, 460],
                "south": [320, 330, 335, 350, 360],
                "edge": [510, 515, 520, 525, 530],
            }
        )
        st.area_chart(chart_data, height=340, use_container_width=True)
    with right:
        st.subheader("Recent Alerts")
        with st.expander("Security Feed", expanded=True):
            alerts = [
                "02:10 - Edge gateway elevated CPU",
                "01:52 - Auth service latency spike resolved",
                "01:33 - New node authenticated (plant-west)",
                "01:05 - Billing service deployed revision v3.4.1",
            ]
            for alert in alerts:
                st.write(f"• {alert}")


# --- MAIN NAVIGATION ---
if not st.session_state.logged_in:
    if st.session_state.show_signup:
        show_signup()
    else:
        show_login()
else:
    show_dashboard()
    with st.sidebar:
        st.title("Console")
        st.caption("Session Controls")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()
