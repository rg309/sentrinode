import streamlit as st

# --- INITIAL STATE ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False


# --- UI LOGIC ---
def show_login():
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


# --- MAIN NAVIGATION ---
if not st.session_state.logged_in:
    if st.session_state.show_signup:
        show_signup()
    else:
        show_login()
else:
    st.title("SentriNode Dashboard")
    st.write("Welcome to the production environment.")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
