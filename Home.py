# Home.py
import streamlit as st
import pandas as pd
import io
from utils import set_df_in_state, get_df, clear_df, to_csv_bytes
import streamlit.components.v1 as components  # <-- add this
from utils import render_footer

st.set_page_config(page_title="EDA Multi-Page App — Home", layout="wide")
st.title("📦 EDA Multi-Page App (Multi-Page Streamlit)")

# =========================
# Particles.js Banner
# =========================
banner_height = 250
components.html(
    f"""
    <div id="particles-js" style="height:{banner_height}px;"></div>
    <script src="https://cdn.jsdelivr.net/npm/particles.js"></script>
    <script>
        particlesJS("particles-js", {{
          "particles": {{
            "number": {{ "value": 80, "density": {{ "enable": true, "value_area": 800 }} }},
            "color": {{ "value": "#d2d6d6" }},
            "shape": {{ "type": "circle" }},
            "opacity": {{ "value": 0.5 }},
            "size": {{ "value": 3, "random": true }},
            "line_linked": {{
              "enable": true,
              "distance": 150,
              "color": "#d2d6d6",
              "opacity": 0.6,
              "width": 1.5
            }},
            "move": {{
              "enable": true,
              "speed": 2,
              "direction": "none",
              "random": false,
              "straight": false,
              "out_mode": "bounce"
            }}
          }},
          "interactivity": {{
            "detect_on": "canvas",
            "events": {{
              "onhover": {{ "enable": true, "mode": "repulse" }},
              "onclick": {{ "enable": true, "mode": "push" }}
            }},
            "modes": {{
              "repulse": {{ "distance": 100 }},
              "push": {{ "particles_nb": 4 }}
            }}
          }},
          "retina_detect": true
        }});
    </script>
    <style>
      #particles-js {{
        width: 100%;
        z-index: -1;
      }}
    </style>
    """,
    height=banner_height,
    width=1600,
)

st.write("Welcome — upload your CSV on this Home page and then switch to the other pages in the sidebar (Overview, EDA, Visuals, Cleaning, ML).")

# Upload widget
uploaded_file = st.file_uploader("Upload a CSV file to begin", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        set_df_in_state(df)
        st.success("✅ File uploaded and stored in session state.")
        st.write("Preview:")
        df_display = df.copy()
        bool_cols = df_display.select_dtypes(include=['bool']).columns
        df_display[bool_cols] = df_display[bool_cols].astype(str)
        st.dataframe(df_display.head())
        st.markdown(f"**Rows**: {df.shape[0]} — **Columns**: {df.shape[1]}")
        st.download_button("Download original CSV", data=to_csv_bytes(df), file_name="original_upload.csv", mime="text/csv")
    except Exception as e:
        st.error("Couldn't read the CSV. Try a different file.")
        st.exception(e)
# ============================
# Sample datasets section
# ============================
st.markdown("### Or try with a sample dataset")

sample_choice = st.selectbox(
    "Choose a sample dataset",
    ["None", "Iris", "Titanic", "Tips", "Telco Churn"]
)

# Load preview (in memory only)
preview_df = None
if sample_choice != "None":
    try:
        if sample_choice == "Iris":
            preview_df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
        elif sample_choice == "Titanic":
            preview_df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
        elif sample_choice == "Tips":
            preview_df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")
        elif sample_choice == "Telco Churn":
            preview_df = pd.read_csv("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv")

        st.write(f"### Preview of {sample_choice} dataset")
        df_display = preview_df.copy()
        bool_cols = df_display.select_dtypes(include=['bool']).columns
        df_display[bool_cols] = df_display[bool_cols].astype(str)

        st.dataframe(df_display.head())
        st.markdown(f"**Rows:** {preview_df.shape[0]} — **Columns:** {preview_df.shape[1]}")

    except Exception as e:
        st.error("Could not preview the sample dataset.")
        st.exception(e)

st.markdown("---")

# Save selected dataset to session state
if st.button("Load sample dataset"):
    if sample_choice == "None":
        st.warning("Please choose a dataset.")
    else:
        try:
            set_df_in_state(preview_df)
            # store success message
            st.session_state["sample_loaded"] = sample_choice
            st.rerun()

        except Exception as e:
            st.error("Could not load the sample dataset.")
            st.exception(e)

# Show success message after rerun
if "sample_loaded" in st.session_state:
    st.success(f"Loaded sample dataset: {st.session_state['sample_loaded']}")
    del st.session_state["sample_loaded"]


##

if st.button("Clear uploaded data from session"):
    clear_df()
    st.success("Cleared.")

# Footer
render_footer()
