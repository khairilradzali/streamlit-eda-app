# pages/1_Overview.py
import streamlit as st
import pandas as pd
from utils import get_df, to_csv_bytes
import io
from utils import render_footer

st.set_page_config(page_title="Overview", layout="wide")
st.title("🧭 Data Overview")

df = get_df()
if df is None:
    st.info("No dataset found. Upload a CSV on the Home page.")
else:
    st.subheader("Preview")
    df_display = df.copy()
    bool_cols = df_display.select_dtypes(include=['bool']).columns
    df_display[bool_cols] = df_display[bool_cols].astype(str)
    st.dataframe(df_display.head())

    st.subheader("Basic Info")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing values", int(df.isnull().sum().sum()))

    st.subheader("Data Types")
    st.dataframe(pd.DataFrame(df.dtypes, columns=["dtype"]))

    st.subheader("Duplicate rows")
    dup_count = df.duplicated().sum()
    st.write(f"Duplicated rows: {dup_count}")
    if dup_count > 0:
        if st.button("Show duplicated rows"):
            st.dataframe(df[df.duplicated(keep=False)].sort_index())

    st.subheader("Column level missing values")
    miss = df.isnull().sum().sort_values(ascending=False)
    st.dataframe(miss)
    st.download_button("Download overview CSV", data=to_csv_bytes(df), file_name="overview_dataset.csv", mime="text/csv")

# Footer
render_footer()