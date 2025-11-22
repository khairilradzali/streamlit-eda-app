# pages/3_Visuals.py
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import get_df
from utils import render_footer

st.set_page_config(page_title="Visuals", layout="wide")
st.title("📈 Visualizations (Interactive)")

df = get_df()
if df is None:
    st.info("Upload a dataset on Home page first.")
else:
    st.sidebar.header("Plot controls")
    plot_type = st.sidebar.selectbox("Plot type", ["Scatter", "Line", "Bar", "Box", "Histogram", "Pie"])
    cols = df.columns.tolist()
    x = st.sidebar.selectbox("X column", cols, index=0)
    y = st.sidebar.selectbox("Y column", cols, index=1)
    color = st.sidebar.selectbox("Color (optional)", [None] + cols, index=0)

    st.write(f"### {plot_type} of {y} vs {x}")
    try:
        if plot_type == "Scatter":
            fig = px.scatter(df, x=x, y=y, color=color, hover_data=df.columns)
        elif plot_type == "Line":
            fig = px.line(df, x=x, y=y, color=color)
        elif plot_type == "Bar":
            fig = px.bar(df, x=x, y=y, color=color)
        elif plot_type == "Box":
            fig = px.box(df, x=x, y=y, color=color)
        elif plot_type == "Histogram":
            fig = px.histogram(df, x=x, color=color)
        elif plot_type == "Pie":
            fig = px.pie(df, names=x, values=y)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error("Could not build the plot. Try different columns.")
        st.exception(e)

# Footer
render_footer()