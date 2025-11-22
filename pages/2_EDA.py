# pages/2_EDA.py
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_df
from utils import render_footer
st.set_page_config(page_title="EDA", layout="wide")
st.title("📊 Exploratory Data Analysis (EDA)")

df = get_df()
if df is None:
    st.info("Upload dataset on Home page to start EDA.")
else:
    st.subheader("Select column for univariate analysis")
    col = st.selectbox("Column", df.columns)
    if col:
        if pd.api.types.is_numeric_dtype(df[col]):
            fig = px.histogram(df, x=col, nbins=30, title=f"Histogram of {col}")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"**Summary statistics for {col}**")
            st.write(df[col].describe())
        else:
            fig = px.bar(df[col].value_counts().reset_index().rename(columns={"index":col, col:"count"}), x=col, y="count", title=f"Bar counts of {col}")
            st.plotly_chart(fig, use_container_width=True)
            st.write(df[col].value_counts().head(20))

    st.markdown("---")
    st.subheader("Bivariate analysis (interactive)")
    cols = df.columns.tolist()
    x = st.selectbox("X-axis", cols, index=0, key="eda_x")
    y = st.selectbox("Y-axis", cols, index=1, key="eda_y")

    if pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]):
        fig = px.scatter(df, x=x, y=y, hover_data=df.columns, title=f"{y} vs {x}")
        st.plotly_chart(fig, use_container_width=True)
    elif pd.api.types.is_numeric_dtype(df[y]) and not pd.api.types.is_numeric_dtype(df[x]):
        fig = px.box(df, x=x, y=y, title=f"{y} by {x}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        ctab = pd.crosstab(df[x], df[y])
        fig = px.imshow(ctab, text_auto=True, title=f"Heatmap of {x} vs {y}")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Correlation heatmap (numerical columns only)")
    numeric = df.select_dtypes(include=['int64','float64'])
    if numeric.shape[1] >= 2:
        corr = numeric.corr()
        fig = px.imshow(corr, text_auto=True, title="Correlation matrix")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numerical columns for correlation.")

# Footer
render_footer()