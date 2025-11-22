# pages/4_Cleaning.py
import streamlit as st
import pandas as pd
from utils import render_footer
from utils import get_df, set_df_in_state, to_csv_bytes

st.set_page_config(page_title="Cleaning", layout="wide")
st.title("🧹 Data Cleaning Tools")

df = get_df()
if df is None:
    st.info("Upload a CSV in Home first.")
else:
    st.write("Current dataset shape:", df.shape)
    st.subheader("Preview")
    st.dataframe(df.head())

    st.markdown("### Column operations")
    cols = st.multiselect("Select columns to drop", options=df.columns.tolist())
    if st.button("Drop selected columns"):
        if cols:
            df2 = df.drop(columns=cols)
            set_df_in_state(df2)
            st.success(f"Dropped columns: {cols}")
            st.experimental_rerun()
        else:
            st.info("No columns selected.")

    st.markdown("### Missing value handling")
    col_fill = st.selectbox("Select column to fill NA", options=[None] + list(df.columns))
    if col_fill:
        fill_val = st.text_input("Fill value (string or numeric). Leave blank to skip.")
        if st.button("Fill NA in column"):
            if fill_val == "":
                st.info("No fill value provided.")
            else:
                # attempt numeric cast
                try:
                    val = float(fill_val)
                except:
                    val = fill_val
                df[col_fill] = df[col_fill].fillna(val)
                set_df_in_state(df)
                st.success(f"Filled NA in {col_fill} with {val}")

    st.markdown("### Convert dtype")
    col_convert = st.selectbox("Select column to convert dtype", options=[None] + list(df.columns))
    dtype = st.selectbox("Convert to", ["int", "float", "str", "category", "datetime"])
    if st.button("Convert dtype"):
        if col_convert:
            try:
                if dtype == "int":
                    df[col_convert] = df[col_convert].astype("Int64")
                elif dtype == "float":
                    df[col_convert] = df[col_convert].astype(float)
                elif dtype == "str":
                    df[col_convert] = df[col_convert].astype(str)
                elif dtype == "category":
                    df[col_convert] = df[col_convert].astype("category")
                elif dtype == "datetime":
                    df[col_convert] = pd.to_datetime(df[col_convert], errors="coerce")
                set_df_in_state(df)
                st.success(f"Converted {col_convert} to {dtype}")
            except Exception as e:
                st.error("Conversion failed.")
                st.exception(e)

    st.markdown("### Duplicates and dropna")
    if st.button("Drop duplicate rows"):
        df2 = df.drop_duplicates()
        set_df_in_state(df2)
        st.success("Dropped duplicates.")
    if st.button("Drop rows with any NA"):
        df2 = df.dropna()
        set_df_in_state(df2)
        st.success("Dropped rows with NA.")

    st.markdown("---")
    st.subheader("Download cleaned dataset")
    st.download_button("Download cleaned CSV", data=to_csv_bytes(get_df()), file_name="cleaned_data.csv", mime="text/csv")

# Footer
render_footer()