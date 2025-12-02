# utils.py
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io

def set_df_in_state(df):
    st.session_state['df'] = df.copy()
    st.session_state['df_display'] = df.copy()

def get_df():
    return st.session_state.get('df', None)

def get_display_df():
    return st.session_state.get('df_display', None)

def clear_df():
    if 'df' in st.session_state: del st.session_state['df']
    if 'df_display' in st.session_state: del st.session_state['df_display']

def to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

def train_test_split_xy(df, target, test_size=0.2, random_state=42, scale=False):
    X = df.drop(columns=[target])
    y = df[target]
    # simple: drop non-numeric columns (user should convert in cleaning page)
    X_num = X.select_dtypes(include=['int64','float64'])
    X_train, X_test, y_train, y_test = train_test_split(X_num, y, test_size=test_size, random_state=random_state)
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train, X_test, y_train, y_test

def render_footer():
    import streamlit as st
    st.markdown("---")
    st.markdown(
        '<p style="text-align:right; font-size:12px; color:gray;">© khairilradzali</p>',
        unsafe_allow_html=True
    )

