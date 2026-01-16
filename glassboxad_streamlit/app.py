import streamlit as st

st.set_page_config(page_title="GlassboxAD", layout="wide")

st.title("GlassboxAD")
st.caption("Streamlit app for HYDRA interactive inspection + benchmark evaluation")

st.markdown(
    """
Use the sidebar to open:
- **HYDRA**: select a preloaded time series (CSV) and run detection, then explore VUS-PR.
- **Benchmark**: select datasets/methods/metrics and view CD plot, boxplot, tables, and runtime.

**Repo layout requirement for HYDRA:** put the **HYDRA** and **TSB-AD** folders next to this app so imports work.
"""
)
