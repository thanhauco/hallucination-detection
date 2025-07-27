import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from hallucination.metrics import HallucinationMetrics

st.set_page_config(page_title="Hallucination Monitor", layout="wide")

st.title("🤖 Hallucination Rate Monitor (2026)")

st.markdown("""
This dashboard visualizes the **Hallucination Rate** of your LLM application in real-time.
It tracks metrics across:
- **Semantic Uncertainty** (Entropy)
- **RAG Context Adherence** (Entailment)
- **Agentic Verification** (Web Facts)
- **Advanced Consistency** (CoVe & NER)
""")

# Mock Data Generation for Demo
@st.cache_data
def load_mock_data():
    dates = pd.date_range(start="2026-01-01", periods=100, freq="H")
    data = []
    for d in dates:
        # Simulate varying hallucination rates
        uncertainty = np.random.beta(2, 5) # Skewed towards low
        rag_fail = np.random.choice([0, 1], p=[0.9, 0.1])
        fact_fail = np.random.choice([0, 1], p=[0.95, 0.05])
        
        # Advanced Metrics
        cove_score = np.random.uniform(0.7, 1.0) # High consistency usually
        if fact_fail: cove_score = np.random.uniform(0.0, 0.4)
        
        ner_consistency = np.random.uniform(0.8, 1.0)
        
        data.append({
            "timestamp": d,
            "uncertainty_score": uncertainty,
            "rag_fail": rag_fail,
            "fact_check_fail": fact_fail,
            "cove_score": cove_score,
            "ner_consistency": ner_consistency,
            "model_version": "gpt-4o-2026"
        })
    return pd.DataFrame(data)

df = load_mock_data()

# Key Metrics
col1, col2, col3, col4 = st.columns(4)

avg_uncertainty = df["uncertainty_score"].mean()
rag_fail_rate = df["rag_fail"].mean()
fact_fail_rate = df["fact_check_fail"].mean()
avg_cove = df["cove_score"].mean()

total_hallucination_rate = (df["rag_fail"] | df["fact_check_fail"]).mean()

col1.metric("Hallucination Rate", f"{total_hallucination_rate:.1%}", delta="-2%")
col2.metric("Avg Uncertainty", f"{avg_uncertainty:.2f}")
col3.metric("RAG Adherence", f"{1.0 - rag_fail_rate:.1%}")
col4.metric("CoVe Consistency", f"{avg_cove:.2f}")

# Detailed Metrics
st.markdown("### Advanced Verification Metrics")
col5, col6 = st.columns(2)
col5.metric("Fact Accuracy", f"{1.0 - fact_fail_rate:.1%}")
col6.metric("Entity Consistency", f"{df['ner_consistency'].mean():.2f}")

# Charts
st.subheader("Trends Over Time")
st.line_chart(df.set_index("timestamp")[["uncertainty_score", "cove_score"]])

st.subheader("Failure Modes")
chart_data = pd.DataFrame({
    "Category": ["RAG Context", "Factuality", "Logic (CoVe)", "Visual/Multi-modal", "Tool Misuse"],
    "Count": [
        df["rag_fail"].sum(), 
        df["fact_check_fail"].sum(), 
        len(df[df["cove_score"] < 0.5]),
        3, 
        5
    ]
})
st.bar_chart(chart_data.set_index("Category"))

st.subheader("Recent Alerts")
st.error("Detected potential hallucination: 'The capital of Paris is London' (High Uncertainty)")
st.warning("Low confidence response detected in RAG pipeline.")
