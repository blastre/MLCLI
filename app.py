import streamlit as st
import pandas as pd
import tempfile
import os
import seaborn as sns
import matplotlib.pyplot as plt

from MLtool import MLCLIPipeline
from Datatool import DataTool

# ----- ML pipeline runner -----
def run_ml(df, target, api_key, temperature, test_size):
    # write the uploaded DataFrame to a temporary CSV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        df.to_csv(tmp.name, index=False)
        data_path = tmp.name

    # run the CLI-equivalent pipeline
    pipe = MLCLIPipeline(api_key=api_key, test_size=test_size, temperature=temperature)
    res = pipe.run_pipeline(data_path, target, output_dir="saved_models")

    # display results
    st.subheader("📊 Raw Data Summary")
    st.write(f"**Shape:** {res['raw_data_info']['shape'][0]} rows, {res['raw_data_info']['shape'][1]} columns")
    st.json(res["raw_data_info"]["missing_values"])

    st.subheader("🚀 Pipeline Execution Details")
    st.write(f"**Task Type:** {res['task_type'].capitalize()}")
    st.write(f"**Selected Model:** {res['model'].__class__.__name__}")

    st.subheader("🏆 Performance Metrics")
    st.json(res["metrics"])

    st.success(f"Model saved at: `{res['model_path']}`")

    # clean up temp file
    os.remove(data_path)


# ----- Data analysis runner -----
def run_data(df, target, api_key, temperature, top_n):
    # write the uploaded DataFrame to a temporary CSV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        df.to_csv(tmp.name, index=False)
        data_path = tmp.name

    dt = DataTool(api_key=api_key)
    data = dt.load_data(data_path)
    info = dt.analyze_data(data, target)
    corr = dt.compute_correlations(data)

    # Overview
    st.subheader("📋 Dataset Overview")
    st.write(
        f"Rows: {info['rows']}  |  "
        f"Columns: {info['columns']}  |  "
        f"Duplicates: {info['duplicates']['duplicate_rows_pct']:.2f}%"
    )

    # Numeric stats
    st.subheader("🔢 Numeric Statistics")
    num_df = pd.DataFrame(info["numeric_stats"]).T
    st.dataframe(num_df)

    # Categorical stats
    st.subheader("🗂️ Categorical Statistics")
    cat_df = pd.DataFrame(info["categorical_stats"]).T
    st.dataframe(cat_df)

    # Correlation heatmap
    st.subheader("🔥 Correlation Heatmap")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax1)
    st.pyplot(fig1)

    # Top scatter plots
    st.subheader(f"📊 Top {top_n} Features vs {target}")
    top_feats = corr[target].abs().nlargest(top_n).index
    for feat in top_feats:
        fig_sc, ax_sc = plt.subplots()
        ax_sc.scatter(data[feat], data[target], alpha=0.6)
        ax_sc.set_title(f"{feat} vs {target} (corr={corr.loc[feat, target]:.2f})")
        ax_sc.set_xlabel(feat)
        ax_sc.set_ylabel(target)
        st.pyplot(fig_sc)

    # Missing data bar chart
    st.subheader("❌ Missing Data (%) by Column")
    miss = (data.isnull().sum() / len(data) * 100).sort_values(ascending=False)
    miss = miss[miss > 0]
    if not miss.empty:
        fig2, ax2 = plt.subplots(figsize=(6, len(miss) * 0.4))
        sns.barplot(x=miss.values, y=miss.index, ax=ax2)
        ax2.set_xlabel("% Missing")
        ax2.set_ylabel("Column")
        st.pyplot(fig2)
    else:
        st.write("No missing data.")

    # Target distribution
    st.subheader(f"📈 Distribution of Target: {target}")
    tinfo = info.get("target_analysis", {})
    fig3, ax3 = plt.subplots()
    if tinfo.get("type") == "categorical":
        dist = tinfo.get("distribution", {})
        sns.barplot(x=list(dist.keys()), y=list(dist.values()), ax=ax3)
    else:
        sns.histplot(data[target].dropna(), kde=True, ax=ax3)
    ax3.set_title(f"Distribution of {target}")
    st.pyplot(fig3)

    # cache for chat
    os.remove(data_path)
    return dt, info, corr


def main():
    st.set_page_config(page_title="ML & Data Tool GUI", layout="wide")
    st.title("🚀 ML & Data Automation GUI")

    # instructions
    with st.expander("📖 Instructions", expanded=True):
        st.markdown(
            """
            - Upload a CSV via the sidebar  
            - (Optional) Enter your GROQ API Key (defaults to env var)  
            - Choose ML Pipeline or Data Analysis mode  
            - Select the target column and adjust settings  
            - Click ▶️ to run, then chat interactively in Data Analysis mode  
            """
        )

    # sidebar inputs
    api_key = st.sidebar.text_input("🔑 API Key", type="password") or None
    st.sidebar.info("If left blank, uses GROQ_API / GROQ_API_KEY environment variable.")
    temperature = st.sidebar.slider("🌡️ Temperature", 0.0, 1.0, 0.7, help="LLM temperature")
    uploaded = st.sidebar.file_uploader("📂 Upload CSV", type=["csv"])
    mode = st.sidebar.radio("⚙️ Mode", ["ML Pipeline", "Data Analysis"])

    # ML Pipeline settings
    if mode == "ML Pipeline":
        test_size = st.sidebar.slider("📊 Test Size", 0.1, 0.5, 0.2, 0.05)
        run_ml_btn = st.sidebar.button("▶️ Run ML Pipeline")

    # Data Analysis settings
    else:
        top_n = st.sidebar.slider("🔍 Top N Features", 3, 15, 5)
        show_summary = st.sidebar.checkbox("🤖 Show LLM Summary", value=True)
        run_data_btn = st.sidebar.button("▶️ Run Data Analysis")

    # when a file is uploaded
    if uploaded:
        df = pd.read_csv(uploaded)
        target = st.sidebar.selectbox("🎯 Target Column", df.columns)

        if mode == "ML Pipeline" and run_ml_btn:
            run_ml(df, target, api_key, temperature, test_size)

        if mode == "Data Analysis" and run_data_btn:
            dt, info, corr = run_data(df, target, api_key, temperature, top_n)

            # store for chat
            st.session_state.dt = dt
            st.session_state.info = info
            st.session_state.corr = corr
            st.session_state.target = target
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            if show_summary:
                st.subheader("🤖 LLM Summary")
                summary = dt.get_llm_summary(info, corr, target)
                st.info(summary)

    else:
        st.info("Upload a CSV to get started.")

    # chat interface for Data Analysis
    if mode == "Data Analysis" and "dt" in st.session_state:
        st.subheader("💬 Chat with your Data")
        for msg in st.session_state.chat_history:
            speaker = "You" if msg["role"] == "user" else "AI"
            st.markdown(f"**{speaker}:** {msg['content']}")

        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Ask anything about the dataset…", key="chat_input")
            submit = st.form_submit_button("Send")

        if submit and user_input:
            # append user question
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            # get AI response using cached context
            ai_resp = st.session_state.dt.ask_query(user_input)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_resp})
            # rerun to update the chat
            st.rerun()


if __name__ == "__main__":
    main()
