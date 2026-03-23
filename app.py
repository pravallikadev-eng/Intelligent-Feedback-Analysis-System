"""
Streamlit Dashboard — Feedback Analysis & Ticket Monitoring
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os
import json
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

# ─── Page Config ───
st.set_page_config(
    page_title="FeedbackAI — Ticket System",
    page_icon="🎫",
    layout="wide",
)

# ─── Constants ───
TICKETS_PATH = "output/generated_tickets.csv"
LOG_PATH = "output/processing_log.csv"
METRICS_PATH = "output/metrics.csv"
REVIEWS_PATH = "data/app_store_reviews.csv"
EMAILS_PATH = "data/support_emails.csv"
EXPECTED_PATH = "data/expected_classifications.csv"

CATEGORY_COLORS = {
    "Bug": "🔴",
    "Feature Request": "🔵",
    "Praise": "🟢",
    "Complaint": "🟡",
    "Spam": "⚫",
}
PRIORITY_COLORS = {
    "Critical": "🔴",
    "High": "🟠",
    "Medium": "🟡",
    "Low": "🟢",
}

# ─── Helpers ───
def load_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path).fillna("")
        except:
            return pd.DataFrame()
    return pd.DataFrame()

def save_tickets(df):
    df.to_csv(TICKETS_PATH, index=False)

# ─── Sidebar ───
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/ticket.png", width=60)
    st.title("FeedbackAI")
    st.caption("Multi-Agent Feedback System")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🎫 Tickets", "📥 Raw Feedback", "⚙️ Run Pipeline", "📈 Analytics"],
        label_visibility="collapsed"
    )

    st.divider()

    # Config panel
    with st.expander("⚙️ Configuration"):
        model = st.selectbox("Model", ["claude-opus-4-5", "gpt-4o", "gemini-2.5-flash"])
        batch_size = st.slider("Batch Size", 1, 20, 5)
        threshold = st.slider("Classification Confidence Threshold", 0.5, 1.0, 0.7)
        st.session_state["config"] = {
            "model": model,
            "batch_size": batch_size,
            "threshold": threshold
        }
        if st.button("💾 Save Config"):
            st.success("Config saved!")

    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

# ─── Dashboard ───
if page == "📊 Dashboard":
    st.title("📊 Feedback Intelligence Dashboard")

    tickets_df = load_csv(TICKETS_PATH)
    metrics_df = load_csv(METRICS_PATH)

    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    total = len(tickets_df)
    bugs = len(tickets_df[tickets_df.get("category", pd.Series()) == "Bug"]) if total else 0
    features = len(tickets_df[tickets_df.get("category", pd.Series()) == "Feature Request"]) if total else 0
    critical = len(tickets_df[tickets_df.get("priority", pd.Series()) == "Critical"]) if total else 0
    open_t = len(tickets_df[tickets_df.get("status", pd.Series()) == "Open"]) if total else 0

    col1.metric("Total Tickets", total)
    col2.metric("🔴 Bugs", bugs)
    col3.metric("🔵 Features", features)
    col4.metric("🚨 Critical", critical)
    col5.metric("📂 Open", open_t)

    st.divider()

    if total > 0:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("By Category")
            cat_counts = tickets_df["category"].value_counts()
            st.bar_chart(cat_counts)

        with col_b:
            st.subheader("By Priority")
            pri_counts = tickets_df["priority"].value_counts()
            st.bar_chart(pri_counts)

        # Recent tickets table
        st.subheader("🕐 Recent Tickets")
        display_cols = ["ticket_id", "category", "priority", "title", "status", "created_at"]
        available = [c for c in display_cols if c in tickets_df.columns]
        st.dataframe(
            tickets_df[available].tail(10).iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No tickets yet. Run the pipeline to generate tickets.")
        st.image("https://img.icons8.com/fluency/200/inbox.png", width=120)

# ─── Tickets ───
elif page == "🎫 Tickets":
    st.title("🎫 Ticket Management")
    tickets_df = load_csv(TICKETS_PATH)

    if tickets_df.empty:
        st.warning("No tickets found. Run the pipeline first.")
    else:
        # Filters
        col1, col2, col3 = st.columns(3)
        categories = ["All"] + sorted(tickets_df["category"].unique().tolist()) if "category" in tickets_df.columns else ["All"]
        priorities = ["All"] + sorted(tickets_df["priority"].unique().tolist()) if "priority" in tickets_df.columns else ["All"]
        statuses = ["All"] + sorted(tickets_df["status"].unique().tolist()) if "status" in tickets_df.columns else ["All"]

        cat_filter = col1.selectbox("Category", categories)
        pri_filter = col2.selectbox("Priority", priorities)
        sta_filter = col3.selectbox("Status", statuses)

        filtered = tickets_df.copy()
        if cat_filter != "All" and "category" in filtered.columns:
            filtered = filtered[filtered["category"] == cat_filter]
        if pri_filter != "All" and "priority" in filtered.columns:
            filtered = filtered[filtered["priority"] == pri_filter]
        if sta_filter != "All" and "status" in filtered.columns:
            filtered = filtered[filtered["status"] == sta_filter]

        st.caption(f"Showing {len(filtered)} of {len(tickets_df)} tickets")

        # Ticket cards + inline edit
        for idx, row in filtered.iterrows():
            cat_icon = CATEGORY_COLORS.get(row.get("category", ""), "⚪")
            pri_icon = PRIORITY_COLORS.get(row.get("priority", ""), "⚪")

            with st.expander(
                f"{cat_icon} {pri_icon} **{row.get('ticket_id','')}** — {row.get('title','')[:70]}"
            ):
                col_l, col_r = st.columns([2, 1])
                with col_l:
                    new_title = st.text_input("Title", value=row.get("title", ""), key=f"title_{idx}")
                    new_desc = st.text_area("Description", value=row.get("description", ""), key=f"desc_{idx}", height=100)
                    new_steps = st.text_area("Steps to Reproduce", value=row.get("steps_to_reproduce", ""), key=f"steps_{idx}", height=80)
                with col_r:
                    new_cat = st.selectbox("Category", ["Bug", "Feature Request", "Praise", "Complaint", "Spam"],
                        index=["Bug","Feature Request","Praise","Complaint","Spam"].index(row.get("category","Bug")) if row.get("category","Bug") in ["Bug","Feature Request","Praise","Complaint","Spam"] else 0,
                        key=f"cat_{idx}")
                    new_pri = st.selectbox("Priority", ["Critical", "High", "Medium", "Low"],
                        index=["Critical","High","Medium","Low"].index(row.get("priority","Medium")) if row.get("priority","Medium") in ["Critical","High","Medium","Low"] else 2,
                        key=f"pri_{idx}")
                    new_status = st.selectbox("Status", ["Open", "In Progress", "Resolved", "Closed", "Rejected"],
                        index=["Open","In Progress","Resolved","Closed","Rejected"].index(row.get("status","Open")) if row.get("status","Open") in ["Open","In Progress","Resolved","Closed","Rejected"] else 0,
                        key=f"sta_{idx}")
                    qs = row.get("quality_score", "")
                    st.metric("Quality Score", f"{qs}/10" if qs else "N/A")

                if st.button(f"💾 Save Changes", key=f"save_{idx}"):
                    tickets_df.at[idx, "title"] = new_title
                    tickets_df.at[idx, "description"] = new_desc
                    tickets_df.at[idx, "steps_to_reproduce"] = new_steps
                    tickets_df.at[idx, "category"] = new_cat
                    tickets_df.at[idx, "priority"] = new_pri
                    tickets_df.at[idx, "status"] = new_status
                    save_tickets(tickets_df)
                    st.success(f"✅ Ticket {row.get('ticket_id','')} updated!")
                    st.rerun()

# ─── Raw Feedback ───
elif page == "📥 Raw Feedback":
    st.title("📥 Raw Feedback Data")
    tab1, tab2, tab3 = st.tabs(["App Store Reviews", "Support Emails", "Expected Classifications"])
    with tab1:
        df = load_csv(REVIEWS_PATH)
        st.caption(f"{len(df)} reviews loaded")
        st.dataframe(df, use_container_width=True, hide_index=True)
    with tab2:
        df = load_csv(EMAILS_PATH)
        st.caption(f"{len(df)} emails loaded")
        st.dataframe(df, use_container_width=True, hide_index=True)
    with tab3:
        df = load_csv(EXPECTED_PATH)
        st.caption(f"{len(df)} expected classifications")
        st.dataframe(df, use_container_width=True, hide_index=True)

# ─── Run Pipeline ───
elif page == "⚙️ Run Pipeline":
    st.title("⚙️ Run Pipeline")
    st.info(
        "This page triggers the multi-agent pipeline. "
        "Make sure your API key is set as an environment variable before running."
    )

    cfg = st.session_state.get("config", {})
    col1, col2 = st.columns(2)
    col1.metric("Model", cfg.get("model", "claude-opus-4-5"))
    col2.metric("Batch Size", cfg.get("batch_size", 5))

    api_key = st.text_input("API Key (optional — or set via env var)", type="password")
    source = st.multiselect("Process from:", ["App Store Reviews", "Support Emails"], default=["App Store Reviews", "Support Emails"])

    if st.button("🚀 Run Pipeline", type="primary"):
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key

        with st.spinner("Running multi-agent pipeline... This may take a minute."):
            try:
                from agents.pipeline import run_pipeline, CONFIG
                CONFIG["model"] = cfg.get("model", "claude-opus-4-5")
                result = run_pipeline(batch_size=cfg.get("batch_size", 5))
                st.success(f"✅ Pipeline complete! Processed {result.get('items_processed')} items in {result.get('elapsed')}")
                st.json(result)
            except ImportError as e:
                st.error(f"Import error: {e}. Make sure crewai is installed: pip install crewai")
            except Exception as e:
                st.error(f"Pipeline error: {e}")

    st.divider()
    st.subheader("📋 Processing Log")
    log_df = load_csv(LOG_PATH)
    if not log_df.empty:
        st.dataframe(log_df.tail(20).iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.caption("No logs yet.")

# ─── Analytics ───
elif page == "📈 Analytics":
    st.title("📈 Analytics & Performance")

    tickets_df = load_csv(TICKETS_PATH)
    metrics_df = load_csv(METRICS_PATH)
    expected_df = load_csv(EXPECTED_PATH)

    if not metrics_df.empty:
        st.subheader("Pipeline Runs")
        st.dataframe(metrics_df.iloc[::-1], use_container_width=True, hide_index=True)

    if not tickets_df.empty and not expected_df.empty:
        st.subheader("Classification Accuracy")
        merged = tickets_df.merge(
            expected_df[["source_id", "category"]].rename(columns={"category": "expected_category"}),
            on="source_id", how="inner"
        )
        if not merged.empty and "category" in merged.columns:
            correct = (merged["category"] == merged["expected_category"]).sum()
            accuracy = round(correct / len(merged) * 100, 1)
            st.metric("Accuracy vs Expected", f"{accuracy}%", f"{correct}/{len(merged)} correct")
            st.dataframe(
                merged[["source_id", "category", "expected_category"]].assign(
                    match=merged["category"] == merged["expected_category"]
                ),
                use_container_width=True, hide_index=True
            )

    if not tickets_df.empty and "quality_score" in tickets_df.columns:
        scores = pd.to_numeric(tickets_df["quality_score"], errors="coerce").dropna()
        if len(scores) > 0:
            st.subheader("Quality Scores")
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Score", f"{scores.mean():.1f}/10")
            col2.metric("Min Score", f"{scores.min():.0f}/10")
            col3.metric("Flagged (<7)", len(scores[scores < 7]))
