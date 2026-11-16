import pandas as pd
import streamlit as st
from datetime import date
from io import BytesIO
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


st.set_page_config(
    page_title="Pennywise | Expense Tracker",
    page_icon="💸",
    layout="wide",
)


CATEGORIES = ["Food", "Transport", "Shopping", "Bills", "Education", "Health", "Entertainment", "Other"]
TRAINING_DATA_PATH = Path(__file__).with_name("expense_training_data.csv")


@st.cache_resource
def train_category_model(training_csv: bytes):
    examples = pd.read_csv(BytesIO(training_csv))
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 3), strip_accents="unicode", sublinear_tf=True),
        MultinomialNB(),
    )
    return model.fit(examples["description"].astype(str), examples["category"].astype(str))


training_data = TRAINING_DATA_PATH.read_bytes()
training_examples = pd.read_csv(BytesIO(training_data))
category_model = train_category_model(training_data)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 3rem; }
      .hero { padding: 1.5rem 1.8rem; border-radius: 18px; background: linear-gradient(120deg, #142d4e, #246b72); color: white; margin-bottom: 1.25rem; }
      .hero h1 { margin: 0; font-size: 2.2rem; }
      .hero p { margin: .45rem 0 0; color: #dcecf0; }
      div[data-testid="stMetric"] { background: #f5f8fa; border: 1px solid #e7edf0; padding: 1rem; border-radius: 14px; }
      .st-key-summary_metrics [data-testid="stMetric"] { background: #e8f5e9; border-color: #a5d6a7; }
      .st-key-summary_metrics [data-testid="stMetricLabel"],
      .st-key-summary_metrics [data-testid="stMetricLabel"] *,
      .st-key-summary_metrics [data-testid="stMetricValue"],
      .st-key-summary_metrics [data-testid="stMetricValue"] * { color: #166534 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "expenses" not in st.session_state:
    st.session_state.expenses = []

st.markdown(
    """
    <div class="hero">
      <h1>💸 Pennywise</h1>
      <p>An ML-powered expense tracker. Add a purchase and see where your money goes.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("➕ Add an expense")
    if st.session_state.pop("clear_expense_note", False):
        st.session_state["expense_note"] = ""

    amount = st.number_input("Amount (₹)", min_value=0.01, value=100.0, step=10.0, format="%.2f")
    note = st.text_input(
        "What was it for?",
        placeholder="e.g. Lunch with friends",
        key="expense_note",
    )
    if note.strip():
        probabilities = category_model.predict_proba([note])[0]
        best_index = probabilities.argmax()
        suggested_category = category_model.named_steps["multinomialnb"].classes_[best_index]
        confidence = float(probabilities[best_index])
        st.info(f"🤖 Model suggestion: **{suggested_category}** ({confidence:.0%} confidence)")
    else:
        suggested_category = "Other"
        confidence = 0.0
        st.caption("Enter a description to get an ML category suggestion.")

    st.caption(
        f"The classifier learns from {len(training_examples)} labeled descriptions "
        f"across {training_examples['category'].nunique()} categories."
    )

    category_mode = st.radio(
        "Choose category",
        ["Use ML suggestion", "Choose manually"],
        horizontal=True,
    )
    if category_mode == "Choose manually":
        category = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index(suggested_category))
    else:
        category = suggested_category

    with st.form("expense_form", clear_on_submit=True):
        spent_on = st.date_input("Date", value=date.today())
        submitted = st.form_submit_button("Add expense", use_container_width=True, type="primary")

    if submitted:
        st.session_state.expenses.append(
            {
                "Date": spent_on,
                "Description": note.strip() or category,
                "Category": category,
                "Amount (₹)": float(amount),
            }
        )
        st.session_state["clear_expense_note"] = True
        st.rerun()


if not st.session_state.expenses:
    st.info("Your tracker is ready. Add your first expense using the form in the sidebar.")
else:
    expenses = pd.DataFrame(st.session_state.expenses)
    expenses["Month"] = pd.to_datetime(expenses["Date"]).dt.strftime("%B %Y")
    month_options = sorted(expenses["Month"].unique(), key=lambda value: pd.to_datetime(value, format="%B %Y"), reverse=True)

    filter_col, _ = st.columns([1, 3])
    with filter_col:
        selected_month = st.selectbox("Show month", ["All months", *month_options])

    visible = expenses if selected_month == "All months" else expenses[expenses["Month"] == selected_month]
    total = visible["Amount (₹)"].sum()
    count = len(visible)
    average = total / count if count else 0

    with st.container(key="summary_metrics"):
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Total spent", f"₹{total:,.2f}")
        metric2.metric("Expenses", f"{count}")
        metric3.metric("Average per expense", f"₹{average:,.2f}")

    chart_col, top_col = st.columns(2)
    with chart_col:
        st.subheader("Spending by category")
        category_totals = visible.groupby("Category")["Amount (₹)"].sum().sort_values(ascending=False)
        st.bar_chart(category_totals, color="#4b2a80")
    with top_col:
        st.subheader("Quick insight")
        if visible.empty:
            st.write("No expenses recorded for this month yet.")
        else:
            biggest_category = category_totals.idxmax()
            share = category_totals.max() / total * 100 if total else 0
            st.write(f"Your highest spending category is **{biggest_category}** at **₹{category_totals.max():,.2f}**.")
            st.progress(min(int(share), 100), text=f"{share:.0f}% of this period's spending")
            st.caption("Tip: Add expenses as you go to keep this snapshot up to date.")

    st.subheader("Expense history")
    history = visible.sort_values("Date", ascending=False)[["Date", "Description", "Category", "Amount (₹)"]]
    st.dataframe(history, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download this view as CSV",
        data=history.to_csv(index=False).encode("utf-8"),
        file_name="pennywise_expenses.csv",
        mime="text/csv",
    )

st.caption("Mini project · Built with Streamlit, pandas, and scikit-learn")
