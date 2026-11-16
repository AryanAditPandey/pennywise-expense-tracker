# Pennywise — Expense Tracker

A small Streamlit mini project for recording daily expenses, viewing totals by month and category, and downloading the visible history as CSV. It uses a scikit-learn text classifier to suggest an expense category from the description.

## Run it

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Use the sidebar to enter an expense description. A TF-IDF + Multinomial Naive Bayes model suggests a category; you can use that suggestion or choose a category manually. The dashboard updates with monthly totals, a category chart, and an expense history table. Entries are kept in Streamlit's session state, so download the CSV if you want to save them beyond the current browser session.

The model is trained locally from `expense_training_data.csv`, a curated dataset with 320 example descriptions (40 per category). It does not call an external AI service. TF-IDF extracts one-, two-, and three-word features, and Multinomial Naive Bayes predicts the category. The examples are hand-built for this demo, so predictions can still be wrong for descriptions the dataset does not cover.
