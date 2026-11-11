# Employee Attrition Risk (Streamlit)

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy on Streamlit Community Cloud:
1. Push these files to a GitHub repo (keep decision_tree_model.pkl next to app.py).
2. Go to share.streamlit.io -> New app -> pick the repo, branch, and `app.py`.
3. Deploy.

Note: scikit-learn is pinned to 1.6.1 because the model was saved with that version.
