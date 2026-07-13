import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Feedback Intelligence", layout="wide")


@st.cache_data
def load_data():
    d = pd.read_csv("reviews_analyzed.csv")
    d["date"] = pd.to_datetime(d["date"])
    return d


@st.cache_resource
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


names = joblib.load("theme_names.pkl")
kmeans = joblib.load("kmeans_model.pkl")
df = load_data()

st.title("\U0001F4F1 Customer Feedback Intelligence")
st.write("Turning 1,940 real ChatGPT app reviews into prioritized product insights.")

summary = df.groupby("theme").agg(count=("review", "size"), avg_rating=("rating", "mean"))
summary["theme_name"] = summary.index.map(names)
summary = summary.sort_values("avg_rating")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Theme priority map")
    fig, ax = plt.subplots(figsize=(7, 5))
    for t, row in summary.iterrows():
        ax.scatter(row["avg_rating"], row["count"], s=140,
                   color="#e74c3c" if row["avg_rating"] < 3 else "#3498db")
        ax.annotate(names[t], (row["avg_rating"] + 0.02, row["count"]))
    ax.axvline(3, color="grey", linestyle="--")
    ax.set_xlabel("Average rating  (← angrier)")
    ax.set_ylabel("Number of reviews")
    st.pyplot(fig)
with c2:
    st.subheader("Themes by rating")
    st.dataframe(summary[["theme_name", "count", "avg_rating"]].round(2))

st.subheader("Login complaints over time (% of weekly reviews)")
df["week"] = df["date"].dt.to_period("W").dt.start_time
weekly_total = df.groupby("week").size()
login_share = (df[df["theme"] == 2].groupby("week").size() / weekly_total * 100).fillna(0)
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(login_share.index, login_share.values, marker="o", color="#e74c3c")
ax2.set_ylabel("% about login")
st.pyplot(fig2)

st.subheader("\U0001F4CB Top recommendations")
st.markdown("""
1. **Account & login** (169 reviews, 1.53★) — #1 fix; spiked to ~40% of reviews in July (likely a release regression).
2. **App freezing / stuck screen** (50, 1.74★) — a distinct crash bug.
3. **Overheating / battery** (109, 2.93★) — high volume.
4. **Quick wins:** birthdate bug, invalid-request error.
5. **Roadmap:** iPad-optimized version (most-requested feature).
""")

st.subheader("\U0001F50E Classify your own review")
user_text = st.text_area("Paste a review and see which theme it matches:")
if st.button("Analyze") and user_text.strip():
    emb = load_embedder().encode([user_text])
    theme = int(kmeans.predict(emb)[0])
    st.success(f"This review falls under: **{names[theme]}**")
