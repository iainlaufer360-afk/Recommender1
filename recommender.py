# -*- coding: utf-8 -*-
import os
import json
import requests

import streamlit as st
import pandas as pd
from pydantic import BaseModel, Field
from surprise import Dataset, Reader, BaselineOnly

# ---------------------------------------------------------------- page setup
st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="wide",
)

# ------------------------------------------------------------ data and model
@st.cache_data
def load_data():
    books = pd.read_csv("Books.csv", encoding="utf-8-sig", lineterminator="\n")
    ratings = pd.read_csv("Ratings.csv", encoding="utf-8-sig", lineterminator="\n")

    books.columns = books.columns.str.replace("\r", "").str.strip()
    ratings.columns = ratings.columns.str.replace("\r", "").str.strip()

    if "title" in books.columns:
        books["title"] = books["title"].str.replace("\r", "").str.strip()
    if "genres" in books.columns:
        books["genres"] = books["genres"].str.replace("\r", "").str.strip()

    title_of = dict(zip(books["book_id"], books["title"]))
    return books, ratings, title_of


@st.cache_resource
def fit_ubcf_model():
    _, ratings, _ = load_data()
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[["book_id", "user_id", "rating"]], reader)
    trainset = data.build_full_trainset()
    model = BaselineOnly(verbose=False)
    model.fit(trainset)
    return model


def recommend(model, user_id, min_ratings, top_n=10):
    books, ratings, title_of = load_data()

    counts = ratings["book_id"].value_counts()
    avg_rating = ratings.groupby("book_id")["rating"].mean()
    popular = set(counts[counts >= min_ratings].index)
    seen = set(ratings.loc[ratings["user_id"] == user_id, "book_id"])

    scored = [
        {
            "book_id": m,
            "title": title_of[m],
            "predicted": model.predict(user_id, m).est,
            "n_ratings": int(counts.get(m, 0)),
            "avg_rating": float(avg_rating.get(m, 0.0)),
        }
        for m in books["book_id"]
        if m not in seen and m in popular
    ]
    scored.sort(key=lambda r: -r["predicted"])
    return scored[:top_n]


# -------------------------------------------------------------- LLM section
GEMINI_MODEL = "gemini-2.5-flash-lite"


def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except (FileNotFoundError, KeyError):
        key = None
    return (key or os.environ.get("GEMINI_API_KEY", "")).strip()


def rerank_with_gemini(candidates, mood):
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("Gemini API key not configured.")

    catalog = "\n".join([f"- {c['title']}" for c in candidates])

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

    payload = {
        "contents": [{
            "parts": [{
                "text": f"User's request: {mood}\n\nBooks:\n{catalog}"
            }]
        }],
        "systemInstruction": {
            "parts": [{
                "text": (
                    "You are a book concierge. The candidate books below were already picked "
                    "for this user by collaborative filtering. Re-rank them by how well they fit "
                    "the user's request, best first, and give each a one-sentence reason. "
                    "Return the data strictly as a JSON array of objects, where each object has "
                    "the keys 'title' and 'reason'."
                )
            }]
        },
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, params={"key": api_key}, data=json.dumps(payload))

    if response.status_code != 200:
        raise RuntimeError(f"Gemini API Error ({response.status_code}): {response.text}")

    response_data = response.json()
    try:
        raw_text = response_data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(raw_text)

        class Pick:
            def __init__(self, d):
                self.title = d.get("title", "Unknown Title")
                self.reason = d.get("reason", "")

        return [Pick(item) for item in parsed]

    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to parse Gemini output: {e}")


# ------------------------------------------------------------------- UI ----
st.title("📚 Book Recommender")
st.caption("Collaborative filtering with optional Gemini re-ranking")

with st.spinner("Loading data and fitting model…"):
    books, ratings, title_of = load_data()
    ubcf = fit_ubcf_model()

# ---- Sidebar ----
with st.sidebar:
    st.header("Settings")

    user_ids = sorted(ratings["user_id"].unique())
    min_id, max_id = int(user_ids[0]), int(user_ids[-1])
    user_id_input = st.text_input(
        "User ID",
        value=str(min_id),
        help="Enter the ID of the user you want recommendations for.",
    )
    st.caption(f"Valid IDs: {min_id}–{max_id} · {len(user_ids)} users in the dataset")

    min_ratings = st.slider(
        "Minimum ratings per book",
        min_value=1,
        max_value=200,
        value=20,
        step=1,
        help="Hide rarely-rated books — CF can over-score titles with very few ratings.",
    )

# ---- Validate user ID ----
if not user_id_input.strip().isdigit():
    st.warning(f"Enter a numeric user ID between {min_id} and {max_id}.")
    st.stop()

user_id = int(user_id_input)
if user_id not in set(user_ids):
    st.warning(
        f"No ratings found for user {user_id}. "
        f"Enter an ID between {min_id} and {max_id} that exists in the data."
    )
    st.stop()

# ---- Recommendations ----
st.subheader(f"Top 10 recommendations for user {user_id}")

recs = recommend(ubcf, user_id, min_ratings, top_n=10)

if not recs:
    st.warning("No books pass the filter — try lowering the minimum rating count.")
    st.session_state.pop("candidates", None)
else:
    rec_df = pd.DataFrame([
        {
            "Book": c["title"],
            "# Ratings": c["n_ratings"],
            "Avg Rating (all users)": round(c["avg_rating"], 2),
            "Predicted (this user)": round(c["predicted"], 2),
        }
        for c in recs
    ])
    rec_df.index = rec_df.index + 1
    rec_df.index.name = "Rank"
    st.dataframe(rec_df, use_container_width=True)
    st.session_state["candidates"] = recs

# ---- User history ----
with st.expander(f"See what user {user_id} has already rated"):
    user_history = (
        ratings[ratings["user_id"] == user_id]
        .merge(books, on="book_id")[["title", "rating"]]
        .sort_values("rating", ascending=False)
        .reset_index(drop=True)
    )
    user_history.columns = ["Book", "Their Rating"]
    user_history.index = user_history.index + 1
    st.dataframe(user_history, use_container_width=True)

# ---- Gemini re-ranking ----
if "candidates" in st.session_state:
    st.divider()
    st.subheader("✨ Personalize with Gemini")
    st.caption(
        "Re-rank the recommendations above by your current mood. "
        "Each pick comes with a short explanation."
    )

    mood = st.text_input(
        "What are you in the mood for?",
        placeholder="e.g., something light and funny after a long day",
    )

    if st.button("Re-rank with Gemini", type="primary"):
        if not get_api_key():
            st.error(
                "Gemini API key not found. Add it to "
                "`.streamlit/secrets.toml` as `GEMINI_API_KEY = \"...\"` "
                "or set the `GEMINI_API_KEY` environment variable, then restart the app."
            )
        elif not mood.strip():
            st.warning("Tell me what you're in the mood for first.")
        else:
            with st.spinner("Asking Gemini to re-rank…"):
                try:
                    result = rerank_with_gemini(st.session_state["candidates"], mood.strip())
                    out_df = pd.DataFrame([
                        {"Book": m.title, "Why it fits": m.reason}
                        for m in result
                    ])
                    out_df.index = out_df.index + 1
                    out_df.index.name = "Rank"
                    st.dataframe(out_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Re-ranking failed: {e}")
