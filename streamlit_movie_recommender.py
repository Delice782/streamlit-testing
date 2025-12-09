# streamlit_movie_recommender.py
# Single-file Streamlit app: lightweight movie recommender + search + insights
# Requirements: streamlit, pandas, scikit-learn, numpy, matplotlib

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Small Movie Recommender")

@st.cache_data
def load_sample_dataset():
    # Small sample dataset if user doesn't provide one
    data = [
        {"movieId": 1, "title": "Toy Story (1995)", "genres": "Animation|Children|Comedy", "description": "A story of toys that come to life."},
        {"movieId": 2, "title": "Jumanji (1995)", "genres": "Adventure|Children|Fantasy", "description": "A magical board game releases jungle dangers."},
        {"movieId": 3, "title": "Grumpier Old Men (1995)", "genres": "Comedy|Romance", "description": "Two old rivals still bicker and fight."},
        {"movieId": 4, "title": "Waiting to Exhale (1995)", "genres": "Comedy|Drama|Romance", "description": "Four women navigate relationships and careers."},
        {"movieId": 5, "title": "Father of the Bride Part II (1995)", "genres": "Comedy", "description": "Father deals with pregnancy surprises and laughter."},
        {"movieId": 6, "title": "Heat (1995)", "genres": "Action|Crime|Thriller", "description": "An intense cat-and-mouse between cops and robbers."},
        {"movieId": 7, "title": "Sabrina (1995)", "genres": "Comedy|Romance", "description": "A story of love, class, and finding happiness."},
        {"movieId": 8, "title": "Tom and Huck (1995)", "genres": "Adventure|Children", "description": "Young adventurers in a classic tale."},
        {"movieId": 9, "title": "Sudden Death (1995)", "genres": "Action|Thriller", "description": "An explosive action film set in a stadium."},
        {"movieId": 10, "title": "GoldenEye (1995)", "genres": "Action|Adventure|Thriller", "description": "A James Bond action thriller."},
    ]
    return pd.DataFrame(data)

@st.cache_data
def prepare_movies(df_movies: pd.DataFrame):
    # Ensure required columns exist
    df = df_movies.copy()
    if 'movieId' not in df.columns:
        df['movieId'] = range(len(df))
    if 'title' not in df.columns:
        df['title'] = df['movieId'].astype(str)
    if 'genres' not in df.columns:
        df['genres'] = ''
    if 'description' not in df.columns:
        df['description'] = ''

    # Create a text field for content-based features
    df['content'] = (df['title'].fillna('') + ' ' + df['genres'].fillna('') + ' ' + df['description'].fillna(''))
    df['content'] = df['content'].str.replace('\|', ' ', regex=False)
    return df

@st.cache_resource
def build_tfidf_model(contents, max_features=5000):
    vec = TfidfVectorizer(max_features=max_features, stop_words='english')
    X = vec.fit_transform(contents)
    return vec, X

@st.cache_resource
def build_knn_index(X, n_neighbors=11):
    nn = NearestNeighbors(metric='cosine', algorithm='brute')
    nn.fit(X)
    return nn

@st.cache_data
def compute_popularity(df_movies, df_ratings=None):
    # If ratings provided, compute popularity by average rating and count
    df = df_movies.copy()
    if df_ratings is not None and not df_ratings.empty:
        agg = df_ratings.groupby('movieId').agg({'rating': ['mean','count']})
        agg.columns = ['rating_mean','rating_count']
        agg = agg.reset_index()
        df = df.merge(agg, on='movieId', how='left')
    else:
        df['rating_mean'] = np.nan
        df['rating_count'] = 0
    # fillna
    df['rating_count'] = df['rating_count'].fillna(0)
    df['rating_mean'] = df['rating_mean'].fillna(df['rating_mean'].mean() if not df['rating_mean'].isna().all() else 0)
    return df.sort_values(['rating_count','rating_mean'], ascending=False)

# UI layout
st.title("🎬 Mini Movie Recommender — Streamlit (Lightweight)")
st.markdown("Use a small content-based model (TF-IDF + KNN) to get similar movies, Top-N, and insights.")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload movies CSV (columns: movieId, title, genres, description)", type=['csv'])
    uploaded_ratings = st.file_uploader("(Optional) Upload ratings CSV (columns: userId, movieId, rating)", type=['csv'])
    sample_btn = st.button("Load sample dataset")

# Load dataset
if uploaded is not None:
    try:
        movies = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read uploaded movies file: {e}")
        movies = load_sample_dataset()
elif sample_btn or uploaded is None:
    movies = load_sample_dataset()

if uploaded_ratings is not None:
    try:
        ratings = pd.read_csv(uploaded_ratings)
    except Exception as e:
        st.error(f"Failed to read uploaded ratings file: {e}")
        ratings = pd.DataFrame()
else:
    ratings = pd.DataFrame()

movies = prepare_movies(movies)
vec, X = build_tfidf_model(movies['content'].fillna(''))
nn = build_knn_index(X)

# Main app tabs
tab1, tab2, tab3 = st.tabs(["Recommend (Top-N)", "Search & Similar", "Insights Dashboard"])

with tab1:
    st.header("User selection → Top‑N recommendations")
    col1, col2 = st.columns([2,1])
    with col1:
        # allow user to pick one or more favorite movies
        choices = movies['title'].tolist()
        favorites = st.multiselect("Select one or more movies you like (use these as seeds):", choices, max_selections=5)
        n_rec = st.slider("Number of recommendations (Top-N):", 1, 20, 5)
    with col2:
        st.write("\n")
        st.write("Tip: pick a few movies you like and click 'Recommend'.")
        if st.button("Recommend"):
            if not favorites:
                st.warning("Please select at least one movie to get recommendations.")
            else:
                # find indices of favorite movies
                idxs = movies[movies['title'].isin(favorites)].index.tolist()
                # compute mean vector of chosen movies
                seed_vecs = X[idxs]
                mean_vec = seed_vecs.mean(axis=0)
                distances, indices = nn.kneighbors(mean_vec, n_neighbors=n_rec+len(idxs))
                indices = indices.flatten()
                # exclude the seeds
                recs = [i for i in indices if i not in idxs]
                recs = recs[:n_rec]
                st.subheader("Recommendations")
                for i in recs:
                    row = movies.iloc[i]
                    st.markdown(f"**{row['title']}** — {row['genres']}  ")
                    if row['description']:
                        st.write(row['description'])

with tab2:
    st.header("Movie search → Similar movies list")
    query = st.text_input("Search movie title (type a few words):")
    n_sim = st.number_input("How many similar movies to show:", min_value=1, max_value=20, value=5)
    if st.button("Search"):
        if not query:
            st.warning("Type a movie title or a few words to search.")
        else:
            # simple substring search (case-insensitive)
            mask = movies['title'].str.contains(query, case=False, na=False)
            results = movies[mask]
            if results.empty:
                st.info("No exact matches found. Showing fuzzy-substring matches by token overlap.")
                tokens = query.lower().split()
                scores = movies['title'].str.lower().apply(lambda t: sum(1 for tok in tokens if tok in t))
                results = movies.loc[scores[scores>0].sort_values(ascending=False).index]

            if results.empty:
                st.error("No matches found.")
            else:
                chosen = st.selectbox("Select the movie to find similar ones:", results['title'].tolist())
                # find similar
                idx = movies[movies['title']==chosen].index[0]
                dist, inds = nn.kneighbors(X[idx], n_neighbors=n_sim+1)
                inds = inds.flatten()
                inds = [i for i in inds if i != idx][:n_sim]
                st.subheader(f"Movies similar to {chosen}")
                for i in inds:
                    r = movies.iloc[i]
                    st.markdown(f"**{r['title']}** — {r['genres']}")
                    if r['description']:
                        st.write(r['description'])

with tab3:
    st.header("Insight Dashboard")
    st.markdown("Quick insights about the loaded dataset and simple visualizations.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset summary")
        st.write(f"Total movies: {len(movies)}")
        top_genres = movies['genres'].str.split('|').explode().value_counts().head(10)
        st.write("Top genres:")
        st.dataframe(top_genres.reset_index().rename(columns={'index':'genre', 'genres':'count'}))

        st.subheader("Top popular (if ratings provided)")
        pop = compute_popularity(movies, ratings)
        if 'rating_count' in pop.columns and pop['rating_count'].sum() > 0:
            st.dataframe(pop[['title','rating_count','rating_mean']].head(10))
        else:
            st.info("No ratings provided — showing sample popular list by dataset order.")
            st.dataframe(movies[['title','genres']].head(10))

    with col2:
        st.subheader("Embeddings projection (SVD)")
        n_components = st.slider("SVD components (for visualization):", 2, 50, 8)
        svd = TruncatedSVD(n_components=min(n_components, X.shape[1]-1))
        X_reduced = svd.fit_transform(X)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(X_reduced[:,0], X_reduced[:,1], alpha=0.7)
        # label a few points
        for i in range(min(20, len(movies))):
            ax.annotate(movies.iloc[i]['title'].split('(')[0].strip(), (X_reduced[i,0], X_reduced[i,1]), fontsize=8)
        ax.set_xlabel('SVD-1')
        ax.set_ylabel('SVD-2')
        st.pyplot(fig)

    st.subheader("Genre distribution")
    genre_counts = movies['genres'].str.split('|').explode().value_counts()
    fig2, ax2 = plt.subplots(figsize=(8,3))
    genre_counts.plot(kind='bar', ax=ax2)
    ax2.set_ylabel('count')
    st.pyplot(fig2)

st.sidebar.markdown("---")
st.sidebar.write("Developed as a minimal example: TF-IDF content-based recommendations + KNN.\nYou can upload your own movies.csv and (optionally) ratings.csv to improve results.")

st.markdown("---")
st.caption("Notes: For larger datasets (MovieLens 20M etc.) you'd want to persist trained matrices and use a more efficient index (FAISS) and more features (tags, cast, collaborative filtering). This app is intentionally tiny and educational.")
