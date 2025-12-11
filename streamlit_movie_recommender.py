import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Movie Analytics & Recommender",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #ff0000, #ff69b4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# Generate synthetic data
@st.cache_data
def create_synthetic_data():
    np.random.seed(42)
    
    # 10 movies
    movies = pd.DataFrame({
        'movieId': range(1, 11),
        'title': [
            'The Shawshank Redemption', 'The Godfather', 'Pulp Fiction',
            'Inception', 'The Dark Knight', 'Forrest Gump',
            'The Matrix', 'Interstellar', 'Fight Club', 'Goodfellas'
        ],
        'genre': [
            'Drama', 'Crime|Drama', 'Crime|Thriller', 'Sci-Fi|Thriller',
            'Action|Crime', 'Drama|Romance', 'Action|Sci-Fi', 'Sci-Fi|Drama',
            'Drama|Thriller', 'Crime|Drama'
        ]
    })
    
    # Ratings
    ratings = []
    for user in range(1, 51):  # 50 users
        for movie in range(1, 11):  # 10 movies
            if np.random.rand() > 0.3:  # 70% chance of rating
                rating = np.random.choice([3.0, 3.5, 4.0, 4.5, 5.0], 
                                         p=[0.1, 0.2, 0.3, 0.25, 0.15])
                ratings.append({'userId': user, 'movieId': movie, 'rating': rating})
    
    ratings_df = pd.DataFrame(ratings)
    
    # Rating distribution
    rating_dist = ratings_df['rating'].value_counts().sort_index().reset_index()
    rating_dist.columns = ['rating', 'count']
    
    # Genre performance
    genre_perf = pd.DataFrame({
        'genre': ['Drama', 'Crime', 'Thriller', 'Sci-Fi', 'Action'],
        'avg_rating': [4.2, 4.1, 4.0, 4.3, 3.9],
        'count': [120, 95, 80, 85, 75]
    })
    
    # Movie stats
    movie_stats = ratings_df.groupby('movieId').agg({
        'rating': ['count', 'mean']
    }).reset_index()
    movie_stats.columns = ['movieId', 'num_ratings', 'avg_rating']
    movie_stats = movie_stats.merge(movies, on='movieId')
    
    return movies, ratings_df, rating_dist, genre_perf, movie_stats

movies, ratings, rating_dist, genre_perf, movie_stats = create_synthetic_data()

# Model info
models_info = {
    'Baseline (Weighted Popularity)': {
        'rmse': 0.89, 'mae': 0.68, 'accuracy': 72, 'status': '✅ Ready',
        'color': '#10B981', 'type': 'Non-Personalized'
    },
    'KNN': {
        'rmse': 0.85, 'mae': 0.65, 'accuracy': 75, 'status': '⏳ Training',
        'color': '#3B82F6', 'type': 'Content-Based'
    },
    'Random Forest': {
        'rmse': 0.82, 'mae': 0.62, 'accuracy': 78, 'status': '⏳ Training',
        'color': '#8B5CF6', 'type': 'Content-Based'
    },
    'XGBoost': {
        'rmse': 0.80, 'mae': 0.60, 'accuracy': 80, 'status': '⏳ Training',
        'color': '#F59E0B', 'type': 'Content-Based'
    },
    'Logistic Regression': {
        'rmse': None, 'mae': None, 'accuracy': 76, 'status': '⏳ Training',
        'color': '#EC4899', 'type': 'Content-Based'
    }
}

# Mock recommendations for each model
mock_recommendations = {
    'Baseline (Weighted Popularity)': [
        {'title': 'The Shawshank Redemption', 'predicted': 4.5, 'confidence': 0.92},
        {'title': 'Inception', 'predicted': 4.4, 'confidence': 0.89},
        {'title': 'The Dark Knight', 'predicted': 4.3, 'confidence': 0.87}
    ],
    'KNN': [
        {'title': 'Interstellar', 'predicted': 4.4, 'confidence': 0.85},
        {'title': 'The Matrix', 'predicted': 4.3, 'confidence': 0.83},
        {'title': 'Inception', 'predicted': 4.2, 'confidence': 0.81}
    ],
    'Random Forest': [
        {'title': 'Fight Club', 'predicted': 4.3, 'confidence': 0.88},
        {'title': 'The Godfather', 'predicted': 4.2, 'confidence': 0.86},
        {'title': 'Pulp Fiction', 'predicted': 4.1, 'confidence': 0.84}
    ],
    'XGBoost': [
        {'title': 'The Dark Knight', 'predicted': 4.4, 'confidence': 0.90},
        {'title': 'Inception', 'predicted': 4.3, 'confidence': 0.88},
        {'title': 'Interstellar', 'predicted': 4.2, 'confidence': 0.85}
    ],
    'Logistic Regression': [
        {'title': 'Forrest Gump', 'predicted': 4.3, 'confidence': 0.79},
        {'title': 'The Shawshank Redemption', 'predicted': 4.2, 'confidence': 0.77},
        {'title': 'Goodfellas', 'predicted': 4.1, 'confidence': 0.75}
    ]
}

# ==================== NEW: Similar Movies Helper ====================
def find_similar_movies(movie_title):
    """Return 3 similar movies based on shared genre and highest ratings."""
    selected = movies[movies['title'] == movie_title].iloc[0]
    genres = selected['genre'].split('|')

    # Filter movies with any genre overlap
    genre_matches = movies[movies['genre'].apply(lambda g: any(x in g.split('|') for x in genres))]
    
    # Remove the selected movie
    genre_matches = genre_matches[genre_matches['title'] != movie_title]

    # Merge with stats for ratings
    genre_matches = genre_matches.merge(movie_stats[['movieId', 'avg_rating']], on='movieId')

    # Choose top 3
    return genre_matches.sort_values(by='avg_rating', ascending=False).head(3)

# Sidebar
with st.sidebar:
    st.title("🎬 Navigation")
    page = st.radio(
        "Select Page",
        ["🏠 Overview", "👥 User Insights", "🎬 Content Analysis", 
         "🎮 Model Playground", "🔍 Movie Search", "📊 Model Comparison"],  # NEW
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("Movie Analytics v1.0")

# Header
st.markdown('<p class="main-header">🎬 Movie Analytics & Recommender</p>', unsafe_allow_html=True)
st.markdown("**ML-Powered Business Intelligence Platform**")
st.warning("⚠️ **Demo Mode:** Using synthetic data with 10 movies & 50 users")

# ==================== OVERVIEW ====================
# (unchanged… skipped for brevity)

# ==================== USER INSIGHTS ====================
# (unchanged… skipped)

# ==================== CONTENT ANALYSIS ====================
# (unchanged… skipped)

# ==================== MODEL PLAYGROUND ====================
# (unchanged… skipped)

# ==================== NEW: MOVIE SEARCH ====================
elif page == "🔍 Movie Search":
    st.header("🔍 Movie Search & Similar Movies")

    search_query = st.text_input("Search for a movie", placeholder="Type movie title...")

    if search_query:
        results = movies[movies['title'].str.contains(search_query, case=False)]

        if len(results) == 0:
            st.error("No movies found.")
        else:
            st.subheader("Search Results")
            for _, row in results.iterrows():
                st.markdown(f"### 🎬 {row['title']}")
                st.caption(f"Genres: {row['genre']}")

                # Find similar movies
                st.markdown("##### ⭐ Similar Movies")
                sims = find_similar_movies(row['title'])

                for _, sim in sims.iterrows():
                    st.write(f"- **{sim['title']}** — ⭐ {sim['avg_rating']:.1f}")

                st.divider()

# ==================== MODEL COMPARISON ====================
# (unchanged… skipped)

st.divider()
st.caption("Built with Streamlit | Synthetic MovieLens Data | 2024")
