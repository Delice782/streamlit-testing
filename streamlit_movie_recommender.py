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

# Sidebar
with st.sidebar:
    st.title("🎬 Navigation")
    page = st.radio(
        "Select Page",
        ["🏠 Overview", "👥 User Insights", "🎬 Content Analysis",
         "🎮 Model Playground", "🔍 Movie Search", "📊 Model Comparison"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("Movie Analytics v1.0")


# Header
st.markdown('<p class="main-header">🎬 Movie Analytics & Recommender</p>', unsafe_allow_html=True)
st.markdown("**ML-Powered Business Intelligence Platform**")
st.warning("⚠️ **Demo Mode:** Using synthetic data with 10 movies & 50 users")

################################################################################
# OVERVIEW
################################################################################
if page == "🏠 Overview":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Movies", "10")
    col2.metric("Active Users", "50")
    col3.metric("Total Ratings", f"{len(ratings)}")
    col4.metric("Models", "5", delta="1 ready, 4 training")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Rating Distribution")
        fig = px.bar(rating_dist, x='rating', y='count', color_discrete_sequence=['#ff0000'])
        fig.update_layout(template='plotly_dark', height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("⭐ Top Movies")
        top_5 = movie_stats.nlargest(5, 'avg_rating')
        for _, row in top_5.iterrows():
            a, b = st.columns([3, 1])
            a.write(f"**{row['title']}**")
            a.caption(f"{int(row['num_ratings'])} ratings")
            b.metric("⭐", f"{row['avg_rating']:.1f}")


################################################################################
# USER INSIGHTS
################################################################################
elif page == "👥 User Insights":
    st.header("User Behavior Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("User Segmentation")
        segments = pd.DataFrame({
            'segment': ['Casual', 'Regular', 'Power', 'Critics'],
            'percentage': [45, 35, 15, 5]
        })
        fig = px.pie(segments, values='percentage', names='segment')
        fig.update_layout(template='plotly_dark', height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Key Insights")
        st.success("**Generous Raters:** 23% rate above 4.0 stars")
        st.error("**Critical Raters:** 12% rate below 3.0")
        st.info("**Power Users:** Top 15% give 60% of all ratings")


################################################################################
# CONTENT ANALYSIS
################################################################################
elif page == "🎬 Content Analysis":
    st.header("Content Performance")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Genre Performance")
        fig = px.bar(
            genre_perf, x='avg_rating', y='genre', orientation='h',
            color='avg_rating', color_continuous_scale='RdYlGn'
        )
        fig.update_layout(template='plotly_dark', height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Insights")
        st.write("🎯 **Top Genre:** Sci-Fi (4.3⭐)")
        st.write("💎 **Hidden Gems:** 3 movies rated ≥4.5 with <30 reviews")


################################################################################
# MODEL PLAYGROUND
################################################################################
elif page == "🎮 Model Playground":
    st.header("Interactive Model Testing")

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_model = st.selectbox("Select Model", list(models_info.keys()))

    with col2:
        user_id = st.number_input("User ID", min_value=1, max_value=50, value=1)

    with col3:
        top_n = st.selectbox("Top N", [3, 5, 10], index=0)

    if st.button("🚀 Generate Recommendations", type="primary", use_container_width=True):
        status = models_info[selected_model]['status']

        if '✅' in status:
            st.success(f"{status} - {selected_model}")
        else:
            st.warning(f"{status} - Showing demo predictions")

        st.divider()
        st.subheader(f"Top {top_n} for User {user_id}")

        recs = mock_recommendations[selected_model][:top_n]
        for i, rec in enumerate(recs, 1):
            a, b, c = st.columns([2, 1, 1])
            a.markdown(f"**{i}. {rec['title']}**")
            b.metric("Predicted", f"⭐ {rec['predicted']}")
            c.metric("Confidence", f"{rec['confidence']*100:.0f}%")


################################################################################
# MOVIE SEARCH — NEW FEATURE
################################################################################
elif page == "🔍 Movie Search":
    st.header("Movie Search & Similar Movies")

    search_query = st.text_input("Search for a movie")

    if search_query.strip():
        results = movies[movies["title"].str.contains(search_query, case=False)]

        if len(results) == 0:
            st.warning("No movies found.")
        else:
            st.subheader("Search Results")
            for _, row in results.iterrows():
                st.markdown(f"### 🎬 {row['title']}")
                st.caption(f"Genres: {row['genre']}")

                # Similar movies — by matching first genre token
                main_genre = row["genre"].split("|")[0]
                similar = movies[movies["genre"].str.contains(main_genre, case=False)]
                similar = similar[similar["movieId"] != row["movieId"]].head(3)

                st.markdown("**Similar Movies:**")
                for _, sim in similar.iterrows():
                    st.write(f"- {sim['title']} ({sim['genre']})")

                st.divider()


################################################################################
# MODEL COMPARISON
################################################################################
elif page == "📊 Model Comparison":
    st.header("Model Performance Comparison")

    st.subheader("Select Models to Compare")
    selected = st.multiselect(
        "Choose models",
        list(models_info.keys()),
        default=list(models_info.keys())[:3]
    )

    if len(selected) >= 2:
        comparison_data = []
        for model in selected:
            info = models_info[model]
            comparison_data.append({
                'Model': model,
                'Type': info['type'],
                'RMSE': info['rmse'] if info['rmse'] else 'N/A',
                'MAE': info['mae'] if info['mae'] else 'N/A',
                'Accuracy': f"{info['accuracy']}%",
                'Status': info['status']
            })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Radar chart
        st.subheader("🎯 Multi-Dimensional Comparison")

        categories = ['Accuracy', 'Speed', 'Scalability', 'Personalization', 'Interpretability']

        fig = go.Figure()

        model_scores = {
            'Baseline (Weighted Popularity)': [72, 95, 100, 20, 100],
            'KNN': [75, 70, 50, 65, 60],
            'Random Forest': [78, 60, 70, 70, 50],
            'XGBoost': [80, 65, 75, 75, 45],
            'Logistic Regression': [76, 85, 80, 60, 80]
        }

        for model in selected:
            if model in model_scores:
                values = model_scores[model] + [model_scores[model][0]]
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=model
                ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            template='plotly_dark',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Please select at least 2 models to compare")


################################################################################
# FOOTER
################################################################################
st.divider()
st.caption("Built with Streamlit | Synthetic MovieLens Data | 2024")

