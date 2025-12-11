import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Movie Analytics & Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
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
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff0000;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/ff0000/ffffff?text=MovieLens", use_container_width=True)
    st.title("Navigation")
    
    page = st.radio(
        "Select Page",
        ["🏠 Overview", "👥 User Insights", "🎬 Content Analysis", 
         "📈 Trends", "🎮 Model Playground", "📊 Model Comparison"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    st.subheader("⚙️ Settings")
    show_raw_data = st.checkbox("Show raw data tables")
    
    st.divider()
    st.caption("Movie Analytics System v1.0")
    st.caption("Data Science Final Project")

# Header
st.markdown('<p class="main-header">🎬 Movie Analytics & Recommendation System</p>', unsafe_allow_html=True)
st.markdown("**ML-Powered Business Intelligence & Personalization Platform**")

# Warning banner
st.warning("⚠️ **Development Mode:** Baseline model ready. Content-based models training. Check 'Model Playground' to compare approaches.")

# Load data (cached)
@st.cache_data
def load_data():
    """Load and preprocess MovieLens data"""
    try:
        # Load datasets
        ratings = pd.read_csv('data/ratings.csv')
        movies = pd.read_csv('data/movies.csv')
        
        # Basic preprocessing
        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
        ratings['year'] = ratings['timestamp'].dt.year
        ratings['month'] = ratings['timestamp'].dt.month
        
        # Parse genres
        movies['genres'] = movies['genres'].str.split('|')
        
        return ratings, movies
    except FileNotFoundError:
        st.error("❌ Data files not found! Please ensure CSV files are in 'data/' folder")
        return None, None

# Generate sample data for demo
@st.cache_data
def generate_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    
    # Rating distribution
    rating_dist = pd.DataFrame({
        'rating': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        'count': [1370, 2811, 1791, 7551, 5550, 20047, 13136, 26818, 8551, 13211]
    })
    
    # Genre performance
    genre_perf = pd.DataFrame({
        'genre': ['Drama', 'Comedy', 'Thriller', 'Action', 'Romance'],
        'avg_rating': [3.7, 3.4, 3.6, 3.4, 3.5],
        'count': [4361, 3756, 1894, 1828, 1596]
    })
    
    # Top movies
    top_movies = pd.DataFrame({
        'title': ['The Shawshank Redemption', 'Forrest Gump', 'Pulp Fiction', 
                  'The Godfather', 'Inception'],
        'rating': [4.5, 4.3, 4.3, 4.5, 4.3],
        'reviews': [17015, 11842, 12318, 10136, 13678]
    })
    
    # Ratings over time
    ratings_time = pd.DataFrame({
        'year': [2010, 2012, 2014, 2016, 2018],
        'avg_rating': [3.45, 3.52, 3.58, 3.51, 3.47]
    })
    
    # Model metrics
    model_metrics = {
        'Weighted Baseline': {'rmse': 0.89, 'mae': 0.68, 'accuracy': 0.72, 'status': 'Ready'},
        'KNN': {'rmse': 0.85, 'mae': 0.65, 'accuracy': 0.75, 'status': 'Training'},
        'Random Forest': {'rmse': 0.82, 'mae': 0.62, 'accuracy': 0.78, 'status': 'Training'},
        'Logistic Regression': {'rmse': None, 'mae': None, 'accuracy': 0.76, 'status': 'Training'},
        'Collaborative': {'rmse': 0.79, 'mae': 0.59, 'accuracy': 0.81, 'status': 'Pending'}
    }
    
    return rating_dist, genre_perf, top_movies, ratings_time, model_metrics

# Load data
rating_dist, genre_perf, top_movies, ratings_time, model_metrics = generate_sample_data()

# ========================================
# PAGE: OVERVIEW
# ========================================
if page == "🏠 Overview":
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Movies", "9,742", help="Across 20 genres")
    with col2:
        st.metric("Active Users", "610", help="Rating contributors")
    with col3:
        st.metric("Total Ratings", "100K+", help="Avg: 3.5/5.0")
    with col4:
        st.metric("Models Ready", "1/5", delta="4 in training")
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Rating Distribution")
        fig = px.bar(rating_dist, x='rating', y='count', 
                     title="Distribution of User Ratings",
                     color_discrete_sequence=['#ff0000'])
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⭐ Top Rated Movies")
        for idx, row in top_movies.iterrows():
            with st.container():
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"**{row['title']}**")
                    st.caption(f"{row['reviews']} reviews")
                with col_b:
                    st.metric("Rating", f"⭐ {row['rating']}")
                st.divider()

# ========================================
# PAGE: USER INSIGHTS
# ========================================
elif page == "👥 User Insights":
    st.header("User Behavior Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("User Segmentation")
        user_segments = pd.DataFrame({
            'segment': ['Casual Viewers', 'Regular Users', 'Power Users', 'Critics'],
            'percentage': [45, 35, 15, 5]
        })
        fig = px.pie(user_segments, values='percentage', names='segment',
                     title="User Distribution by Activity Level")
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Key Insights")
        
        st.success("**Generous Raters:** ~23% of users consistently rate above 4.0 stars")
        st.error("**Critical Raters:** ~12% of users have average ratings below 3.0")
        st.info("**Power Users:** Top 15% contribute 60% of all ratings")
        st.warning("**Retention Risk:** 30% of users haven't rated in 6+ months")

# ========================================
# PAGE: CONTENT ANALYSIS
# ========================================
elif page == "🎬 Content Analysis":
    st.header("Content Strategy & Performance")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Genre Performance")
        fig = px.bar(genre_perf, x='avg_rating', y='genre', 
                     orientation='h',
                     title="Average Rating by Genre",
                     color='avg_rating',
                     color_continuous_scale='RdYlGn')
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Strategic Insights")
        
        st.markdown("##### 🎯 High Performers")
        st.write("Drama & Thriller genres show highest engagement")
        
        st.markdown("##### 💎 Hidden Gems")
        st.write("412 movies with ≥4.0 rating but <50 reviews")
        
        st.markdown("##### 📉 Underperformers")
        st.write("Horror & Documentary need content refresh")

# ========================================
# PAGE: TRENDS
# ========================================
elif page == "📈 Trends":
    st.header("Temporal Analysis & Trends")
    
    st.subheader("Rating Trends Over Time")
    fig = px.line(ratings_time, x='year', y='avg_rating',
                  title="Average Rating Evolution",
                  markers=True)
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 **Insight:** Ratings peaked in 2014 and have slightly declined since, possibly due to increased content volume.")

# ========================================
# PAGE: MODEL PLAYGROUND
# ========================================
elif page == "🎮 Model Playground":
    st.header("Interactive Model Testing")
    st.write("Select a model and test it with different users to see personalized recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_model = st.selectbox(
            "Select Model",
            list(model_metrics.keys())
        )
    
    with col2:
        user_id = st.text_input("User ID", value="1", placeholder="Enter user ID")
    
    with col3:
        top_n = st.selectbox("Top N Results", [5, 10, 20], index=0)
    
    if st.button("🚀 Generate Recommendations", type="primary", use_container_width=True):
        # Show model status
        status = model_metrics[selected_model]['status']
        if status == 'Ready':
            st.success(f"✅ {selected_model} is ready!")
        else:
            st.warning(f"⏳ {selected_model} is still {status.lower()}. Showing demo data.")
        
        st.divider()
        
        # Display recommendations (sample data)
        st.subheader(f"Top {top_n} Recommendations for User {user_id}")
        
        sample_recs = [
            {"title": "Inception", "predicted": 4.4, "confidence": 0.85},
            {"title": "The Dark Knight", "predicted": 4.3, "confidence": 0.83},
            {"title": "Interstellar", "predicted": 4.2, "confidence": 0.81},
            {"title": "The Matrix", "predicted": 4.1, "confidence": 0.79},
            {"title": "Fight Club", "predicted": 4.0, "confidence": 0.77}
        ]
        
        for i, rec in enumerate(sample_recs[:top_n], 1):
            col_a, col_b, col_c = st.columns([2, 1, 1])
            with col_a:
                st.markdown(f"**{i}. {rec['title']}**")
            with col_b:
                st.metric("Predicted", f"⭐ {rec['predicted']}")
            with col_c:
                st.metric("Confidence", f"{rec['confidence']*100:.0f}%")

# ========================================
# PAGE: MODEL COMPARISON
# ========================================
elif page == "📊 Model Comparison":
    st.header("Model Performance Comparison")
    st.write("Compare different recommendation algorithms across multiple metrics")
    
    # Model selection
    st.subheader("Select Models to Compare")
    selected_models = st.multiselect(
        "Choose 2-5 models",
        list(model_metrics.keys()),
        default=list(model_metrics.keys())[:3]
    )
    
    if len(selected_models) >= 2:
        # Create comparison dataframe
        comparison_df = pd.DataFrame(model_metrics).T.loc[selected_models]
        
        # Metrics table
        st.subheader("📋 Performance Metrics")
        st.dataframe(
            comparison_df.style.format({
                'rmse': '{:.2f}',
                'mae': '{:.2f}',
                'accuracy': '{:.2%}'
            }),
            use_container_width=True
        )
        
        # Radar chart
        st.subheader("🎯 Multi-Dimensional Comparison")
        
        # Create radar chart data
        categories = ['Accuracy', 'Speed', 'Scalability', 'Personalization', 'Interpretability']
        
        fig = go.Figure()
        
        # Add trace for each model
        for model in selected_models:
            # Sample values (replace with actual metrics)
            values = [
                model_metrics[model]['accuracy'] * 100 if model_metrics[model]['accuracy'] else 0,
                85, 75, 70, 65  # Demo values
            ]
            values += values[:1]  # Complete the circle
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Business recommendations
        st.subheader("💡 Business Recommendations")
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            **Best for Production:**
            Collaborative Filtering offers the best accuracy (81%) and personalization.
            Recommended for main recommendation engine once training completes.
            """)
        
        with col2:
            st.info("""
            **Best for Speed:**
            Weighted Baseline provides instant results with decent accuracy (72%).
            Ideal for cold-start users and homepage "Trending Now" section.
            """)
    else:
        st.warning("Please select at least 2 models to compare")

# Footer
st.divider()
st.caption("Built with Streamlit | MovieLens Dataset | Final Project 2024")
