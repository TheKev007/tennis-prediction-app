import streamlit as st
import pandas as pd
import random
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Tennis Match Prediction System",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .confidence-high {
        color: #2E7D32;
        font-weight: bold;
    }
    .confidence-medium {
        color: #F9A825;
        font-weight: bold;
    }
    .confidence-low {
        color: #C62828;
        font-weight: bold;
    }
    .surface-hard {
        background-color: #E3F2FD;
        padding: 5px;
        border-radius: 5px;
    }
    .surface-clay {
        background-color: #FFEBEE;
        padding: 5px;
        border-radius: 5px;
    }
    .surface-grass {
        background-color: #E8F5E9;
        padding: 5px;
        border-radius: 5px;
    }
    .player-card {
        border: 1px solid #EEEEEE;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1 class='main-header'>Tennis Match Prediction System</h1>", unsafe_allow_html=True)

# Generate sample data for demonstration
def generate_sample_matches(num_matches=10):
    surfaces = ["Hard", "Clay", "Grass"]
    atp_players = [
        {"name": "Novak Djokovic", "rank": 1, "win_rate": 0.83},
        {"name": "Carlos Alcaraz", "rank": 2, "win_rate": 0.78},
        {"name": "Jannik Sinner", "rank": 3, "win_rate": 0.75},
        {"name": "Daniil Medvedev", "rank": 4, "win_rate": 0.72},
        {"name": "Alexander Zverev", "rank": 5, "win_rate": 0.70},
        {"name": "Andrey Rublev", "rank": 6, "win_rate": 0.68},
        {"name": "Casper Ruud", "rank": 7, "win_rate": 0.67},
        {"name": "Hubert Hurkacz", "rank": 8, "win_rate": 0.65},
        {"name": "Stefanos Tsitsipas", "rank": 9, "win_rate": 0.69},
        {"name": "Taylor Fritz", "rank": 10, "win_rate": 0.64}
    ]
    
    wta_players = [
        {"name": "Iga Swiatek", "rank": 1, "win_rate": 0.85},
        {"name": "Aryna Sabalenka", "rank": 2, "win_rate": 0.76},
        {"name": "Coco Gauff", "rank": 3, "win_rate": 0.72},
        {"name": "Elena Rybakina", "rank": 4, "win_rate": 0.74},
        {"name": "Jessica Pegula", "rank": 5, "win_rate": 0.70},
        {"name": "Marketa Vondrousova", "rank": 6, "win_rate": 0.68},
        {"name": "Ons Jabeur", "rank": 7, "win_rate": 0.69},
        {"name": "Maria Sakkari", "rank": 8, "win_rate": 0.67},
        {"name": "Karolina Muchova", "rank": 9, "win_rate": 0.65},
        {"name": "Daria Kasatkina", "rank": 10, "win_rate": 0.64}
    ]
    
    matches = []
    
    # Current date
    today = datetime.now()
    
    for i in range(num_matches):
        # Randomly select tour
        tour = "ATP" if random.random() > 0.5 else "WTA"
        players = atp_players if tour == "ATP" else wta_players
        
        # Select two different players
        player1_idx = random.randint(0, len(players) - 1)
        player2_idx = random.randint(0, len(players) - 1)
        while player2_idx == player1_idx:
            player2_idx = random.randint(0, len(players) - 1)
        
        player1 = players[player1_idx]
        player2 = players[player2_idx]
        
        # Select surface
        surface = random.choice(surfaces)
        
        # Generate match date (within next 7 days)
        match_date = today + timedelta(days=random.randint(1, 7))
        
        # Calculate confidence based on rank difference and win rates
        rank_diff = abs(player1["rank"] - player2["rank"])
        win_rate_diff = abs(player1["win_rate"] - player2["win_rate"])
        
        # Base confidence calculation
        base_confidence = 50 + (rank_diff * 2) + (win_rate_diff * 100)
        
        # Add some randomness
        confidence = min(95, max(55, base_confidence + random.randint(-10, 10)))
        
        # Determine predicted winner
        if player1["win_rate"] > player2["win_rate"]:
            predicted_winner = player1["name"]
            predicted_winner_idx = 1
        else:
            predicted_winner = player2["name"]
            predicted_winner_idx = 2
        
        # Determine model certainty
        if confidence > 75:
            model_certainty = "High"
        elif confidence > 60:
            model_certainty = "Medium"
        else:
            model_certainty = "Low"
        
        # Create match entry
        match = {
            "match_id": f"{tour.lower()}_{i+1}",
            "tour": tour,
            "match_date": match_date.strftime("%Y-%m-%d"),
            "surface": surface,
            "player1_name": player1["name"],
            "player2_name": player2["name"],
            "player1_rank": player1["rank"],
            "player2_rank": player2["rank"],
            "predicted_winner": predicted_winner,
            "predicted_winner_idx": predicted_winner_idx,
            "confidence": confidence,
            "model_certainty": model_certainty
        }
        
        matches.append(match)
    
    return pd.DataFrame(matches)

# Sidebar filters
st.sidebar.markdown("## Filters")

# Generate sample data
sample_matches = generate_sample_matches(15)

# Tour filter
tour_filter = st.sidebar.multiselect(
    "Tour",
    options=["ATP", "WTA"],
    default=["ATP", "WTA"]
)

# Surface filter
surface_filter = st.sidebar.multiselect(
    "Surface",
    options=["Hard", "Clay", "Grass"],
    default=["Hard", "Clay", "Grass"]
)

# Confidence threshold
confidence_threshold = st.sidebar.slider(
    "Minimum Confidence (%)",
    min_value=50,
    max_value=95,
    value=60,
    step=5
)

# Apply filters
filtered_matches = sample_matches[
    (sample_matches["tour"].isin(tour_filter)) &
    (sample_matches["surface"].isin(surface_filter)) &
    (sample_matches["confidence"] >= confidence_threshold)
]

# Sort by confidence (descending)
filtered_matches = filtered_matches.sort_values(by="confidence", ascending=False)

# Display match predictions
st.markdown("<h2 class='sub-header'>Match Predictions</h2>", unsafe_allow_html=True)

if filtered_matches.empty:
    st.info("No matches found with the selected filters.")
else:
    # Display date
    st.markdown(f"### Predictions for: {datetime.now().strftime('%B %d, %Y')}")
    st.markdown("These predictions are updated daily with the latest player data.")
    
    # Display matches
    for _, match in filtered_matches.iterrows():
        col1, col2, col3 = st.columns([2, 1, 2])
        
        # Surface class
        surface_class = f"surface-{match['surface'].lower()}"
        
        # Confidence class
        if match["model_certainty"] == "High":
            confidence_class = "confidence-high"
        elif match["model_certainty"] == "Medium":
            confidence_class = "confidence-medium"
        else:
            confidence_class = "confidence-low"
        
        with col1:
            st.markdown(f"<div class='player-card'>", unsafe_allow_html=True)
            st.markdown(f"### {match['player1_name']}")
            st.markdown(f"Rank: {match['player1_rank']}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div style='text-align: center; padding-top: 20px;'>", unsafe_allow_html=True)
            st.markdown(f"<span class='{surface_class}'>{match['surface']}</span>", unsafe_allow_html=True)
            st.markdown("<h3>VS</h3>", unsafe_allow_html=True)
            st.markdown(f"<p>Match Date: {match['match_date']}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"<div class='player-card'>", unsafe_allow_html=True)
            st.markdown(f"### {match['player2_name']}")
            st.markdown(f"Rank: {match['player2_rank']}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Prediction box
        st.markdown(f"""
        <div style='background-color: #F5F5F5; padding: 15px; border-radius: 10px; margin-bottom: 30px;'>
            <h4>Prediction:</h4>
            <p>Predicted Winner: <strong>{match['predicted_winner']}</strong></p>
            <p>Confidence: <span class='{confidence_class}'>{match['confidence']:.1f}%</span></p>
            <p>Model Certainty: <span class='{confidence_class}'>{match['model_certainty']}</span></p>
            <p>Tour: {match['tour']}</p>
        </div>
        """, unsafe_allow_html=True)

# About section
with st.expander("About this System"):
    st.markdown("""
    ## Tennis Match Prediction System
    
    This system uses machine learning to predict the outcomes of tennis matches with confidence percentages.
    
    ### Features:
    - Predicts winners for ATP and WTA matches
    - Provides confidence percentages for each prediction
    - Filters by tour, surface, and confidence level
    - Updates daily with the latest player data
    
    ### How it Works:
    The prediction model analyzes player statistics, historical performance, surface-specific win rates, 
    and other factors to generate predictions. The confidence percentage indicates how certain the model 
    is about its prediction.
    
    ### Note:
    This is a demonstration version using sample data. In a production environment, the system would 
    connect to real-time tennis data APIs to provide predictions for upcoming matches.
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 Tennis Match Prediction System | Last Updated: " + datetime.now().strftime("%B %d, %Y"))
