import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Netflix Finder", page_icon="🍿", layout="wide")

# Style CSS pour améliorer le visuel
st.markdown("""
    <style>
    .main { background-color: #141414; color: white; }
    .stButton>button { background-color: #E50914; color: white; border-radius: 5px; border: none; }
    .stSelectbox { color: black; }
    </style>
    """, unsafe_allow_html=True)

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data():
    # Lien vers un dataset Netflix public pour que ça marche immédiatement
    url = "https://raw.githubusercontent.com/611683930/Netflix-Visualizations-Recommendation/master/netflix_titles.csv"
    df = pd.read_csv(url)
    df = df.fillna('')
    # On combine les infos pour l'algorithme
    df['combined_features'] = df['listed_in'] + " " + df['description'] + " " + df['cast']
    return df

@st.cache_data
def load_data():
    # Nouveau lien vérifié
    url = "https://raw.githubusercontent.com/shivamb/netflix-shows/master/netflix_titles.csv"
    df = pd.read_csv(url)
    df = df.fillna('')
    # On combine les infos pour l'algorithme
    df['combined_features'] = df['listed_in'] + " " + df['description'] + " " + df['cast']
    return df

cosine_sim = compute_similarity(df)

# --- INTERFACE ---
st.title("🎬 Quel film ou série regarder sur Netflix ?")
st.subheader("L'IA qui analyse vos goûts pour vous proposer le meilleur catalogue.")

# Sélection de l'utilisateur
movie_list = df['title'].values
selected_movie = st.selectbox("Tapez ou sélectionnez un titre que vous avez aimé :", movie_list)

if st.button('Générer des recommandations'):
    # Logique de recommandation
    idx = df[df['title'] == selected_movie].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:7]
    
    st.write("---")
    st.header(f"Parce que vous avez aimé '{selected_movie}', vous devriez adorer :")
    
    # Affichage en colonnes (3 par ligne)
    cols = st.columns(3)
    for i, score in enumerate(sim_scores):
        movie_idx = score[0]
        title = df.iloc[movie_idx]['title']
        desc = df.iloc[movie_idx]['description']
        year = df.iloc[movie_idx]['release_year']
        category = df.iloc[movie_idx]['listed_in']
        
        with cols[i % 3]:
            st.info(f"**{title}** ({year})")
            st.caption(f"Genre : {category}")
            st.write(f"{desc[:150]}...")
