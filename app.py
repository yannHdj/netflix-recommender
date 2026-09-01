import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Netflix Finder", page_icon="🍿", layout="wide")

# Style CSS pour le look Netflix
st.markdown("""
    <style>
    .main { background-color: #141414; color: white; }
    .stButton>button { background-color: #E50914; color: white; border-radius: 5px; border: none; width: 100%; }
    .stSelectbox div[data-baseweb="select"] { color: black; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data():
    # Lien stable et vérifié vers le dataset
    url = "https://raw.githubusercontent.com/shivamb/netflix-shows/master/netflix_titles.csv"
    df = pd.read_csv(url)
    df = df.fillna('')
    # On crée une colonne qui regroupe les infos textuelles
    df['combined_features'] = df['listed_in'] + " " + df['description'] + " " + df['cast'] + " " + df['director']
    return df

df = load_data()

# --- 2. CALCUL DE LA SIMILARITÉ (Le Cerveau de l'IA) ---
@st.cache_resource
def compute_similarity(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['combined_features'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

# On lance le calcul
cosine_sim = compute_similarity(df)

# --- 3. INTERFACE UTILISATEUR ---
st.title("🎬 Netflix Finder")
st.write("Trouvez votre prochain film ou série basé sur vos goûts.")

# Barre de sélection
movie_list = df['title'].values
selected_movie = st.selectbox("Sélectionnez un titre que vous avez aimé :", movie_list)

if st.button('Recommander des titres similaires'):
    try:
        # Trouver l'index du film sélectionné
        idx = df[df['title'] == selected_movie].index[0]
        
        # Calculer les scores de similarité
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:7]
        
        st.write("---")
        st.subheader(f"Si vous avez aimé '{selected_movie}', vous devriez regarder :")
        
        # Affichage en grille de 3 colonnes
        cols = st.columns(3)
        for i, score in enumerate(sim_scores):
            movie_idx = score[0]
            row = df.iloc[movie_idx]
            
            with cols[i % 3]:
                st.info(f"**{row['title']}**")
                st.caption(f"{row['release_year']} | {row['type']}")
                st.write(f"*{row['listed_in']}*")
                with st.expander("Voir le résumé"):
                    st.write(row['description'])
    except Exception as e:
        st.error("Une erreur est survenue lors de la génération des recommandations.")

