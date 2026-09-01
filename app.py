import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration large pour voir les films côte à côte
st.set_page_config(page_title="Netflix AI Finder", page_icon="🍿", layout="wide")

@st.cache_data
def load_data():
    try:
        # On lit le fichier directement depuis ton dépôt GitHub
        df = pd.read_csv('netflix_titles.csv')
        df = df.fillna('')
        # Création de l'intelligence : mélange des genres, description et acteurs
        df['combined_features'] = df['listed_in'] + " " + df['description'] + " " + df['cast']
        return df
    except Exception as e:
        return None

df = load_data()

if df is None:
    st.error("⚠️ Le fichier 'netflix_titles.csv' n'est pas encore sur ton GitHub. Clique sur 'Add file' > 'Upload' sur GitHub.")
    st.stop()

# --- BARRE LATÉRALE : LES FILTRES ---
st.sidebar.title("⚙️ Tes Préférences")

# 1. Filtre Film ou Série
type_filter = st.sidebar.radio("Format :", ["Tout", "Movie", "TV Show"])

# 2. Filtre Pays (on nettoie la liste des pays)
countries = set()
for c in df['country'].unique():
    for name in str(c).split(', '):
        if name: countries.add(name)
selected_country = st.sidebar.selectbox("Pays d'origine :", ["Tous les pays"] + sorted(list(countries)))

# --- FILTRAGE DES DONNÉES ---
filtered_df = df.copy()
if type_filter != "Tout":
    filtered_df = filtered_df[filtered_df['type'] == type_filter]
if selected_country != "Tous les pays":
    filtered_df = filtered_df[filtered_df['country'].str.contains(selected_country)]

# --- MOTEUR D'IA ---
@st.cache_resource
def get_similarity_matrix(data):
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(data['combined_features'])
    return cosine_similarity(matrix, matrix)

# On prépare les titres pour la barre de recherche
if not filtered_df.empty:
    # On réinitialise l'index pour que l'IA ne se mélange pas les pinceaux
    display_df = filtered_df.reset_index(drop=True)
    cosine_sim = get_similarity_matrix(display_df)
    
    st.title("🎬 Mon Recommandeur Netflix")
    st.write(f"Analyse de {len(display_df)} titres disponibles selon tes filtres.")

    # Choix du film
    target_movie = st.selectbox("Sélectionne un film que tu as aimé :", display_df['title'].values)

    if st.button('Trouver des recommandations 🚀'):
        # On trouve la position du film
        idx = display_df[display_df['title'] == target_movie].index[0]
        
        # On calcule les 6 films les plus proches
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:7]
        
        st.write("---")
        st.subheader(f"Si tu as aimé '{target_movie}', tu aimeras aussi :")
        
        # Affichage en grille
        cols = st.columns(3)
        for i, score in enumerate(sim_scores):
            movie_row = display_df.iloc[score[0]]
            with cols[i % 3]:
                st.info(f"**{movie_row['title']}**")
                st.caption(f"{movie_row['release_year']} | {movie_row['country']}")
                with st.expander("Voir le résumé"):
                    st.write(movie_row['description'])
else:
    st.warning("Aucun film ne correspond à tes filtres. Essaie d'élargir ta recherche.")
