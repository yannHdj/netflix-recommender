import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Netflix AI Finder", page_icon="🍿", layout="wide")

@st.cache_data
def load_data():
    # Lien stable vers le dataset complet (8800 titres)
    url = "https://raw.githubusercontent.com/shivamb/netflix-shows/master/netflix_titles.csv"
    df = pd.read_csv(url)
    df = df.fillna('')
    # On prépare la "soupe de mots" pour l'algorithme
    df['combined_features'] = df['listed_in'] + " " + df['description'] + " " + df['cast'] + " " + df['director']
    return df

df = load_data()

# --- BARRE LATÉRALE (FILTRES) ---
st.sidebar.header("⚙️ Filtres de recherche")

# Filtre par Type
type_filter = st.sidebar.radio("Que cherchez-vous ?", ["Tout", "Movie", "TV Show"])

# Filtre par Pays (on nettoie les noms de pays car certains titres en ont plusieurs)
all_countries = set()
for c in df['country'].unique():
    for sub_c in str(c).split(', '):
        if sub_c: all_countries.add(sub_c)

selected_country = st.sidebar.selectbox("Filtrer par pays d'origine :", ["Tous les pays"] + sorted(list(all_countries)))

# --- APPLICATION DES FILTRES ---
filtered_df = df.copy()

if type_filter != "Tout":
    filtered_df = filtered_df[filtered_df['type'] == type_filter]

if selected_country != "Tous les pays":
    filtered_df = filtered_df[filtered_df['country'].str.contains(selected_country)]

# --- MOTEUR DE RECOMMANDATION ---
@st.cache_resource
def compute_sim(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['combined_features'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

# On recalcule la similarité sur les données filtrées
if not filtered_df.empty:
    cosine_sim = compute_sim(filtered_df.reset_index())
    titles = filtered_df['title'].values
else:
    st.error("Aucun film ne correspond à ces filtres.")
    st.stop()

# --- INTERFACE PRINCIPALE ---
st.title("🎬 Netflix AI Recommender")
st.write(f"Analyse de **{len(filtered_df)}** titres disponibles pour vos critères.")

# Sélection du film de référence
selected_movie = st.selectbox("Sélectionnez un film/série que vous aimez :", titles)

if st.button('Trouver des pépites similaires 🚀'):
    # Trouver l'index dans le dataframe filtré
    idx = filtered_df[filtered_df['title'] == selected_movie].index[0]
    # Récupérer l'index relatif pour la matrice de similarité
    rel_idx = list(filtered_df.index).index(idx)
    
    sim_scores = list(enumerate(cosine_sim[rel_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:7]

    st.write("---")
    st.subheader(f"Basé sur vos goûts et vos filtres ({selected_country}) :")
    
    cols = st.columns(3)
    for i, score in enumerate(sim_scores):
        movie_idx = score[0]
        row = filtered_df.iloc[movie_idx]
        
        with cols[i % 3]:
            st.info(f"**{row['title']}**")
            st.caption(f"📅 {row['release_year']} | 🌍 {row['country']}")
            st.write(f"*{row['listed_in']}*")
            with st.expander("Lire le synopsis"):
                st.write(row['description'])
