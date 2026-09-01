import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
st.set_page_config(page_title="Netflix Finder", page_icon="🍿")

@st.cache_data
def load_data():
    # Liste de liens de secours au cas où l'un tombe en 404
    links = [
        "https://raw.githubusercontent.com/diego-vicente/netflix-report/master/data/netflix_titles.csv",
        "https://raw.githubusercontent.com/shivamb/netflix-shows/master/netflix_titles.csv",
        "https://raw.githubusercontent.com/611683930/Netflix-Visualizations-Recommendation/master/netflix_titles.csv"
    ]
    
    for url in links:
        try:
            df = pd.read_csv(url)
            df = df.fillna('')
            # Création de la soupe de mots pour l'algorithme
            df['combined'] = df['listed_in'] + " " + df['description'] + " " + df['cast']
            return df
        except:
            continue
    return None

# Chargement
df = load_data()

if df is None:
    st.error("⚠️ Impossible de charger les données. Veuillez vérifier votre connexion ou réessayer plus tard.")
else:
    # --- CALCUL ---
    @st.cache_resource
    def get_sim_matrix(_data):
        tfidf = TfidfVectorizer(stop_words='english')
        matrix = tfidf.fit_transform(_data['combined'])
        return cosine_similarity(matrix, matrix)

    cosine_sim = get_sim_matrix(df)

    # --- INTERFACE ---
    st.title("🎬 Mon Recommandeur Netflix")
    
    title = st.selectbox("Quel film/série avez-vous aimé ?", df['title'].values)

    if st.button("Trouver des idées"):
        idx = df[df['title'] == title].index[0]
        scores = list(enumerate(cosine_sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:7]
        
        st.write("### Voici des titres similaires :")
        cols = st.columns(2)
        for i, s in enumerate(scores):
            movie = df.iloc[s[0]]
            with cols[i % 2]:
                st.success(f"**{movie['title']}**")
                st.write(f"*{movie['listed_in']}*")
                st.caption(movie['description'][:150] + "...")
