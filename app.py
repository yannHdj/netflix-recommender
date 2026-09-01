import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

st.set_page_config(page_title="Netflix Finder", page_icon="🍿")

# --- DONNÉES INTÉGRÉES (Pour être sûr que ça marche) ---
csv_data = """type,title,listed_in,description
Movie,Dick Johnson Is Dead,Documentaries,As a daughter nears a milestone she stages her father's death.
TV Show,Blood & Water,"International TV Shows, TV Dramas",A teen sets out to prove if a swimming star is her abducted sister.
TV Show,Ganglands,"Crime TV Shows, Action",A skilled thief and his team are pulled into a deadly turf war.
TV Show,Jailbirds New Orleans,"Docuseries, Reality TV",Feuds and flirtations abound among incarcerated women.
TV Show,Midnight Mass,"TV Dramas, TV Horror",A charismatic priest brings miracles and mysteries to a dying town.
Movie,My Little Pony,"Children & Family Movies",A hero believes ponies and unicorns should be friends.
Movie,Sankofa,"Dramas, International",An enslaved woman journeys back in time to experience the past.
Movie,The Starling,Comedies,A woman adjusting to loss battles a cheeky bird in her garden.
Movie,Jeux d'enfants,"Dramas, Romantic",A childhood game of dare continues into adulthood.
Movie,Inception,"Action, Sci-Fi",A thief who steals secrets through dreams is given a final chance.
"""

@st.cache_data
def load_data():
    # On lit le texte ci-dessus comme si c'était un fichier
    df = pd.read_csv(io.StringIO(csv_data))
    df = df.fillna('')
    df['combined'] = df['listed_in'] + " " + df['description']
    return df

df = load_data()

# --- CALCUL ---
@st.cache_resource
def get_sim_matrix(_data):
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(_data['combined'])
    return cosine_similarity(matrix, matrix)

cosine_sim = get_sim_matrix(df)

# --- INTERFACE ---
st.title("🎬 Mon Recommandeur Netflix")
st.write("L'application fonctionne avec un échantillon de test !")

title = st.selectbox("Choisissez un film/série :", df['title'].values)

if st.button("Trouver des idées"):
    idx = df[df['title'] == title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:4]
    
    st.write("### Recommandations :")
    for i, s in enumerate(scores):
        movie = df.iloc[s[0]]
        st.success(f"**{movie['title']}**")
        st.caption(f"Genre: {movie['listed_in']}")
        st.write(movie['description'])
