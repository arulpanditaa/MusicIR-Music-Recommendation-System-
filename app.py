"""
Run with:  streamlit run app.py

Requirements in the same folder
    - artist_metadata_clean.csv
    - baseline_similarity.npy
    - dataset.csv
"""
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="MusicIR",
    page_icon="🎵",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    padding-top: 2rem;
    max-width: 760px;
}

h1 {
    font-family: 'Space Mono', monospace !important;
    font-size: 2rem !important;
    letter-spacing: -1px;
    color: #f0f0f0 !important;
}

.subtitle {
    color: #888;
    font-size: 0.95rem;
    margin-top: -12px;
    margin-bottom: 28px;
    font-weight: 300;
}

.rec-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    display: flex;
    align-items: center;
    gap: 14px;
    transition: border-color 0.2s;
}

.rec-card:hover {
    border-color: #555;
}

.rec-rank {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #555;
    min-width: 24px;
}

.rec-name {
    font-size: 1rem;
    font-weight: 600;
    color: #f0f0f0;
    flex: 1;
}

.rec-score {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #666;
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #555;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
    margin-top: 24px;
}

.badge {
    display: inline-block;
    background: #222;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    color: #888;
    font-family: 'Space Mono', monospace;
    margin-right: 6px;
}

.badge-green {
    border-color: #2a4a2a;
    color: #5a9a5a;
    background: #1a2e1a;
}

.improvement {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #5a9a5a;
    margin-top: 16px;
    padding: 10px 14px;
    background: #1a2e1a;
    border: 1px solid #2a4a2a;
    border-radius: 8px;
}

.stTextInput > div > div > input {
    background: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
    color: #f0f0f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 12px 16px !important;
}

.stTextInput > div > div > input:focus {
    border-color: #666 !important;
    box-shadow: none !important;
}

.stButton > button {
    background: #f0f0f0 !important;
    color: #111 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    padding: 10px 24px !important;
    width: 100%;
    letter-spacing: 1px;
}

.stButton > button:hover {
    background: #ddd !important;
}

.stSlider > div {
    color: #888 !important;
}

div[data-testid="stMarkdownContainer"] p {
    color: #ccc;
}

.error-box {
    background: #2e1a1a;
    border: 1px solid #4a2a2a;
    border-radius: 8px;
    padding: 12px 16px;
    color: #c77;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

#data loading

@st.cache_resource
def load_data():
    df_meta   = pd.read_csv("artist_metadata_clean.csv")
    sim_matrix = np.load("baseline_similarity.npy")
    df_songs  = pd.read_csv("dataset.csv")
    df_songs["primary_artist"] = df_songs["artists"].str.split(";").str[0].str.strip()

    #build artist - genres map
    artist_genres = (
        df_songs.groupby("primary_artist")["track_genre"]
        .apply(lambda x: set(x.dropna()))
        .to_dict()
    )

    #build graph
    G = nx.Graph()
    n = len(df_meta)
    for i in range(n):
        G.add_node(i, artist=df_meta["artist"].iloc[i])

    for i in range(n):
        for j in range(i + 1, n):
            ai = df_meta["artist"].iloc[i]
            aj = df_meta["artist"].iloc[j]
            gi = artist_genres.get(ai, set())
            gj = artist_genres.get(aj, set())
            if gi & gj:
                G.add_edge(i, j, weight=0.6)
            if sim_matrix[i][j] > 0.12:
                if G.has_edge(i, j):
                    G[i][j]["weight"] = min(1.0, G[i][j]["weight"] + sim_matrix[i][j])
                else:
                    G.add_edge(i, j, weight=float(sim_matrix[i][j]))

    return df_meta, sim_matrix, G, artist_genres


def baseline_recommend(artist_name, df, sim_matrix, top_n):
    matches = df[df["artist"].str.lower() == artist_name.strip().lower()]
    if matches.empty:
        return None, None
    idx = matches.index[0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    results = [(df["artist"].iloc[i], round(float(s), 4)) for i, s in scores if i != idx]
    return results[:top_n], idx


def hybrid_recommend(artist_name, df, sim_matrix, G, alpha, top_n):
    matches = df[df["artist"].str.lower() == artist_name.strip().lower()]
    if matches.empty:
        return None, None
    idx = matches.index[0]
    candidates = [(u, v, p) for u, v, p in nx.jaccard_coefficient(
        G, [(idx, j) for j in range(len(df)) if j != idx]
    )]
    jaccard_lookup = {v: p for u, v, p in candidates}
    scores = []
    for j in range(len(df)):
        if j == idx:
            continue
        ir    = float(sim_matrix[idx][j])
        graph = jaccard_lookup.get(j, 0.0)
        score = (alpha * ir) + ((1 - alpha) * graph)
        scores.append((df["artist"].iloc[j], round(score, 4)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n], idx


def get_artist_genres(artist_name, artist_genres):
    genres = artist_genres.get(artist_name, set())
    return sorted(list(genres))[:3]  # show up to 3


#interface

st.markdown("# 🎵 MusicIR")
st.markdown('<p class="subtitle">Graph-Based Music Recommendation &nbsp;·&nbsp; Gilberto Rios and Arul Pandita ·&nbspJHU 601.466</p>',
            unsafe_allow_html=True)

# Load data
with st.spinner("Loading models..."):
    try:
        df_meta, sim_matrix, G, artist_genres = load_data()
        data_loaded = True
    except FileNotFoundError as e:
        data_loaded = False
        missing = str(e)

if not data_loaded:
    st.markdown(f'<div class="error-box"> Missing file: {missing}<br>Make sure dataset.csv, artist_metadata_clean.csv, and baseline_similarity.npy are in the same folder as app.py.</div>',
                unsafe_allow_html=True)
    st.stop()

all_artists = sorted(df_meta["artist"].tolist())

#search
artist_input = st.text_input(
    "",
    placeholder="Type an artist name (e.g. Bad Bunny, Drake, Sam Smith...)",
    label_visibility="collapsed"
)

col1, col2 = st.columns(2)
with col1:
    top_n = st.slider("Recommendations", min_value=3, max_value=10, value=5)
with col2:
    alpha = st.slider("Graph weight (α)", min_value=0.1, max_value=0.9, value=0.6, step=0.1,
                      help="How much weight to give TF-IDF vs graph score. 0.6 = 60% TF-IDF, 40% graph.")

search = st.button("GET RECOMMENDATIONS")

#results

if search and artist_input.strip():
    b_results, b_idx = baseline_recommend(artist_input, df_meta, sim_matrix, top_n)
    h_results, h_idx = hybrid_recommend(artist_input, df_meta, sim_matrix, G, alpha, top_n)

    if b_results is None:
        #fuzzy match sugg
        close = [a for a in all_artists if artist_input.lower() in a.lower()]
        st.markdown(f'<div class="error-box">Artist "<b>{artist_input}</b>" not found in metadata.</div>',
                    unsafe_allow_html=True)
        if close:
            st.markdown(f"**Did you mean:** {', '.join(close[:5])}")
    else:
        #artist info
        queried_artist = df_meta["artist"].iloc[b_idx]
        genres = get_artist_genres(queried_artist, artist_genres)
        genre_badges = "".join([f'<span class="badge">{g}</span>' for g in genres])

        st.markdown(f"<br>Showing recommendations for &nbsp;<b>{queried_artist}</b>&nbsp; {genre_badges}",
                    unsafe_allow_html=True)

        #side by side columns
        col_b, col_h = st.columns(2)

        with col_b:
            st.markdown('<p class="section-label">Baseline · TF-IDF only</p>', unsafe_allow_html=True)
            for i, (artist, score) in enumerate(b_results, 1):
                genres_r = get_artist_genres(artist, artist_genres)
                genre_str = genres_r[0] if genres_r else ""
                st.markdown(f"""
                <div class="rec-card">
                    <span class="rec-rank">#{i}</span>
                    <span class="rec-name">{artist}<br>
                    <span style="font-size:0.75rem;color:#555;font-weight:400">{genre_str}</span></span>
                    <span class="rec-score">{score}</span>
                </div>""", unsafe_allow_html=True)

        with col_h:
            st.markdown('<p class="section-label">Hybrid · TF-IDF + Graph</p>', unsafe_allow_html=True)
            for i, (artist, score) in enumerate(h_results, 1):
                genres_r = get_artist_genres(artist, artist_genres)
                genre_str = genres_r[0] if genres_r else ""
                st.markdown(f"""
                <div class="rec-card">
                    <span class="rec-rank">#{i}</span>
                    <span class="rec-name">{artist}<br>
                    <span style="font-size:0.75rem;color:#555;font-weight:400">{genre_str}</span></span>
                    <span class="rec-score">{score}</span>
                </div>""", unsafe_allow_html=True)

        #overlap analysis
        b_set = set([a for a, _ in b_results])
        h_set = set([a for a, _ in h_results])
        shared = b_set & h_set
        new_in_hybrid = h_set - b_set

        if new_in_hybrid:
            st.markdown(
                f'<div class="improvement">💠 Graph added {len(new_in_hybrid)} new artist(s) not in baseline: '
                f'{", ".join(sorted(new_in_hybrid))}</div>',
                unsafe_allow_html=True
            )

elif search and not artist_input.strip():
    st.markdown('<div class="error-box">Please enter an artist name.</div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

