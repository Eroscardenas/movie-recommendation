import pickle
from pathlib import Path
from difflib import get_close_matches

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_DIR = PROJECT_DIR / "models"


# ---------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------

def load_pickle(filename):
    with open(MODEL_DIR / filename, "rb") as file:
        return pickle.load(file)


knn_model = load_pickle("knn_model.pkl")
movie_matrix = load_pickle("movie_matrix.pkl")
df = load_pickle("movies.pkl")

df = df.reset_index(drop=True)


ACTOR_COLUMNS = [
    "Star1",
    "Star2",
    "Star3",
    "Star4"
]


# ---------------------------------------------------------
# Movie search
# ---------------------------------------------------------

def find_movie(query):
    query = query.strip().lower()

    if not query:
        return None

    titles = (
        df["Series_Title"]
        .fillna("")
        .astype(str)
    )

    # Exact match
    exact = titles.str.lower().eq(query)

    if exact.any():
        return exact[exact].index[0]

    # Partial match
    partial = titles.str.contains(
        query,
        case=False,
        na=False,
        regex=False
    )

    if partial.any():
        return partial[partial].index[0]

    # Fuzzy match
    lookup = {
        title.lower(): index
        for index, title in titles.items()
    }

    matches = get_close_matches(
        query,
        lookup.keys(),
        n=1,
        cutoff=0.55
    )

    if not matches:
        return None

    return lookup[matches[0]]


# ---------------------------------------------------------
# Actor search
# ---------------------------------------------------------

def find_actor(query):
    query = query.strip().lower()

    if not query:
        return None

    actors = set()

    for column in ACTOR_COLUMNS:
        actors.update(
            df[column]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )

    lookup = {
        actor.lower(): actor
        for actor in actors
    }

    # Exact match
    if query in lookup:
        return lookup[query]

    # Partial match
    partial = [
        actor
        for key, actor in lookup.items()
        if query in key
    ]

    if partial:
        return partial[0]

    # Fuzzy match
    matches = get_close_matches(
        query,
        lookup.keys(),
        n=1,
        cutoff=0.55
    )

    if not matches:
        return None

    return lookup[matches[0]]


def movies_by_actor(actor_name):
    matched_actor = find_actor(actor_name)

    if matched_actor is None:
        return pd.DataFrame(), None

    mask = pd.Series(
        False,
        index=df.index
    )

    for column in ACTOR_COLUMNS:
        mask |= df[column].str.contains(
            matched_actor,
            case=False,
            na=False,
            regex=False
        )

    return df[mask], matched_actor


# ---------------------------------------------------------
# Result helper
# ---------------------------------------------------------

def build_result(row, distance):
    poster = ""

    if "Poster_Link" in row.index:
        poster = row["Poster_Link"]

    return {
        "Title": row["Series_Title"],
        "Year": row["Released_Year"],
        "Genre": row["Genre"],
        "Director": row["Director"],
        "IMDB": row["IMDB_Rating"],
        "Poster": poster,
        "Similarity": round(
            (1 - distance) * 100,
            1
        )
    }


# ---------------------------------------------------------
# Recommendation by title
# ---------------------------------------------------------

def recommend_by_title(title, n=5):
    movie_index = find_movie(title)

    if movie_index is None:
        return pd.DataFrame(), None

    distances, indices = knn_model.kneighbors(
        movie_matrix[movie_index],
        n_neighbors=n + 1
    )

    results = []

    for distance, index in zip(
        distances[0][1:],
        indices[0][1:]
    ):
        row = df.iloc[index]

        results.append(
            build_result(
                row,
                distance
            )
        )

    reference = df.iloc[movie_index]

    return (
        pd.DataFrame(results),
        reference
    )


# ---------------------------------------------------------
# Recommendation by actor
# ---------------------------------------------------------

def recommend_by_actor(actor_name, n=5):
    actor_movies, matched_actor = movies_by_actor(
        actor_name
    )

    if actor_movies.empty:
        return pd.DataFrame(), None, None

    source_indices = actor_movies.index.tolist()

    mean_vector = movie_matrix[
        source_indices
    ].mean(axis=0)

    mean_vector = csr_matrix(
        np.asarray(mean_vector)
    )

    distances, indices = knn_model.kneighbors(
        mean_vector,
        n_neighbors=min(
            len(df),
            n + len(source_indices) + 10
        )
    )

    results = []

    for distance, index in zip(
        distances[0],
        indices[0]
    ):
        if index in source_indices:
            continue

        row = df.iloc[index]

        results.append(
            build_result(
                row,
                distance
            )
        )

        if len(results) >= n:
            break

    return (
        pd.DataFrame(results),
        actor_movies,
        matched_actor
    )