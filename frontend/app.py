from pathlib import Path

import streamlit as st

from recommender import (
    recommend_by_actor,
    recommend_by_title
)

from ui import (
    render_actor_reference,
    render_best_match,
    render_footer,
    render_header,
    render_instructions,
    render_movie_reference,
    render_recommendations,
    render_similarity_chart
)


BASE_DIR = Path(__file__).resolve().parent
CSS_PATH = BASE_DIR / "style.css"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


def load_css():
    if CSS_PATH.exists():
        with open(
            CSS_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            st.markdown(
                f"<style>{file.read()}</style>",
                unsafe_allow_html=True
            )


load_css()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

render_header()
render_instructions()


# ---------------------------------------------------------
# Search
# ---------------------------------------------------------

st.markdown("## Find Recommendations")

search_type = st.radio(
    "Search by",
    ["Movie title", "Actor"],
    horizontal=True
)


placeholder = (
    "e.g. The Matrix"
    if search_type == "Movie title"
    else "e.g. Christian Bale"
)


query = st.text_input(
    "Search",
    placeholder=placeholder
)


search = st.button(
    "Find Movies",
    type="primary"
)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

if search:

    if not query.strip():
        st.warning(
            "Enter a movie title or actor first."
        )

    else:

        if search_type == "Actor":

            (
                results,
                reference,
                matched_actor
            ) = recommend_by_actor(
                query,
                n=5
            )

        else:

            results, reference = recommend_by_title(
                query,
                n=5
            )

            matched_actor = None


        if results.empty:

            st.error(
                "No close match was found. "
                "Try another search."
            )

        else:

            if search_type == "Movie title":

                render_movie_reference(
                    reference,
                    query
                )

            else:

                render_actor_reference(
                    reference,
                    query,
                    matched_actor
                )


            st.divider()


            best = results.iloc[0]

            render_best_match(best)

            render_similarity_chart(
                results
            )

            render_recommendations(
                results
            )


render_footer()