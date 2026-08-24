import streamlit as st


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

def render_header():
    st.markdown(
        """
<div class="hero">
<div class="hero-label">MOVIE DISCOVERY</div>
<h1>Movie Recommendation System</h1>
<p class="hero-text">
Discover movies based on a title you already like or explore
recommendations through an actor's filmography.
</p>
<div class="hero-meta">
KNN · Cosine Similarity · Movie Dataset
</div>
<p class="hero-note">Dataset snapshot: 2020</p>
</div>
""",
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# Instructions
# ---------------------------------------------------------

def render_instructions():
    with st.container(border=True):
        st.markdown(
            """
<div class="section-label">GETTING STARTED</div>
<h3 class="compact-title">How to use it</h3>
""",
            unsafe_allow_html=True
        )

        movie_col, actor_col = st.columns(2)

        with movie_col:
            st.markdown("**Search by movie**")
            st.caption(
                "Enter a title such as The Matrix, "
                "The Godfather or Inception."
            )

        with actor_col:
            st.markdown("**Search by actor**")
            st.caption(
                "Enter an actor such as Christian Bale, "
                "Al Pacino or Leonardo DiCaprio."
            )

        st.caption(
            "Small spelling mistakes are accepted. "
            "The system will try to find the closest match."
        )


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def format_year(year):
    try:
        return str(int(float(year)))
    except (ValueError, TypeError):
        return str(year)


# ---------------------------------------------------------
# Movie reference
# ---------------------------------------------------------

def render_movie_reference(reference, query):
    matched_title = reference["Series_Title"]

    if query.strip().lower() != str(matched_title).lower():
        st.info(
            f'Showing results for "{matched_title}"'
        )

    st.markdown(
        '<div class="section-label">YOUR SELECTION</div>',
        unsafe_allow_html=True
    )

    poster = (
        reference["Poster_Link"]
        if "Poster_Link" in reference.index
        else None
    )

    poster_col, info_col = st.columns(
        [1, 4],
        vertical_alignment="center"
    )

    with poster_col:
        if poster:
            st.image(
                poster,
                width=130
            )

    with info_col:
        st.caption("BASED ON")

        st.markdown(
            f"### {matched_title}"
        )

        year = format_year(
            reference["Released_Year"]
        )

        st.markdown(
            f"**{year} · {reference['Genre']}**"
        )

        st.markdown(
            f"**Directed by:** "
            f"{reference['Director']}"
        )

        if "IMDB_Rating" in reference.index:
            st.markdown(
                f"**IMDB Rating:** "
                f"{reference['IMDB_Rating']}"
            )


# ---------------------------------------------------------
# Actor reference
# ---------------------------------------------------------

def render_actor_reference(
    reference,
    query,
    matched_actor
):
    if query.strip().lower() != matched_actor.lower():
        st.info(
            f'Showing results for "{matched_actor}"'
        )

    st.markdown(
        '<div class="section-label">YOUR SELECTION</div>',
        unsafe_allow_html=True
    )

    st.caption("BASED ON ACTOR")

    st.markdown(
        f"### {matched_actor}"
    )

    titles = (
        reference["Series_Title"]
        .head(5)
        .tolist()
    )

    st.caption(
        "Movies available in the dataset"
    )

    st.write(
        " · ".join(titles)
    )


# ---------------------------------------------------------
# Best match
# ---------------------------------------------------------

def render_best_match(best):
    st.markdown(
        """
<div class="section-heading">
<div class="section-label">TOP RESULT</div>
<h2>Best Match</h2>
<p>The closest movie found by the recommendation model.</p>
</div>
""",
        unsafe_allow_html=True
    )

    with st.container(border=True):

        poster_col, info_col, score_col = st.columns(
            [1.1, 3, 1.2],
            vertical_alignment="center"
        )

        with poster_col:
            if best.get("Poster"):
                st.image(
                    best["Poster"],
                    width=130
                )

        with info_col:
            st.caption(
                "BEST RECOMMENDATION"
            )

            st.markdown(
                f"### {best['Title']}"
            )

            year = format_year(
                best["Year"]
            )

            st.markdown(
                f"**{year} · {best['Genre']}**"
            )

            st.markdown(
                f"**Directed by:** "
                f"{best['Director']}"
            )

        with score_col:
            st.metric(
                "Similarity",
                f"{best['Similarity']}%"
            )

            st.metric(
                "IMDB",
                best["IMDB"]
            )

        similarity = min(
            max(
                best["Similarity"] / 100,
                0.0
            ),
            1.0
        )

        st.progress(similarity)


# ---------------------------------------------------------
# Similarity chart
# ---------------------------------------------------------

def render_similarity_chart(results):
    st.markdown(
        """
<div class="section-heading">
<div class="section-label">MODEL RESULTS</div>
<h2>Similarity Comparison</h2>
<p>Compare the similarity score of the top recommendations.</p>
</div>
""",
        unsafe_allow_html=True
    )

    chart_data = (
        results[
            ["Title", "Similarity"]
        ]
        .set_index("Title")
    )

    st.bar_chart(
        chart_data,
        horizontal=True
    )


# ---------------------------------------------------------
# Recommended movies
# ---------------------------------------------------------

def render_recommendations(results):
    st.markdown(
        """
<div class="section-heading">
<div class="section-label">DISCOVER</div>
<h2>Recommended Movies</h2>
<p>Movies ranked by their similarity to your search.</p>
</div>
""",
        unsafe_allow_html=True
    )

    movies = list(
        results.head(5).iterrows()
    )

    columns = st.columns(
        len(movies),
        gap="medium"
    )

    for position, ((_, row), column) in enumerate(
        zip(movies, columns),
        start=1
    ):
        with column:
            with st.container(border=True):

                if row.get("Poster"):
                    st.image(
                        row["Poster"],
                        width=135
                    )

                st.caption(
                    f"RECOMMENDATION #{position}"
                )

                st.markdown(
                    f"### {row['Title']}"
                )

                year = format_year(
                    row["Year"]
                )

                st.markdown(
                    f"**{year} · {row['Genre']}**"
                )

                st.markdown(
                    f"**Directed by:** "
                    f"{row['Director']}"
                )

                st.markdown(
                    f"""
<div class="movie-stats">
    <div>
        <span>MATCH</span>
        <strong>{row['Similarity']}%</strong>
    </div>
    <div>
        <span>IMDB</span>
        <strong>{row['IMDB']}</strong>
    </div>
</div>
""",
                    unsafe_allow_html=True
                )

                similarity = min(
                    max(
                        row["Similarity"] / 100,
                        0.0
                    ),
                    1.0
                )

                st.progress(similarity)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

def render_footer():
    st.divider()

    st.markdown(
        """
<div class="app-footer">
<strong>Movie Recommendation System</strong>
<span>KNN · Cosine Similarity · Streamlit</span>
<small>Dataset snapshot: 2020</small>
</div>
""",
        unsafe_allow_html=True
    )