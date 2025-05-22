# Core Pkg
import pandas as pd
import streamlit as st
import streamlit.components.v1 as stc
from fuzzywuzzy import process
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load Dataset
def load_data(data):
    return pd.read_csv(data)

# Vectorize + Cosine Similarity Matrix
def vectorize_text_to_cosine_mat(data):
    count_vect = CountVectorizer()
    cv_mat = count_vect.fit_transform(data)
    return cosine_similarity(cv_mat)

# Recommendation System
@st.cache_data
def get_recommendation(title, cosine_sim_mat, df, num_of_rec=10):
    course_indices = pd.Series(df.index, index=df['course_title']).drop_duplicates()
    idx = course_indices.get(title)
    if idx is None:
        raise KeyError("Course not found")
    sim_scores = list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    selected_course_indices = [i[0] for i in sim_scores[1:]]
    result_df = df.iloc[selected_course_indices]
    result_df['similarity_score'] = [x[1] for x in sim_scores[1:]]
    return result_df[['course_title', 'similarity_score', 'url', 'price', 'num_subscribers']].head(num_of_rec)

# Search for Closest Matches
@st.cache_data
def search_term_if_not_found(term, df):
    suggestions = process.extract(term, df['course_title'].tolist(), limit=10)
    suggested_courses = [s[0] for s in suggestions if s[1] > 50]

    if suggested_courses:
        result_df = df[df['course_title'].isin(suggested_courses)].copy()
        result_df['similarity_score'] = [s[1] for s in suggestions if s[1] > 50]
        result_df = result_df.sort_values(by='similarity_score', ascending=False)
        return result_df[['course_title', 'similarity_score', 'url', 'price', 'num_subscribers']]
    else:
        return pd.DataFrame()  # Return empty DataFrame if no matches are found

# HTML Template
RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px; box-shadow:0 0 15px 5px #ccc; background-color: #a8f0c6; border-left: 5px solid #6c6c6c;">
    <h4>{}</h4>
    <p style="color:blue;"><span style="color:black;">📈Score:</span>{}</p>
    <p style="color:blue;"><span style="color:black;">🔗</span><a href="{}" target="_blank">Link</a></p>
    <p style="color:blue;"><span style="color:black;">💲Price:</span>{}</p>
    <p style="color:blue;"><span style="color:black;">🧑‍🎓 Students:</span>{}</p>
</div>
"""

# Main App
def main():
    st.title("Course Recommendation App")

    menu = ["Home", "Recommend", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    df = load_data("data/udemy_course_data.csv")

    if choice == "Home":
        st.subheader("Home")
        for _, row in df.head(10).iterrows():
            rec_title = row['course_title']
            rec_url = row['url']
            rec_price = row['price']
            rec_num_sub = row['num_subscribers']

            st.markdown(
                f"""
                <div class="course-card">
                    <h4>{rec_title}</h4>
                    <p><span style="color:black;">🔗</span><a href="{rec_url}" target="_blank">Link</a></p>
                    <p><span style="color:black;">💲Price:</span> {rec_price}</p>
                    <p><span style="color:black;">🧑‍🎓 Students:</span> {rec_num_sub}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    elif choice == "Recommend":
        st.subheader("Recommend Courses")
        cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])
        search_term = st.text_input("Search")
        num_of_rec = st.sidebar.number_input("Number", 4, 30, 7)

        if st.button("Recommend"):
            if not search_term.strip():
                st.warning("Please enter a search term.")
            else:
                try:
                    # Main recommendation results
                    results = get_recommendation(search_term, cosine_sim_mat, df, num_of_rec)
                    with st.expander("Results as JSON"):
                        st.write(results.to_dict('index'))

                    for _, row in results.iterrows():
                        rec_title = row['course_title']
                        rec_score = row['similarity_score']
                        rec_url = row['url']
                        rec_price = row['price']
                        rec_num_sub = row['num_subscribers']
                        stc.html(RESULT_TEMP.format(rec_title, rec_score, rec_url, rec_price, rec_num_sub), height=350)

                except KeyError:
                    # If course is not found, display suggested options
                    st.warning("Course not found. Please refine your search.")
                    st.info("Suggested Options include:")

                    # Use the search function to get similar courses
                    result_df = search_term_if_not_found(search_term, df)
                    if not result_df.empty:
                        for _, row in result_df.iterrows():
                            rec_title = row['course_title']
                            rec_score = row['similarity_score']
                            rec_url = row['url']
                            rec_price = row['price']
                            rec_num_sub = row['num_subscribers']
                            stc.html(RESULT_TEMP.format(rec_title, rec_score, rec_url, rec_price, rec_num_sub), height=350)
                    else:
                        st.error("No similar courses found.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    else:
        st.subheader("About")
        st.text("Built with Streamlit & Pandas")
        st.markdown("### Project Members")
        st.write("""
        - **Ritik Mishra**: Frontend Developer
        - **Mohit Singh**: Frontend Developer
        - **Rohit Saseendran**: Backend Developer
        - **Sachin Mishra**: Backend Developer
        """)
        st.image("gl.jpeg", caption="GL Bajaj Institute Of Technology and Management, Greater Noida", width=800)

if __name__ == '__main__':
    main()
