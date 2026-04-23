import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title('🎬 AI Movie Recommender System')
st.write('Créé par les étudiants Younes , Zineb , Yasmine , Lina')

movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

selected_movie_name = st.selectbox(
    ' Sélectionnez un film :',
    movies['title'].values
)

if st.button('Recommander 🚀'):
    recommendations = recommend(selected_movie_name)
    st.write("### Movie Recommender")
    for i in recommendations:
        st.success(i)
