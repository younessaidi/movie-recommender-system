import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 1. إعطاء عنوان للموقع
st.title('🎬 AI Movie Recommender System')
st.write('Créé par Younes , zaineb , Yasmine , Lina')

# 2. جلب البيانات (الجدول والمتجهات المضغوطة)
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
vectors = pickle.load(open('vectors.pkl', 'rb'))

# 3. حساب مصفوفة التشابه هنا (بسرعة فائقة)
similarity = cosine_similarity(vectors)

# 4. دالة الاقتراح
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

# 5. تصميم الواجهة
selected_movie_name = st.selectbox(
    'اختر فيلما يعجبك / Sélectionnez un film :',
    movies['title'].values
)

# 6. زر الاقتراح
if st.button('Recommander 🚀'):
    recommendations = recommend(selected_movie_name)
    
    st.write("### الأفلام المقترحة لك:")
    for i in recommendations:
        st.success(i)
