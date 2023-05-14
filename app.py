import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# Load the data
youtube_data = pd.read_csv('youtube_dataset.csv')
nptel_data = pd.read_csv('nptel_dataset.csv')
udemy_data = pd.read_csv('udemy_dataset.csv')
coursera_data = pd.read_csv('coursera_dataset.csv')

# Calculate cosine similarity matrix for YouTube and NPTEL data
similarity_matrix_youtube = cosine_similarity(youtube_data.iloc[:, 1:-1])
similarity_matrix_nptel = cosine_similarity(nptel_data.iloc[:, 1:-1])

# Given a user ID, find similar users from both matrices
def get_similar_users(user_id):
    youtube_row = similarity_matrix_youtube[user_id - 1, :]  # Fix indexing issue
    nptel_row = similarity_matrix_nptel[user_id - 1, :]  # Fix indexing issue
    # Find indices of top two most similar users for each matrix
    youtube_indices = youtube_row.argsort()[-3:][::-1] + 1
    nptel_indices = nptel_row.argsort()[-3:][::-1] + 1
    # Combine the two lists of indices and remove the user_id itself
    similar_user_indices = list(set(youtube_indices) | set(nptel_indices))
    similar_user_indices = [i for i in similar_user_indices if i != user_id]  # Fix indexing issue
    return similar_user_indices

# Given a user ID and a list of similar users, recommend a platform based on average attention level
def recommend_platform(user_id, similar_users):
    user_group = similar_users.copy()  # Create a copy of similar_users list
    user_group.append(user_id)  # Append user_id to the list
    udemy_attention = udemy_data.loc[udemy_data['User_id'].isin(user_group), 'Attention'].mean()
    coursera_attention = coursera_data.loc[coursera_data['User_id'].isin(user_group), 'Attention'].mean()
    if udemy_attention > coursera_attention:
        return 'Udemy'
    else:
        return 'Coursera'


image1 = Image.open('images/nptel_heatmap.png')
image2 = Image.open('images/youtube_heatmap.png')
image3 = Image.open('images/youtube_bar.png')
image4 = Image.open('images/nptel_bar.png')
image5 = Image.open('images/udemy_bar.png')
image6 = Image.open('images/coursera_bar.png')
image7 = Image.open('images/overview.png')



# Streamlit app
def app():
    # Create a multi-page app
    st.title("Platform Recommendation System using EEG")
    pages = ['Recommendation', 'Exploratory Data Analysis','Description of the Dataset','About']
    choice = st.sidebar.selectbox('Select Page', pages)

    # Page for recommendation
    if choice == 'Recommendation':
        st.header('Recommendation')
        username = st.text_input("Enter Username")
        user_id = st.number_input("Enter User ID", value=12, step=1, min_value=12, max_value=24)
        course_name = st.text_input("Enter Course Name")

        if st.button("Recommend"):
            similar_users = get_similar_users(user_id)
            platform = recommend_platform(user_id, similar_users)
            st.success(f"Recommended platform for {username} ({user_id}) taking {course_name} is {platform}.")


    # Page for statistics
    elif choice == 'Exploratory Data Analysis':
        st.header('Exploratory Data Analysis')
        st.text("Heatmap of NPTEL Video")
        st.image(image1,  use_column_width=True)
        st.text("Heatmap of YouTube Video")
        st.image(image2,  use_column_width=True)
        st.header("Video set 1: ")
        st.header("Topic: Introduction to Big Data")
        st.text("Bar charts representing the count of number of attentive and not attentive samples")
        st.header("NPTEL:")
        st.image(image4,  use_column_width=True)
        st.header("YouTube:")
        st.image(image3,  use_column_width=True)

        st.header("Video set 2: ")
        st.header("Topic: Introduction to Agile")
        st.text("Bar charts representing the count of number of attentive and not attentive samples")
        st.header("Udemy:")
        st.image(image5,  use_column_width=True)
        st.header("Coursera:")
        st.image(image6,  use_column_width=True)


    elif choice == "Description of the Dataset":
        st.subheader("Description of the Dataset")
        # st.image("C:\Users\bhumi\OneDrive\Desktop\fyp_code\images\yt_ss.png")
        st.text("User_id - A unique reference ID to every sample")
        st.text("Delta waves")
        st.text("Theta Waves")
        st.text("Low Alpha Waves")
        st.text("High Alpha Waves")
        st.text("Low Beta Waves")
        st.text("High Beta Waves")
        st.text("Low Gamma Waves") 
        st.text("High Gamma Waves")   
        st.text("Attention Waves")
        st.text("Meditation Waves")
        st.text("Attentive - Whether a sample is attentive or not(Yes/No)")

    
    elif choice == "About":
        st.image(image7,  use_column_width=True)

if __name__ == '__main__':
    app()
