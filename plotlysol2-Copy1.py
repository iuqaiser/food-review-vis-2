import os
import numpy as np
import pandas as pd
import wget
import ast
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import streamlit as st
from streamlit_jupyter import StreamlitPatcher
StreamlitPatcher().jupyter()

df = pd.read_csv("food_review.csv")
df = df.drop_duplicates(keep = 'first')
embedding_array = np.array(df['embedding'].apply(ast.literal_eval).to_list())

# word_vectors = dict(zip(df['Word'], embedding_array))

# Define a function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def create_embedding(data): 
    # Tokenize the input text
    tokens = word_tokenize(data.lower())
    vector_size = embedding_array.shape[1]
    
    # Train Word2Vec model
    model = gensim.models.Word2Vec([tokens], min_count=1, vector_size=vector_size, window=5, sg=1)
    
    # Aggregate embeddings for all tokens into a single embedding
    embedding = np.zeros((model.vector_size,), dtype=np.float32)
    for token in tokens:
        if token in model.wv:
            embedding += model.wv[token]
    
    return embedding.reshape(1, -1)  # Reshape to a 2D array

# Define your visualization function
def visualize(query):
    # Your visualization code here
    print("THIS IS THE QUERY WORD:")
    print(query)
    if not query:
        query = "Wolfgang Puck"
    print(query)
    st.write("Visualizing for query:", query)
    query_embedding = np.array(create_embedding(query))
    print(query_embedding)
    df['distance'] = cdist(embedding_array, query_embedding)

    scaler = MinMaxScaler()
    scaler.fit(df[['distance']])
    df['normalised'] = scaler.transform(df[['distance']])

    tsne_model = TSNE(
        n_components = 2,
        perplexity = 15,
        random_state = 42,
        init = 'random',
        learning_rate = 200
    )
    tsne_embeddings = tsne_model.fit_transform(embedding_array)

    visualisation_data = pd.DataFrame(
        {'x': tsne_embeddings[:, 0],
         'y': tsne_embeddings[:, 1],
         'Similarity': df['normalised'],
         'Summary': df['Summary'],
         'Text': df['Text']}
    )
    #visualisation_data
    visualisation_data = visualisation_data.drop_duplicates(keep = 'first')

    # +
    plot = px.scatter(
        visualisation_data,
        x = 'x',
        y = 'y',
        color = 'Similarity',
        hover_name = "Summary",
        color_continuous_scale = 'rainbow',
        opacity = 0.3,
        title = f"Similarity to '{query}' visualised using t-SNE"
    )

    plot.update_layout(
        width = 650,
        height = 650
    )
    # Show the plot
    #plot.show()
    # -

    df2 = visualisation_data[['Similarity', 'Summary', 'Text']]
    df2 = df2.sort_values(by = 'Similarity', ascending = False)
    df2 = df2.drop_duplicates(keep = 'first')
    df2 = df2.reset_index()
    df2 = df2.drop(columns=['index'])

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df2.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df2.Similarity, df2.Summary, df2.Text],
                   fill_color='lavender',
                   align='left'))
    ])
    #fig.show()

    st.plotly_chart(plot, use_container_width=True)
    df2


# Main function for Streamlit app
def main():
    st.title("Food Review Vector Embedding Search")

    # Text input for search query
    if "search_query_entered" not in st.session_state:
        st.session_state.search_query_entered = False

    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
        
    search_query = st.text_input("Enter your search query:", "")
    #visualize()
    # Button to trigger search
    if st.button("Search") or st.session_state.search_query_entered:
        # Update query and visualize
        st.session_state.search_query = search_query
    visualize(search_query)

    # Handle Enter key press event
    if st.session_state.search_query != search_query:
        st.session_state.search_query_entered = True
    else:
        st.session_state.search_query_entered = False

if __name__ == "__main__":
    main()
