import pandas as pd
import numpy as np
import ast
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import logging

df = pd.read_csv("food_review.csv")
df['combined'] = df['Summary'] + " " + df['Text']
df['combined'] = df['combined'].astype(str)
df['combined'] = df['combined'].str.lower()
df = df[['combined', 'embedding']]

embedding_array = np.array(df['embedding'].apply(ast.literal_eval).to_list())
vector_size = embedding_array.shape[1]

# Configure logging to see training progress
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

# Your training data
sentences = [sentence.split() for sentence in df['combined']]

# Define your Word2Vec model
model = Word2Vec(sentences, vector_size=vector_size, window=5, min_count=1, workers=4)

# Define a callback to track training progress (optional)
class LossLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.losses = []
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        print(f"Loss after epoch {self.epoch}: {loss}")
        self.epoch += 1

# Train the Word2Vec model
loss_logger = LossLogger()
model.train(sentences, total_examples=model.corpus_count, epochs=10, callbacks=[loss_logger])

# Save the trained model to disk
model.save("word2vec_model.bin")