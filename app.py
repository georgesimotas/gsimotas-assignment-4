from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)


newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=1000)
X = vectorizer.fit_transform(documents)

# Apply Truncated SVD (LSA)
lsa = TruncatedSVD(n_components=100)
X_lsa = lsa.fit_transform(X)


def search_engine(query):
    # Transform the query into the TF-IDF space
    query_vec = vectorizer.transform([query])
    
    # Project the query into the LSA space
    query_lsa = lsa.transform(query_vec)
    
    # Compute cosine similarity between query and documents
    cosine_similarities = cosine_similarity(query_lsa, X_lsa).flatten()
    
    # Get the top 5 most similar documents
    top_indices = cosine_similarities.argsort()[-5:][::-1]
    
    top_documents = [documents[i] for i in top_indices]
    top_similarities = cosine_similarities[top_indices]
    
    return top_documents, top_similarities.tolist(), top_indices.tolist()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True)
