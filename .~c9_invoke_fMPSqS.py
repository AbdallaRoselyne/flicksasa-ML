from flask import Flask, request, jsonify
import pandas as pd
import boto3
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Initialize boto3 client with region (if needed)
s3 = boto3.client('s3', region_name='us-west-2')  # Change 'us-west-2' to the correct region

# S3 bucket and file details
bucket_name = 'myflicksasa'
file_key = 'imdb/combined_processed.csv'

# Fetch the file from S3
def fetch_movie_data():
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    data = obj['Body'].read().decode('utf-8')
    return pd.read_csv(StringIO(data))

# Load and preprocess data
movies_df = fetch_movie_data()
movies_df['description'] = movies_df['description'].fillna('')

# Initialize and fit the TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['description'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Strip prefixes from movie titles
movies_df['clean_title'] = movies_df['title'].str.replace(r'^\d+\.\s*', '', regex=True)

# Create a reverse map of indices and cleaned movie titles
indices = pd.Series(movies_df.index, index=movies_df['clean_title']).drop_duplicates()

# Function to get movie recommendations based on description similarity
def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in indices:
        return []
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    recommendations = movies_df['clean_title'].iloc[movie_indices]
    return recommendations.tolist()

@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title')
    if not title:
        return jsonify({'error': 'Title parameter is required'}), 400

    recommendations = get_recommendations(title)
    if not recommendations:
        return jsonify({'error': 'Movie title not found'}), 404

    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)
