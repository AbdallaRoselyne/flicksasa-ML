import pandas as pd
import boto3
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import argparse

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

# Function to get movie recommendations based on description similarity
def get_recommendations(title, movies_df, cosine_sim):
    # Create a reverse map of indices and cleaned movie titles
    indices = pd.Series(movies_df.index, index=movies_df['clean_title']).drop_duplicates()

    # Check if the movie title exists in the dataset
    if title not in indices:
        return []

    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    recommendations = movies_df['clean_title'].iloc[movie_indices]
    return recommendations.tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get movie recommendations based on a given title.")
    parser.add_argument("title", type=str, help="Title of the movie to get recommendations for")
    args = parser.parse_args()

    try:
        # Load the data into a DataFrame
        movies_df = fetch_movie_data()

        # Preprocess the description column
        movies_df['description'] = movies_df['description'].fillna('')

        # Initialize and fit the TF-IDF vectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies_df['description'])

        # Compute the cosine similarity matrix
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        # Strip prefixes from movie titles
        movies_df['clean_title'] = movies_df['title'].str.replace(r'^\d+\.\s*', '', regex=True)

        # Get recommendations for the input movie title
        recommendations = get_recommendations(args.title, movies_df, cosine_sim)

        # Print the recommendations
        for movie in recommendations:
            print(movie)

    except s3.exceptions.NoSuchBucket:
        print(f"Error: The bucket '{bucket_name}' does not exist.")
    except s3.exceptions.NoSuchKey:
        print(f"Error: The key '{file_key}' does not exist in the bucket '{bucket_name}'.")

