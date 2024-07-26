import pandas as pd
import boto3
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Initialize boto3 client with region (if needed)
s3 = boto3.client('s3', region_name='us-west-2')  # Change 'us-west-2' to the correct region

# S3 bucket and file details
bucket_name = 'myflicksasa'
file_key = 'imdb/combined_processed.csv'

# List buckets to verify access
response = s3.list_buckets()
buckets = [bucket['Name'] for bucket in response['Buckets']]
print("Available buckets:", buckets)

# Fetch the file from S3
try:
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    data = obj['Body'].read().decode('utf-8')

    # Load the data into a DataFrame
    movies_df = pd.read_csv(StringIO(data))

    # Preprocess the description column
    movies_df['description'] = movies_df['description'].fillna('')

    # Initialize and fit the TF-IDF vectorizer
    print("Initializing the TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['description'])
    print("TF-IDF matrix shape:", tfidf_matrix.shape)

    # Compute the cosine similarity matrix
    print("Computing the cosine similarity matrix...")
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    print("Cosine similarity matrix shape:", cosine_sim.shape)

    # Strip prefixes from movie titles
    movies_df['clean_title'] = movies_df['title'].str.replace(r'^\d+\.\s*', '', regex=True)

    # Create a reverse map of indices and cleaned movie titles
    indices = pd.Series(movies_df.index, index=movies_df['clean_title']).drop_duplicates()

    # Print the first few cleaned movie titles to verify
    print("Available movie titles:")
    print(movies_df['clean_title'].head(20))

    # Function to get movie recommendations based on description similarity
    def get_recommendations(title, cosine_sim=cosine_sim):
        print(f"Getting recommendations for: {title}")
        
        # Check if the movie title exists in the dataset
        if title not in indices:
            print("Movie title not found in the dataset.")
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
        print("Recommendations found:")
        print(recommendations)
        return recommendations

    # Example: Get recommendations for a specific movie
    user_favorite_movie = 'IF'  # Change to a valid movie title
    recommendations = get_recommendations(user_favorite_movie)

    # Display the recommendations
    print("\nTop 10 movie recommendations:")
    for i, movie in enumerate(recommendations, 1):
        print(f"{i}. {movie}")

except s3.exceptions.NoSuchBucket:
    print(f"Error: The bucket '{bucket_name}' does not exist.")
