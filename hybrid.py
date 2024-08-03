<<<<<<< HEAD
import pandas as pd
import boto3
from io import StringIO
from pymongo import MongoClient
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import argparse

# Initialize the S3 client
s3_client = boto3.client('s3')

# Function to load CSV data from S3
def load_csv_from_s3(bucket_name, file_key):
    csv_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    return pd.read_csv(StringIO(csv_string))

# Load datasets from S3
bucket_name = 'myflicksasa'
movies_df = load_csv_from_s3(bucket_name, 'imdb/combined_processed.csv')

# Add the movieId column to movies_df
movies_df['movieId'] = range(1, len(movies_df) + 1)

# Initialize and fit the TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')
movies_df['description'] = movies_df['description'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies_df['description'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a reverse map of indices and movie titles
movies_df['clean_title'] = movies_df['title'].str.replace(r'^\d+\.\s*', '', regex=True)
indices = pd.Series(movies_df.index, index=movies_df['clean_title']).drop_duplicates()

# Connect to MongoDB
client = MongoClient('mongodb+srv://planetmoses12:Lahaja40@ipinfoclone.awzagna.mongodb.net/Netflix?retryWrites=true&w=majority')
db = client['Netflix']
interactions_collection = db['userinteractions']

# Function to fetch interactions from MongoDB and update the DataFrame
def fetch_interactions_from_mongodb(user_id):
    interactions = list(interactions_collection.find({'user': user_id}))
    interactions_df = pd.DataFrame(interactions)
    if not interactions_df.empty:
        interactions_df['rating'] = interactions_df.apply(
            lambda x: 5 if x['loved'] else (4 if x['liked'] else (1 if x['disliked'] else 3)), axis=1
        )
    return interactions_df

# Function to get movie recommendations for a specific user using collaborative filtering
def get_collaborative_recommendations(user_id, num_recommendations=10):
    interactions_df = fetch_interactions_from_mongodb(user_id)
    
    if interactions_df.empty:
        return pd.DataFrame(columns=['movieId', 'collaborative_score'])  # Return an empty DataFrame with necessary columns
    
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(interactions_df[['user', 'movieId', 'rating']], reader)
    
    trainset = data.build_full_trainset()
    
    algo = SVD()
    algo.fit(trainset)
    
    all_movie_ids = movies_df['movieId'].tolist()
    rated_movies = interactions_df['movieId'].tolist()
    predictions = [algo.predict(user_id, movie_id) for movie_id in all_movie_ids if movie_id not in rated_movies]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n_predictions = predictions[:num_recommendations]
    top_n_movie_ids = [pred.iid for pred in top_n_predictions]
    recommended_movies = movies_df[movies_df['movieId'].isin(top_n_movie_ids)].copy()
    recommended_movies['collaborative_score'] = [pred.est for pred in top_n_predictions]
    return recommended_movies

# Function to get movie recommendations based on content similarity
def get_content_recommendations(title, num_recommendations=10):
    if title not in indices:
        return pd.DataFrame(columns=['movieId', 'content_score'])  # Return an empty DataFrame with necessary columns
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    recommendations = movies_df.iloc[movie_indices].copy()
    recommendations['content_score'] = [score[1] for score in sim_scores]
    return recommendations

# Weighted Hybrid recommendation function
def get_hybrid_recommendations(user_id, favorite_movie, num_recommendations=10, alpha=0.5):
    collaborative_recs = get_collaborative_recommendations(user_id, num_recommendations)
    content_recs = get_content_recommendations(favorite_movie, num_recommendations)
    
    if collaborative_recs.empty and content_recs.empty:
        return pd.DataFrame(columns=['title', 'description', 'final_score'])  # Return an empty DataFrame with necessary columns
    
    if collaborative_recs.empty:
        content_recs['final_score'] = content_recs['content_score'] * (1 - alpha)
        return content_recs.sort_values(by='final_score', ascending=False).head(num_recommendations)
    
    if content_recs.empty:
        collaborative_recs['final_score'] = collaborative_recs['collaborative_score'] * alpha
        return collaborative_recs.sort_values(by='final_score', ascending=False).head(num_recommendations)
    
    hybrid_recs = pd.merge(collaborative_recs, content_recs, on='movieId', how='outer', suffixes=('_collab', '_content'))
    hybrid_recs['final_score'] = hybrid_recs['collaborative_score'].fillna(0) * alpha + hybrid_recs['content_score'].fillna(0) * (1 - alpha)
    hybrid_recs = hybrid_recs.sort_values(by='final_score', ascending=False).head(num_recommendations)
    
    required_columns = ['title', 'description', 'final_score']
    for col in required_columns:
        if col not in hybrid_recs.columns:
            hybrid_recs[col] = movies_df.set_index('movieId').loc[hybrid_recs['movieId']][col].values
    
    return hybrid_recs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get movie recommendations")
    parser.add_argument("--user_id", type=str, help="User ID to get recommendations for")
    parser.add_argument("--favorite_movie", type=str, help="Favorite movie title to base content recommendations on")
    args = parser.parse_args()
    
    recommendations = None
    
    if not args.user_id and not args.favorite_movie:
        raise ValueError("Please provide a user ID and/or a favorite movie title")
    else:
        if args.user_id and args.favorite_movie:
            recommendations = get_hybrid_recommendations(args.user_id, args.favorite_movie)
        elif args.user_id:
            recommendations = get_collaborative_recommendations(args.user_id)
        elif args.favorite_movie:
            recommendations = get_content_recommendations(args.favorite_movie)
    
    if recommendations is not None:
        print(recommendations[['title', 'description', 'final_score']])

=======
import pandas as pd
import boto3
from io import StringIO
from pymongo import MongoClient
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import argparse

# Initialize the S3 client
s3_client = boto3.client('s3')

# Function to load CSV data from S3
def load_csv_from_s3(bucket_name, file_key):
    csv_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    return pd.read_csv(StringIO(csv_string))

# Load datasets from S3
bucket_name = 'myflicksasa'
movies_df = load_csv_from_s3(bucket_name, 'imdb/combined_processed.csv')

# Add the movieId column to movies_df
movies_df['movieId'] = range(1, len(movies_df) + 1)

# Initialize and fit the TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')
movies_df['description'] = movies_df['description'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies_df['description'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a reverse map of indices and movie titles
movies_df['clean_title'] = movies_df['title'].str.replace(r'^\d+\.\s*', '', regex=True)
indices = pd.Series(movies_df.index, index=movies_df['clean_title']).drop_duplicates()

# Connect to MongoDB
client = MongoClient('mongodb+srv://planetmoses12:Lahaja40@ipinfoclone.awzagna.mongodb.net/Netflix?retryWrites=true&w=majority')
db = client['Netflix']
interactions_collection = db['userinteractions']

# Function to fetch interactions from MongoDB and update the DataFrame
def fetch_interactions_from_mongodb(user_id):
    interactions = list(interactions_collection.find({'user': user_id}))
    interactions_df = pd.DataFrame(interactions)
    if not interactions_df.empty:
        interactions_df['rating'] = interactions_df.apply(
            lambda x: 5 if x['loved'] else (4 if x['liked'] else (1 if x['disliked'] else 3)), axis=1
        )
    return interactions_df

# Function to get movie recommendations for a specific user using collaborative filtering
def get_collaborative_recommendations(user_id, num_recommendations=10):
    interactions_df = fetch_interactions_from_mongodb(user_id)
    
    if interactions_df.empty:
        return pd.DataFrame(columns=['movieId', 'collaborative_score'])  # Return an empty DataFrame with necessary columns
    
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(interactions_df[['user', 'movieId', 'rating']], reader)
    
    trainset = data.build_full_trainset()
    
    algo = SVD()
    algo.fit(trainset)
    
    all_movie_ids = movies_df['movieId'].tolist()
    rated_movies = interactions_df['movieId'].tolist()
    predictions = [algo.predict(user_id, movie_id) for movie_id in all_movie_ids if movie_id not in rated_movies]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n_predictions = predictions[:num_recommendations]
    top_n_movie_ids = [pred.iid for pred in top_n_predictions]
    recommended_movies = movies_df[movies_df['movieId'].isin(top_n_movie_ids)].copy()
    recommended_movies['collaborative_score'] = [pred.est for pred in top_n_predictions]
    return recommended_movies

# Function to get movie recommendations based on content similarity
def get_content_recommendations(title, num_recommendations=10):
    if title not in indices:
        return pd.DataFrame(columns=['movieId', 'content_score'])  # Return an empty DataFrame with necessary columns
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    recommendations = movies_df.iloc[movie_indices].copy()
    recommendations['content_score'] = [score[1] for score in sim_scores]
    return recommendations

# Weighted Hybrid recommendation function
def get_hybrid_recommendations(user_id, favorite_movie, num_recommendations=10, alpha=0.5):
    collaborative_recs = get_collaborative_recommendations(user_id, num_recommendations)
    content_recs = get_content_recommendations(favorite_movie, num_recommendations)
    
    if collaborative_recs.empty and content_recs.empty:
        return pd.DataFrame(columns=['title', 'description', 'final_score'])  # Return an empty DataFrame with necessary columns
    
    if collaborative_recs.empty:
        content_recs['final_score'] = content_recs['content_score'] * (1 - alpha)
        return content_recs.sort_values(by='final_score', ascending=False).head(num_recommendations)
    
    if content_recs.empty:
        collaborative_recs['final_score'] = collaborative_recs['collaborative_score'] * alpha
        return collaborative_recs.sort_values(by='final_score', ascending=False).head(num_recommendations)
    
    hybrid_recs = pd.merge(collaborative_recs, content_recs, on='movieId', how='outer', suffixes=('_collab', '_content'))
    hybrid_recs['final_score'] = hybrid_recs['collaborative_score'].fillna(0) * alpha + hybrid_recs['content_score'].fillna(0) * (1 - alpha)
    hybrid_recs = hybrid_recs.sort_values(by='final_score', ascending=False).head(num_recommendations)
    
    required_columns = ['title', 'description', 'final_score']
    for col in required_columns:
        if col not in hybrid_recs.columns:
            hybrid_recs[col] = movies_df.set_index('movieId').loc[hybrid_recs['movieId']][col].values
    
    return hybrid_recs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get movie recommendations")
    parser.add_argument("--user_id", type=str, help="User ID to get recommendations for")
    parser.add_argument("--favorite_movie", type=str, help="Favorite movie title to base content recommendations on")
    args = parser.parse_args()
    
    recommendations = None
    
    if not args.user_id and not args.favorite_movie:
        raise ValueError("Please provide a user ID and/or a favorite movie title")
    else:
        if args.user_id and args.favorite_movie:
            recommendations = get_hybrid_recommendations(args.user_id, args.favorite_movie)
        elif args.user_id:
            recommendations = get_collaborative_recommendations(args.user_id)
        elif args.favorite_movie:
            recommendations = get_content_recommendations(args.favorite_movie)
    
    if recommendations is not None:
        print(recommendations[['title', 'description', 'final_score']])
>>>>>>> origin/master
