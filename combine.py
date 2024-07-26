import pandas as pd
import boto3
from io import StringIO
import re

# Initialize the S3 client
s3 = boto3.client('s3')
s3_bucket_name = 'myflicksasa'
csv_files = [
    'imdb/processed_0.csv',
    'imdb/processed_1.csv',
    'imdb/processed_2.csv',
    'imdb/processed_3.csv',
    'imdb/processed_4.csv',
    'imdb/processed_5.csv',
    'imdb/processed_6.csv',
    'imdb/processed_7.csv',
    'imdb/processed_8.csv',
    'imdb/processed_9.csv'
]
combined_csv_path = 'imdb/combined_processed.csv'

# Function to read CSV from S3
def read_csv_from_s3(bucket_name, file_path):
    obj = s3.get_object(Bucket=bucket_name, Key=file_path)
    return pd.read_csv(obj['Body'])

# Function to clean the year data
def clean_year(year):
    if pd.isna(year):
        return year
    year = str(year)
    # Replace non-standard dash characters with standard dash
    year = re.sub(r'[â€“–—]', '-', year)
    # Remove any non-numeric and non-dash characters
    year = re.sub(r'[^\d-]', '', year)
    return year

# Read and combine all CSV files
data_frames = []
for file in csv_files:
    df = read_csv_from_s3(s3_bucket_name, file)
    # Clean the 'year' column
    if 'year' in df.columns:
        df['year'] = df['year'].apply(clean_year)
    data_frames.append(df)

combined_df = pd.concat(data_frames, ignore_index=True)

# Save the combined DataFrame to CSV in memory
csv_buffer = StringIO()
combined_df.to_csv(csv_buffer, index=False)

# Upload the combined CSV to S3
s3.put_object(Bucket=s3_bucket_name, Key=combined_csv_path, Body=csv_buffer.getvalue())

print("Combined CSV has been saved to S3 bucket.")
