import os
import boto3
import json
import datetime
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    region_name=os.getenv("AWS_REGION")
)

bucket_name = os.getenv("AWS_BUCKET_NAME")
folder_path = "research_papers"  # local folder path
vectorstore_key = f"{os.path.basename(folder_path)}/vectorstore.json"
local_vectorstore_path = "vectorstore.json"  # Local path to save vectorstore

def check_file_exists(bucket, key):
    """Check if a file exists in the S3 bucket."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False

def get_vectorstore():
    """Retrieve existing vectorstore or create a new one.
    First checks locally, then S3, and creates new if neither exists."""
    
    # First check if vectorstore exists locally
    if os.path.exists(local_vectorstore_path):
        print(f"✅ Found local vectorstore at {local_vectorstore_path}")
        with open(local_vectorstore_path, 'r') as f:
            try:
                vectorstore_data = json.load(f)
                print(f"✅ Loaded local vectorstore with {len(vectorstore_data.get('documents', []))} documents")
                return vectorstore_data
            except json.JSONDecodeError:
                print(f"⚠️ Local vectorstore file is corrupted. Will check S3.")
    
    # If not local, check S3
    if check_file_exists(bucket_name, vectorstore_key):
        print(f"✅ Existing vectorstore found at s3://{bucket_name}/{vectorstore_key}")
        response = s3.get_object(Bucket=bucket_name, Key=vectorstore_key)
        vectorstore_data = json.loads(response['Body'].read().decode('utf-8'))
        print(f"✅ Loaded existing vectorstore with {len(vectorstore_data.get('documents', []))} documents")
        
        # Save a local copy
        with open(local_vectorstore_path, 'w') as f:
            json.dump(vectorstore_data, f, indent=2)
        print(f"✅ Saved a local copy of the vectorstore to {local_vectorstore_path}")
        
        return vectorstore_data
    else:
        print(f"ℹ️ No existing vectorstore found. Will create a new one.")
        timestamp = datetime.datetime.now().isoformat()
        return {"documents": [], "metadata": {"created": timestamp}}

def ensure_csv_format():
    """Ensure the CSV file has the correct format (title, url columns)"""
    csv_local_path = "Articles_urls.csv"
    
    if os.path.exists(csv_local_path):
        try:
            df = pd.read_csv(csv_local_path)
            
            # Check if the required columns exist
            if not ('title' in df.columns and 'url' in df.columns):
                print("⚠️ CSV file doesn't have the required columns (title, url)")
                print("Creating a new CSV file with the correct format...")
                
                # Create a new DataFrame with the correct columns
                new_df = pd.DataFrame(columns=['title', 'url'])
                
                # Try to map existing columns if possible
                if 'title' not in df.columns and df.shape[1] >= 1:
                    new_df['title'] = df.iloc[:, 0]  # Use first column as title
                
                if 'url' not in df.columns and df.shape[1] >= 2:
                    new_df['url'] = df.iloc[:, 1]  # Use second column as url
                
                # Save the new CSV
                new_df.to_csv(csv_local_path, index=False)
                print(f"✅ Created new CSV file with correct format")
        except Exception as e:
            print(f"⚠️ Error reading CSV: {e}")
            print("Creating a new CSV file with the correct format...")
            pd.DataFrame(columns=['title', 'url']).to_csv(csv_local_path, index=False)
    else:
        print(f"⚠️ CSV file not found. Creating a new one with the correct format...")
        pd.DataFrame(columns=['title', 'url']).to_csv(csv_local_path, index=False)
        print(f"✅ Created new CSV file: {csv_local_path}")

# Main execution
ensure_csv_format()
vectorstore = get_vectorstore()
uploaded_documents = set(doc["filename"] for doc in vectorstore.get("documents", []))

# Upload all PDFs from the folder to the S3 bucket if they don't exist
uploaded_count = 0
skipped_count = 0

# Load the CSV data for titles and URLs
csv_local_path = "Articles_urls.csv"
url_mapping = {}
if os.path.exists(csv_local_path):
    try:
        df = pd.read_csv(csv_local_path)
        for index, row in df.iterrows():
            if 'title' in df.columns and 'url' in df.columns:
                url_mapping[row['title']] = row['url']
    except Exception as e:
        print(f"⚠️ Error reading CSV: {e}")

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".pdf"):
            local_path = os.path.join(root, file)
            s3_key = f"{os.path.basename(folder_path)}/{file}"  # e.g., research_papers/filename.pdf
            
            if check_file_exists(bucket_name, s3_key):
                print(f"ℹ️ File {file} already exists in S3. Skipping upload.")
                skipped_count += 1
            else:
                s3.upload_file(local_path, bucket_name, s3_key)
                print(f"✅ Uploaded {file} to s3://{bucket_name}/{s3_key}")
                uploaded_count += 1
            
            # Add to vectorstore if not already there
            if file not in uploaded_documents:
                # Extract title (filename without extension)
                title = os.path.splitext(file)[0]
                
                # Check if URL exists for this title
                url = url_mapping.get(title, "")
                
                # Add document to vectorstore
                vectorstore["documents"].append({
                    "filename": file,
                    "s3_key": s3_key,
                    "title": title,
                    "url": url,
                    "status": "pending_vectorization"
                })
                
                # Add to CSV if not already there
                if title not in url_mapping and os.path.exists(csv_local_path):
                    try:
                        df = pd.read_csv(csv_local_path)
                        new_row = pd.DataFrame([{'title': title, 'url': ''}])
                        df = pd.concat([df, new_row], ignore_index=True)
                        df.to_csv(csv_local_path, index=False)
                        print(f"✅ Added {title} to CSV file")
                    except Exception as e:
                        print(f"⚠️ Error updating CSV: {e}")

# Upload the Articles_urls.csv file if it doesn't exist or has been updated
csv_local_path = "Articles_urls.csv"
if os.path.exists(csv_local_path):
    s3_csv_key = f"{os.path.basename(folder_path)}/Articles_urls.csv"
    
    if check_file_exists(bucket_name, s3_csv_key):
        print(f"ℹ️ Updating Articles_urls.csv in S3.")
        s3.upload_file(csv_local_path, bucket_name, s3_csv_key)
        print(f"✅ Updated Articles_urls.csv in s3://{bucket_name}/{s3_csv_key}")
    else:
        s3.upload_file(csv_local_path, bucket_name, s3_csv_key)
        print(f"✅ Uploaded Articles_urls.csv to s3://{bucket_name}/{s3_csv_key}")
else:
    print("❌ Articles_urls.csv not found in the current directory")

# Update the vectorstore in S3 and locally
if vectorstore["documents"]:
    # Update timestamp
    vectorstore["metadata"]["last_updated"] = datetime.datetime.now().isoformat()
    
    # Save to S3
    s3.put_object(
        Bucket=bucket_name,
        Key=vectorstore_key,
        Body=json.dumps(vectorstore, indent=2),
        ContentType='application/json'
    )
    print(f"✅ Updated vectorstore at s3://{bucket_name}/{vectorstore_key}")
    
    # Save locally
    with open(local_vectorstore_path, 'w') as f:
        json.dump(vectorstore, f, indent=2)
    print(f"✅ Updated local vectorstore at {local_vectorstore_path}")
    
    print(f"📊 Summary: {uploaded_count} files uploaded, {skipped_count} files skipped (already existed)")
    print(f"📚 Vectorstore now contains {len(vectorstore['documents'])} documents")
else:
    print("❌ No documents found to upload or update in vectorstore")

print("\n✅ Upload process complete.")
