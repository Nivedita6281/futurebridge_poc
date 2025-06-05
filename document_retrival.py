import os
import boto3
import json
import pandas as pd
from dotenv import load_dotenv
import datetime
import io
from flask import Flask, render_template, request, redirect, jsonify, send_file

# Load environment variables
load_dotenv()

# S3 Setup
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    region_name=os.getenv("AWS_REGION")
)

# Config
bucket_name = os.getenv("AWS_BUCKET_NAME")
folder_path = "research_papers"  # S3 folder path
vectorstore_key = f"{folder_path}/vectorstore.json"
local_vectorstore_path = "vectorstore.json"
urls_csv_key = f"{folder_path}/Articles_urls.csv"
local_urls_csv = "Articles_urls.csv"

# Initialize Flask app
app = Flask(__name__)

def check_file_exists(bucket, key):
    """Check if a file exists in the S3 bucket."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False

def get_vectorstore():
    """Retrieve existing vectorstore or create a new one."""
    if os.path.exists(local_vectorstore_path):
        with open(local_vectorstore_path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("Local vectorstore corrupted, checking S3...")
    
    if check_file_exists(bucket_name, vectorstore_key):
        response = s3.get_object(Bucket=bucket_name, Key=vectorstore_key)
        vectorstore_data = json.loads(response['Body'].read().decode('utf-8'))
        
        # Save locally
        with open(local_vectorstore_path, 'w') as f:
            json.dump(vectorstore_data, f, indent=2)
        
        return vectorstore_data
    
    # Create new if not found
    return {"documents": [], "metadata": {"created": datetime.datetime.now().isoformat()}}

def get_urls_mapping():
    """Get mapping of document titles to URLs from CSV."""
    urls_mapping = {}
    
    # Try to get from local file first
    if os.path.exists(local_urls_csv):
        try:
            df = pd.read_csv(local_urls_csv)
            for index, row in df.iterrows():
                if 'title' in df.columns and 'url' in df.columns:
                    urls_mapping[row['title']] = row['url']
            return urls_mapping
        except Exception as e:
            print(f"Error reading local CSV: {e}")
    
    # If not found locally, try S3
    if check_file_exists(bucket_name, urls_csv_key):
        try:
            response = s3.get_object(Bucket=bucket_name, Key=urls_csv_key)
            df = pd.read_csv(io.BytesIO(response['Body'].read()))
            
            # Save locally
            df.to_csv(local_urls_csv, index=False)
            
            for index, row in df.iterrows():
                if 'title' in df.columns and 'url' in df.columns:
                    urls_mapping[row['title']] = row['url']
        except Exception as e:
            print(f"Error reading S3 CSV: {e}")
    
    return urls_mapping

def search_documents(query, vectorstore):
    """
    Search for documents related to the query.
    In a real implementation, this would use proper vector search.
    This simplified version just does basic keyword matching.
    """
    results = []
    query_lower = query.lower()
    
    for doc in vectorstore.get("documents", []):
        # Simple keyword search in filename (would be replaced with vector search)
        if query_lower in doc.get("filename", "").lower():
            results.append(doc)
    
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('q', '')
    if not query:
        return jsonify({"results": []})
    
    vectorstore = get_vectorstore()
    results = search_documents(query, vectorstore)
    
    # Get URL mapping
    urls = get_urls_mapping()
    
    # Add URLs to results if available
    for result in results:
        title = result.get("filename", "").replace(".pdf", "")
        if title in urls:
            result["url"] = urls[title]
    
    return jsonify({"results": results})

@app.route('/document/<filename>')
def get_document(filename):
    """
    Fetch and return the document content.
    If URL is available, redirect to it.
    """
    # Check if there's a URL mapping for this document
    urls = get_urls_mapping()
    title = filename.replace(".pdf", "")
    
    if title in urls:
        return redirect(urls[title])
    
    # If no URL found, try to fetch from S3
    s3_key = f"{folder_path}/{filename}"
    
    if check_file_exists(bucket_name, s3_key):
        # For PDF files, return the file
        if filename.endswith('.pdf'):
            response = s3.get_object(Bucket=bucket_name, Key=s3_key)
            return send_file(
                io.BytesIO(response['Body'].read()),
                mimetype='application/pdf',
                as_attachment=True,
                download_name=filename
            )
    
    return "Document not found", 404

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create basic HTML template
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Document Search</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .search-container { margin-bottom: 30px; }
        input[type="text"] { width: 70%; padding: 10px; font-size: 16px; }
        button { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
        .result-item { margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        .result-title { color: #0066cc; cursor: pointer; text-decoration: underline; }
    </style>
</head>
<body>
    <h1>Document Search</h1>
    
    <div class="search-container">
        <input type="text" id="searchInput" placeholder="Enter keywords to search...">
        <button onclick="searchDocuments()">Search</button>
    </div>
    
    <div id="results"></div>
    
    <script>
        function searchDocuments() {
            const query = document.getElementById('searchInput').value;
            if (!query) return;
            
            fetch(`/search?q=${encodeURIComponent(query)}`)
                .then(response => response.json())
                .then(data => {
                    const resultsDiv = document.getElementById('results');
                    resultsDiv.innerHTML = '';
                    
                    if (data.results.length === 0) {
                        resultsDiv.innerHTML = '<p>No results found.</p>';
                        return;
                    }
                    
                    data.results.forEach(doc => {
                        const resultItem = document.createElement('div');
                        resultItem.className = 'result-item';
                        
                        const title = document.createElement('h3');
                        title.className = 'result-title';
                        title.textContent = doc.filename;
                        
                        title.onclick = function() {
                            if (doc.url) {
                                window.open(doc.url, '_blank');
                            } else {
                                window.location.href = `/document/${doc.filename}`;
                            }
                        };
                        
                        resultItem.appendChild(title);
                        resultsDiv.appendChild(resultItem);
                    });
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('results').innerHTML = 
                        '<p>An error occurred while searching. Please try again.</p>';
                });
        }
        
        // Allow pressing Enter key to search
        document.getElementById('searchInput').addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                searchDocuments();
            }
        });
    </script>
</body>
</html>
        ''')
    
    # Start the Flask app
    print("Starting server at http://localhost:5000")
    app.run(debug=True)