import boto3
import os
import PyPDF2
from openai import OpenAI
import numpy as np
import faiss
import time
import re
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from datetime import datetime
import pickle
from fuzzywuzzy import fuzz, process  # Added fuzzywuzzy for fuzzy string matching
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

# Load environment variables
load_dotenv()
# Initialize clients
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    region_name=os.getenv("AWS_REGION")
)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Local embedding model
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CACHE_SIZE = 200
FUZZY_MATCH_THRESHOLD = 80  # Minimum threshold for fuzzy matching (0-100)
# Define a cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
# Create the cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)
# Use absolute paths for cache files
EMBEDDINGS_CACHE_FILE = os.path.join(CACHE_DIR, "embeddings_cache.pkl")
INDEX_CACHE_FILE = os.path.join(CACHE_DIR, "faiss_index.pkl")
DOCUMENTS_CACHE_FILE = os.path.join(CACHE_DIR, "documents_cache.pkl")
TFIDF_VECTORIZER_FILE = os.path.join(CACHE_DIR, "tfidf_vectorizer.pkl")
TFIDF_MATRIX_FILE = os.path.join(CACHE_DIR, "tfidf_matrix.npz")

def s3_check_file_exists(bucket, key):
    """Check if a file exists in the S3 bucket."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False

class PaperProcessor:
    def __init__(self):
        self._init_models()
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.metadata_cache = {}
        self.query_cache = {}
        self.last_processed_time = 0
        self.url_mapping = {}
        self.embeddings_modified = False
        self.term_variants={}
        self.term_frequencies={}
        # Add TF-IDF related attributes
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        # Add dynamic term tracking
        self.term_importance = {}  # Tracks term importance scores
        self.query_history = []    # Stores recent queries
        self.term_cooccurrence = defaultdict(lambda: defaultdict(int))  # Term relationships
        self.last_boost_update = time.time()
        # Load URL mappings immediately
        self.load_url_mappings()
        try:
            print("Loading embedding model...")
            self.embedding_model = SentenceTransformer(
                EMBEDDING_MODEL, 
                device='cpu'  # Force CPU if GPU issues occur
            )
            print("Model loaded successfully")
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            raise
    def _update_term_variants(self, text: str):
        """Analyze text to discover term variants dynamically"""
        words = re.findall(r'\b\w{4,}\b', text.lower())  # Only consider words with 4+ chars
        stems = [self._stem_word(w) for w in words]  # Basic stemming
        # Group similar terms
        for i, stem in enumerate(stems):
            if stem not in self.term_variants:
                self.term_variants[stem] = set()
            self.term_variants[stem].add(words[i])
            
            # Update frequency counts
            self.term_frequencies[words[i]] = self.term_frequencies.get(words[i], 0) + 1
            
    def _stem_word(self, word: str) -> str:
        """Basic stemming to group similar terms"""
        # Implement or use a library for more sophisticated stemming
        return re.sub(r'(ing|s|es|ed|ly)$', '', word)
        
    def process_document(self, text: str):
        """Process each document to learn term variants"""
        self._update_term_variants(text)
        
    def create_dummy_data(self):
        """Create dummy data for testing when S3 access fails"""
        print("Creating dummy research paper data for testing...")
        
        # Create 3 dummy papers
        dummy_papers = [
            {
                "title": "Advances in Cancer Immunotherapy",
                "authors": "Smith J, Johnson R",
                "year": "2023",
                "text": "Cancer immunotherapy has shown remarkable progress in recent years. The development of checkpoint inhibitors has revolutionized treatment approaches. Clinical trials demonstrate improved survival rates across multiple cancer types.",
                "journal": "Journal of Oncology Research",
                "volume": "42",
                "issue": "3",
                "pages": "156-172",
                "doi": "10.1234/jor.2023.42.3.156",
                "SRID": "SRID12345678",
                "abstract": "This paper reviews recent advances in cancer immunotherapy with focus on checkpoint inhibitors and CAR-T cell approaches.",
                "source": "cancer_immunotherapy.pdf"
            },
            {
                "title": "COVID-19 Vaccine Efficacy",
                "authors": "Lee H, Garcia M",
                "year": "2022",
                "text": "mRNA vaccines demonstrated high efficacy in preventing severe COVID-19. Long-term immunity studies show sustained antibody response. Booster doses further enhance protection against emerging variants.",
                "journal": "Vaccine Science",
                "volume": "15",
                "issue": "2",
                "pages": "45-61",
                "doi": "10.5678/vs.2022.15.2.45",
                "SRID": "SRID87654321",
                "abstract": "This study evaluates the long-term efficacy of mRNA vaccines against COVID-19 and its variants.",
                "source": "covid_vaccine_efficacy.pdf"
            },
            {
                "title": "Alzheimer's Disease Biomarkers",
                "authors": "Wong P, Miller K",
                "year": "2023",
                "text": "Blood-based biomarkers show promise for early Alzheimer's detection. Phosphorylated tau proteins correlate with disease progression. Combined biomarker panels improve diagnostic accuracy in preclinical stages.",
                "journal": "Neurology Research",
                "volume": "28",
                "issue": "4",
                "pages": "210-225",
                "doi": "10.7890/nr.2023.28.4.210",
                "SRID": "SRID23456789",
                "abstract": "This research identifies novel blood-based biomarkers for early detection of Alzheimer's disease.",
                "source": "alzheimers_biomarkers.pdf"
            }
        ]
        
        # Create embeddings
        self.documents = dummy_papers
        texts = [doc['text'] for doc in self.documents]
        embeddings = self.embedding_model.encode(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        print("✅ Created dummy data with 3 research papers")

    @lru_cache(maxsize=1)
    def _init_models(self):
        """Cache model initialization with TF-IDF"""
        start = time.time()
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_df=0.85,           # Ignore terms that appear in >85% of documents
            min_df=2,              # Ignore terms that appear in fewer than 2 documents
            max_features=20000,    # Limit features to prevent memory issues
            stop_words='english',  # Remove English stopwords
            ngram_range=(1, 2)     # Include both unigrams and bigrams
        )
        print(f"Models loaded in {time.time()-start:.2f}s")
    
    def load_url_mappings(self):
        """Load URL mappings from CSV with fuzzy matching capabilities"""
        self.url_mapping = {}
        csv_local_path = "Articles_urls.csv"
    
        # Try to load from local file first
        if os.path.exists(csv_local_path):
            try:
                df = pd.read_csv(csv_local_path)
                if 'title' in df.columns and 'url' in df.columns:
                    for index, row in df.iterrows():
                        # Create multiple matching keys for each title
                        title = str(row['title']).strip()
                        url = str(row['url']).strip()
                    
                        # Add original title
                        self.url_mapping[title] = url
                    
                        # Add lowercase version
                        self.url_mapping[title.lower()] = url
                    
                        # Add filename-like version (replace spaces with underscores)
                        filename_version = title.replace(" ", "_") + ".pdf"
                        self.url_mapping[filename_version] = url
                    
                        # Add filename without extension
                        self.url_mapping[title.replace(" ", "_")] = url
                    
                    print(f"✅ Loaded {len(df)} URL mappings with multiple matching variants")
                    return
            except Exception as e:
                print(f"⚠️ Error reading local CSV: {e}")
    
        # If local file failed or doesn't exist, try S3
        try:
            s3_csv_key = "research_papers/Articles_urls.csv"
            if s3_check_file_exists(BUCKET_NAME, s3_csv_key):
                response = s3.get_object(Bucket=BUCKET_NAME, Key=s3_csv_key)
                df = pd.read_csv(BytesIO(response['Body'].read()))
            
                if 'title' in df.columns and 'url' in df.columns:
                    for index, row in df.iterrows():
                        title = str(row['title']).strip()
                        url = str(row['url']).strip()
                    
                        # Add original title
                        self.url_mapping[title] = url
                    
                        # Add lowercase version
                        self.url_mapping[title.lower()] = url
                    
                        # Add filename-like version
                        filename_version = title.replace(" ", "_") + ".pdf"
                        self.url_mapping[filename_version] = url
                    
                        # Add filename without extension
                        self.url_mapping[title.replace(" ", "_")] = url
                    
                    print(f"✅ Loaded {len(df)} URL mappings from S3 with variants")
                else:
                    print("⚠️ CSV in S3 doesn't have required columns")
            else:
                print("ℹ️ No URL mapping CSV found in S3")
        except Exception as e:
            print(f"⚠️ Error loading URL mappings from S3: {e}")

    def get_best_url_match(self, title: str) -> str:
        """Find the best matching URL using fuzzy string matching"""
        if not title or not self.url_mapping:
            return ""
            
        # Try exact match first
        if title in self.url_mapping:
            return self.url_mapping[title]
            
        # Try lowercased match
        if title.lower() in self.url_mapping:
            return self.url_mapping[title.lower()]
            
        # Try filename-like version (with and without extension)
        filename_version = title.replace(" ", "_") + ".pdf"
        if filename_version in self.url_mapping:
            return self.url_mapping[filename_version]
            
        filename_no_ext = title.replace(" ", "_")
        if filename_no_ext in self.url_mapping:
            return self.url_mapping[filename_no_ext]
            
        # Use fuzzy matching as last resort
        choices = list(self.url_mapping.keys())
        if not choices:
            return ""
            
        # Get the best fuzzy match
        best_match, score = process.extractOne(title, choices, scorer=fuzz.token_sort_ratio)
        
        # Only return if score is above threshold
        if score >= FUZZY_MATCH_THRESHOLD:
            return self.url_mapping[best_match]
        return ""

    def correct_spelling(self, query: str) -> str:
        """More aggressive spelling correction using learned variants"""
        if not query or not self.term_variants:
            return query
    
        corrected_words = []
        for word in query.split():
            if len(word) <= 3:  # Skip short words
                corrected_words.append(word)
                continue
        
            lower_word = word.lower()
            stem = self._stem_word(lower_word)
    
            # Find best match among known variants
            if stem in self.term_variants:
                variants = list(self.term_variants[stem])
                # Find the most frequent variant with good match
                best_match, best_score = process.extractOne(
                    lower_word, 
                    variants, 
                    scorer=fuzz.token_sort_ratio
                )
            
                # Only correct if significantly better match
                if best_score >= 85:  # Increased from original threshold
                    # Preserve original capitalization
                    if word.isupper():
                        corrected_words.append(best_match.upper())
                    elif word.istitle():
                        corrected_words.append(best_match.title())
                    else:
                        corrected_words.append(best_match)
                    continue
                
            corrected_words.append(word)

        return " ".join(corrected_words)
        
    def _build_tfidf_matrix(self):
        """Build TF-IDF matrix with dynamic term boosting"""
        if not self.documents:
            return

        texts = [doc.get('text', '') for doc in self.documents]
        self.tfidf_vectorizer = TfidfVectorizer(
            max_df=0.85,
            min_df=2,
            max_features=20000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
    
        # Apply dynamic boosts
        if hasattr(self, 'term_importance') and self.term_importance:
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            for term, importance in self.term_importance.items():
                if term in feature_names and importance > 1.0:
                    idx = list(feature_names).index(term)
                    self.tfidf_matrix[:, idx] *= min(importance, 2.5)  # Cap boost at 2.5x
                    
    def save_cache(self):
        """Save embeddings, index, documents and TF-IDF to disk cache"""
        if self.embeddings_modified and self.index is not None and len(self.documents) > 0:
            try:
                # Save FAISS index
                faiss.write_index(self.index, INDEX_CACHE_FILE)
                print(f"✅ Saved FAISS index to {INDEX_CACHE_FILE}")
            
                # Save documents (but without embeddings to save space)
                with open(DOCUMENTS_CACHE_FILE, 'wb') as f:
                    pickle.dump(self.documents, f)
                print(f"✅ Saved {len(self.documents)} documents to {DOCUMENTS_CACHE_FILE}")
            
                # Save TF-IDF vectorizer and matrix
                if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
                    with open(TFIDF_VECTORIZER_FILE, 'wb') as f:
                        pickle.dump(self.tfidf_vectorizer, f)
                
                    # Save as sparse matrix to conserve space
                    sp.save_npz(TFIDF_MATRIX_FILE, self.tfidf_matrix)
                    print(f"✅ Saved TF-IDF vectorizer and matrix")
            
                # Reset the modified flag
                self.embeddings_modified = False
            except Exception as e:
                print(f"⚠️ Error saving cache: {e}")

    def load_cache(self):
        """Load embeddings, index, documents and TF-IDF from disk cache"""
        # Check if at least basic files exist
        if os.path.exists(INDEX_CACHE_FILE) and os.path.exists(DOCUMENTS_CACHE_FILE):
            try:
                # Load FAISS index
                self.index = faiss.read_index(INDEX_CACHE_FILE)
                print(f"✅ Loaded FAISS index from {INDEX_CACHE_FILE}")
            
                # Load documents
                with open(DOCUMENTS_CACHE_FILE, 'rb') as f:
                    self.documents = pickle.load(f)
                print(f"✅ Loaded {len(self.documents)} documents from cache")
            
                # Try to load TF-IDF data if available
                if os.path.exists(TFIDF_VECTORIZER_FILE) and os.path.exists(TFIDF_MATRIX_FILE):
                    with open(TFIDF_VECTORIZER_FILE, 'rb') as f:
                        self.tfidf_vectorizer = pickle.load(f)
                
                    self.tfidf_matrix = sp.load_npz(TFIDF_MATRIX_FILE)
                    print(f"✅ Loaded TF-IDF vectorizer and matrix")
                else:
                    # If TF-IDF files don't exist, build them
                    print("⚠️ TF-IDF files not found, rebuilding...")
                    self._build_tfidf_matrix()
            
                return True
            except Exception as e:
                print(f"⚠️ Error loading cache: {e}")
                self.index = None
                self.documents = []
                self.tfidf_matrix = None
                return False
        return False
        
    def force_build_cache(self):
        """Force rebuild and save the cache files"""
        print("Force building cache files...")
        # Build TF-IDF matrix
        self._build_tfidf_matrix()
        # Mark as modified to ensure saving
        self.embeddings_modified = True
        
        # Save cache
        self.save_cache()
        
        return True

    def extract_text_from_s3_pdf(self, s3_key: str) -> str:
        """Extract text from a PDF directly in S3 without downloading"""
        response = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        pdf_content = response['Body'].read()
        
        text = ""
        with BytesIO(pdf_content) as pdf_file:
            try:
                reader = PyPDF2.PdfReader(pdf_file)
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    # clean up the text as we extract it
                    page_text = re.sub(r'\s+', ' ', page_text)
                    page_text = re.sub(r'(\w+)-\s+(\w+)', r'\1 \2', page_text)  # Remove extra spaces
                    text += page_text + "\n"
            except Exception as e:
                print(f"Error extracting text from {s3_key}: {str(e)}")
                return ""
        return text.strip()
    
    def extract_metadata(self, text: str, filename: str) -> Dict[str, Any]:
        """Enhanced metadata extraction with improved reference handling"""
        if filename in self.metadata_cache:
            return self.metadata_cache[filename]
        # Improved title extraction from filename
        title_default = (
            os.path.splitext(filename)[0]
            .replace("_", " ")
            .replace("-", " ")
            .replace(".", " ")
            .title()
        )
        # Try to extract year from filename
        year_match = re.search(r"(19|20)\d{2}", filename)
        year = year_match.group(0) if year_match else str(datetime.now().year - 1)
        # Generate consistent IDs
        paper_id = abs(hash(filename)) % 100000000
        srid = f"SRID{paper_id:08d}"
        try:
        # First try with GPT extraction
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """Extract from research paper:
                        1. Standard metadata (title, authors, etc.)
                        2. Main content text (excluding references)
                        2. Sections (each with heading and 1-2 sentence summary)
                        3. References (seperate from main content)
                        Return as JSON with these fields:
                        - title, authors, year, journal, abstract
                        - sections: [{heading, content}]
                        - references: [strings]
                        """
                    },
                    {
                        "role": "user",
                        "content": f"Extract from:\n{text[:15000]}"  # More context
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=3000
            )
            metadata = json.loads(response.choices[0].message.content)
            # Validate references
            if 'references' not in metadata or len(metadata['references']) < 3:
                # Use improved fallback with reference section detection
                ref_section = self._locate_reference_section(text)
                metadata['references'] = self._extract_references_fallback(ref_section)
            if 'content' not in metadata or not metadata['content']:
                metadata['content'] = self._extract_main_content(text)
        except Exception as e:
            print(f"Metadata extraction error: {e}")
            main_content = self._extract_main_content(text)
            references = self._extract_references_fallback(text)
            #try to extract title from first line if GPT failed
            first_line = text.split('\n')[0].strip()
            title = first_line if len(first_line) < 120 and not first_line.isupper() else filename
            # Try to create fallback sections based on headings
            sections = []
            split_sections = re.split(r'\n\s*(Introduction|Methods?|Results?|Discussion|Conclusion)\s*\n', text, flags=re.I)
            if len(split_sections) >= 3:
                for i in range(1, len(split_sections), 2):
                    heading = split_sections[i]
                    content = split_sections[i+1]
                    sections.append({
                        'heading': heading,
                        'content': content.strip()
                    })

            metadata = {
                "title": title_default,
                "authors": "Unknown Authors",
                "year": year,
                "journal": "Unknown Journal",
                "text":main_content,
                "content":main_content,
                "references": references,
                "sections": sections if sections else [{
                    'heading':'Content',
                    'content': main_content
                }], 

            }
            # Ensure we have good content sections
            if 'sections' not in metadata or not metadata['sections']:
                # Fallback content extraction
                main_content = self._extract_main_content(text)
                if main_content:
                    metadata['sections'] = [{
                    'heading': 'Content',
                    'content': main_content
                }]
            # Ensure we have some text content
            if 'text' not in metadata or not metadata['text']:
                metadata['text'] = self._extract_main_content(text) or text[:2000]
        # Add standard fields
        metadata.update({
            "volume": "1",
            "issue": "1",
            "pages": "1-10",
            "doi": f"10.0000/journal.{paper_id}",
            "SRID": srid,
            "abstract": metadata.get('abstract', text[:200] + "..."),
            "url": self.get_best_url_match(metadata.get("title", title_default)),
            "source": filename
        })

        self.metadata_cache[filename] = metadata
        return metadata
    def _extract_main_content(self, text: str) -> str:
        """Extract main content by removing reference section"""
        sections = re.split(r'\n\s*(Abstract|Introduction|Methods?|Results?|Discussion|Conclusion|References)\s*\n', 
            text, flags=re.I)
    
        # If we found proper sections
        if len(sections) > 1:
            main_content = []
            for i in range(1, len(sections), 2):
                heading = sections[i]
                content = sections[i+1]
                if not self._is_mostly_references(content) and not heading.lower().startswith('ref'):
                    main_content.append(f"{heading}\n{content}")
            return "\n\n".join(main_content[:3])  # Return first 3 sections
    
        # Fallback for poorly structured papers
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
        return "\n\n".join(paragraphs[:5]) if paragraphs else text[:2000]
        
    def _extract_references_fallback(self, text: str) -> List[str]:
        """Improved fallback reference extraction with better splitting"""
        # First try to split by common reference patterns
        ref_pattern = r'(?:\n|^)\s*\d+\.\s+(.*?)(?=\n\s*\d+\.\s|\n\s*\[?\d+\]?\s|\n\s*[A-Z][a-z]+|$)'
        references = re.findall(ref_pattern, text, re.DOTALL)
        # If we didn't find enough, try more aggressive splitting
        if len(references) < 3:
            # Split by lines that start with numbers
            numbered_lines = re.findall(r'(?:\n|^)\s*\d+\.\s+(.*)', text)
            references.extend(numbered_lines)
    
        # Additional cleanup
        cleaned_refs = []
        for ref in references:
            # Remove any trailing periods or spaces
            ref = ref.strip().rstrip('.')
            # Remove any citation markers like [1] or (Author, 2023)
            ref = re.sub(r'\[[\d,]+\]', '', ref)
            ref = re.sub(r'\([^)]*\d{4}\)', '', ref)
            if ref and len(ref) > 20:  # Minimum length for a reference
                cleaned_refs.append(ref)
    
        return cleaned_refs[:100]  # Return up to 100 references
  # Return up to 50 references
    def _locate_reference_section(self, text: str) -> str:
        """Try to locate the reference section in paper text"""
        # Common section headers that might precede references
        ref_headers = [
            'references',
            'bibliography',
            'literature cited',
            'citations',
            'reference list'
        ]
        # Look for section headers
        for header in ref_headers:
            match = re.search(
                rf'\n\s*{header}\s*[\n:].*?(?=\n\s*\w|$)',
                text, 
                re.IGNORECASE | re.DOTALL
            )
            if match:
                return match.group(0)
        # Fallback: last 20% of document (where references usually are)
        return text[-int(len(text)*0.2):]
    def _is_mostly_references(self, text: str) -> bool:
        """Detect if a given text block is mostly made up of references."""
        if not text or len(text) < 100:  # Don't mark short texts as references
            return False
        
        # Count reference indicators
        ref_indicators = 0
        lines = [line.strip() for line in text.split('\n') if line.strip()]
    
        for line in lines:
            # Skip empty lines and section headers
            if re.match(r'^\s*(abstract|introduction|methods?|results?|discussion|conclusion)\s*$', line, re.I):
                return False
            
        # Check for reference patterns
        if (re.search(r'^\d+\.\s+[A-Z]', line) or  # 1. Author
            re.search(r'\[[\d,]+\]', line) or      # [1] or [1,2]
            re.search(r'\bdoi:\s*10\.', line) or   # DOI
            re.search(r'pp?\.\s*\d+', line)):     # Page numbers
            ref_indicators += 1
    
        # Only mark as references if >70% of lines match patterns
        return (ref_indicators / len(lines)) > 0.7
    def classify_article_type(self, text: str) -> str:
        """Simple heuristic to classify article type from content"""
        text_lower = text.lower()
        if "case report" in text_lower or "case study" in text_lower:
            return "Case Study"
        elif "randomized trial" in text_lower or "clinical trial" in text_lower:
            return "Clinical Trial"
        elif "review" in text_lower or "systematic review" in text_lower:
            return "Review"
        elif "research" in text_lower or "experiment" in text_lower:
            return "Research Article"
        else:
            return "Research Article"
    def process_s3_folder(self, s3_folder: str):
        """Process all PDFs in an S3 folder with improved caching"""
        # First check if we should reprocess
        if os.path.exists(DOCUMENTS_CACHE_FILE) and not self.should_reprocess(s3_folder):
            if self.load_cache():
                print("✅ Loaded from cache (no new files detected)")
            return
        # First, try to load from cache
        if self.load_cache():
            print("✅ Successfully loaded processor from cache")
            # Build TF-IDF matrix from loaded documents
            self._build_tfidf_matrix()
            # Still load URL mappings to ensure they're up to date
            self.load_url_mappings()
            return
            
        # Ensure folder path ends with '/'
        if not s3_folder.endswith('/'):
            s3_folder += '/'
        # Check if S3 credentials are set
        if not all([os.getenv("AWS_ACCESS_KEY"), os.getenv("AWS_SECRET_KEY"), os.getenv("AWS_REGION"), os.getenv("AWS_BUCKET_NAME")]):
            print("Warning: S3 credentials not fully configured. Using dummy data.")
            self.create_dummy_data()
            return
            
        try:
            # First try a simple operation to test S3 connection
            s3.head_bucket(Bucket=BUCKET_NAME)
            # List objects in the bucket
            response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=s3_folder)
            pdf_files = []
            for obj in response.get('Contents', []):
                if obj['Key'].endswith('.pdf'):
                    pdf_files.append(obj['Key'])
    
            if not pdf_files:
                print("No PDF files found in S3 folder. Creating dummy data.")
                self.create_dummy_data()
                return
    
            # For debugging, print what we found
            print(f"Found {len(pdf_files)} PDF files in S3: {', '.join(os.path.basename(f) for f in pdf_files)}")
    
            # Process each PDF
            successful_files = 0
            for pdf_key in pdf_files:
                try:
                    text = self.extract_text_from_s3_pdf(pdf_key)
                    if not text.strip():
                        print(f"Warning: No text extracted from {pdf_key}")
                        continue
            
                    chunks = self.text_splitter.split_text(text)
                    filename = os.path.basename(pdf_key)
            
                    # Extract metadata for this file
                    metadata = self.extract_metadata(text, filename)
            
                    # Add chunks to documents with metadata
                    for i, chunk in enumerate(chunks):
                        document_data = {
                            'text': chunk,
                            'source': filename,
                            'chunk_id': i,
                            'references': metadata.get('references', [])  # ✅ ADD THIS LINE
                        }
                        document_data.update(metadata)
                        self.documents.append(document_data)
                        # Add article type classification
                        document_data['article_type'] = self.classify_article_type(text)

            
                    successful_files += 1
            
                except Exception as e:
                    print(f"Error processing {pdf_key}: {str(e)}")
                    continue
    
            print(f"Successfully processed {successful_files} out of {len(pdf_files)} PDF files")
    
            if not self.documents:
                print("No valid documents were processed. Creating dummy data.")
                self.create_dummy_data()
                return
    
            # Generate embeddings
            texts = [doc['text'] for doc in self.documents]
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
    
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
        
            # After creating FAISS index, build TF-IDF matrix
            if self.documents:
                self._build_tfidf_matrix()
                
            # Mark embeddings as modified
            self.embeddings_modified = True
        
            # Save to cache
            self.save_cache()
            
            # Force rebuild cache if TF-IDF still not available
            if self.tfidf_matrix is None:
                print("TF-IDF matrix still not available after save, forcing rebuild")
                self.force_build_cache()

        except Exception as e:
            print(f"Error accessing S3: {str(e)}")
            print("Falling back to dummy data.")
            self.create_dummy_data()

    def preprocess_query(self, query: str) -> str:
        """Dynamic query processing with term boosting"""
        # Track this query
        self.query_history.append(query)
        if len(self.query_history) > 100:  # Keep only recent 100 queries
            self.query_history.pop(0)
    
        # Basic cleaning
        query = query.lower()
        query = re.sub(r'[^a-z0-9\s\-]', '', query)
    
        # Dynamic spelling correction
        query = self.correct_spelling(query)
    
        # Analyze and boost important terms
        terms = query.split()
        boosted_terms = []
    
        for term in terms:
            # Boost terms that appear frequently in important contexts
            importance = self.term_importance.get(term, 1.0)
            if importance > 1.2:  # Significant boost threshold
                boosted_terms.extend([term] * min(int(importance), 3))  # Repeat important terms
            
            # Add synonyms from co-occurrence analysis
            for co_term, count in self.term_cooccurrence[term].items():
                if count > 2 and co_term not in terms:  # Significant co-occurrence
                    boosted_terms.append(co_term)
                
            boosted_terms.append(term)
    
        # Add related terms from similar successful queries
        if len(self.query_history) > 5:
            similar_queries = process.extract(query, self.query_history, limit=5, scorer=fuzz.token_sort_ratio)
            for q, score in similar_queries:
                if score > 70:  # Similar enough
                    for t in q.split():
                        if t not in boosted_terms:
                            boosted_terms.append(t)
    
        return ' '.join(boosted_terms)
        
    def _update_term_importance(self, query: str):
        """Update term importance based on query"""
        terms = query.split()
        for term in terms:
            # Basic importance increment
            self.term_importance[term] = self.term_importance.get(term, 1.0) + 0.1
        
            # Update co-occurrence
            for other_term in terms:
                if term != other_term:
                    self.term_cooccurrence[term][other_term] += 1
                    
        # Gradual decay of importance over time
        if time.time() - self.last_boost_update > 3600:  # Every hour
            self.last_boost_update = time.time()
            for term in list(self.term_importance.keys()):
                self.term_importance[term] *= 0.9  # Decay
                if self.term_importance[term] < 1.0:
                    del self.term_importance[term]
                    
    def _learn_from_successful_query(self, query: str, results: List[Dict]):
        """Learn from documents returned for good queries"""
        important_terms = set(query.split())
    
        # Add terms from top results
        for doc in results[:3]:
            # Get top TF-IDF terms from this doc
            if hasattr(self, 'tfidf_vectorizer') and self.tfidf_vectorizer:
                try:
                    doc_idx = next(i for i, d in enumerate(self.documents) if d.get('text', '') == doc['text'])
                    tfidf_scores = self.tfidf_matrix[doc_idx].toarray().flatten()
                    top_features = np.argsort(-tfidf_scores)[:5]  # Top 5 terms
                    feature_names = self.tfidf_vectorizer.get_feature_names_out()
                    for idx in top_features:
                        term = feature_names[idx]
                        important_terms.add(term)
                except:
                    pass
    
        # Boost these terms
        for term in important_terms:
            if term in self.term_importance:
                self.term_importance[term] += 0.2
            else:
                self.term_importance[term] = 1.2
        
    def query_papers(self, question: str, k: int = 10, hybrid_weight: float = 0.7, 
                year_filter: Optional[str] = None,
                journal_filter: Optional[str] = None, 
                article_type: Optional[str] = None,
                subject: Optional[str] = None,
                min_year: Optional[int] = None,
                max_year: Optional[int] = None) -> List[Dict]:
        """Search for relevant papers with filtering options
        
        Args:
            question: The search query
            k: Number of results to return
            hybrid_weight: Weight for hybrid search (0-1)
            year_filter: Exact year to filter by (e.g., "2023")
            journal_filter: Journal name to filter by
            author_filter: Author name to filter by
            min_year: Minimum year for range filter
            max_year: Maximum year for range filter
        """
        title_match = False
        abstract_match = False
        text_match = False
        # Check if question is a single word or very short
        words = question.strip().split()
        is_short_query = len(words) <= 2

        # Learn from previous queries
        self._update_term_variants(question)

        # Apply dynamic correction
        corrected = self.correct_spelling(question)
        question = self.preprocess_query(question)
        if corrected != question:
            print(f"Corrected query: {question} → {corrected}")
            question = corrected    
        if self.index is None:
            return self.get_dummy_results(question, k)
    
        # Check query cache first
        cache_key = f"{question}_{k}_{hybrid_weight}_{year_filter}_{journal_filter}_{article_type}_{subject}_{min_year}_{max_year}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        try:
            # For short queries, expand with related terms
            if is_short_query:
                # First, try to find exact matches in document titles
                title_matches = []
                for idx, doc in enumerate(self.documents):
                    title = doc.get('title', '').lower()
                    text = doc.get('text', '').lower()
                    abstract = doc.get('abstract', '').lower()
                    references = doc.get('references', [])
                    # Check if any word in the query is in the title, abstract or first 100 chars of text
                    if any(word.lower() in title for word in words) or \
                        any(word.lower() in abstract[:200] for word in words) or \
                        any(word.lower() in text[:500] for word in words) or \
                        any(word.lower() in ref.lower()[:400] for ref in references for word in words):
                        title_matches.append(idx)
        
                # If we found exact matches, prioritize them
                if title_matches:
                    print(f"Found {len(title_matches)} direct matches for '{question}'")
        
            # Get semantic embedding results
            query_embedding = self.embedding_model.encode([question])
            emb_distances, emb_indices = self.index.search(query_embedding.astype('float32'), k*3)
    
            # Get TF-IDF results if available
            tfidf_scores = None
            if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
                try:
                    query_tfidf = self.tfidf_vectorizer.transform([question])
                    if sp.issparse(self.tfidf_matrix):  # Check if matrix is sparse
                        tfidf_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
                    else:
                        tfidf_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix.toarray())[0]
                    # Convert similarities to scores (higher is better)
                    tfidf_scores = tfidf_similarities
                except Exception as e:
                    print(f"Error in TF-IDF processing: {str(e)}")
                    tfidf_scores = None
            
            # Combine scores with hybrid weighting
            combined_scores = {}
            seen_sources = set()
            # For short queries, lower the thresholds to be more inclusive
            if is_short_query:
                emb_score_threshold = 0.5  # Lower threshold for short queries
                tfidf_score_threshold = 0.05
            else:
                emb_score_threshold = 0.7
                tfidf_score_threshold = 0.15   
            
            # Add bonus for direct title matches for short queries
            if is_short_query and title_matches:
                for idx in title_matches:
                    combined_scores[idx] = 2.0  # Give high priority to direct matches
                    doc = self.documents[idx]
                    seen_sources.add(doc['source'])
                
            # Process embedding results with filters
            for idx, distance in zip(emb_indices[0], emb_distances[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    source_filename = doc['source']
                
                    # Apply filters
                    if journal_filter and journal_filter.lower() != "all":
                        if journal_filter.lower() not in doc.get('journal', '').lower():
                            continue  
                    if article_type and article_type.lower() != "all":
                        if article_type.lower() not in doc.get('article_type', '').lower():
                            continue
                    if subject and subject.lower() != "all":
                        subject_terms = subject.lower().split()
                        title_match = any(term in doc.get('title', '').lower() for term in subject_terms)
                        abstract_match = any(term in doc.get('abstract', '').lower() for term in subject_terms)
                        text_match = any(term in doc.get('text', '').lower() for term in subject_terms)
                    if not (title_match or abstract_match or text_match):
                        continue    
                    if min_year and int(doc.get('year', 0)) < min_year:
                        continue   
                    if max_year and int(doc.get('year', 9999)) > max_year:
                        continue
                    # Skip if we've already seen this source document
                    if source_filename in seen_sources:
                        continue
                    # Convert distance to similarity score (0-1)
                    emb_score = 1 - (distance / 4)
                    # For short queries, check if the query exists in the title or text
                    extra_score = 0
                    if is_short_query:
                        title = doc.get('title', '').lower()
                        text = doc.get('text', '').lower()
                        # Bonus for term appearing in title or text
                        for word in words:
                            if word.lower() in title:
                                extra_score += 0.3
                            if word.lower() in text[:500]:  # Check start of text
                                extra_score += 0.2
                    # Initialize with embedding score
                    if emb_score >= emb_score_threshold:  # Threshold for semantic relevance
                        combined_scores[idx] = emb_score * hybrid_weight + extra_score
                        seen_sources.add(source_filename)
            # Add TF-IDF scores if available (with filters)
            if tfidf_scores is not None:
                # Get top TF-IDF scores
                tfidf_top_indices = np.argsort(-tfidf_scores)[:k*3]
                for idx in tfidf_top_indices:
                    if idx < len(self.documents):
                        doc = self.documents[idx]
                        source_filename = doc['source']
                        # Apply filters
                        if year_filter and str(doc.get('year', '')) != year_filter:
                            continue
                        if journal_filter and journal_filter.lower() not in doc.get('journal', '').lower():
                            continue
                        if article_type and article_type.lower() not in doc.get('articles', '').lower():
                            continue
                        if subject and subject.lower() not in doc.get('subject', '').lower():
                            continue
                        if min_year and int(doc.get('year', 0)) < min_year:
                            continue
                        if max_year and int(doc.get('year', 9999)) > max_year:
                            continue
                        # Apply article type filter if specified
                        if article_type and article_type.lower() != "all":
                            if not any(article_type.lower() in doc.get('article_type', '').lower() 
                                for doc in self.documents):
                                    print(f"Warning: No documents found with article type '{article_type}'")
                                    if article_type.lower() not in doc.get('article_type', '').lower():
                                        continue
                                        # Apply subject filter if specified
                                    if subject and subject.lower() != "all":
                                        subject_terms = subject.lower().split()
                                        title_match = any(term in doc.get('title', '').lower() for term in subject_terms)
                                        abstract_match = any(term in doc.get('abstract', '').lower() for term in subject_terms)
                                        text_match = any(term in doc.get('text', '').lower() for term in subject_terms)
                                    if not (title_match or abstract_match or text_match):
                                         continue
                        # Get TF-IDF score
                        tfidf_score = tfidf_scores[idx]
                        if tfidf_score > tfidf_score_threshold:  # Threshold for keyword relevance
                            if idx in combined_scores:
                                # Add weighted TF-IDF score to existing embedding score
                                combined_scores[idx] += tfidf_score * (1 - hybrid_weight)
                            else:
                                # Only include if source hasn't been seen yet
                                if source_filename not in seen_sources:
                                    combined_scores[idx] = tfidf_score * (1 - hybrid_weight)
                                    seen_sources.add(source_filename)             
            # Sort by combined score
            sorted_indices = sorted(combined_scores.keys(), key=lambda idx: combined_scores[idx], reverse=True)
            # For logging/debugging
            if is_short_query:
                print(f"Short query '{question}' - found {len(sorted_indices)} results with adjusted thresholds")
            
            # Prepare results
            results = []
            seen_sources = set()  # Reset for final collection
            for idx in sorted_indices[:k*2]:
                doc = self.documents[idx]
                source_filename = doc['source']
                # Skip if we've already seen this source document in final results
                if source_filename in seen_sources:
                    continue
                # Get the combined score
                score = combined_scores[idx]
                # Use fuzzy matching to find the URL
                url = self.get_best_url_match(doc.get('title', '')) or self.get_best_url_match(source_filename)
                if not url:
                    url = source_filename  # Default to filename if no URL found
                result = {
                    'text': doc['text'],
                    'source': url,  # Will be URL if found, otherwise filename
                    'score': score,
                    'references': doc.get('references', []) 
                }
                print(f"📎 DEBUG: Final result for {doc.get('title', '')[:30]}... has {len(result['references'])} references.")
                print("🔍 Sample reference:", result['references'][0] if result['references'] else "None")
                # Ensure SRID is properly transferred
                srid_value = doc.get('SRID', '')
                if not srid_value:
                    # Generate a fallback SRID if missing
                    paper_id = abs(hash(source_filename)) % 100000000
                    srid_value = f"SRID{paper_id:08d}"
                # Include all metadata fields
                for key in ['title', 'authors', 'year', 'journal', 'volume', 
                            'issue', 'pages', 'doi', 'abstract']:
                    result[key] = doc.get(key, '')
        
                # Set SRID separately to ensure it's not empty
                result['SRID'] = srid_value
                # Ensure URL is set properly
                result['url'] = url
                results.append(result)
                seen_sources.add(source_filename)
                if len(results) >= k:
                    break
                
            # Cache the results
            self.query_cache[cache_key] = results
            # Log the number of results for debugging
            print(f"Query: '{question}' returned {len(results)} results")
            return results
        except Exception as e:
            print(f"Error in hybrid query_papers: {str(e)}")
            return self.get_dummy_results(question, k)
    def _get_contextual_terms(self, word: str, window=5):
        """Analyze terms that frequently appear together"""
        # Implement co-occurrence analysis
        pass
    def _auto_adjust_thresholds(self):
        """Automatically adjust matching thresholds based on performance"""
        # Could analyze correction accuracy over time
        pass
    def show_term_variants(self, stem: str):
        """Display discovered variants for a term"""
        variants = self.term_variants.get(stem, set())
        print(f"Variants for '{stem}':")
        for v in sorted(variants, key=lambda x: -self.term_frequencies.get(x, 0)):
            print(f"- {v} ({self.term_frequencies.get(v, 0)} uses)")
    def get_dummy_results(self, question: str, k: int = 3) -> List[Dict]:
        """Get dummy results when real search isn't available"""
        # This is a new helper method to return something when index fails
        if not hasattr(self, 'documents') or not self.documents:
            self.create_dummy_data()
    
        # Just return available documents as results (limited to k)
        results = []
        for doc in self.documents[:k]:
            result = doc.copy()
            result['score'] = 0.9  # High dummy score
            results.append(result)
    
        return results
    def should_reprocess(self, s3_folder: str) -> bool:
        """Check if new files exist that aren't in cache"""
        if not os.path.exists(DOCUMENTS_CACHE_FILE):
            return True
            
        try:
            # Get list of files in S3
            response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=s3_folder)
            s3_files = {obj['Key'] for obj in response.get('Contents', []) 
                        if obj['Key'].endswith('.pdf')}
            
            # Get list of processed files from cache
            with open(DOCUMENTS_CACHE_FILE, 'rb') as f:
                cached_docs = pickle.load(f)
            processed_files = {doc['source'] for doc in cached_docs}
            
            # Check if any S3 files aren't in processed files
            return len(s3_files - processed_files) > 0
        except Exception as e:
            print(f"Error checking for reprocess: {e}")
            return True
def generate_answer(self, question: str, context: List[str]) -> str:
    """Generate answer using OpenAI with retrieved context and improved relevance check"""
    filtered_context = []
    for ctx in context:
        # Less aggressive filtering for short queries
        query_terms = set(term.lower() for term in question.split() if len(term) > 2)
        if any(term in ctx.lower() for term in query_terms):
            filtered_context.append(ctx[:2000])
        """# Skip if more than 30% of the context looks like references
        if not self._is_mostly_references(ctx):
            # Additional check for question terms in context
            if any(term.lower() in ctx.lower() for term in question.split() if len(term) > 3):
                filtered_context.append(ctx)
        """
    if not filtered_context:
        return "No relevant content found in the research papers."

    try:
        context_str = "\n\n".join(f"Excerpt {i+1}: {ctx}" for i, ctx in enumerate(filtered_context[:3]))
        system_prompt = f"""You are a research assistant. Provide a detailed summary about "{question}" based on these research excerpts:
        - Focus specifically on information related to {question}
        - Include only information found in the provided context
        - Format your response in clear paragraphs"""
        
        prompt = f"""Research context:
{context_str}

Question: {question}
Provide a detailed summary:"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content or "No summary could be generated."
        
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        return "An error occurred while generating the research summary."
PaperProcessor.generate_answer = generate_answer
def cleanup(self):
    """Clean up and save resources on shutdown"""
    self.save_cache()
    print("Resources saved to cache")

PaperProcessor.cleanup = cleanup
def suggest_next_questions(self, current_query: str, num_suggestions=3) -> List[str]:
    """Generate clear, short follow-up questions based on the current query and context papers."""
    if not self.documents:
        return []

    try:
        # Get the most relevant papers
        relevant_papers = self.query_papers(current_query, k=3)
        if not relevant_papers:
            return []

        # Create a concise summary of relevant topics
        related_titles = [paper.get("title", "")[:100] for paper in relevant_papers if paper.get("title")]
        context_summary = "\n".join(related_titles[:3])

        # New system prompt with better instructions
        system_prompt = (
            "You are an AI research assistant. Based on the provided topic and paper titles, "
            "generate exactly 3 follow-up research questions. "
            "Each question must be short (max 10 words), clear, and end with a question mark. "
            "Avoid repeating words or phrases. "
            "Return each question on a new line, numbered (1., 2., 3.)."
        )


        user_prompt = f"""Original query: "{current_query}"
Related paper titles:
{context_summary}

Generate 3 follow-up questions:"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )

        # Parse LLM response
        suggestions = []
        if response.choices[0].message.content:
            lines = response.choices[0].message.content.strip().split("\n")
            for line in lines:
                question = re.sub(r"^\d+\.\s*", "", line).strip()
                if question and question.endswith("?") and 5 <= len(question.split()) <= 16:
                    suggestions.append(question)
        # Fallback if no good questions
        if not suggestions:
            return [
                f"New treatments for {current_query}?",
                f"How is {current_query} diagnosed?",
                f"Research updates on {current_query}?"
            ][:num_suggestions]

        return suggestions[:num_suggestions]

    except Exception as e:
        print(f"Error generating suggestions: {e}")
        return [
            f"New treatments for {current_query}?",
            f"How is {current_query} diagnosed?",
            f"Research updates on {current_query}?"
        ][:num_suggestions]
PaperProcessor.suggest_next_questions = suggest_next_questions
