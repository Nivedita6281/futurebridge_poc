import streamlit as st
import time
from process_papers import PaperProcessor
from typing import Dict, List
from fastapi import HTTPException
from datetime import datetime
import re
# Page config
st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for Nature-like design
st.markdown("""
<style>
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Main paper layout */
    .paper-header {
        padding: 2rem 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 2rem;
    }
    
    .paper-title {
        font-size: 1.1rem;
        font-weight: 700;
        line-height: 1.2;
        color: #111827;
        margin-bottom: 1.5rem;
    }
    
    .paper-authors {
        font-size: 1rem;
        color: #0066cc;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    
    .paper-metadata {
        display: flex;
        align-items: center;
        gap: 1rem;
        font-size: 0.875rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    
    .journal-name {
        font-weight: 600;
        color: #dc2626;
    }
    
    .paper-abstract {
        background-color: #f8fafc;
        padding: 1.5rem;
        border-left: 4px solid #0066cc;
        border-radius: 0 8px 8px 0;
        margin-bottom: 2rem;
    }
    
    .abstract-title {
        font-weight: 700;
        font-size: 1.1rem;
        color: #111827;
        margin-bottom: 1rem;
    }
    
    .abstract-content {
        line-height: 1.7;
        color: #374151;
        font-size: 0.95rem;
    }
    
    /* Content sections */
    .content-section {
        margin-bottom: 2.5rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid #f3f4f6;
    }
    
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0066cc;
        display: inline-block;
    }
    
    .section-content {
    line-height: 1.7;
    color: #374151;
    font-size: 0.95rem;
    white-space: pre-wrap;       /* Preserve line breaks and wrap text */
    word-break: break-word;      /* Break long words if needed */
}

    .highlight {
            background-color: transparent;
            padding: 0.1rem 0.2rem;
            border-radius: 3px;
    }
    /* Answer container */
    .answer-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    
    .answer-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .answer-content {
        line-height: 1.7;
        font-size: 1rem;
    }
    
    .reference-item {
        margin-bottom: 0.75rem;
        padding: 0.5rem;
        background-color: #f8fafc;
        border-radius: 4px;
        font-size: 0.85rem;
        line-height: 1.4;
        page-break-inside: avoid; /* Prevent references from breaking across pages */
    }

    .reference-number {
        font-weight: 600;
        color: #0066cc;
        margin-right: 0.5rem;
    }
    .reference-nav {
        max-height: 400px;
        overflow-y: auto;
        column-count: 1; /* Single column for better readability */
        column-gap: 1rem;
    }

</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
    st.session_state.initialization_status = {
        "status": "not_started", 
        "message": "Initialization not started",
        "documents_loaded": 0,
        "index_ready": False
    }
# Initialize query state properly
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'pending_query' not in st.session_state:
    st.session_state.pending_query = ""

def initialize_processor():
    """Initialize the paper processor"""
    try:
        st.session_state.processor = PaperProcessor()
        st.session_state.processor.process_s3_folder("research_papers/")
        st.success("Processor initialized successfully!")
        print("Processor initialized - documents loaded:", len(st.session_state.processor.documents))
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        raise
    
def ask_question(question: str, journal_filter=None, article_type=None,subject = None, min_year=None, max_year=None):
    """Ask a question to the research assistant with filters"""
    processor = st.session_state.processor
    
    # Check if processor is ready
    if processor is None:
        status = st.session_state.initialization_status["status"]
        message = st.session_state.initialization_status["message"]
        
        if status == "failed":
            raise HTTPException(
                status_code=500,
                detail=f"Research paper database initialization failed: {message}"
            )
        else:
            raise HTTPException(
                status_code=503,
                detail="Research paper database is still initializing. Please try again in a moment."
            )
    
    question = question.strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    words = question.split()
    is_short_query = len(words) <= 2
    
    try:
        k = 5 if is_short_query else 3
        results = processor.query_papers(
            question, 
            k,
            journal_filter=journal_filter,
            article_type=article_type,
            subject=subject,
            min_year=min_year,
            max_year=max_year
        )
        
        # For very short queries, analyze result relevance
        relevant_results = []
        if is_short_query:
            query_terms = set(question.lower().split())
            for result in results:
                text = result.get('text', '').lower()
                title = result.get('title', '').lower()
                abstract = result.get('abstract', '').lower()
                
                # Check if any query term appears in the document
                if any(term in text for term in query_terms) or \
                   any(term in title for term in query_terms) or \
                   any(term in abstract for term in query_terms):
                    relevant_results.append(result)
                    
            # If we found relevant results, use only those
            if relevant_results:
                results = relevant_results
                
        context = [r['text'] for r in results]
        answer = processor.generate_answer(question, context)
        
        # Prepare response
        response = {
            "answer": answer,
            "papers": results[:1],  # First result is primary paper
            "similar_papers": results[1:]  # Remaining are similar papers
        }
            
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )

def highlight_query_terms(text: str, query: str) -> str:
    """Highlight query terms in text"""
    if not text or not query:
        return text
    
    for word in query.split():
        if len(word) > 2:  # Only highlight words longer than 3 characters
            text = text.replace(word, f'<span class="highlight">{word}</span>')
            text = text.replace(word.title(), f'<span class="highlight">{word.title()}</span>')
    return text
def clean_text_content(text:str) -> str:
    """Clean and format the text content for better display"""
    # Fix missing spaces between words (insert space before capital letters after lowercase)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Fix spacing around references (e.g., "stigma10,12" → "stigma [10,12]")
    text = re.sub(r'(\D)(\d{1,3})(,\d{1,3})*', r'\1[\2\3]', text)
    
    # Fix hyphenated ranges (e.g., "10–13" → "10–13.")
    text = re.sub(r'(\d+-\d+)\.', r'\1', text)
    
    # Fix URL formatting
    #text = re.sub(r'(https?://\S+)', r'<a href="\1" target="_blank">\1</a>', text)
    return text
def display_nature_style_paper_fixed(paper: Dict, query: str = ""):
    """Fixed version that handles different data structures"""
    # Paper header with fallbacks for all fields
    title = paper.get('title', 'Untitled Paper')
    authors = paper.get('authors', 'Unknown Authors') or 'Unknown Authors'
    journal = paper.get('journal', 'Unknown Journal') or 'Unknown Journal'
    year = paper.get('year', 'Unknown Year') or 'Unknown Year'
    doi = paper.get('doi', '')
    url = paper.get('url', '')
    references = paper.get('references', [])
    if not isinstance(references, list):
        references = []
    st.markdown(f"""
    <div class="paper-header">
        <h2 class="paper-title">{title}</h2>
        <div class="paper-authors">{authors}</div>
        <div class="paper-metadata">
            <span class="journal-name">{journal}</span>
            <span>•</span>
            <span>{year}</span>
            {f'<span>•</span><span>DOI: {doi}</span>' if doi else ''}
            {f'<span>•</span><span><a href="{url}" target="_blank">View Article</a></span>' if url else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Abstract with fallback
    abstract = paper.get('abstract', '')
    if not abstract and 'text' in paper:
        abstract = paper['text'][:200] + "..." if len(paper['text']) > 500 else paper['text']
        
    if abstract:
        st.markdown(f"""
        <div class="paper-abstract">
            <div class="abstract-title">Abstract</div>
            <div class="abstract-content">{highlight_query_terms(abstract, query)}</div>
        </div>
        """, unsafe_allow_html=True)
    # Main content display - handle multiple possible formats
    content_to_display = []
    
    # Case 1: Check if we have actual content sections
    if 'sections' in paper and paper['sections']:
        for section in paper['sections']:
            if section.get('content'):
                # Skip sections that are mostly references
                if not st.session_state.processor._is_mostly_references(section['content']):
                    content_to_display.append((
                        section.get('heading', 'Section'), 
                        section['content']
                    ))
   # Case 2: Check text field for actual content (not just references)
    if not content_to_display and 'text' in paper and paper['text']:
        text = paper['text']
        print("🔍 DEBUG: Content preview:", text[:500])
        print("🔍 DEBUG: Mostly references?", st.session_state.processor._is_mostly_references(text))
        # Check if text is mostly references
        if not st.session_state.processor._is_mostly_references(text):
            cleaned_text = clean_text_content(text)
            content_to_display.append(('Content', cleaned_text))
    
    # Case 3: Fallback with first 500 characters (excluding references)
    if not content_to_display:
        fallback_text = paper.get('abstract', '') or paper.get('text', '')[:1000]
        if fallback_text:
            content_to_display.append(('Content', clean_text_content(fallback_text)))
    # ultimate fallback
    # Case 4: Ultimate fallback
    if not content_to_display:
        content_to_display.append(('Content', 
            "Full text content could not be extracted. Please view the original paper for details."))
    # Display all found content 
    if content_to_display:
        for heading, content in content_to_display:
            st.markdown(f"""
            <div class="paper-abstract">
                <div class="abstract-title">{heading}</div>
                <div class="abstract-content">{highlight_query_terms(content, query)}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No content available to display")
       
    # References section
    if content_to_display and paper.get('references'):
        references = paper.get('references', [])
        if references:
            # Clean up references by ensuring each one is on its own line
            cleaned_references = []
            for ref in references:
                # Split references that are concatenated together
                # This handles cases where multiple references appear in one string
                # Split aggressively on numbered references
                split_refs = re.split(r'\s*(?=\d{1,3}\.\s+[A-Z])', ref) if isinstance(ref, str) else []
                for r in split_refs:
                    clean_ref = re.sub(r'^\d+\.\s*', '', r).strip()
                    if clean_ref and len(clean_ref) > 20:
                        cleaned_references.append(clean_ref)
            st.markdown(f"""
            <div class="paper-abstract">
                <div class="abstract-title">References</div>
                <div class="abstract-content">
                    {''.join([
                        f'<div class="reference-item"><span class="reference-number">{i+1}.</span> {highlight_query_terms(ref, query)}</div>'
                        for i, ref in enumerate(cleaned_references[:100])
                    ]) if cleaned_references else '<div class="reference-item">No references were extracted.</div>'}
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    st.title("📚 Research Paper Assistant")
    st.markdown("""
    This assistant helps you find relevant research papers and get summaries of their content. 
    Ask a question about any medical or scientific topic to get started.
    """)
    # if 'last_query' in st.session_state:
    #     st.session_state.last_query = 
    # if 'query' in st.session_state:
    #     st.session_state.query = st.session_state.get('query', "")
    
    # Initialize session state
    if 'query' not in st.session_state:
        st.session_state.query = ""
        st.session_state.last_query = ""
        st.session_state.results = None
        st.session_state.filters = {
            'journal': "All",
            'article_type': "All",
            'subject': "All",
            'year_range': (2000, datetime.now().year)
        }
        st.session_state.filters_applied = False

    # Initialize processor if not already done
    if st.session_state.processor is None:
        with st.spinner("Initializing research paper database..."):
            initialize_processor()

    # Filter sidebar [Keep the same until line 487]
    with st.sidebar:
        st.subheader("🔍 Filters")
        
        # Journal filter
        journal_options = ["All"] + (
            sorted(list(set(doc.get('journal', '') for doc in st.session_state.processor.documents)))
            if st.session_state.processor and hasattr(st.session_state.processor, 'documents')
            else []
        )
        selected_journal = st.selectbox(
            "Journal", 
            journal_options,
            index=journal_options.index(st.session_state.filters['journal'])
        )
        
        # Article type filter
        article_types = ["All", "Research Article", "Review", "Case Study", "Clinical Trial"]
        selected_type = st.selectbox(
            "Article type", 
            article_types,
            index=article_types.index(st.session_state.filters['article_type'])
        )
        
        # Subject filter
        subjects = ["All", "Diabetes", "Cancer", "Cardiology", "Neurology", "Immunology","Geology","Geography"]
        selected_subject = st.selectbox(
            "Subject", 
            subjects,
            index=subjects.index(st.session_state.filters['subject'])
        )
        
        # Date filter
        min_year = 2000
        max_year = datetime.now().year
        selected_years = st.slider(
            "Publication Year",
            min_value=min_year,
            max_value=max_year,
            value=st.session_state.filters['year_range']
        )
        
        # Create columns for buttons
        col1, col2 = st.columns(2)
    
        with col1:
            if st.button("Clear all filters"):
                st.session_state.filters = {
                    'journal': "All",
                    'article_type': "All",
                    'subject': "All",
                    'year_range': (2000, datetime.now().year)
                }
                st.session_state.filters_applied = True
                # Reset the query to trigger a new search
                if st.session_state.query:
                    st.session_state.last_query = st.session_state.query
                    st.session_state.results = None
                st.rerun()
        with col2:
            apply_clicked = st.button("Apply Filters")
            if apply_clicked:
                current_filters = st.session_state.filters  # Define current_filters here
                new_filters = {
                    'journal': selected_journal,
                    'article_type': selected_type,
                    'subject': selected_subject,
                    'year_range': selected_years
                }
                if current_filters != new_filters:
                    st.session_state.filters = new_filters
                    st.session_state.filters_applied = True
                    st.rerun()
            # Show filter status outside the button blocks
            if st.session_state.filters_applied:
                current_filters = []  # This is a different variable than the one above
                if st.session_state.filters['journal'] != "All":
                    current_filters.append(f"Journal: {st.session_state.filters['journal']}")
                if st.session_state.filters['article_type'] != "All":
                    current_filters.append(f"Type: {st.session_state.filters['article_type']}")
                if st.session_state.filters['subject'] != "All":
                    current_filters.append(f"Subject: {st.session_state.filters['subject']}")
                if st.session_state.filters['year_range'] != (2000, datetime.now().year):
                    current_filters.append(f"Years: {st.session_state.filters['year_range'][0]}-{st.session_state.filters['year_range'][1]}") 
                if current_filters:
                    st.sidebar.success("Filters applied: " + ", ".join(current_filters))
                else:
                    st.sidebar.success("All filters cleared!")
                st.session_state.filters_applied = False
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.markdown("""
        **Research Paper Assistant** uses advanced AI to search and analyze thousands of research papers.
        
        **Features:**
        - 🔍 Semantic search with keyword matching
        - 🤖 AI-powered summarization  
        - 📄 Full paper content display
        - 📑 Organized sections navigation
        - 📚 Complete reference lists
        - 🎯 Advanced filtering options
        """)

    # Main content area
    query = st.text_input(
        "Ask a question about scientific research:", 
        placeholder="e.g. What are the latest treatments for Alzheimer's disease?",
        value=st.session_state.query,
        key="search_input"
        
    )
    if query != st.session_state.query:
        st.session_state.query = query
        st.rerun()
    # Process query when it changes
    if query:
        st.session_state.query = query
        # st.session_state.last_query = query
        
        with st.spinner(f"🔍 Searching research papers for: {query}..."):
            try:
                # Apply filters
                filters = {
                    'journal_filter': selected_journal if selected_journal != "All" else None,
                    'article_type': selected_type if selected_type != "All" else None,
                    'subject': selected_subject if selected_subject != "All" else None,
                    'min_year': selected_years[0],
                    'max_year': selected_years[1]
                }
                
                response = ask_question(query, **filters)
                st.session_state.results = response
                
            except HTTPException as e:
                st.error(f"❌ Error: {e.detail}")
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {str(e)}")
                st.session_state.results = None

    # Display results if available
    if st.session_state.results:
        response = st.session_state.results
        
        # Display AI summary
        st.markdown(f"""
        <div class="answer-container">
            <div class="answer-title">🤖 Summary</div>
            <div class="answer-content">{response['answer'].replace(chr(10), '<br>')}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display primary paper
        if response['papers']:
            st.markdown("---")
            st.subheader("📄 Most Relevant Paper")
            display_nature_style_paper_fixed(response['papers'][0], st.session_state.query)
        
        # Display additional papers
        if response['similar_papers']:
            st.markdown("---")
            st.subheader(f"📚 Additional Relevant Papers ({len(response['similar_papers'])})")
            for i, paper in enumerate(response['similar_papers']):
                with st.expander(f"📄 {paper.get('title', 'Untitled')[:80]}...", expanded=False):
                    display_nature_style_paper_fixed(paper, st.session_state.query)
        
        # Suggest next questions
        if response and response['answer']:
            next_questions = st.session_state.processor.suggest_next_questions(st.session_state.query)
            if next_questions:
                st.markdown("---")
                st.subheader("💡 You Might Also Ask")
                cols = st.columns(min(3, len(next_questions)))
                for i, question in enumerate(next_questions):
                    with cols[i % len(cols)]:
                        clean_question = re.sub(r'^\d+\.\s*', '', question).strip()
                        if st.button(
                            clean_question if clean_question.endswith("?") else f"{clean_question}?",
                            key=f"nextq_{i}",
                            help=f"Search for: {clean_question}",
                            on_click=lambda q=clean_question: st.session_state.update({
                                'query': q,
                                'last_query': q,
                                'results': None
                            }),
                            use_container_width=True
                        ):
                            # st.session_state.last_query = st.session_state.query
                            st.session_state.query = clean_question
                            
                            # st.session_state.results = None
                            st.rerun()

if __name__ == "__main__":
    main()