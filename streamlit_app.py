import streamlit as st
import os
import json
import shutil
from datetime import datetime
from pathlib import Path
import base64
from PIL import Image
import io
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create necessary directories on startup
def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        "uploads",
        "extracted_images"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

# Setup directories
setup_directories()

def cleanup_old_databases():
    """Clean up any old database directories that were renamed during previous cleanup attempts"""
    try:
        import glob
        import shutil
        
        # Find any old database directories
        old_patterns = [
            "./old_chroma_*",
            "./chroma_marked_for_deletion_*"
        ]
        
        for pattern in old_patterns:
            for old_dir in glob.glob(pattern):
                try:
                    if os.path.isdir(old_dir):
                        shutil.rmtree(old_dir)
                        print(f"Cleaned up old database: {old_dir}")
                except Exception as e:
                    print(f"Could not clean up {old_dir}: {e}")
                    
    except Exception as e:
        print(f"Error during old database cleanup: {e}")

# Cleanup old databases on startup
cleanup_old_databases()

# Import our RAG components
try:
    from basic import RAGPipeline, DocumentProcessor, ImageAnalyzer
except ImportError as e:
    st.error(f"Error importing RAG components: {e}")
    st.error("Make sure basic.py is in the same directory as this app.")
    st.stop()

# Page config
st.set_page_config(
    page_title="Responsive RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apple-style CSS
def load_css():
    # CSS for AutoGen-style chat interface
    st.markdown("""
    <style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Prevent sidebar from collapsing */
    section[data-testid="stSidebar"] {
        min-width: 300px !important;
        width: 300px !important;
    }
    
    /* Hide sidebar collapse button */
    button[kind="header"] {
        display: none !important;
    }
    
    /* Main content styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* Input field styling */
    .stTextInput input {
        border-radius: 25px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 12px 20px !important;
        font-size: 16px !important;
        background-color: #f8f9fa !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput input:focus {
        border-color: #ff4444 !important;
        box-shadow: 0 0 0 2px rgba(255, 68, 68, 0.1) !important;
        background-color: white !important;
    }
    
    /* Submit button styling */
    .stButton button[kind="primary"] {
        background-color: #ff4444 !important;
        border: none !important;
        border-radius: 20px !important;
        width: 100% !important;
        height: 48px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button[kind="primary"]:hover {
        background-color: #e63939 !important;
        transform: translateY(-1px) !important;
    }
    
    /* Clear chat button */
    .stButton button[kind="secondary"] {
        border-radius: 8px !important;
        border: 1px solid #ddd !important;
        background-color: white !important;
        color: #666 !important;
        padding: 8px 16px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button[kind="secondary"]:hover {
        background-color: #f5f5f5 !important;
        border-color: #ccc !important;
    }
    
    /* Form styling */
    .stForm {
        border: none !important;
        background: none !important;
    }
    
    /* Remove default margins */
    .stMarkdown h1 {
        margin-top: 0 !important;
        color: #333 !important;
        font-weight: 600 !important;
    }
    
    /* Improve overall spacing */
    .element-container {
        margin-bottom: 1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'vectordb_created' not in st.session_state:
        st.session_state.vectordb_created = False
    if 'processing' not in st.session_state:
        st.session_state.processing = False

# File management functions
def get_uploads_dir():
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    return uploads_dir



def save_uploaded_file(uploaded_file):
    """Save uploaded file to uploads directory"""
    uploads_dir = get_uploads_dir()
    file_path = uploads_dir / uploaded_file.name
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)

def get_existing_files():
    """Get list of existing files in uploads directory"""
    uploads_dir = get_uploads_dir()
    if not uploads_dir.exists():
        return []
    
    files = []
    for file_path in uploads_dir.iterdir():
        if file_path.is_file() and not file_path.name.startswith('.'):
            files.append({
                'name': file_path.name,
                'path': str(file_path),
                'size': file_path.stat().st_size,
                'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
            })
    
    return sorted(files, key=lambda x: x['modified'], reverse=True)

def delete_file(file_path):
    """Delete a file from uploads directory"""
    try:
        os.remove(file_path)
        return True
    except Exception as e:
        st.error(f"Error deleting file: {e}")
        return False

def cleanup_locked_database():
    """Clean up locked database files with enhanced Windows handling"""
    try:
        # Step 1: Properly close all ChromaDB connections
        if hasattr(st.session_state, 'rag_pipeline') and st.session_state.rag_pipeline:
            if hasattr(st.session_state.rag_pipeline, 'vectorstore') and st.session_state.rag_pipeline.vectorstore:
                try:
                    # Enhanced ChromaDB connection cleanup
                    vectorstore = st.session_state.rag_pipeline.vectorstore
                    
                    # Close client connection if it exists
                    if hasattr(vectorstore, '_client') and vectorstore._client:
                        try:
                            vectorstore._client.reset()
                            vectorstore._client = None
                        except:
                            pass
                    
                    # Close collection if it exists
                    if hasattr(vectorstore, '_collection') and vectorstore._collection:
                        try:
                            vectorstore._collection = None
                        except:
                            pass
                    
                    # Try to call delete_collection if available
                    try:
                        if hasattr(vectorstore, 'delete_collection'):
                            vectorstore.delete_collection()
                    except:
                        pass
                        
                except Exception as e:
                    print(f"Error closing vectorstore: {e}")
                
                # Clear the vectorstore reference
                st.session_state.rag_pipeline.vectorstore = None
            
            # Clear retriever
            if hasattr(st.session_state.rag_pipeline, 'retriever'):
                st.session_state.rag_pipeline.retriever = None
        
        # Step 2: Clear all session state references
        st.session_state.rag_pipeline = None
        
        # Step 3: Force garbage collection aggressively
        import gc
        for _ in range(5):  # More aggressive GC
            gc.collect()
        
        # Step 4: Close any remaining ChromaDB processes (Windows)
        import subprocess
        try:
            # Kill any ChromaDB processes
            subprocess.run(['taskkill', '/f', '/im', 'chroma.exe'], 
                         capture_output=True, check=False)
            subprocess.run(['taskkill', '/f', '/im', 'python.exe', '/fi', 'WINDOWTITLE eq *chroma*'], 
                         capture_output=True, check=False)
        except:
            pass
        
        # Step 5: Wait for processes to release files
        import time
        time.sleep(2)  # Give time for processes to clean up
        
        # Step 6: Enhanced file deletion with Windows-specific strategies
        for attempt in range(10):  # More attempts for Windows
            try:
                if os.path.exists("./streamlit_chroma_db"):
                    if attempt < 2:
                        # First attempts: try direct removal
                        shutil.rmtree("./streamlit_chroma_db")
                    elif attempt < 5:
                        # Middle attempts: try with file attribute changes
                        for root, dirs, files in os.walk("./streamlit_chroma_db", topdown=False):
                            for file in files:
                                file_path = os.path.join(root, file)
                                try:
                                    # Remove read-only attribute and take ownership
                                    os.chmod(file_path, 0o777)
                                    # Force close any file handles (Windows specific)
                                    import subprocess
                                    subprocess.run(['handle', '-p', str(os.getpid()), '-c', file_path], 
                                                 capture_output=True, check=False)
                                    os.remove(file_path)
                                except Exception as e:
                                    print(f"Could not delete file {file_path}: {e}")
                                    continue
                            for dir_name in dirs:
                                dir_path = os.path.join(root, dir_name)
                                try:
                                    os.rmdir(dir_path)
                                except:
                                    continue
                        # Try to remove the root directory
                        try:
                            os.rmdir("./streamlit_chroma_db")
                        except:
                            pass
                    else:
                        # Last attempts: use robocopy to delete (Windows specific)
                        try:
                            import subprocess
                            # Create empty temp directory
                            temp_empty = "./temp_empty_dir"
                            os.makedirs(temp_empty, exist_ok=True)
                            
                            # Use robocopy to mirror empty directory (effectively deletes)
                            subprocess.run([
                                'robocopy', temp_empty, './streamlit_chroma_db', 
                                '/MIR', '/NP', '/NFL', '/NDL', '/NJH', '/NJS'
                            ], capture_output=True, check=False)
                            
                            # Remove both directories
                            try:
                                os.rmdir(temp_empty)
                                os.rmdir("./streamlit_chroma_db")
                            except:
                                pass
                                
                        except Exception as e:
                            print(f"Robocopy method failed: {e}")
                            # Final fallback: rename directory
                            try:
                                import uuid
                                backup_name = f"./old_chroma_{uuid.uuid4().hex[:8]}"
                                os.rename("./streamlit_chroma_db", backup_name)
                                st.warning(f"Database renamed to {backup_name} - will be cleaned up on next restart")
                            except:
                                pass
                    
                    # Check if deletion was successful
                    if not os.path.exists("./streamlit_chroma_db"):
                        print(f"Successfully deleted streamlit_chroma_db on attempt {attempt + 1}")
                        break
                else:
                    break
                    
            except (PermissionError, OSError) as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                
                # Progressive wait times
                wait_time = min(2 + attempt, 8)  # 2, 3, 4, 5, 6, 7, 8, 8, 8, 8 seconds
                time.sleep(wait_time)
                
                # Force garbage collection between attempts
                import gc
                gc.collect()
                
                if attempt == 9:  # Last attempt
                    # Final fallback: create marker file and rename
                    try:
                        import uuid
                        marker_name = f"./chroma_marked_for_deletion_{uuid.uuid4().hex[:8]}"
                        os.rename("./streamlit_chroma_db", marker_name)
                        st.warning("Database marked for deletion and moved. It will be cleaned up automatically.")
                        return True
                    except:
                        st.error("Cannot delete database - files are locked by another process")
                        st.info("**Solution:** Close all browser tabs, stop Streamlit (Ctrl+C), and restart the app")
                        return False
        
        # Verify deletion
        if os.path.exists("./streamlit_chroma_db") and not os.path.exists("./streamlit_chroma_db/.deleted"):
            st.warning("Database directory still exists but should be ignored")
            
        return True
        
    except Exception as e:
        st.error(f"Error cleaning up database: {e}")
        st.warning("**Solution:** Close all browser tabs, stop Streamlit (Ctrl+C), and restart the app")
        return False

def initialize_rag_pipeline():
    """Initialize RAG pipeline with API keys from environment"""
    try:
        # Get API keys from environment variables
        COHERE_API_KEY = os.getenv("COHERE_API_KEY")
        
        if not COHERE_API_KEY:
            st.error("COHERE_API_KEY not found in environment variables")
            st.info("Please add COHERE_API_KEY to your .env file")
            return None
        
        rag = RAGPipeline(cohere_api_key=COHERE_API_KEY)
        # Override the persist directory for Streamlit to avoid conflicts with basic.py
        rag._persist_directory = "./streamlit_chroma_db"
        return rag
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {e}")
        return None

def create_or_update_vectordb(file_paths):
    """Create or update vector database with current files"""
    try:
        with st.spinner("Processing documents and creating vector database..."):
            if st.session_state.rag_pipeline is None:
                st.session_state.rag_pipeline = initialize_rag_pipeline()
            
            if st.session_state.rag_pipeline is None:
                st.session_state.processing = False
                return False
            
            # Process documents - this creates ./streamlit_chroma_db  
            # We need to override the vectorstore creation to use our custom directory
            from langchain_community.vectorstores import Chroma
            
            # Process documents first to get document chunks
            documents, visual_metadata = st.session_state.rag_pipeline.document_processor.process_documents(file_paths)
            
            if not documents:
                st.error("No documents were processed successfully")
                st.session_state.processing = False
                return False
            
            # Create vector store with custom directory
            st.session_state.rag_pipeline.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=st.session_state.rag_pipeline.embeddings,
                persist_directory="./streamlit_chroma_db"
            )
            
            # Create retriever
            st.session_state.rag_pipeline.retriever = st.session_state.rag_pipeline.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}
            )
            
            num_docs = len(documents)
            
            # Set the flag to show chat interface
            st.session_state.vectordb_created = True
            st.session_state.processing = False
            st.success(f" Vector database created successfully! Processed {num_docs} document chunks.")
            return True
            
    except Exception as e:
        st.session_state.processing = False
        st.error(f"Error creating vector database: {e}")
        return False

def load_existing_vectordb():
    """Load existing vector database if available"""
    try:
        # Check if streamlit_chroma_db exists in current directory
        if os.path.exists("./streamlit_chroma_db"):
            try:
                # Initialize RAG pipeline and load existing vectordb
                if st.session_state.rag_pipeline is None:
                    st.session_state.rag_pipeline = initialize_rag_pipeline()
                
                if st.session_state.rag_pipeline is not None:
                    # Recreate retriever from existing vectordb
                    from langchain_community.vectorstores import Chroma
                    from langchain_openai import AzureOpenAIEmbeddings
                    
                    # Check for required Azure OpenAI environment variables
                    required_env_vars = [
                        "AZURE_EMBEDDING_DEPLOYMENT",
                        "AZURE_EMBEDDING_API_VERSION", 
                        "AZURE_EMBEDDING_API_KEY",
                        "AZURE_EMBEDDING_ENDPOINT"
                    ]
                    
                    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
                    if missing_vars:
                        st.error(f"Missing environment variables: {', '.join(missing_vars)}")
                        st.info("Please add these variables to your .env file")
                        return False
                    
                    embeddings = AzureOpenAIEmbeddings(
                        azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
                        api_version=os.getenv("AZURE_EMBEDDING_API_VERSION"),
                        api_key=os.getenv("AZURE_EMBEDDING_API_KEY"),
                        azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT")
                    )
                    
                    st.session_state.rag_pipeline.vectorstore = Chroma(
                        persist_directory="./streamlit_chroma_db",
                        embedding_function=embeddings
                    )
                    
                    st.session_state.rag_pipeline.retriever = st.session_state.rag_pipeline.vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 50}
                    )
                    
                    st.session_state.vectordb_created = True
                    return True
                    
            except Exception as e:
                # If database is locked, clean it up
                st.warning("Database files may be locked. Cleaning up...")
                cleanup_locked_database()
                st.info("Database cleaned up. Please re-upload your documents.")
                return False
        
        return False
        
    except Exception as e:
        st.error(f"Error loading existing vector database: {e}")
        return False

def query_rag(question):
    """Query the RAG pipeline"""
    try:
        if st.session_state.rag_pipeline is None or not st.session_state.vectordb_created:
            return None
        
        result = st.session_state.rag_pipeline.query(
            question=question,
            fetch_k=50,
            rerank_k=10,
            use_reranker=True,
            include_visual_citation=True
        )
        
        return result
        
    except Exception as e:
        st.error(f"Error querying RAG pipeline: {e}")
        return None

def display_chat_message(message, is_user=False):
    """Display a chat message with normal styling"""
    if is_user:
        st.write(f"**You:** {message}")
    else:
        st.write(f"**Responsive AI:** {message}")

def display_sources(sources):
    """Display source citations"""
    if sources:
        st.subheader("Sources")
        for i, source in enumerate(sources, 1):
            content_type_display = f" [{source['content_type']}]" if source['content_type'] != 'text' else ""
            page_display = f"Page {source['page']}" if source['page'] != "Unknown" else "Unknown page"
            relevance_indicator = "Most Relevant" if i == 1 else "📋 Secondary"
            
            st.write(f"**{relevance_indicator}:** {source['source']} ({page_display}){content_type_display}")

def display_visual_citations(visual_citations):
    """Display visual citations"""
    if visual_citations:
        st.subheader("Visual Citations")
        
        for i, citation in enumerate(visual_citations, 1):
            with st.expander(f"Image {i}: {citation['source']} (Page {citation['page']})"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if citation.get('image_base64'):
                        try:
                            image_data = base64.b64decode(citation['image_base64'])
                            image = Image.open(io.BytesIO(image_data))
                            st.image(image, caption=f"Size: {citation.get('image_size', 'Unknown')}")
                        except Exception as e:
                            st.error(f"Error displaying image: {e}")
                
                with col2:
                    st.write(f"**Description:** {citation['chunk_text']}")
                    st.write(f"**Source:** {citation['source']}")
                    st.write(f"**Page:** {citation['page']}")

def main():
    # Load CSS
    load_css()
    
    # Initialize session state
    init_session_state()
    

    
    # Header
    st.title("Responsive AI")
    st.markdown("**Intelligent Document Analysis with Visual Understanding**")
    
    # Check for required environment variables and show status
    required_vars = {
        "AZURE_OPENAI_ENDPOINT": "Azure OpenAI Endpoint",
        "AZURE_OPENAI_API_KEY": "Azure OpenAI API Key", 
        "AZURE_OPENAI_DEPLOYMENT": "Azure OpenAI Deployment",
        "AZURE_EMBEDDING_ENDPOINT": "Azure Embedding Endpoint",
        "AZURE_EMBEDDING_API_KEY": "Azure Embedding API Key",
        "AZURE_EMBEDDING_DEPLOYMENT": "Azure Embedding Deployment",
        "COHERE_API_KEY": "Cohere API Key"
    }
    
    missing_vars = [name for var, name in required_vars.items() if not os.getenv(var)]
    
    if missing_vars:
        st.error(f"Missing environment variables: {', '.join(missing_vars)}")
        with st.expander("Environment Setup Instructions"):
            st.markdown("""
            **Create a `.env` file in the project root with the following variables:**
            
            ```
            # Azure OpenAI Embeddings Configuration  
            AZURE_EMBEDDING_ENDPOINT=https://your-embedding-endpoint.openai.azure.com/
            AZURE_EMBEDDING_API_KEY=your-embedding-api-key
            AZURE_EMBEDDING_DEPLOYMENT=your-embedding-deployment-name
            AZURE_EMBEDDING_API_VERSION=2024-10-01-preview

            # Cohere API Configuration (for reranking)
            COHERE_API_KEY=your-cohere-api-key
            ```
            
            **Then restart the Streamlit app.**
            """)
    else:
        st.success("All environment variables configured")
    
    st.markdown("---")
    
    # Sidebar for file management
    with st.sidebar:
        st.markdown("## Document Management")
        
        # Load existing vectordb on startup - but only if database exists and isn't marked for deletion
        if (not st.session_state.vectordb_created and 
            os.path.exists("./streamlit_chroma_db") and 
            not os.path.exists("./streamlit_chroma_db/.deleted")):
            if load_existing_vectordb():
                st.success("Loaded existing vector database")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx', 'pptx', 'xlsx', 'csv', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, PPTX, XLSX, CSV, TXT"
        )
        
        # Process uploaded files
        if uploaded_files:
            new_files = []
            for uploaded_file in uploaded_files:
                file_path = save_uploaded_file(uploaded_file)
                new_files.append(file_path)
            
            # Disable button while processing
            if st.button("Process & Update Database", type="primary", disabled=st.session_state.processing):
                st.session_state.processing = True
                # Only process the newly uploaded files, not all files in uploads directory
                create_or_update_vectordb(new_files)
                st.session_state.processing = False
        

        
        # Database status
        st.subheader("Database Status")
        if st.session_state.vectordb_created:
            st.success("Vector Database Ready")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear Database", help="Clear the vector database"):
                    try:
                        # Use the robust cleanup function
                        cleanup_locked_database()
                        
                        # Reset session state
                        st.session_state.vectordb_created = False
                        st.session_state.rag_pipeline = None
                        st.session_state.chat_history = []
                        
                        st.success("Database cleared successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing database: {e}")
            
            with col2:
                if st.button("Force Reset", help="Force reset everything"):
                    # Clear all session state
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.success("App reset! Please refresh the page.")
                    st.rerun()
        else:
            st.error("No Vector Database")
    
    # Main content area
    if st.session_state.vectordb_created:
        
        # Chat messages
        if st.session_state.chat_history:
            for i, chat in enumerate(st.session_state.chat_history):
                # User message on the right with red circle
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-end; margin: 20px 0;">
                    <div style="display: flex; align-items: flex-start; max-width: 70%;">
                        <div style="background-color: #ff4444; color: white; padding: 12px 18px; border-radius: 18px 18px 4px 18px; font-size: 16px; line-height: 1.4; margin-right: 8px;">
                            {chat['question']}
                        </div>
                        <div style="width: 12px; height: 12px; background-color: #ff4444; border-radius: 50%; margin-top: 4px; flex-shrink: 0;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # AI response on the left with yellow/orange circle
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-start; margin: 20px 0;">
                    <div style="display: flex; align-items: flex-start; max-width: 70%;">
                        <div style="width: 12px; height: 12px; background-color: #ffa500; border-radius: 50%; margin-right: 8px; margin-top: 4px; flex-shrink: 0;"></div>
                        <div style="background-color: #f8f9fa; padding: 12px 18px; border-radius: 18px 18px 18px 4px; font-size: 16px; line-height: 1.4; color: #333; border: 1px solid #e9ecef;">
                            {chat['answer']}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Visual citations only (no sources)
                if chat.get('visual_citations'):
                    st.markdown('<div style="margin-left: 20px; margin-bottom: 20px;">', unsafe_allow_html=True)
                    display_visual_citations(chat['visual_citations'])
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Add some spacing between conversations
                st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
        
        else:
            st.markdown("""
            <div style="text-align: center; color: #888; font-style: italic; margin: 60px 0; font-size: 16px;">
                Start a conversation by typing a message below
            </div>
            """, unsafe_allow_html=True)
        
        # Spacer
        st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)
        
        # Input form with better styling
        with st.form("query_form", clear_on_submit=True):
            col1, col2 = st.columns([0.92, 0.08])
            
            with col1:
                query = st.text_input(
                    "",
                    placeholder="Type something...",
                    label_visibility="collapsed",
                    key="chat_input"
                )
            
            with col2:
                submit_button = st.form_submit_button("➤", type="primary")
        
        # Clear chat button
        if st.button("Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Process query
        if submit_button and query.strip():
            with st.spinner("Thinking..."):
                result = query_rag(query.strip())
                
                if result:
                    # Add to chat history
                    chat_entry = {
                        'question': query.strip(),
                        'answer': result['answer'],
                        'sources': result.get('sources', []),
                        'visual_citations': result.get('visual_citations', []),
                        'timestamp': datetime.now()
                    }
                    st.session_state.chat_history.append(chat_entry)
                    st.rerun()
                else:
                    st.error("Failed to get response. Please try again.")
    
    else:
        # Welcome screen
        st.header("Get Started")
        st.write("Welcome to Responsive AI! Upload your documents to begin intelligent analysis.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Quick Setup")
            st.markdown("""
            1. **Upload Documents**: Use the sidebar to upload PDF, DOCX, PPTX, XLSX, CSV, or TXT files
            2. **Process Database**: Click "Process & Update Database" to create your vector database  
            3. **Start Chatting**: Ask questions about your documents and get intelligent answers with citations
            """)
            
            st.subheader("Example Questions")
            st.markdown("""
            - "What are the main topics discussed in these documents?"
            - "Show me any charts or images related to financial data"
            - "What information is available about project timelines?"
            - "Summarize the key findings from the research papers"
            """)
        
        with col2:
            st.subheader("Features")
            st.markdown("""
            - **Multi-format Support**: PDF, Word, PowerPoint, Excel, CSV, and Text files
            - **Visual Understanding**: Extracts and analyzes images from documents
            - **Chat Interface**: Natural conversation with follow-up questions
            - **Smart Citations**: Shows exact sources for every answer
            - **Persistent Storage**: Your database is saved and reloaded automatically
            - **File Management**: Easy upload, view, and delete documents
            """)

if __name__ == "__main__":
    main() 