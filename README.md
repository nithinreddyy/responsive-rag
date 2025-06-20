# Responsive AI - Intelligent Document Analysis

**Responsive AI** is a powerful RAG (Retrieval-Augmented Generation) pipeline that enables intelligent document analysis with visual understanding. Built with Azure OpenAI and Cohere reranking, it processes multiple document formats and provides accurate, contextual answers with proper citations.

## Features

- **Multi-Format Support**: PDF, Word, PowerPoint, Excel, CSV, and Text files
- **Visual Understanding**: Extracts and analyzes images, charts, and diagrams from documents
- **Smart Retrieval**: Advanced vector search with Cohere reranking for improved accuracy
- **Chat Interface**: Natural conversation with follow-up questions
- **Source Citations**: Shows exact sources and page numbers for every answer
- **Dual Interfaces**: Command-line tool (`basic.py`) and web UI (`streamlit_app.py`)
- **Persistent Storage**: Vector databases are saved and reloaded automatically
- **Professional Logging**: Comprehensive logs for debugging and monitoring

## Requirements

- **Python**: 3.10.11
- **Operating System**: Windows, macOS, or Linux
- **API Keys**: Azure OpenAI and Cohere accounts required

## Installation

### 1. Clone the Repository
```bash
git clone nithinreddyy/responsive-rag
cd Responsive
```

### 2. Create Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root with your API credentials:

```env
# Azure OpenAI Configuration (for LLM)
AZURE_OPENAI_ENDPOINT=https://your-openai-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
AZURE_OPENAI_DEPLOYMENT=your-chat-deployment-name
AZURE_OPENAI_API_VERSION=2024-10-01-preview

# Azure OpenAI Embeddings Configuration  
AZURE_EMBEDDING_ENDPOINT=https://your-embedding-endpoint.openai.azure.com/
AZURE_EMBEDDING_API_KEY=your-embedding-api-key
AZURE_EMBEDDING_DEPLOYMENT=your-embedding-deployment-name
AZURE_EMBEDDING_API_VERSION=2024-10-01-preview

# Cohere API Configuration (for reranking)
COHERE_API_KEY=your-cohere-api-key
```

## How It Works

### Architecture Overview

1. **Document Processing**: 
   - Extracts text and images from various document formats
   - Uses conditional chunking based on content sufficiency
   - Analyzes embedded images with Azure OpenAI Vision

2. **Vector Storage**: 
   - Creates embeddings using Azure OpenAI
   - Stores in ChromaDB for fast similarity search
   - Separate databases for CLI and web interfaces

3. **Smart Retrieval**:
   - Initial retrieval: Fetches top-k documents using similarity search
   - Reranking: Uses Cohere to rerank documents for better relevance
   - Context preparation: Combines selected documents for LLM

4. **Answer Generation**:
   - Sends context to Azure OpenAI for answer generation
   - Provides source citations and visual references
   - Logs complete audit trail

### Processing Flow

```
Documents → Text/Image Extraction → Chunking → Embeddings → Vector DB
                                                                ↓
User Query → Vector Search → Cohere Reranking → LLM → Answer + Citations
```

## Usage

### Option 1: Web Interface (Recommended)

Run the Streamlit web application:

```bash
streamlit run streamlit_app.py
```

**Features:**
- Drag-and-drop file upload
- Real-time chat interface
- Visual citations with image previews
- Database management (clear, reset)
- Separate database (`streamlit_chroma_db/`)

### Option 2: Command Line Interface

Run the basic Python script:

```bash
python basic.py
```

**Features:**
- Processes files from `files/` folder
- Detailed console output
- Comprehensive logging to `logs/` folder
- Separate database (`chroma_db/`)

## Project Structure

```
Responsive/
├── basic.py                    # CLI interface
├── streamlit_app.py           # Web interface  
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── Streamlit UI.PNG          # UI screenshot
├── .env                      # Environment variables (create this)
├── files/                    # Documents for CLI processing
├── uploads/                  # Streamlit uploaded files
├── logs/                     # Processing logs
├── extracted_images/         # Saved image summaries
├── chroma_db/               # CLI vector database
├── streamlit_chroma_db/     # Streamlit vector database
└── __pycache__/             # Python cache
```

## 🔧 Configuration

### Supported File Formats
- **PDF**: Text and embedded images
- **Word (.docx)**: Text and embedded images  
- **PowerPoint (.pptx)**: Text and slide images
- **Excel (.xlsx)**: Text and embedded charts
- **CSV**: Structured data
- **TXT**: Plain text

### Retrieval Parameters
You can customize the retrieval behavior:

```python
result = rag.query(
    question="Your question",
    fetch_k=50,           # Initial documents to retrieve
    rerank_k=10,          # Final documents after reranking
    use_reranker=True,    # Enable Cohere reranking
    include_visual_citation=True  # Include image citations
)
```

## Example Queries

- "What are the main topics discussed in these documents?"
- "Show me any charts or images related to financial data"
- "What information is available about project timelines?"
- "Summarize the key findings from the research papers"
- "What is mentioned about revenue in Q3?"

## Logging

Comprehensive logs are automatically saved to:
- **Location**: `logs/rag_pipeline_YYYYMMDD_HHMMSS.log`
- **Content**: Document processing, retrieval steps, generated answers
- **Format**: Timestamped, structured, searchable

Example log output:
```
2024-01-15 10:30:45 | INFO     | __main__ | Document Summary for report.pdf:
2024-01-15 10:30:45 | INFO     | __main__ |    Total pages: 12
2024-01-15 10:30:45 | INFO     | __main__ |    Pages with text only: 10
2024-01-15 10:30:45 | INFO     | __main__ |    Pages with images only: 1
2024-01-15 10:30:45 | INFO     | __main__ |    Pages with both text and images: 1
```

## Troubleshooting

### Common Issues

**1. Environment Variables Not Found**
- Verify your `.env` file is in the project root
- Check all required API keys are present
- Restart the application after changes

**2. Database Cleanup Issues (Windows)**
- The app includes robust cleanup mechanisms
- If database won't delete, it will be renamed and cleaned up on restart
- Solution: Close browser tabs, stop Streamlit (Ctrl+C), restart

**3. Import Errors**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check Python version is 3.10.11

**4. API Rate Limits**
- Reduce `fetch_k` parameter for fewer API calls
- Check your Azure OpenAI and Cohere quotas
- Monitor usage in respective dashboards

## Security

- API keys are stored in `.env` file (not in code)
- Add `.env` to `.gitignore` to prevent committing secrets
- Use different keys for development/production environments
- Regularly rotate API keys for security

## Performance Tips

1. **Optimize Document Size**: Large PDFs may take longer to process
2. **Adjust Parameters**: Lower `fetch_k` for faster queries
3. **Use Reranking**: Improves accuracy but adds latency
4. **Monitor Logs**: Check processing times in log files
5. **Database Management**: Clear old databases periodically

![Streamlit UI](Streamlit%20UI.PNG)

---

**Built using Azure OpenAI, Cohere, and Streamlit** 
