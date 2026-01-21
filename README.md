# 🚀 Sci-Fi Concept Explorer

A specialized AI-powered Question-Answering system built with Retrieval-Augmented Generation (RAG) that serves as a creative assistant and literary analyst for classic science fiction literature.

## 📖 Project Overview

The Sci-Fi Concept Explorer helps writers overcome creative blocks by exploring how pioneering sci-fi authors handled different themes. By querying a curated collection of classic, public-domain science fiction stories, users can discover patterns, draw inspiration, and analyze literary concepts without manually rereading hundreds of pages.

This project was developed as part of the AI Engineer training program at Code.Hub (Accenture), demonstrating practical implementation of RAG pipelines using LangChain, ChromaDB, and OpenAI embeddings.

## ✨ Features

- **Document Ingestion**: Automatically loads and processes multiple science fiction texts from a directory
- **Intelligent Chunking**: Splits documents into semantically meaningful chunks for optimal retrieval
- **Vector Database**: Utilizes ChromaDB for persistent storage of document embeddings
- **Dual Retrieval Strategies**: 
  - Standard similarity search retriever
  - MMR (Maximal Marginal Relevance) retriever for diverse results
- **RAG Pipeline**: Complete retrieval-augmented generation chain using LangChain Expression Language (LCEL)
- **Prompt Engineering**: Custom-designed prompts that ensure context-grounded responses
- **Comprehensive Logging**: Tracks all key operations for debugging and monitoring

## 🏗️ Architecture

The application follows a class-based architecture with the `SciFiQA` class encapsulating all core functionality:

```
SciFiQA
├── __init__()          # Initialize LLM model
├── set_data()          # Load documents from directory
├── split_data()        # Chunk documents for processing
├── create_embeddings() # Generate and store embeddings in ChromaDB
└── RAG_chain()         # Execute dual retrieval and generation pipeline
```

### RAG Pipeline Flow

1. **User Query** → 2. **Vector Store Retrieval** → 3. **Context Injection** → 4. **LLM Generation** → 5. **Response**

Both standard similarity search and MMR retrieval strategies are employed to provide comprehensive answers.

## 📚 Dataset

The project includes classic science fiction texts from Project Gutenberg:

- *Twenty Thousand Leagues Under the Sea* by Jules Verne
- *The Island of Doctor Moreau* by H.G. Wells
- *The Time Machine* by H.G. Wells
- *The War of the Worlds* by H.G. Wells
- *Frankenstein, or The Modern Prometheus* by Mary Shelley
- *A Princess of Mars* by Edgar Rice Burroughs
- *We* by Evgenii Ivanovich Zamiatin
- *Little Brother* by Cory Doctorow
- *The Marching Morons* by C.M. Kornbluth
- *Famous Modern Ghost Stories* (Collection)

## 🛠️ Technology Stack

- **Python 3.12+**
- **LangChain** - Framework for LLM application development
- **LangChain Community** - Community-contributed integrations
- **LangChain OpenAI** - OpenAI model integration
- **ChromaDB** - Vector database for embeddings storage
- **OpenAI API** - Embeddings (text-embedding-3-small) and LLM (gpt-4-turbo)
- **Unstructured** - Document loading and parsing
- **Poetry** - Dependency management
- **pytest** - Testing framework

## 📋 Prerequisites

- Python 3.12 or higher
- Poetry (for dependency management)
- OpenAI API key (or compatible Azure OpenAI endpoint)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sci-fi-concept-explorer.git
cd sci-fi-concept-explorer
```

### 2. Install Poetry (if not already installed)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Or follow the [official Poetry installation guide](https://python-poetry.org/docs/#installation).

### 3. Install Dependencies

```bash
poetry install
```

This will create a virtual environment and install all required dependencies specified in `pyproject.toml`.

### 4. Configure Environment Variables

Create a `.env` file in the project root directory:

```bash
touch .env
```

Add your OpenAI credentials:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_ENDPOINT=https://api.openai.com/v1
```

**Note**: If you're using Azure OpenAI, adjust the endpoint accordingly:
```env
OPENAI_API_KEY=your_azure_api_key
OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
```

### 5. Prepare the Data Directory

Ensure your science fiction texts are in a directory named `Data Files` in the project root, or modify the `data_path` parameter in the code to point to your data location.

## 🎯 Usage

### Running the Main Application

Activate the Poetry virtual environment and run the application:

```bash
poetry run python src/first_assigment/application.py
```

This will:
1. Load all documents from the `Data Files` directory
2. Split them into chunks
3. Create embeddings and store them in ChromaDB
4. Execute a sample query
5. Log all operations to the `logfile`

### Example Queries

The application answers questions like:

- "What are some common themes in Sci-Fi literature?"
- "What are some interesting examples of first contact with alien life?"
- "Describe different types of futuristic societies imagined by early 20th-century authors."
- "Find passages that describe space travel before the invention of computers."

### Customizing the Query

Modify the `query` variable in `application.py`:

```python
query = "Your custom question here"
response_1, response_2 = controller.RAG_chain(question=query)
```

### Using as a Library

You can also import and use the `SciFiQA` class in your own scripts:

```python
from first_assigment.application import SciFiQA

# Initialize the system
explorer = SciFiQA(model="gpt-4-turbo")

# Load and process documents
explorer.set_data(data_path="Data Files")
explorer.split_data(explorer.documents, chunk_size=500, chunk_overlap=50)
explorer.create_embeddings(model="text-embedding-3-small")

# Ask questions
response_standard, response_mmr = explorer.RAG_chain(
    "What are the main themes in H.G. Wells' works?"
)

print("Standard Retriever Response:", response_standard)
print("\nMMR Retriever Response:", response_mmr)
```

## 🧪 Testing

The project includes comprehensive unit tests to verify core functionality.

### Run All Tests

```bash
poetry run pytest tests/test_1.py -v
```

### Test Coverage

The test suite includes:

1. **test_set_data**: Verifies document chunking with proper size constraints
2. **test_create_embeddings**: Ensures embeddings are created and stored correctly
3. **test_RAG_chain_responses**: Validates the complete RAG pipeline generates responses

### Example Test Output

```
tests/test_1.py::test_set_data PASSED
tests/test_1.py::test_create_embeddings PASSED
tests/test_1.py::test_RAG_chain_responses PASSED
```

## 📁 Project Structure

```
sci-fi-concept-explorer/
│
├── src/
│   └── first_assigment/
│       ├── __init__.py
│       └── application.py          # Main application logic
│
├── tests/
│   ├── __init__.py
│   └── test_1.py                   # Unit tests
│
├── Data Files/                     # Science fiction texts (.txt files)
│   ├── Twenty_Thousand_Leagues_under_the_Sea_by_Jules_Verne.txt
│   ├── The_Time_Machine_by_H__G__Wells.txt
│   └── ...
│
├── db/
│   └── chroma_db/                  # ChromaDB vector store (generated)
│
├── pyproject.toml                  # Poetry dependencies and config
├── poetry.lock                     # Locked dependency versions
├── README.md                       # Project documentation
├── .env                            # Environment variables (create this)
├── logfile                         # Application logs (generated)
└── .gitignore                      # Git ignore rules
```

## ⚙️ Configuration Options

### Chunking Parameters

Adjust in `split_data()` method:

```python
chunk_size = 500       # Maximum characters per chunk
chunk_overlap = 50     # Overlap between consecutive chunks
```

### Model Selection

Change the LLM model in initialization:

```python
explorer = SciFiQA(model="gpt-4-turbo")  # or "gpt-3.5-turbo", "gpt-4", etc.
```

### Embedding Model

Modify in `create_embeddings()`:

```python
model = "text-embedding-3-small"  # or "text-embedding-3-large", "text-embedding-ada-002"
```

### Retrieval Parameters

Customize MMR retrieval in `RAG_chain()`:

```python
mmr_retriever = self.vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}  # Number of documents to retrieve
)
```

## 🔍 How It Works

### 1. Document Processing

Documents are loaded using LangChain's `DirectoryLoader` and split into manageable chunks using `RecursiveCharacterTextSplitter`. This ensures semantic coherence while maintaining optimal chunk sizes for embedding generation.

### 2. Embedding Generation

Text chunks are converted into vector embeddings using OpenAI's `text-embedding-3-small` model. These embeddings capture semantic meaning and enable similarity-based retrieval.

### 3. Vector Storage

Embeddings are stored in ChromaDB, a persistent vector database that enables efficient similarity search and retrieval operations.

### 4. Retrieval Strategies

- **Standard Retriever**: Finds the most semantically similar passages to the query
- **MMR Retriever**: Balances relevance with diversity to provide varied perspectives

### 5. Generation

Retrieved context is injected into a carefully crafted prompt template along with the user's question. The LLM generates responses strictly based on the provided context.

## 📝 Prompt Engineering

The system uses a specialized prompt that:

- Defines the AI's role as a creative assistant for writers
- Emphasizes context-grounded responses
- Prevents hallucination by requiring answers to be based only on retrieved passages
- Instructs the model to clearly state when information isn't available

```python
system_message = "You are a helpful creative assistant that gives ideas " \
                 "to help writers create new Sci-Fi stories. " \
                 "You are given a context and a question"

prompt_template = """
    Answer the question based only on the following context: {context}. 
    Do not generate additional questions or answers.
    Question: {question}
"""
```

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError

**Solution**: Ensure you're running commands within the Poetry environment:
```bash
poetry shell  # Activate the environment
python src/first_assigment/application.py
```

### Issue: OpenAI API Errors

**Solution**: Verify your `.env` file is correctly configured and your API key is valid. Check your OpenAI account has sufficient credits.

### Issue: ChromaDB Not Persisting

**Solution**: Ensure the `db/chroma_db` directory has write permissions. The vector store should persist automatically after the first run.

### Issue: Out of Memory During Embedding Creation

**Solution**: Reduce the number of documents or process them in batches. You can also adjust the `chunk_size` parameter to create fewer, larger chunks.

## 🔮 Future Enhancements

- [ ] **Conversational Memory**: Add chat history to support follow-up questions
- [ ] **Interactive CLI**: Build a command-line interface for real-time querying
- [ ] **Web Interface**: Create a Streamlit or FastAPI frontend
- [ ] **Multiple Data Sources**: Extend support for PDFs, EPUBs, and web scraping
- [ ] **Query Optimization**: Implement query rewriting and expansion
- [ ] **Source Attribution**: Display specific passages and book titles in responses
- [ ] **Evaluation Metrics**: Add RAGAS or similar frameworks for RAG evaluation
- [ ] **Caching Layer**: Implement semantic caching for repeated queries
- [ ] **Multi-language Support**: Extend to handle non-English sci-fi literature

## 📄 License

This project uses public domain texts from Project Gutenberg. The code is available for educational purposes.

## 🙏 Acknowledgments

- **Code.Hub & Accenture** for the AI Engineer training program
- **Project Gutenberg** for providing access to classic literature
- **LangChain** community for excellent documentation and tools
- **OpenAI** for embeddings and language model APIs

## 👤 Author

**Stavros Vlach**
- Email: stavrosvlach34@gmail.com
- GitHub: [@stavros-vlach](https://github.com/stavros-vlach)

## 📞 Contact & Feedback

For questions, suggestions, or collaboration opportunities, please reach out via email or open an issue on GitHub.

---

*Built with ❤️ as part of the AI Engineer training at Code.Hub*
