# SmartDoc RAG

SmartDoc RAG is a document question-answering application that uses Retrieval-Augmented Generation (RAG) to answer questions from uploaded documents.

Instead of relying only on the language model's general knowledge, the application retrieves relevant document chunks and uses them as context for generating answers.

## Features

- Upload PDF, TXT, and Markdown documents.
- Load, clean, and split documents into smaller chunks.
- Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- Store and retrieve document chunks using Chroma.
- Support three retrieval strategies:
  - **Vector retrieval:** Finds semantically similar chunks.
  - **Keyword retrieval:** Finds chunks based on matching words.
  - **Hybrid retrieval:** Combines vector and keyword retrieval.
- Generate answers using the Groq API.
- Return source citations and a simple grounding/confidence estimate.
- Evaluate retrieval using metrics such as Hit@K, Precision@K, Recall@K, and latency.
- Provide a FastAPI backend and a web frontend.

## Technology Stack

- **Language:** Python
- **Backend:** FastAPI
- **LLM API:** Groq
- **Embeddings:** Sentence Transformers
- **Vector Database:** Chroma
- **Validation:** Pydantic
- **Frontend:** HTML, CSS, JavaScript

## Project Workflow

1. Upload a document.
2. Load and clean the document text.
3. Split the text into chunks.
4. Convert the chunks into embeddings.
5. Store the chunks and embeddings in Chroma.
6. Retrieve relevant chunks using the selected strategy.
7. Pass the retrieved context and question to the language model.
8. Return the answer with its source information.

## Project Structure

```text
week-4/
├── app/
│   └── main.py
├── data/
│   ├── documents/
│   └── evaluation/
├── experiments/
├── src/
│   └── smartdoc/
│       ├── generation/
│       ├── ingestion/
│       ├── retrieval/
│       ├── evaluation/
│       ├── config.py
│       ├── document_pipeline.py
│       └── pipeline.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Setup and Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd week-4
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Create a local `.env` file in the project root and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit your actual `.env` file or API key to GitHub.

### 5. Run the application

```bash
python -m uvicorn app.main:app --reload
```

Open the application at:

- Frontend: http://127.0.0.1:8000/
- API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/api/health

## Evaluation

The project includes an evaluation dataset and scripts for measuring retrieval performance.

Metrics include:

- **Hit@K:** Whether a relevant chunk appears in the top K results.
- **Precision@K:** The proportion of retrieved top-K chunks that are relevant.
- **Recall@K:** The proportion of relevant chunks retrieved in the top K.
- **Latency:** The time taken to perform an operation.

Evaluation results depend on the document dataset, retrieval configuration, and environment.

## Limitations

- Retrieval quality depends on document content, chunking, and retrieval strategy.
- Generated answers may be incomplete or incorrect.
- Citation checks and confidence estimates are heuristic checks, not guarantees of factual correctness.
- The application requires a valid Groq API key for answer generation.

## Future Improvements

- Improve vector and hybrid retrieval quality.
- Experiment with chunk sizes and embedding models.
- Improve answer-grounding validation.
- Expand the evaluation dataset and compare retrieval strategies.

## Security

API keys, virtual environments, uploaded PDFs, the local Chroma database, and generated evaluation results should remain local and must not be committed to the repository.
