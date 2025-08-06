# Speed-RAG: Efficient Conversational AI

This project implements a highly efficient, conversational RAG (Retrieval-Augmented Generation) pipeline and serves it via a FastAPI web application. It's designed to answer questions about a private collection of documents with speed and accuracy, leveraging binarized embeddings for performance and a conversational chain to maintain context.

## Key Features

- **Efficient Retrieval**: Uses binarized embeddings to significantly speed up vector search and reduce memory footprint.
- **Conversational Context**: Employs a `ConversationalRetrievalChain` to remember chat history, allowing for natural follow-up questions.
- **FastAPI Backend**: Exposes the RAG pipeline through a robust and modern asynchronous API.
- **Modular Code**: The RAG logic (`rag.py`) is cleanly separated from the web server (`app.py`), making the system easy to maintain and extend.
- **Powered by Groq**: Leverages the high-speed Groq API for near-instant language model inference.

## How It Works

The project follows a standard RAG pattern but with key optimizations:

1.  **Indexing**: Documents in the `docs_dir/` are loaded, split into chunks, and then converted into binary vector embeddings using a Hugging Face model. These are stored in a local Chroma vector database (`chroma_db/`). This happens automatically on the first run.
2.  **Runtime**:
    - When a user query is received, the FastAPI server passes it to the RAG pipeline.
    - The pipeline converts the query into a standalone question using the chat history.
    - It then retrieves the most relevant document chunks from the Chroma database based on this question.
    - The retrieved chunks, the chat history, and the original query are all sent to the Groq LLM to generate a context-aware answer.
    - The final answer is sent back to the user.

## Project Structure

```
Speed-RAG/
│
├── app.py              # The FastAPI web server and API endpoint.
├── rag.py              # The core RAG pipeline logic and classes.
├── requirements.txt    # All Python dependencies for the project.
├── docs_dir/           # Place your PDF documents here.
├── chroma_db/          # The vector database (created automatically).
├── .env                # Your environment variables (e.g., API keys).
└── README.md           # This file.
```

## Setup and Installation

Follow these steps to get the project running on your local machine.

### 1. Prerequisites

- Python 3.8+
- An API key from [Groq](https://console.groq.com/keys)

### 2. Clone the Repository

```bash
git clone https://github.com/21lakshh/Speed-RAG
cd Speed-RAG
```

### 3. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv venv
venv\\Scripts\\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

Install all the required Python packages.

```bash
pip install -r requirements.txt
```

### 5. Set Up Environment Variables

Create a file named `.env` in the root of the project directory and add your Groq API key:

```
GROQ_API_KEY="your-groq-api-key-here"
```

### 6. Add Your Documents

Place any PDF files you want the AI to learn from into the `docs_dir/` directory. The application will automatically process them.

## Running the Application

Once everything is set up, you can start the FastAPI server using Uvicorn.

```bash
uvicorn app:app --reload
```

-   `uvicorn app:app`: Tells Uvicorn to run the `app` object from the `app.py` file.
-   `--reload`: Automatically restarts the server whenever you make changes to the code.

The server will be available at `http://127.0.0.1:8000`. The first time you run it, you will see messages indicating that the vector database is being created. This is a one-time process.

## Using the API

You can interact with the RAG pipeline by sending `POST` requests to the `/api/chat` endpoint.

-   **URL**: `http://127.0.0.1:8000/api/chat`
-   **Method**: `POST`
-   **Body**: Raw JSON

### Example Request Body (New Conversation)

```json
{
  "query": "What are the key skills listed in the resume?",
  "chat_history": []
}
```
