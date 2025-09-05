# Document Chatbot with RAG

A "smart" chatbot that can answer questions about your documents using Retrieval-Augmented Generation (RAG).

## Overview

This project implements a chatbot that can:
- Process your documents (PDFs, text files)
- Answer questions about their content
- Maintain context in a conversation

It uses LangChain, OpenAI, and ChromaDB to implement a RAG (Retrieval-Augmented Generation) system.

## How RAG Works

RAG combines the power of retrieval systems with generative AI:

1. **Document Processing**: Your documents are split into chunks and converted into vector embeddings
2. **Retrieval**: When you ask a question, the system finds the most relevant chunks from your documents
3. **Generation**: The LLM generates an answer based on the retrieved information and your question

This approach allows the chatbot to provide accurate, contextual answers based on your specific documents.

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your OpenAI API key:
   ```
   cp .env.example .env
   ```
   Then edit the `.env` file to add your actual OpenAI API key.

4. Add your documents to the `documents` directory (will be created on first run):
   - Supported formats: PDF (.pdf), Text (.txt)

## Usage

Run the chatbot:
```
python main.py
```

The first time you run it, the system will:
1. Create a `documents` directory if it doesn't exist
2. Process any documents in that directory
3. Create embeddings and store them in a vector database

You can then ask questions about your documents through the command-line interface.

Example conversation:
```
You: What is the main topic of my resume?
Chatbot: Based on your resume, the main topic appears to be your experience as a software engineer with expertise in...

You: What projects have I worked on?
Chatbot: According to your resume, you've worked on several projects including...
```

## Extending the Project

Here are some ways you could extend this project:

1. **Web Interface**: Add a simple web UI using Flask or Streamlit
2. **More Document Types**: Add support for more document formats (DOCX, HTML, etc.)
3. **Alternative Models**: Add support for local models via Hugging Face
4. **Document Management**: Add features to add/remove documents without reprocessing everything

## How It Works

The system follows these steps:

1. **Document Loading**: Uses LangChain's document loaders to read files
2. **Text Splitting**: Breaks documents into smaller chunks for processing
3. **Embedding Creation**: Converts text chunks into vector embeddings using OpenAI
4. **Vector Storage**: Stores embeddings in ChromaDB for efficient retrieval
5. **Question Answering**: Uses LangChain's ConversationalRetrievalChain to:
   - Find relevant document chunks based on the question
   - Generate an answer using the retrieved context
   - Maintain conversation history for follow-up questions

## License

[MIT License](LICENSE)
