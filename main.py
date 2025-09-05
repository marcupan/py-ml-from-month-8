import os
import argparse
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

class DocumentChatbot:
    def __init__(self, documents_dir="./documents", use_openai=True,
                 hf_embedding_model="sentence-transformers/all-mpnet-base-v2",
                 hf_llm_model="google/flan-t5-large"):
        """
        Initialize the chatbot with a directory of documents.

        Args:
            documents_dir (str): Path to the directory containing documents
            use_openai (bool): Whether to use OpenAI models (True) or Hugging Face models (False)
            hf_embedding_model (str): Hugging Face embedding model to use if not using OpenAI
            hf_llm_model (str): Hugging Face LLM model to use if not using OpenAI
        """
        self.documents_dir = documents_dir
        self.use_openai = use_openai
        self.hf_embedding_model = hf_embedding_model
        self.hf_llm_model = hf_llm_model
        self.vector_store = None
        self.conversation_chain = None
        self.chat_history = []

        # Ensure documents directory exists
        if not os.path.exists(documents_dir):
            os.makedirs(documents_dir)
            print(f"Created documents directory at {documents_dir}")
            print("Please add your documents to this directory and run the program again.")
            exit()

        # Initialize the chatbot
        self._initialize()

    def _initialize(self):
        """Set up the document processing pipeline and conversation chain."""
        # Load documents
        documents = self._load_documents()
        if not documents:
            print("No documents found. Please add documents to the documents directory.")
            exit()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)

        # Create vector store with appropriate embeddings
        if self.use_openai:
            print("Using OpenAI for embeddings...")
            embeddings = OpenAIEmbeddings()
        else:
            print(f"Using Hugging Face for embeddings (model: {self.hf_embedding_model})...")
            embeddings = HuggingFaceEmbeddings(model_name=self.hf_embedding_model)

        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        # Create conversation chain with appropriate LLM
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        if self.use_openai:
            print("Using OpenAI for language model...")
            llm = ChatOpenAI(temperature=0)
        else:
            print(f"Using Hugging Face for language model (model: {self.hf_llm_model})...")
            llm = HuggingFaceHub(
                repo_id=self.hf_llm_model,
                model_kwargs={"temperature": 0.5, "max_length": 512}
            )

        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(),
            memory=memory
        )

    def _load_documents(self):
        """Load documents from the documents directory."""
        documents = []

        # Load PDFs
        pdf_loader = DirectoryLoader(
            self.documents_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents.extend(pdf_loader.load())

        # Load text files
        text_loader = DirectoryLoader(
            self.documents_dir,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents.extend(text_loader.load())

        return documents

    def ask(self, question):
        """
        Ask a question to the chatbot.

        Args:
            question (str): The question to ask

        Returns:
            str: The chatbot's response
        """
        if not self.conversation_chain:
            return "Chatbot not initialized properly. Please check if documents are loaded."

        response = self.conversation_chain({"question": question})
        return response["answer"]

def main():
    print("Initializing Document Chatbot...")
    chatbot = DocumentChatbot()
    print("Chatbot initialized! You can now ask questions about your documents.")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nYou: ")
        if question.lower() in ["exit", "quit", "q"]:
            break

        answer = chatbot.ask(question)
        print(f"\nChatbot: {answer}")

if __name__ == "__main__":
    main()
