from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_pdf_file(data):
    """
    Extract data from PDF files in the specified directory.
    """
    loader = DirectoryLoader(
        data, 
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

def text_split(extracted_data):
    """
    Split documents into smaller text chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def download_hugging_face_embeddings():
    """
    Download and initialize Hugging Face embeddings.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings
