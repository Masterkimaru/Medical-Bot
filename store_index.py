import os
import time
from dotenv import load_dotenv
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from tenacity import retry, stop_after_attempt, wait_exponential
from pinecone import Pinecone, ServerlessSpec

# Monkey-patch pinecone.Index to use the new client’s index type
from pinecone import data
import pinecone
pinecone.Index = data.index.Index

def main():
    # Load environment variables
    load_dotenv()
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    
    try:
        # Initialize Pinecone client (new v3.0+ style)
        pc = Pinecone(api_key=PINECONE_API_KEY)
    except Exception as e:
        print(f"Failed to initialize Pinecone: {e}")
        raise

    # Load and process data
    print("\n1. Loading PDF documents...")
    extracted_data = load_pdf_file(data='C:/Users/Kelvin/OneDrive/Desktop/Medical Chatbot/Data/')
    
    print("\n2. Splitting text into chunks...")
    text_chunks = text_split(extracted_data)
    
    print("\n3. Downloading embeddings...")
    embeddings = download_hugging_face_embeddings()

    index_name = "doctorbot"

    # Index management
    print("\n4. Checking Pinecone index...")
    try:
        if index_name not in [index_info["name"] for index_info in pc.list_indexes()]:
            print(f"Creating index {index_name}...")
            pc.create_index(
                name=index_name,
                dimension=384,  # Update this to match your embedding dimensions
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print("Waiting 60 seconds for index initialization...")
            time.sleep(60)
    except Exception as e:
        print(f"Index management failed: {e}")
        raise

    # Document upsert with retry logic
    @retry(stop=stop_after_attempt(3), 
           wait=wait_exponential(multiplier=1, min=4, max=10))
    def upsert_documents():
        print("\n5. Upserting documents to Pinecone...")
        return LangchainPinecone.from_documents(
            documents=text_chunks,
            index_name=index_name,
            embedding=embeddings
        )

    try:
        # Perform upsert and verify
        docsearch = upsert_documents()
        print("\n6. Indexing completed successfully!")
        
        # Verify index stats
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(f"Total vectors in index: {stats['total_vector_count']}")
        print(f"Total documents processed: {len(text_chunks)}")
        
    except Exception as e:
        print(f"\nError during document upsert: {e}")
        raise

if __name__ == "__main__":
    main()