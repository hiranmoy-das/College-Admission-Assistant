import os
from supabase.client import Client, create_client
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore

def _validate_supabase_credentials():
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    
    if not supabase_url or "your-project-id" in supabase_url:
        raise ValueError("SUPABASE_URL is currently set to a placeholder. Please set your actual Supabase project URL in the .env file.")
    if not supabase_key or "your_supabase_service_role_key" in supabase_key:
        raise ValueError("SUPABASE_SERVICE_KEY is currently set to a placeholder. Please set your actual Supabase service key in the .env file.")
    return supabase_url, supabase_key

def get_vector_store():
    """Initializes the connection to Supabase and the Embedding model."""
    supabase_url, supabase_key = _validate_supabase_credentials()
    
    supabase: Client = create_client(supabase_url, supabase_key)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents"
    )
    return vector_store

def clear_db():
    """Deletes all previous records from the documents table to refresh it."""
    supabase_url, supabase_key = _validate_supabase_credentials()
    supabase: Client = create_client(supabase_url, supabase_key)
    
    try:
        # This deletes all rows where ID is greater than 0 (which is everything)
        supabase.table("documents").delete().gt("id", 0).execute()
        print("Database cleared successfully!")
    except Exception as e:
        if "PGRST205" in str(e) or "Could not find the table" in str(e):
            raise ValueError("The 'documents' table is missing in your Supabase database. Please run the queries from 'supabase_setup.sql' in your Supabase SQL Editor.")
        raise e

def add_chunks_to_db(chunks):
    """Embeds and uploads chunks to Supabase."""
    vector_store = get_vector_store()
    vector_store.add_texts(texts=chunks)