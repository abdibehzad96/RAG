
import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

load_dotenv() # Load environment variables from .env file

openai_key = os.getenv("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key, model_name="text-embedding-3-small"
) # We create an embedding function using OpenAI's embedding model


# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=openai_ef
) # We create a collection in ChromaDB to store our documents and their embeddings

client = OpenAI(api_key=openai_key) # Initialize OpenAI client
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "What is human life expectancy in the United States?",
        },
    ],
) # Example of using OpenAI's chat completion API

print(response.choices[0].message["content"]) # Print the response from OpenAI