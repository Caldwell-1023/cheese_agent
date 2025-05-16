import json
import os
from typing import List, Dict, Any
import time
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ----- Configuration -----
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# ----- Initialize clients -----
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI()

# ----- Functions -----
def load_json_data(json_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both array and single object formats
    if isinstance(data, dict):
        return [data]
    return data

def generate_embeddings(texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    """Generate embeddings using OpenAI API v1.0+."""
    if not texts:
        print("WARNING: No texts provided for embedding generation")
        return []
    
    embeddings = []
    # Reduce batch size to avoid quota issues
    batch_size = 50  # Reduced from 100 to 50
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"Generating embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    input=batch,
                    model=model
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                print(f"Successfully embedded {len(batch)} texts")
                # Add a small delay between batches to avoid rate limits
                time.sleep(1)
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error: {str(e)}. Retrying in {retry_delay} seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"Failed to generate embeddings after {max_retries} attempts: {str(e)}")
                    raise
    
    return embeddings

def prepare_product_text(product: Dict[str, Any]) -> str:
    """Create a rich text representation of a product for embedding."""
    parts = []

    if product.get("text"):
        parts.append(product.get("text"))
    # if product.get("name"):
    #     parts.append(f"Product: {product['name']}")
    # if product.get("Category"):
    #     parts.append(f"Category: {product['category']}")
    # if product.get("price"):
    #     parts.append(f"Price: ${product['price']}")
    # if product.get("LB_price"):
    #     parts.append(f"Price per pound: ${product['LB_price']}/lb")
    # if product.get("case_size"):
    #     parts.append(f"Case size: {product['case_size']}")
    # if product.get("case_price"):
    #     parts.append(f"Price per case: ${product['case_price']}/case")
    # if product.get("SKU_number"):
    #     parts.append(f"SKU: {product['SKU_number']}")
    # if product.get("UPC_number"):
    #     parts.append(f"UPC: {product['UPC_number']}")
    # if product.get("brand"):
    #     parts.append(f"brand: {product['brand']}")
    # if product.get("weight"):
    #     parts.append(f"weight: {product['weight']}")
    # if product.get("product_url"):
    #     parts.append(f"Product URL: {product['product_url']}")
    
    # if product.get("related_products") and isinstance(product["related_products"], list):
    #     related_products = ", ".join(product["related_products"][:5])
    #     parts.append(f"Related products: {related_products}")
    
    return " ".join(parts)

def initialize_pinecone():
    """Initialize Pinecone with API key."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print(f"Existing indexes: {pc.list_indexes().names()}")
    return pc

def check_and_recreate_index(pc, index_name: str, dimension: int, metric: str = "cosine"):
    """Check if index exists with correct dimensions, delete and recreate if needed."""
    try:
        # Delete the index if it exists
        if index_name in pc.list_indexes().names():
            print(f"Deleting existing index: {index_name}")
            pc.delete_index(index_name)
        
        # Create new index with correct dimensions
        print(f"Creating index: {index_name} with dimension {dimension}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        
        # Wait for index to be ready
        print("Waiting for index to be ready...")
        time.sleep(10)
        
        return pc.Index(index_name)
    except Exception as e:
        print(f"Error creating Pinecone index: {str(e)}")
        print("Please verify your Pinecone API key and account status")
        raise
        
def create_vector_db_from_food_products(json_path: str, index_name: str):
    """Create a vector database from food product JSON data."""
    # Load data
    print(f"Loading JSON data from {json_path}...")
    data = load_json_data(json_path)
    print(f"Loaded {len(data)} product records")
    
    # Prepare text for embedding
    print("Preparing product text for embedding...")
    product_texts = [prepare_product_text(product) for product in data]
    print(f"Created {len(product_texts)} text descriptions")
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(product_texts)
    
    if not embeddings:
        raise ValueError("Failed to generate embeddings")
    
    dimension = len(embeddings[0])
    print(f"Generated embeddings with dimension {dimension}")
    
    # Initialize Pinecone
    print("Initializing Pinecone...")
    pc = initialize_pinecone()
    
    # Create or connect to index - with correct dimensions
    print(f"Setting up index '{index_name}'...")
    index = check_and_recreate_index(pc, index_name, dimension)
    
    # Prepare metadata and IDs
    print("Preparing metadata and IDs...")
    ids = [f"product_{product.get('SKU_number', i)}" for i, product in enumerate(data)]
    
    # Prepare metadata
    metadata_list = []
    for product in data:
        metadata = {
            "name": product.get("name", ""),
            "sku": str(product.get("sku", "")),
            "department": str(product.get("department", "")),
            "brand": str(product.get("brand", "")),
            "showImage": str(product.get("showImage", "")),
            "price": float(product.get("price", 0)),
            "pricePer": float(product.get("priceper", 0)),
            "price_each": float(product.get("price_each", 0)),
            "price_unit": str(product.get("price_unit", "")),
            "price_order": float(product.get("price_order", 0)),
            "weight_each": float(product.get("weight_each", 0)),
            "weight_case": float(product.get("weight_case", 0)),
            "count_unit": str(product.get("count_unit", "")),
            "dimension_each": str(product.get("dimension_each", "")),
            "href": str(product.get("href", "")),
            "popularity_order": float(product.get("popularity_order", 0)),
            
        }
        metadata_list.append(metadata)
    
    # Insert vectors in batches
    print("Inserting vectors into Pinecone...")
    batch_size = 100
    for i in range(0, len(embeddings), batch_size):
        end_idx = min(i + batch_size, len(embeddings))
        batch_vectors = [
            {"id": ids[j], 
             "values": embeddings[j], 
             "metadata": metadata_list[j]
            } for j in range(i, end_idx)
        ]
        
        index.upsert(vectors=batch_vectors)
        print(f"Inserted batch {i//batch_size + 1}/{(len(embeddings)-1)//batch_size + 1}")
    
    print(f"Successfully created vector database with {len(embeddings)} product vectors")
    return pc, index

def query_product_database(pc, index_name, query_text: str, top_k: int = 5):
    """Query the product database with text."""
    # Generate embedding for the query
    query_embedding = generate_embeddings([query_text])[0]
    
    # Get the index
    index = pc.Index(index_name)
    
    # Query the index
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    return results

# ----- Main execution -----
def main():
    # Path to your JSON file
    json_path = "./fixture/products.json"  # Update with your file path
    
    # Create the vector database
    pc, index = create_vector_db_from_food_products(json_path, INDEX_NAME)
    
    # Example query
    results = query_product_database(pc, INDEX_NAME, "mozzarella cheese for pizza")
    
    print("\nSearch Results for 'mozzarella cheese for pizza':")
    for i, match in enumerate(results.matches):
        print(f"\n{i+1}. {match.metadata.get('name', 'No name')}")
        print(f"   Category: {match.metadata.get('category', 'N/A')}")
        print(f"   Price: ${match.metadata.get('price', 'N/A')}")
        print(f"   Similarity Score: {match.score:.4f}")

if __name__ == "__main__":
    main()