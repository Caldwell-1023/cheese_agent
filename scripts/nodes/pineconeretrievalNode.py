from scripts.schema import PlanExecute
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import os

def pineconeretrievalNode(state: PlanExecute):
    """Retrieve the relevant information from the Pinecone database.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state of the plan execution.
    """
    client = OpenAI()
    load_dotenv()
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = os.getenv("INDEX_NAME")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    query_embedding = client.embeddings.create(
        input=state["query_to_retrieve_or_answer"],
        model=EMBEDDING_MODEL
    ).data[0].embedding
        
    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )
    relevant_products = results.matches
    formatted_info = []
    for product in relevant_products:
        info = f"Product: {product.metadata.get('name', 'N/A')}\n"
        info += f"Category: {product.metadata.get('department', 'N/A')}\n"
        info += f"Price: ${product.metadata.get('price', 'N/A')}\n"
        if product.metadata.get('pricePer'):
            info += f"Price per pound: ${product.metadata.get('pricePer')}/lb\n"
        info += f"Brand: {product.metadata.get('brand', 'N/A')}\n"
        info += f"Similarity Score: {product.score:.2f}\n"
        info += f"Product URL: {product.metadata.get('href', 'N/A')}\n"
        info += f"image_url: {product.metadata.get('showImage', 'N/A')}\n"
        info += f"SKU: {product.metadata.get('sku', 'N/A')}\n"
        info += f"Related Products: {product.metadata.get('relateds', 'N/A')}\n"
        formatted_info.append(info)

    context = "\n".join(formatted_info)
    state["curr_context"] = context
    state["aggregated_context"] += "\n" + state["curr_context"]
    state["curr_state"] = "pinecone_retrieval"
    return state
