import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
import json
from typing import Optional

from scripts.schema import PlanExecute

def MongoDBretrievalNode(state: PlanExecute):
    """Retrieve the relevant information from the MongoDB database using ChatGPT to generate queries.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state of the plan execution.
    """
    # Load environment variables
    load_dotenv()
    
    # Connect to MongoDB
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client[os.getenv("MONGODB_DB")]
    collection = db[os.getenv("MONGODB_COLLECTION")]

    # Define the query schema
    class MongoQuery(BaseModel):
        """Schema for MongoDB query generation."""
        filter_conditions: dict = Field(description="MongoDB filter conditions")
        sort_conditions: Optional[dict] = Field(description="MongoDB sort conditions", default=None)
        projection: Optional[dict] = Field(description="Fields to include in results", default=None)

    # Create the parser
    parser = PydanticOutputParser(pydantic_object=MongoQuery)

    # Create the prompt template
    prompt_template = """
    You are a MongoDB query generator. Given a user's message, generate a MongoDB query to find relevant information.

Available fields in the database:
- name: Product name (string)
- brand: Brand name (string)
- price: Price in dollars (float)
- pricePer: Price per LB (float)
- department: Department name (string)
- weight_each: Weight of each unit (float)
- weight_unit: Unit of weight (e.g., lbs, kg) (string)
- text: Full product description (string)
- popularity_order: Popularity ranking (integer, lower is more popular)
- price_order: Price ranking (integer, lower is more expensive)
- item_counts_each: Number of items per unit (integer)
- item_counts_case: Number of items per case (integer)
- weight_case: Weight of entire case (float)
- price_case: Price of entire case (float)
- showImage: Image URL (string)
- empty: out_of_stock (boolean)
Available departments:
- Specialty Cheese
- Sliced Cheese
- Cream Cheese
- Crumbled, Cubed, Grated, Shaved
- Shredded Cheese
- Cottage Cheese
- Cheese Loaf
- Cheese Wheel
User message: {message}

Query Type Selection Rules:
1. Use "aggregate" query_type when:
   - Counting items (e.g., "how many", "count", "number of", "total number of")
   - Calculating averages, sums, or other aggregations
   - Grouping data (e.g., "by brand", "by department")
   - Finding distinct values
   - Any operation that requires $group, $count, $avg, $sum stages
   - When the user asks about quantities or statistics

2. Use "find" query_type when:
   - Retrieving individual documents
   - Filtering by specific criteria
   - Sorting and limiting results
   - Simple text searches
   - No aggregation operations needed
Generate a MongoDB query that will help find the most relevant information. The query should be in the following format:
{{
    "query_type": "find" | "aggregate",  // Choose based on Query Type Selection Rules above
    "filter_conditions": {{  // Always required, even for aggregation queries
        // MongoDB filter conditions as key-value pairs
        // For aggregation queries, this can be empty {{}}
    }},
    "sort_conditions": {{
        // Optional: MongoDB sort conditions
        // Example: "price": -1 for descending order
    }},
    "limit": 0,  // Optional: Number of results to return
    "projection": {{
        // Optional: Fields to include in results
        // Example: "name": 1, "price": 1
    }},
    "aggregation_pipeline": [  // Only used when query_type is "aggregate"
        // Array of aggregation stages
        // Example: [{{"$group": {{"_id": "$brand", "count": {{"$sum": 1}}}}}}]
    ]
}}

Example queries:
1. For "cheese by Galbani out of stock in Specialty Cheese department":
{{
    "query_type": "find",  // Using find because we're filtering by specific criteria
    "filter_conditions": {{
        "brand": {{"$regex": "Galbani", "$options": "i"}},
        "department": {{"$regex": "Specialty Cheese", "$options": "i"}},
        "empty": true
    }},
    "sort_conditions": {{"popularity_order": 1}},
    "projection": {{"name": 1, "brand": 1, "department": 1, "price": 1, "_id": 0}}
}}

2. For "how many brands are there?" or "count number of brands":
{{
    "query_type": "aggregate",  // Using aggregate because we're counting distinct brands
    "filter_conditions": {{}},
    "aggregation_pipeline": [
        {{"$group": {{"_id": "$brand"}}}},  // First group by brand to get distinct brands
        {{"$count": "total_brands"}}  // Then count the number of distinct brands
    ]
}}

3. For "how many cheeses are there?":
{{
    "query_type": "aggregate",  // Using aggregate because we're counting
    "filter_conditions": {{}},
    "aggregation_pipeline": [
        {{"$count": "total_cheeses"}}
    ]
}}

4. For "how many brands have cheese under $10?":
{{
    "query_type": "aggregate",  // Using aggregate because we're counting with a condition
    "filter_conditions": {{}},
    "aggregation_pipeline": [
        {{"$match": {{"price": {{"$lt": 10}}}}}},  // First filter by price
        {{"$group": {{"_id": "$brand"}}}},  // Then group by brand
        {{"$count": "brands_under_10"}}  // Finally count the brands
    ]
}}

5. For "all cheese products out of stock":
{{
    "query_type": "find",  // Using find because we're retrieving individual documents
    "filter_conditions": {{"empty": true}},
    "sort_conditions": {{"popularity_order": 1}},
    "projection": {{"name": 1, "brand": 1, "price": 1, "pricePer": 1, "_id": 0}},
    "limit": 0  // 0 means no limit, return all results
}}

6. For "most expensive cheese":
{{
    "query_type": "find",  // Using find because we're retrieving a single document
    "filter_conditions": {{}},
    "sort_conditions": {{"price": -1}},
    "limit": 1,
    "projection": {{"name": 1, "price": 1, "brand": 1, "pricePer": 1, "_id": 0}}
}}

7. For "all mozzarella cheeses under $50":
{{
    "query_type": "find",  // Using find because we're filtering and retrieving documents
    "filter_conditions": {{
        "name": {{"$regex": "mozzarella", "$options": "i"}},
        "price": {{"$lte": 50}}
    }},
    "sort_conditions": {{"price": -1}},
    "projection": {{"name": 1, "price": 1, "brand": 1, "pricePer": 1, "_id": 0}},
    "limit": 0
}}


8. For "all goat cheeses":
{{
    "query_type": "find",
    "filter_conditions": {{
        "name": {{"$regex": "goat", "$options": "i"}}
    }},
    "projection": {{"name": 1, "price": 1, "brand": 1, "pricePer": 1, "_id": 0}},
    "limit": 0
}}

9. For "all Sliced Cheeses":
{{
    "query_type": "find",
    "filter_conditions": {{
        "department": {{"$regex": "Sliced Cheese", "$options": "i"}}
    }},
    "projection": {{"name": 1, "price": 1, "brand": 1, "pricePer": 1, "_id": 0}},
    "limit": 0
}}

10. For "average price by brand":
{{
    "query_type": "aggregate",  // Using aggregate because we're calculating averages
    "filter_conditions": {{}},
    "aggregation_pipeline": [
        {{"$group": {{
            "_id": "$brand",
            "avg_price": {{"$avg": "$price"}},
            "count": {{"$sum": 1}}
        }}}},
        {{"$sort": {{"avg_price": -1}}}}
    ]
}}


11. For "list all brands":
{{
    "query_type": "aggregate",  // Using aggregate because we're finding distinct values
    "filter_conditions": {{}},
    "aggregation_pipeline": [
        {{"$group": {{"_id": "$brand"}}}},
        {{"$sort": {{"_id": 1}}}}
    ]
}}

12. For "total weight of all cheeses":
{{
    "query_type": "aggregate",  // Using aggregate because we're calculating a sum
    "filter_conditions": {{}},
    "aggregation_pipeline": [
        {{"$group": {{
            "_id": null,
            "total_weight": {{"$sum": "$weight_each"}}
        }}}}
    ]
}}

Important rules:
1. Always include filter_conditions, even for aggregation queries (use empty {{}} if no filtering needed)
2. Use case-insensitive regex matching for text fields
3. Sort by relevant fields (price, popularity, weight) when appropriate
4. Include price and brand in projection unless specifically not needed
5. Use appropriate comparison operators ($gt, $lt, $gte, $lte) for numeric fields
6. For "all" queries, use empty filter_conditions {{}}
7. For counting or grouping operations, use query_type: "aggregate"
8. For aggregation queries, use the aggregation_pipeline array to specify stages
9. Common aggregation operations:
   - Count: {{"$count": "field_name"}}
   - Group: {{"$group": {{"_id": "$field", "count": {{"$sum": 1}}}}}}
   - Average: {{"$avg": "$field"}}
   - Sum: {{"$sum": "$field"}}
   - Distinct: {{"$group": {{"_id": "$field"}}}}
10. For "all" or "list all" queries, set limit to 0 to return all results
11. Always include _id: 0 in projection unless _id is specifically needed
12. For text search, use $regex with $options: "i" for case-insensitive matching
13. For aggregation queries that need filtering, use $match stage in aggregation_pipeline
14. When in doubt about query type:
    - If the query involves counting, grouping, or calculating values → use "aggregate"
    - If the query involves retrieving individual documents → use "find"
15. For counting operations:
    - Use $group stage to get distinct values
    - Use $count stage to count the results
    - Use $match stage for filtering before counting
    - Always use "aggregate" query_type
16. For text search, when using $regex, don't include "cheese".

Your query:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["message"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # Initialize the LLM
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")

    # Generate the query
    query = state["query_to_retrieve_or_answer"]
    prompt_value = prompt.format(message=query)
    response = llm.invoke(prompt_value)
    print("Response:")
    print(response)
    print("--------------------------------")
    mongo_query = json.loads(response.content)
    print("MongoDB query:")
    print(mongo_query)
    print("--------------------------------")
    results = []
    # Execute the query
    if mongo_query.get("query_type") == "aggregate":
        pipeline = []
        if mongo_query.get("filter_conditions"):
            pipeline.append({"$match": mongo_query.get("filter_conditions")})
        
        if mongo_query.get("aggregation_pipeline"):
            pipeline.extend(mongo_query.get("aggregation_pipeline"))
        
        if mongo_query.get("sort_conditions"):
            pipeline.append({"$sort": mongo_query.get("sort_conditions")})
        
        if mongo_query.get("limit") and mongo_query.get("limit") > 0:
            pipeline.append({"$limit": mongo_query.limit})

        results = list(collection.aggregate(pipeline))
    else:
        cursor = collection.find(
            mongo_query.get("filter_conditions"),
            mongo_query.get("projection") or {
                "name": 1,
                "brand": 1,
                "price": 1,
                "pricePer": 1,
                "department": 1,
                "weight_each": 1,
                "weight_unit": 1,
                "text": 1,
                "showImage": 1,
                "_id": 0
            }
        )

        if mongo_query.get("sort_conditions"):
            cursor = cursor.sort(list(mongo_query.get("sort_conditions").items()))
        if mongo_query.get("limit") and mongo_query.get("limit") > 0:
            cursor = cursor.limit(mongo_query.get("limit"))

        results = list(cursor)
    # print("Results:")
    # print(results)
    # print("--------------------------------")

    # Format the results
    formatted_results = []
    if mongo_query.get("query_type") == "find":
        formatted_results.append({
            "the number of products": len(results)  # You might want to format this differently based on the aggregation
        }) 
    for result in results:
        if mongo_query.get("query_type") == "aggregate":
            formatted_result = {
                "result": result  # You might want to format this differently based on the aggregation
            }
        else:
            formatted_result = {
                "name": result.get("name", "N/A"),
                "brand": result.get("brand", "N/A"),
                "price": f"${result.get('price', 'N/A')}",
                "price_per_unit": f"${result.get('pricePer', 'N/A')}/{result.get('weight_unit', 'unit')}",
                "department": result.get("department", "N/A"),
                "weight": f"{result.get('weight_each', 'N/A')} {result.get('weight_unit', 'units')}",
                "description": result.get("text", "N/A"),
                "image": result.get("showImage", "N/A")
            }
        formatted_results.append(formatted_result)
    
    # Update the state with the results
    # print("Formatted results:")
    # print(len(formatted_results))
    # print(formatted_results)
    state["curr_context"]=[{"role": "system", "content": str(formatted_results)}]
    if state["tool"] == "combined_search":
        return {
            "curr_context": state["curr_context"]
        }
    else:
        return state
