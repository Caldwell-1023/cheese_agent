from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from scripts.schema import PlanExecute
from openai import OpenAI
import json
from pydantic import BaseModel, Field

def answerNode(state: PlanExecute):
    """Answer the question from the given context.
    Args:
        state: The current state of the plan execution.
    Returns:
        The updated state of the plan execution.
    """
    # Create a prompt template for generating answers
    answer_prompt_template = """You are a helpful cheese expert assistant. Your task is to answer the user's question about cheese products based on the provided context.

Context:
{context}

User Question: {question}

IMPORTANT:
 If the user asks "how many galbani cheese are there in the stock? what is the average price?" , the chabot must answer "The number of galbani cheese in the stock is 20 and the average price is 44.8"
 If the user asks "How many cheese products are out of stock at the moment?" , the chabot mustanswer "Found 27 products"

Guidelines:
1. First, check if the query is a greeting or small talk:
   - If the query is a greeting (e.g., "hi", "hello", "hey", "good morning") or small talk:
     * DO NOT use any context information
     * Respond with a friendly greeting
     * Politely redirect to cheese-related questions
     * Example: "Hello! I'm your cheese expert assistant. How can I help you find the perfect cheese today?"
   - If the query is about cheese products, continue with the guidelines below

2. For cheese-related queries:
   - Provide clear, concise, and accurate answers
   - If the context doesn't contain enough information, say so
   - Format prices and measurements appropriately
   - Highlight key features and benefits of the products
   - Be friendly and professional in your tone

3. When listing products:
   - IMPORTANT: Use the EXACT total number of products mentioned in the context
   - Start your response with "Found [exact number] products"
   - If the context mentions a specific number, you MUST use that exact number
   - If the number of products is 30 or less, you MUST list ALL products without exception
   - If the number of products is more than 30, list only the first 30 products
   - Always mention if you're showing a partial list

4. Number handling:
   - Never estimate or guess the total number
   - Do not modify or round the number in any way

5. Product listing requirements:
   - For 30 or fewer products:
     * You MUST list EVERY product from the context
     * Do not skip any products
     * Format each product as a separate item
     * Include all available details for each product
   - For more than 30 products:
     * List only the first 30 products
     * Clearly state that you're showing a partial list
     * Mention the total number of products found

6. Formatting:
   - List each product on a new line
   - Use bullet points or numbers for each product
   - Include all available details (price, brand, type, etc.)
   - Maintain consistent formatting throughout the list

Your answer:"""

    # Create the prompt\
    current_context = ''
    print("Current context:", state['curr_context'])
    if len(state['curr_context'])==0:
        current_context = ""
    elif len(state['curr_context'])==1:
        current_context = state['curr_context'][0].content
    else:
        current_context += state['curr_context'][0].content
        current_context += state['curr_context'][1].content
    
    answer_prompt = PromptTemplate(
        template=answer_prompt_template,
        input_variables=["context", "question"]
    )

    print("Current context:", current_context)

    # Initialize the LLM
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

    # Generate the answer
    print("Context:", current_context)
    print("Question:", state["query_to_retrieve_or_answer"])
    prompt_value = answer_prompt.format(
        context=current_context,
        question=state["query_to_retrieve_or_answer"]
    )
    
    response = llm.invoke(prompt_value)
    
    # Update the state with the answer
    state["message"].append(HumanMessage(content=response.content))
    client = OpenAI()
    evaluation_prompt_template = """
    You are an AI assistant evaluating the quality of an answer about cheese products.
    
    Original Question: {message}
    
    Answer to evaluate: {response}
    
    Please evaluate if this answer is satisfactory and provides useful information.
    Return only "GOOD" if the answer is informative and addresses the question well.
    Return only "POOR" if the answer is vague, uninformative, or doesn't properly address the question.
    """
    class reasoningOutput(BaseModel):
        """Output schema for the task handler."""
        analysis: str = Field(description="Brief explanation of why this tool was chosen")
        tool: str = Field(description="The tool to be used should be either GOOD or POOR")

    evaluation_prompt = PromptTemplate(
        template=evaluation_prompt_template,
        input_variables=["message", "response"]
    )
    llm1 = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    evaluation_chain = evaluation_prompt | llm1.with_structured_output(reasoningOutput)
    
    evaluation_response = evaluation_chain.invoke({
        "message": state["message"],
        "response": response.content
    })

    state["reasoning_chain"].append(evaluation_response.analysis)
    quality_assessment = evaluation_response.tool
    print("Quality assessment:", quality_assessment)
    
    # Store the answer and quality assessment in state
    state["answer_quality"] = quality_assessment
    state["message"].append(response.content)
    state["curr_context"] = []
    print("--------------------------------")
    print("Current context:", state["curr_context"])
    print("--------------------------------")

    return state
