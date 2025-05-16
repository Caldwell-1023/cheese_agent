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

Guidelines:
1. Provide clear, concise, and accurate answers
2. If the context doesn't contain enough information, say so
3. Format prices and measurements appropriately
4. Highlight key features and benefits of the products
5. If multiple products are mentioned, compare them when relevant
6. Be friendly and professional in your tone
7. if the user ask for all products, first return the correct number of products and then return all the products.
8. if the number of products is bigger than 30, return the first 30 products.
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
