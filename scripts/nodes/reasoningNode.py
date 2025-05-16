from scripts.schema import PlanExecute
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI


def reasoningNode(state: PlanExecute):
    """ Run the task handler chain to decide which tool to use to execute the task.
    Args:
       state: The current state of the plan execution.
    Returns:
       The updated state of the plan execution.
    """
    state["curr_state"] = "reasoning"
    inputs = {
                "message": state["message"],
                "aggregated_context": state["aggregated_context"],
                "human_feedback": state["human_feedback"]
            }
    
    reasoning_chain = create_reasoning_chain()
    output = reasoning_chain.invoke(inputs)
    print("reasoning output:")
    print(output)
    print("--------------------------------")
    state["reasoning_chain"].append(output.analysis)
    if output.tool == "mongoDB_retrieval":
        state["query_to_retrieve_or_answer"] = output.query
        state["tool"]="MongoDB_retrieval"
    
    elif output.tool == "pinecone_retrieval":
        state["query_to_retrieve_or_answer"] = output.query
        state["tool"]="pinecone_retrieval"

    elif output.tool == "human_in_the_loop":
        state["query_to_retrieve_or_answer"] = output.query
        state["tool"]="human_in_the_loop"

    elif output.tool == "out_of_scope":
        state["query_to_retrieve_or_answer"] = output.query
        state["tool"]="answer"
    elif output.tool == "combined_search":
        state["query_to_retrieve_or_answer"] = output.query
        state["tool"]="combined_search"
    else:
        raise ValueError("Invalid tool was outputed. Must be either 'retrieve' or 'answer_from_context'")
    return state  

def create_reasoning_chain():
    reasoning_prompt_template  = """
    You are an intelligent task handler that analyzes user queries about cheese products and determines the most appropriate tool to use. Your goal is to provide the most relevant and accurate information to the user.

Available Tools:

1. MongoDB Database Search (Tool A)
   - Use this tool when the query requires:
     * Searching for specific cheese products by name, brand, or type
     * Finding products within specific price ranges
     * Filtering by department or weight
     * Sorting by price, popularity, or other attributes
     * Counting the number
   - Example queries: "Find mozzarella under $50", "Show me the most expensive cheese", "What cheese does Galbani make?"

2. Vector Database Search (Tool B)
   - Use this tool when the query requires:
     * Semantic understanding of product descriptions
     * Finding similar products based on characteristics
     * Complex natural language queries about cheese features
     * Searching based on product descriptions or attributes
   - Example queries: "Find cheese that's good for pizza", "What cheese is similar to brie?", "Show me creamy Italian cheeses"

3. Combined Search (Tool E)
   - Use this tool when the query requires:
     * Both specific product information AND semantic understanding
     * Price/attribute filtering AND similarity-based recommendations
     * Complex queries that need both structured and unstructured data
     * Comprehensive product analysis combining multiple aspects
   - Example queries: 
     * "Find Italian cheeses under $30 that are similar to mozzarella"
     * "Show me creamy cheeses from France that cost less than $40"
     * "What are the best-rated cheddar cheeses under $50 that are good for melting?"
     * "Find organic cheeses similar to brie that are less than $35"

4. Out-of-Scope Handler (Tool C)
   - Use this tool when:
     * The query is not related to cheese products or information
     * The query is about other food items or unrelated topics
     * The query is outside the system's domain of expertise
   - Example scenarios: "Tell me about wine", "What's the weather like?", "How to make pasta" or "Hello!"

5. Human-in-the-Loop (Tool D)
   - Use this tool when:
     * The query lacks sufficient information to provide a meaningful response
     * The query is ambiguous or unclear
     * Additional context or clarification is needed
     * The query requires human judgment or expertise
   - Example scenarios: "Find cheese" (too vague), "What's good?" (unclear context), "Tell me about cheese" (too broad)

User Query: {message}
Human Feedback: {human_feedback}
Aggregated Context: {aggregated_context}

Your task:
1. First, analyze the human feedback:
   - If human feedback is provided, it takes precedence over the original query
   - Use the human feedback to clarify or modify the original query
   - If human feedback indicates a different tool should be used, switch to that tool
   - If human feedback provides additional context, incorporate it into your analysis

2. Then, analyze the modified query (original query + human feedback):
   - Determine which tool is most appropriate
   - Consider both the original query and any clarifications from human feedback

3. For Tools A, B, or E:
   - Generate a clear, specific query that will retrieve the most relevant information
   - For Tool E, specify that both MongoDB_retrieval and pinecone_retrieval should be used
   - Specify which tool(s) to use (MongoDB_retrieval, pinecone_retrieval, or both)
   - Incorporate any relevant information from human feedback into the query

4. For Tool C:
   - Clearly state that the query is outside the system's scope
   - Explain that the system is specifically designed for cheese-related queries
   - Politely redirect the user to focus on cheese-related questions
   - If human feedback suggests a cheese-related aspect, reconsider the tool choice

5. For Tool D:
   - Identify what information is missing or unclear
   - Formulate specific questions to ask the user for clarification
   - Explain why the current query is insufficient
   - If human feedback provides some clarification, acknowledge it and ask for remaining information

Important Guidelines:
- Always consider the user's intent and the type of information they're seeking
- For price-related queries, prefer MongoDB search
- For descriptive or similarity-based queries, prefer vector search
- For complex queries requiring both structured and semantic search, use the Combined Search tool
- When in doubt about query clarity or completeness, use the human-in-the-loop tool
- Be specific about what additional information is needed from the user
- Clearly identify and handle out-of-scope queries appropriately
- Human feedback should always be given priority in decision-making
- If human feedback contradicts the initial tool choice, switch to the appropriate tool
- Acknowledge and incorporate any additional context provided in human feedback
    """

    class reasoningOutput(BaseModel):
        """Output schema for the task handler."""
        query: str = Field(description="The specific query or question to be used")
        analysis: str = Field(description="Brief explanation of why this tool was chosen")
        curr_context: str = Field(description="The context to use")
        tool: str = Field(description="The tool to be used should be either mongoDB_retrieval, pinecone_retrieval, human_in_the_loop, combined_search or out_of_scope.")


    reasoning_prompt = PromptTemplate(
        template=reasoning_prompt_template,
        input_variables=["message", "aggregated_context", "human_feedback"],
    )

    reasoning_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=2000)
    reasoning_chain = reasoning_prompt | reasoning_llm.with_structured_output(reasoningOutput)
    return reasoning_chain
