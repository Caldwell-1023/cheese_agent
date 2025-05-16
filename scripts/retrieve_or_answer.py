from scripts.schema import PlanExecute

def retrieve_or_answer(state: PlanExecute):
    """Decide whether to retrieve or answer the question based on the current state.
    Args:
        state: The current state of the plan execution.
    Returns:
        updates the tool to use .
    """
    print("deciding whether to retrieve or answer")
    if state["tool"] == "MongoDB_retrieval":
        return "chosen_tool_is_MongoDB_retrieval"
    elif state["tool"] == "pinecone_retrieval":
        return "chosen_tool_is_pinecone_retrieval"
    elif state["tool"] == "human_in_the_loop":
        return "chosen_tool_is_human_in_the_loop"
    elif state["tool"] == "answer":
        return "chosen_tool_is_answer"
    else:
        raise ValueError("Invalid tool was outputed. Must be either 'retrieve' or 'answer_from_context'")  
