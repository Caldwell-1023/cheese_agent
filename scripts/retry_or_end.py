from scripts.schema import PlanExecute

def retry_or_end(state: PlanExecute):
    """Decide whether to retry or end the workflow.
    Args:
        state: The current state of the plan execution.
    Returns:
        updates the tool to use .
    """
    print("deciding whether to retry or end")
    if state["answer_quality"] == "GOOD":
        return "end_workflow"
    elif state["answer_quality"] == "POOR":
        return "retry_reasoning"
    else:
        raise ValueError("Invalid tool was outputed. Must be either 'retrieve' or 'answer_from_context'")  