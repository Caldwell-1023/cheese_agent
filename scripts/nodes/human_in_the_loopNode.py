from scripts.schema import PlanExecute
from langgraph.types import interrupt

def human_in_the_loopNode(state: PlanExecute):
    """
    This function is used to handle queries that are not clear or ambiguous.
    It will ask the user for more information and then update the state with the new query.
    """

    print("Human in the loop")
    state["curr_state"] = "human_in_the_loop"
    response = interrupt({"query": state["query_to_retrieve_or_answer"]})
    # Command(resume=[{"args":"Help me."}])
    state["human_feedback"] = response[0]["args"]
    print("human_feedback:")
    print(state["human_feedback"])
    print("--------------------------------")
    return state
