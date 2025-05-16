from scripts.schema import PlanExecute

def combinedSearchNode(state: PlanExecute):
    """
    This node is used to search for a product in the database.
    """
    print("combine searching...")
    state["tool"] = "combined_search"
    state["curr_state"] = "combined_search"
    return state


