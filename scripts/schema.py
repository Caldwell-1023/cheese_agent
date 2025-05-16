from typing import TypedDict, List

class PlanExecute(TypedDict):
    curr_state: str
    message: List[str]
    query_to_retrieve_or_answer: str
    curr_context: str
    aggregated_context: str
    tool: str
    human_feedback: str
    answer_quality: str
    reasoning_chain: List[str]