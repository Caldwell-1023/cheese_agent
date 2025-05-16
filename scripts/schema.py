from typing import TypedDict, List, Annotated
from langgraph.graph.message import add_messages

class PlanExecute(TypedDict):
    curr_state: str
    message: List[str]
    query_to_retrieve_or_answer: str
    curr_context: Annotated[List[str], add_messages]
    aggregated_context: str
    tool: str
    human_feedback: str
    answer_quality: str
    reasoning_chain: List[str]