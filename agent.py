from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, TypedDict, Optional
from dotenv import load_dotenv
import os
from pinecone import Pinecone
import json
from openai import OpenAI
from pymongo import MongoClient
from langchain.output_parsers import PydanticOutputParser
from langgraph.types import Command, interrupt


from scripts.schema import PlanExecute
from scripts.nodes.human_in_the_loopNode import human_in_the_loopNode
from scripts.nodes.reasoningNode import reasoningNode
from scripts.nodes.MongoDBretrievalNode import MongoDBretrievalNode
from scripts.nodes.pineconeretrievalNode import pineconeretrievalNode
from scripts.nodes.answerNode import answerNode
from scripts.retrieve_or_answer import retrieve_or_answer
from scripts.retry_or_end import retry_or_end

def make_agent_workflow():
    agent_workflow = StateGraph(PlanExecute)
    agent_workflow.add_node("reasoning", reasoningNode)
    agent_workflow.add_node("MongoDB_retrieval", MongoDBretrievalNode)
    agent_workflow.add_node("pinecone_retrieval", pineconeretrievalNode)
    agent_workflow.add_node("answer", answerNode)
    agent_workflow.add_node("human_in_the_loop", human_in_the_loopNode)

    agent_workflow.add_edge(START, "reasoning")
    agent_workflow.add_conditional_edges(
        "reasoning",
        retrieve_or_answer,
        {
            "chosen_tool_is_MongoDB_retrieval": "MongoDB_retrieval",
            "chosen_tool_is_pinecone_retrieval": "pinecone_retrieval",
            "chosen_tool_is_human_in_the_loop": "human_in_the_loop",
            "chosen_tool_is_answer": "answer"
        },
    )

    # agent_workflow.add_edge("reasoning", "answer")
    agent_workflow.add_edge("human_in_the_loop", "reasoning" )
    agent_workflow.add_edge("MongoDB_retrieval", "answer")
    agent_workflow.add_edge("pinecone_retrieval", "answer") 
    agent_workflow.add_conditional_edges(
        "answer",
        retry_or_end,
        {
            "retry_reasoning": "reasoning",
            "end_workflow": END
        }
    )

    checkpointer = MemorySaver()

    workflow = agent_workflow.compile(checkpointer=checkpointer)
    return workflow

# workflow = make_agent_workflow()
# # Execute the workflow
# while True:
#     input_query = input("Enter your query: ")
#     initial_state = {
#         "message": [input_query],
#         "aggregated_context": "",  # Initialize with empty string
#         "curr_context": "",
#         "query_to_retrieve_or_answer": "",
#         "tool": "",
#         "curr_state": "",
#         "human_feedback": "",
#         "answer_quality": ""
#     }

#     final_state = workflow.invoke(
#         initial_state,
#         config={"configurable": {"thread_id": 40}}
#     )

#     print("--------------------------------")
#    if "__interrupt__" in final_state.keys(): 
#         input_query = input("Enter new query:")
#         final_state = workflow.invoke(Command(resume=[{"args": input_query}]), config={"configurable": {"thread_id": 40}})
#     # Show the final response
    
#     print("final_state:")
#     print(final_state)
#     print(final_state["message"])