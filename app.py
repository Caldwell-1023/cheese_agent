import streamlit as st
from agent import make_agent_workflow
from langgraph.types import Command
import json
import os
import dotenv

image_path = "img/workflow_graph.png"
# Set page config
st.set_page_config(
    page_title="Cheese Sales Assistant",
    page_icon="🧀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ... (keep existing code until the sidebar section)


# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f5f5f5;
        padding: 2rem;
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        font-size: 18px;
        border-radius: 20px;
        padding: 12px 20px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #4a90e2;
        box-shadow: 0 0 8px rgba(74, 144, 226, 0.2);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #4a90e2;
        color: white;
        border-radius: 20px;
        padding: 12px 24px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #357abd;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .chat-message:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    .chat-message.user {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        margin-left: 2rem;
    }
    
    .chat-message.assistant {
        background-color: #4a90e2;
        color: white;
        margin-right: 2rem;
    }
    
    .chat-message .content {
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        gap: 1rem;
    }
    
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        background-color: #f0f0f0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .chat-message.assistant .avatar {
        background-color: #357abd;
    }
    
    .chat-message .message {
        flex: 1;
        line-height: 1.5;
    }
    
    /* Warning message styling */
    .stWarning {
        background-color: #fff3cd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Title styling */
    h1 {
        color: #2c3e50;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    /* Spinner styling */
    .stSpinner>div {
        border-color: #4a90e2;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "workflow" not in st.session_state:
    st.session_state.workflow = make_agent_workflow()
if "input_key" not in st.session_state:
    st.session_state.input_key = 0
if "needs_feedback" not in st.session_state:
    st.session_state.needs_feedback = False
if "feedback_key" not in st.session_state:
    st.session_state.feedback_key = 0
if "reasoning_chain" not in st.session_state:
    st.session_state.reasoning_chain = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = 0

dotenv.load_dotenv()
# Header with gradient background
st.markdown("""
    <div style='background: linear-gradient(45deg, #4a90e2, #357abd); padding: 2rem; border-radius: 15px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0;'>🧀 Cheese Sales Assistant</h1>
        <p style='color: white; opacity: 0.9; margin-top: 0.5rem;'>
            Your personal cheese expert, ready to help you find the perfect cheese!
        </p>
    </div>
""", unsafe_allow_html=True)



# Display chat messages in a container
with st.container():
    for message in st.session_state.messages:
        with st.container():
            st.markdown(f"""
                <div class="chat-message {message['role']}">
                    <div class="content">
                        <div class="avatar">
                            {'🧀' if message['role'] == 'assistant' else '👤'}
                        </div>
                        <div class="message">
                            {message['content']}
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    st.markdown(f"""
        <div class="content">
            <div class="message">
                Reasoning Steps:
            </div>
        </div>
    """, unsafe_allow_html=True)
    for step in st.session_state.reasoning_chain:
        with st.container():
            st.markdown(f"""
                <div class="content">
                    <div class="message">
                        -{step}
                    </div>
                </div>
            """, unsafe_allow_html=True)

# Handle feedback input if needed
if st.session_state.needs_feedback:
    st.warning("I need more information to help you better.")
    feedback_input = st.text_input(
        "Please provide more details:",
        key=f"feedback_input_{st.session_state.feedback_key}"
    )
    
    if feedback_input:
        with st.spinner("Processing..."):
            final_state = st.session_state.workflow.invoke(
                Command(resume=[{"args": feedback_input}]),
                config={"configurable": {"thread_id": st.session_state.thread_id}}
            )
            st.session_state.needs_feedback = False
            st.session_state.feedback_key += 1
            if isinstance(final_state["message"], list):
                response = final_state["message"][-1]
            else:
                response = final_state["message"]
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

# Regular chat input
if not st.session_state.needs_feedback:
    user_input = st.text_input(
        "Ask me about cheese:",
        key=f"user_input_{st.session_state.input_key}",
        placeholder="e.g., 'Find mozzarella under $50' or 'What cheese is similar to brie?'"
    )

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        import random

        # Generate a random integer between 0 and 100
        st.session_state.thread_id = random.randint(0, 1000)
        print(st.session_state.thread_id)  # Example output: 42
        chat_history = []
        for i in st.session_state.messages[-4:]:
            chat_history.append(i["role"] + ": " + i["content"])
        print("chat history:")
        print(chat_history)
        print("--------------------------------")
        initial_state = {
            "curr_state": "",
            "message": chat_history,
            "aggregated_context": "",
            "curr_context": [],
            "query_to_retrieve_or_answer": "",
            "tool": "",
            "human_feedback": "",
            "answer_quality": "",
            "reasoning_chain": []
        }

        with st.spinner("Thinking..."):
            final_state = st.session_state.workflow.invoke(
                initial_state,
                config={"configurable": {"thread_id": st.session_state.thread_id}}
            )
            
            print("reasoning:")
            print(final_state["reasoning_chain"])
            print("--------------------------------")
            print("final state:", final_state)
            print("--------------------------------")

            if "__interrupt__" in final_state:
                st.session_state.needs_feedback = True
                st.session_state.feedback_key += 1
                st.rerun()
            else:
                if isinstance(final_state["message"], list):
                    response = final_state["message"][-1]
                else:
                    response = final_state["message"]
                
                st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.reasoning_chain = final_state["reasoning_chain"]
            st.session_state.input_key += 1
            st.rerun()

# Sidebar with information
with st.sidebar:
    st.markdown("""
        <div style='background: linear-gradient(45deg, #4a90e2, #357abd); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
            <h2 style='color: white; margin: 0;'>Workflow Graph</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Add the workflow graph

    st.image(image_path)
