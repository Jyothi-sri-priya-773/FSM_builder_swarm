import os
import json
import re
import streamlit as st
from fsm_agents import user_interaction_agent, planning_agent
from swarm.core import Swarm

from typing import List, Any
from fsm_agents import user_interaction_agent, planning_agent
from State_builder_agent_1 import GPTClient
from memory import FSMMemory, get_fsm_schemas, ensure_collections_table_exists
from dotenv import load_dotenv



def run_pipeline(user_query: str, transcript: List[dict[str, Any]] | None = None) -> str:
    """Run the Swarm agent pipeline until a final assistant JSON is returned."""
    swarm = Swarm()
    current_agent = user_interaction_agent
    messages = transcript or [{"role": "user", "content": user_query}]

    while True:
        response = swarm.run(agent=current_agent, messages=messages)
        messages = response.messages  # keep the full history for the next turn
        last = messages[-1]

        # If the last message is a tool‑result, check for a hand‑off ↓
        if last["role"] == "tool":
            try:
                payload = json.loads(last["content"])
            except Exception:
                payload = {}
            nxt = payload.get("next_agent")
            if nxt is not None:
                if nxt == "planning_agent":
                    current_agent = planning_agent
                else:
                    current_agent = user_interaction_agent
                continue
        # Otherwise we have an assistant message ⇒ done.
        break

    return messages[-1]["content"]

# --- Initialization ---
load_dotenv()
ensure_collections_table_exists("fsm_memory.db")
fsm_memory = FSMMemory()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "fsm_output" not in st.session_state:
    st.session_state.fsm_output = ""
if "last_answered_query" not in st.session_state:
    st.session_state.last_answered_query = ""
if "last_state_builder_fsm" not in st.session_state:
    st.session_state.last_state_builder_fsm = None
if "fsm_compare_1" not in st.session_state:
    st.session_state.fsm_compare_1 = None
if "fsm_compare_2" not in st.session_state:
    st.session_state.fsm_compare_2 = None
if "chat_expanded" not in st.session_state:
    st.session_state.chat_expanded = True
if "sidebar_collapsed" not in st.session_state:
    st.session_state.sidebar_collapsed = False

# Page config
st.set_page_config(
    page_title="FSM Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main container styling */
    .main-container {
        padding: 1rem;
    }
    
    /* FSM Output container */
    
    .fsm-title {
        color: black;
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Chat interface styling */
    
    
    .chat-title {
        color: black;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    /* Toggle button styling */
    .toggle-btn {
        background: linear-gradient(45deg, #2196f3, #1565c0);
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s ease;
        color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    .toggle-btn:hover {
        background: linear-gradient(45deg, #1e88e5, #0d47a1);
        transform: scale(1.1);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    /* Sidebar buttons */
    .stButton > button {
        background: linear-gradient(45deg, #2196f3, #1565c0) !important;
        width: 100% !important;
        min-height: 42px !important;
    }
    /* Limit sidebar button width so they don't stretch across main area */
    div[data-testid="stSidebar"] .stButton > button {
        width: 100% !important; /* keep full inside sidebar column */
        max-width: 180px !important; /* prevent full-page stretch */
    }
    /* Add extra bottom padding to main FSM container to clear floating chat btn */
    .fsm-container {
        padding-bottom: 80px !important;
    }
    .fsm-container .stButton > button {
        margin-bottom: 0.5rem !important;
      
        color: white !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #1e88e5, #0d47a1) !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
    }

    /* Download buttons (match primary blue style) */
    .stDownloadButton > button {
        background: linear-gradient(45deg, #2196f3, #1565c0) !important;
        width: 100% !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        font-size: 0.9rem !important;
        transition: all 0.3s ease !important;
        min-height: 42px !important;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(45deg, #1e88e5, #0d47a1) !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Tabs in sidebar */
    .stTabs [data-baseweb="tab"] {
        color: #2196f3 !important;
    }
    
  
    
    /* Chat messages styling */
    .chat-message {
        margin: 0.5rem 0;
        padding: 0.75rem 1rem;
        border-radius: 15px;
        max-width: 80%;
        word-wrap: break-word;
        animation: fadeIn 0.3s ease;
    }
    
    .user-message {
        background: #e3f2fd;
        color: #333;
        border: 3px solid #2196f3;
        margin-right: auto;
        text-align: left;
        border-bottom-left-radius: 5px;
    }
    
    .assistant-message {
        background: #bbdefb;
        color: #333;
        border: 3px solid #1565c0;
        margin-left: auto;
        text-align: right;
        border-bottom-right-radius: 5px;
        backdrop-filter: blur(10px);
        white-space: pre-wrap; /* preserve indentation and allow wrapping */
    }

    /* Generating message styling */
    .generating-message {
        background: transparent;
        color: #0d47a1; /* purple */
        margin: 0.5rem auto;
        text-align: center;
        animation: fadeIn 0.3s ease;
        font-style: italic;
    }

    /* Chat row layout and icon */
    .chat-row {
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
    }
    .assistant-row {
        flex-direction: row-reverse;
    }
    .chat-icon {
        font-size: 1.4rem;
        line-height: 1.4rem;
        margin-top: 2px;
    }

    /* Hide default Streamlit spinner at bottom */
    .stSpinner {
        display: none !important;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Input styling */
    .chat-input {
        margin-top: 1rem;
    }
    
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 20px !important;
        border: 4px solid rgba(255, 255, 255, 0.5) !important;
        min-height: 80px !important;
        font-size: 14px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        border-color: rgba(255, 255, 255, 0.8) !important;
        box-shadow: 0 0 20px rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 25px;
        background: linear-gradient(45deg, #ff6b6b, #ee5a24);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        margin-top: 0.5rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar styling */
    .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #1565c0 100%);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Code block styling */
    .stCode {
        border-radius: 10px !important;
        background: rgba(0, 0, 0, 0.05) !important;
        max-height: 500px !important;
        overflow-y: auto !important;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .status-success {
        background: rgba(34, 197, 94, 0.2);
        color: #059669;
    }
    
    .status-error {
        background: rgba(239, 68, 68, 0.2);
        color: #dc2626;
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.2);
        color: #d97706;
    }
    
    /* Chat container styling with increased border thickness */
    .chat-messages-container {
        border: 3px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 20px !important;
        padding: 1rem !important;
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Chat header with improved X button positioning */
    .chat-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        margin-bottom: 1rem;
    }
    
    .chat-header .chat-title-text {
        color: black;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .chat-close-btn {
        background: transparent !important;
        border: 2px solid #ccc !important;
        color: #666 !important;
        border-radius: 50% !important;
        width: 35px !important;
        height: 35px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        font-size: 18px !important;
        padding: 0 !important;
    }
    
    .chat-close-btn:hover {
        background: #f0f0f0 !important;
        border-color: #999 !important;
        transform: scale(1.1) !important;
    }
    
    /* Purple styling for form buttons */
    div[data-testid="stForm"] button[kind="secondaryFormSubmit"],
    div[data-testid="stForm"] button[kind="secondary"] {
        background: linear-gradient(45deg, #2196f3, #1565c0) !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        transition: all 0.3s ease !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
    }
    
    div[data-testid="stForm"] button[kind="secondaryFormSubmit"]:hover,
    div[data-testid="stForm"] button[kind="secondary"]:hover {
        background: linear-gradient(45deg, #1976d2, #0d47a1) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(156, 39, 176, 0.3) !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-container {
            padding: 0.5rem;
        }
        
        .fsm-container, .chat-container {
            margin: 0.5rem 0;
        }
        
        .chat-message {
            max-width: 95%;
        }
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.5);
    }
    
    .user-row .chat-icon {
        color: #1565c0;
    }
    .assistant-row .chat-icon {
        color: #2196f3;
    }

    /* Floating chat button styling */
    .floating-chat-btn {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 60px !important;
        height: 60px !important;
        border-radius: 50% !important;
        background: linear-gradient(45deg, #f093fb, #f5576c) !important;
        border: none !important;
        color: white !important;
        font-size: 24px !important;
        cursor: pointer !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
        transition: all 0.3s ease !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        z-index: 9999 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    .floating-chat-btn:hover {
        transform: scale(1.1) !important;
        box-shadow: 0 6px 25px rgba(0,0,0,0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

def load_fsm_from_file(uploaded_file):
    """Load FSM from uploaded file"""
    try:
        content = uploaded_file.read().decode("utf-8")
        return json.loads(content)
    except Exception as e:
        st.error(f"Error loading FSM: {str(e)}")
        return None

def extract_json_from_markdown(text):
    """Extract JSON from markdown code blocks."""
    if not text or not isinstance(text, str):
        return text
    json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', text)
    if json_match:
        return json_match.group(1)
    json_match = re.search(r'({[\s\S]*})', text)
    if json_match:
        return json_match.group(1)
    return text

def validate_fsm(fsm_data):
    """Validate FSM against schema"""
    try:
        schemas = get_fsm_schemas()
        if isinstance(fsm_data, str):
            fsm_data = extract_json_from_markdown(fsm_data)
            fsm_data = json.loads(fsm_data)
        required_keys = ["name", "description", "version", "finiteStateMachine"]
        if not all(key in fsm_data for key in required_keys):
            return False, "Missing required top-level fields"
        states = fsm_data.get("finiteStateMachine", {}).get("states", [])
        if not isinstance(states, list) or not states:
            return False, "No states found in FSM"
        return True, "FSM is valid"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def compare_fsms(fsm1, fsm2):
    """Compare two FSMs and return detailed differences"""
    def parse_fsm(fsm):
        if isinstance(fsm, str):
            try:
                return json.loads(fsm)
            except json.JSONDecodeError:
                return {}
        return fsm or {}
    
    fsm1 = parse_fsm(fsm1)
    fsm2 = parse_fsm(fsm2)
    
    differences = []
    
    # Compare top-level fields
    for key in ["name", "description", "version"]:
        val1 = fsm1.get(key, "")
        val2 = fsm2.get(key, "")
        if val1 != val2:
            differences.append(f"{key}: '{val1}' vs '{val2}'")
    
    # Get states from both FSMs
    states1 = {s['name']: s for s in fsm1.get("finiteStateMachine", {}).get("states", []) if 'name' in s}
    states2 = {s['name']: s for s in fsm2.get("finiteStateMachine", {}).get("states", []) if 'name' in s}
    
    # Find added/removed states
    added_states = set(states2.keys()) - set(states1.keys())
    removed_states = set(states1.keys()) - set(states2.keys())
    
    # Find modified states
    common_states = set(states1.keys()) & set(states2.keys())
    modified_states = [
        state for state in common_states 
        if json.dumps(states1[state], sort_keys=True) != json.dumps(states2[state], sort_keys=True)
    ]
    
    # Format differences
    if added_states:
        differences.append(f"Added states: {', '.join(added_states)}")
    if removed_states:
        differences.append(f"Removed states: {', '.join(removed_states)}")
    
    # Add details of modified states
    for state in modified_states:
        state1 = states1[state]
        state2 = states2[state]
        
        # Compare state properties
        for prop in ["description", "type", "initial"]:
            if state1.get(prop) != state2.get(prop):
                differences.append(f"State '{state}' {prop} changed: '{state1.get(prop)}' → '{state2.get(prop)}'")
        
        # Compare transitions
        trans1 = {t['event']: t for t in state1.get("transitions", []) if 'event' in t}
        trans2 = {t['event']: t for t in state2.get("transitions", []) if 'event' in t}
        
        added_trans = set(trans2.keys()) - set(trans1.keys())
        removed_trans = set(trans1.keys()) - set(trans2.keys())
        common_trans = set(trans1.keys()) & set(trans2.keys())
        
        for event in added_trans:
            differences.append(f"State '{state}': Added transition on event '{event}' → '{trans2[event].get('target')}'")
        for event in removed_trans:
            differences.append(f"State '{state}': Removed transition on event '{event}'")
        
        for event in common_trans:
            if trans1[event] != trans2[event]:
                t1 = trans1[event]
                t2 = trans2[event]
                if t1.get('target') != t2.get('target'):
                    differences.append(f"State '{state}': Transition '{event}' target changed: '{t1.get('target')}' → '{t2.get('target')}'")
    
    return differences if differences else ["No differences found"]

# The code in main.py is compatible with fsm_agents.py if:
# - fsm_agents.py defines user_interaction_agent and planning_agent as agent objects/classes
#   that can be passed to Swarm().run(agent=..., messages=...) and have the expected interface.
# - user_interaction_agent is used for initial user queries.
# - planning_agent is used for corrections/feedback or when routed by user_interaction_agent.
# - Both agents return responses in the expected format (with a 'messages' list containing dicts
#   with 'role', 'content', and optionally 'sender' keys).

# The main.py code uses:
#   from fsm_agents import user_interaction_agent, planning_agent
# and then uses these agents as arguments to Swarm().run(...).
# It expects the result to have a .messages attribute (list of dicts), and the last message
# to have at least 'role' and 'content' keys.

# If fsm_agents.py provides these two agents as described, the code is compatible.

# No code changes needed for compatibility.

# --- Enhanced Sidebar ---
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown("# 🛠️ FSM Tools")
    tab1, tab2, tab3 = st.tabs(["📁 Load/Save", "✅ Validate", "🔄 Compare"])
    with tab1:
        st.markdown("### Upload FSM")
        uploaded_file = st.file_uploader("Choose FSM JSON file", type=["json", "txt"], key="fsm_upload")
        if uploaded_file is not None:
            fsm_data = load_fsm_from_file(uploaded_file)
            if fsm_data:
                st.session_state.fsm_output = json.dumps(fsm_data, indent=4)
                st.success("✅ FSM loaded successfully!")
        
        st.markdown("### Download FSM")
        if st.session_state.fsm_output:
            fsm_json_str = json.dumps(st.session_state.fsm_output, indent=2)

            st.download_button(
                label="📥 Download Current FSM",
                data=fsm_json_str,
                file_name="fsm_output.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.info("No FSM available for download")
    
    with tab2:
        st.markdown("### FSM Validation")
        # --- FSM Search/Filter ---
        search_query = st.text_input("Search FSM states by ID, type, or field value", key="fsm_search_query")
        search_results = None
        uploaded_states = None
        # Try to get states from current FSM output
        try:
            if st.session_state.fsm_output:
                fsm_obj = json.loads(extract_json_from_markdown(st.session_state.fsm_output))
                uploaded_states = fsm_obj.get("finiteStateMachine", {}).get("states", [])
        except Exception:
            uploaded_states = None

        if search_query:
            if uploaded_states:
                try:
                    results = []
                    for state in uploaded_states:
                        if (
                            search_query.lower() in str(state.get("id", "")).lower()
                            or search_query.lower() in str(state.get("type", "")).lower()
                            or any(search_query.lower() in str(v).lower() for v in state.values())
                        ):
                            results.append(state)
                    search_results = results
                except Exception as e:
                    search_results = [{"error": f"Error searching FSM states: {e}"}]
            else:
                search_results = [{"info": "No FSM loaded or no states to search."}]
        # Display search results if any
        if search_query:
            st.markdown("#### Search Results")
            if search_results:
                for idx, state in enumerate(search_results, 1):
                    st.code(json.dumps(state, indent=2), language="json")
            else:
                st.info("No matching states found.")

        # ...existing validation button and logic...
        if st.button("🔍 Validate Current FSM", use_container_width=True):
            if st.session_state.fsm_output:
                is_valid, message = validate_fsm(st.session_state.fsm_output)
                if is_valid:
                    st.markdown(f'<div class="status-indicator status-success">✅ {message}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="status-indicator status-error">❌ {message}</div>', unsafe_allow_html=True)
    with tab3:
        st.markdown("### FSM Comparison & Merge")
        fsm_file_1 = st.file_uploader("FSM File 1", type=["json", "txt"], key="fsm1")
        fsm_file_2 = st.file_uploader("FSM File 2", type=["json", "txt"], key="fsm2")
        compare_result = None
        merge_result = None
        col_compare, col_merge = st.columns(2)
        with col_compare:
            compare_clicked = st.button("Compare FSMs", key="compare_btn")
        with col_merge:
            merge_clicked = st.button("Merge FSMs", key="merge_btn")
        if compare_clicked:
            if fsm_file_1 and fsm_file_2:
                try:
                    fsm1 = json.loads(fsm_file_1.read().decode("utf-8"))
                    fsm2 = json.loads(fsm_file_2.read().decode("utf-8"))
                    states1 = fsm1.get("finiteStateMachine", {}).get("states", [])
                    states2 = fsm2.get("finiteStateMachine", {}).get("states", [])
                    ids1 = {(s.get("id"), s.get("type")) for s in states1}
                    ids2 = {(s.get("id"), s.get("type")) for s in states2}
                    only_in_1 = ids1 - ids2
                    only_in_2 = ids2 - ids1
                    compare_result = {
                        "States only in FSM 1": list(only_in_1),
                        "States only in FSM 2": list(only_in_2)
                    }
                    # Store compare result in session for main output panel
                    st.session_state.fsm_compare_result = compare_result
                    st.session_state.fsm_merge_result = None
                except Exception as e:
                    compare_result = {"error": f"Error comparing FSMs: {e}"}
                    st.session_state.fsm_compare_result = compare_result
                    st.session_state.fsm_merge_result = None
            else:
                compare_result = {"warning": "Upload two FSM files to compare."}
                st.session_state.fsm_compare_result = compare_result
                st.session_state.fsm_merge_result = None

        if merge_clicked:
            if fsm_file_1 and fsm_file_2:
                try:
                    fsm1 = json.loads(fsm_file_1.read().decode("utf-8"))
                    fsm2 = json.loads(fsm_file_2.read().decode("utf-8"))
                    states1 = fsm1.get("finiteStateMachine", {}).get("states", [])
                    states2 = fsm2.get("finiteStateMachine", {}).get("states", [])
                    merged_states = states1 + [s for s in states2 if (s.get("id"), s.get("type")) not in {(st.get("id"), st.get("type")) for st in states1}]
                    merge_result = {
                        "name": "MergedFSM",
                        "description": "Merged FSM from two files.",
                        "version": 1,
                        "finiteStateMachine": {"states": merged_states}
                    }
                    st.session_state.fsm_merge_result = merge_result
                    st.session_state.fsm_compare_result = None
                except Exception as e:
                    merge_result = {"error": f"Error merging FSMs: {e}"}
                    st.session_state.fsm_merge_result = merge_result
                    st.session_state.fsm_compare_result = None
            else:
                merge_result = {"warning": "Upload two FSM files to merge."}
                st.session_state.fsm_merge_result = merge_result
                st.session_state.fsm_compare_result = None
        # Do NOT display compare_result or merge_result in the sidebar here.
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat toggle button - only show "Open Chat" when collapsed
    st.markdown("---")
    if st.session_state.chat_expanded:
        pass  # Don't show any button when chat is expanded
    else:
        pass
# Main title
st.markdown("# 🤖 FSM Generation Chatbot")

# Create columns for the main content
if st.session_state.chat_expanded:
    col1, col2, col3 = st.columns([2, 0.3, 1.7])
else:
    col1 = st.container()
    col2 = None
    col3 = None

# --- FSM Output Panel ---
with col1:
    st.markdown('<div class="fsm-container">', unsafe_allow_html=True)
    st.markdown('<div class="fsm-title"><img src="https://img.icons8.com/color/48/000000/json--v1.png" width="30" height="30" style="vertical-align: middle; margin-right: 8px;"> FSM Output</div>', unsafe_allow_html=True)

    user_query = None
    for msg in reversed(st.session_state.chat_history):
        if msg["role"] == "user":
            user_query = msg["content"]
            break

    if user_query:
        st.markdown(
            f"""
            <div style="
                border:2px solid #4F8BF9;
                border-radius:8px;
                padding:12px;
                background-color:#f7fafd;
                margin-bottom:10px;
                height:120px;
                overflow-y:auto;
                display:block;
            ">
                <b>User Query:</b><br>
                <span style="font-size:16px; white-space:pre-wrap;">{user_query}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    if st.session_state.fsm_output:
        st.markdown("**FSM Output:**")
        st.markdown(
            f'<div class="chat-message assistant-message" style="text-align:left; white-space:pre-wrap;">{st.session_state.fsm_output}</div>',
            unsafe_allow_html=True
        )
    elif not user_query:
        st.info("💡 No FSM generated yet. Start a conversation to generate your FSM!")

    st.markdown('</div>', unsafe_allow_html=True)

# --- Chat Interface Panel ---
if st.session_state.chat_expanded:
    with col3:
        st.markdown(
            '<div style="height: 100%; display: flex; flex-direction: column; margin-top: 0; padding-top: 0;">',
            unsafe_allow_html=True
        )
        col1_, col2_ = st.columns([5, 1])
        with col1_:
            st.markdown(
                '<div style="font-size: 2.4rem; font-weight: 600; margin: 0 0 1rem 0;">💬 Chat Interface</div>',
                unsafe_allow_html=True
            )
        with col2_:
            if st.button("▶", key="close_chat", help="Close chat"):
                st.session_state.chat_expanded = False
                st.rerun()

        chat_container = st.container()
        chat_box = chat_container.container(height=500, border=True)
        with chat_box:
            # Show chat history (user and validator agent)
            for idx, msg in enumerate(st.session_state.chat_history):
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="chat-row user-row"><span class="chat-icon">👤</span>'
                        f'<div class="chat-message user-message">{msg["content"]}</div></div>',
                        unsafe_allow_html=True
                    )
                elif msg.get("role") == "assistant":
                    content = msg["content"]
                    st.markdown(
                        f'<div class="chat-row assistant-row"><span class="chat-icon">🤖</span>'
                        f'<div class="chat-message assistant-message" style="text-align:left; white-space:pre-wrap;">{content}</div></div>',
                        unsafe_allow_html=True
                    )
            # Show spinner if last message is user and not answered yet
            if (
                st.session_state.chat_history
                and st.session_state.chat_history[-1]["role"] == "user"
                and (
                    "last_answered_query" not in st.session_state or
                    st.session_state.chat_history[-1]["content"] != st.session_state.last_answered_query
                )
            ):
                st.markdown(
                    '<div class="chat-message generating-message">⌛ Generating FSM...</div>',
                    unsafe_allow_html=True
                )

        # Chat input section
        st.markdown("""
        <style>
            .stForm {
                border: 2px solid #555555 !important;
                border-radius: 10px !important;
                padding: 15px !important;
                margin-top: 10px !important;
            }
            .stTextArea textarea {
                border: 2px solid #555555 !important;
                border-radius: 8px !important;
                padding: 10px !important;
            }
            .stTextArea label {
                font-weight: 600 !important;
            }
        </style>
        """, unsafe_allow_html=True)

        with st.form("chat_form", clear_on_submit=True):
            prompt = st.text_area(
                "Enter your query:",
                height=100,
                key="chat_input_main",
                placeholder="Describe the FSM you want to generate..."
            )
            col1, col2 = st.columns([1, 1])
            with col1:
                submitted = st.form_submit_button("🚀 Send", use_container_width=True)
                if submitted and prompt.strip():
                    prompt_clean = " ".join(prompt.split())
                    st.session_state.chat_history.append({"role": "user", "content": prompt_clean})
                    # Route to user_interaction_agent
                    swarm = Swarm()
                    route = swarm.run(
                        agent=user_interaction_agent,
                        messages=[{"role": "user", "content": prompt_clean}]
                    )
                    # If routed to planning_agent, run planning_agent and update FSM output
                    if route.agent == planning_agent:
                        plan = swarm.run(
                            agent=planning_agent,
                            messages=[{"role": "user", "content": prompt_clean}]
                        )
                        content = plan.messages[-1]['content']
                        st.session_state.fsm_output = content
                        st.session_state.chat_history.append({"role": "assistant", "content": content})
                    else:
                        st.session_state.fsm_output = route.value
                        st.session_state.chat_history.append({"role": "assistant", "content": route.value})
                    st.session_state.last_answered_query = prompt_clean
                    st.rerun()
            with col2:
                if st.form_submit_button("🗑️ Clear", use_container_width=True):
                    st.session_state.chat_input_main = ""
                    st.session_state.chat_history = []
                    st.session_state.fsm_output = ""
                    st.session_state.last_answered_query = ""
                    st.rerun()

# Add this helper function near the top of your file (after imports)
def normalize_fsm_structure(fsm):
    """
    Ensures that 'finiteStateMachine' is always a dict with a 'states' array.
    If it's a list, wraps it. If it's missing, adds an empty list.
    """
    if not isinstance(fsm, dict):
        return fsm
    fsm = dict(fsm)  # shallow copy
    fsmachine = fsm.get("finiteStateMachine")
    if isinstance(fsmachine, list):
        fsm["finiteStateMachine"] = {"states": fsmachine}
    elif isinstance(fsmachine, dict):
        if "states" not in fsmachine:
            fsm["finiteStateMachine"]["states"] = []
    else:
        fsm["finiteStateMachine"] = {"states": []}
    return fsm
# --- Agent Execution (UI only, no CLI prompt loop) ---
def agentic_fsm_agent_ui(prompt, previous_fsm_json=None):
    from swarm.core import Swarm
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Only pass previous FSM if it exists and is valid
    previous_fsm_json = None
    previous_fsm = None
    if st.session_state.fsm_output:
        try:
            previous_fsm_json = json.loads(st.session_state.fsm_output) if isinstance(st.session_state.fsm_output, str) else st.session_state.fsm_output
            previous_fsm = normalize_fsm_structure(previous_fsm_json)
            print("jyo", previous_fsm)
        except Exception:
            previous_fsm = None

    context_variables = {
        "query": prompt,
        "context": {},
        "previous_fsm_json": previous_fsm
    }
    
    swarm = Swarm()
    
    # Always use user_interaction_agent for initial routing
    result = swarm.run(
        agent=user_interaction_agent,
        messages=messages,
        context_variables=context_variables,
        execute_tools=True,
        debug=True,
    )
    
    # ALWAYS prefer tool output with FSM JSON over assistant messages
    for msg in reversed(getattr(result, "messages", []) or []):
        if msg.get("role") == "tool" and msg.get("content"):
            try:
                tool_content = json.loads(msg["content"])
                if "fsm_output" in tool_content:
                    print(f"[agentic_fsm_agent_ui] Found FSM in tool output, returning it")
                    return msg["content"]  # Return the tool content directly
            except Exception:
                continue
    
    # Fallback: look for assistant messages only if no tool output found
    validator_response = None
    main_response = None
    for msg in reversed(getattr(result, "messages", []) or []):
        if msg.get("role") == "assistant" and msg.get("sender") == "validator_agent" and not validator_response:
            validator_response = msg.get("content")
        if msg.get("role") == "assistant" and msg.get("sender") == "main_agent" and not main_response:
            main_response = msg.get("content")
        if validator_response and main_response:
            break
    
    if validator_response or main_response:
        return validator_response or main_response
    
    return None

# --- Agent Execution (UI) ---
if (
    st.session_state.chat_history
    and st.session_state.chat_history[-1]["role"] == "user"
    and (
        "last_answered_query" not in st.session_state or
        st.session_state.chat_history[-1]["content"] != st.session_state.last_answered_query
    )
):
    prompt = st.session_state.chat_history[-1]["content"]
    gpt_client = GPTClient()

    with st.spinner("🔄 Generating FSM..."):
        try:
            previous_fsm_json = None
            if st.session_state.fsm_output:
                try:
                    if isinstance(st.session_state.fsm_output, str):
                        previous_fsm_json = json.loads(st.session_state.fsm_output)
                    elif isinstance(st.session_state.fsm_output, dict):
                        previous_fsm_json = st.session_state.fsm_output
                except Exception:
                    previous_fsm_json = None

            # Use the UI-specific agent function to get validator or main result
            agent_result = agentic_fsm_agent_ui(
                prompt,
                previous_fsm_json=previous_fsm_json
            )

            # if agent_result is None:
            #     st.session_state.fsm_output = "No FSM output was generated for your query."
            # else:
            #     st.session_state.fsm_output = agent_result
            if agent_result is None:
                st.session_state.fsm_output = "No FSM output was generated for your query."
            else:
                try:
                    # ✅ IMPROVED FSM EXTRACTION - Handle nested JSON strings
                    if isinstance(agent_result, str):
                        parsed = json.loads(agent_result)
                        
                        # Extract fsm_output from agent response
                        if "fsm_output" in parsed:
                            fsm_data = parsed["fsm_output"]
                            
                            # ✅ Handle nested JSON string (common in agent responses)
                            if isinstance(fsm_data, str):
                                try:
                                    fsm = json.loads(fsm_data)  # Parse the nested JSON string
                                except json.JSONDecodeError:
                                    fsm = fsm_data  # Keep as string if parse fails
                            else:
                                fsm = fsm_data
                        else:
                            fsm = parsed
                    else:
                        fsm = agent_result
                    
                    # ✅ Ensure FSM has proper structure
                    if isinstance(fsm, dict):
                        # Add debug logging
                        print(f"[DEBUG] FSM extracted successfully:")
                        print(f"[DEBUG] Name: {fsm.get('name', 'NOT_FOUND')}")
                        print(f"[DEBUG] Has finiteStateMachine: {'finiteStateMachine' in fsm}")
                        if 'finiteStateMachine' in fsm:
                            states = fsm.get('finiteStateMachine', {}).get('states', [])
                            print(f"[DEBUG] States count: {len(states)}")
                    
                    # ✅ Store as properly formatted JSON string
                    st.session_state.fsm_output = json.dumps(fsm, indent=2, ensure_ascii=False)
                    
                except Exception as e:
                    print(f"[ERROR] FSM extraction failed: {e}")
                    # ✅ Fallback formatting
                    try:
                        if isinstance(agent_result, dict):
                            st.session_state.fsm_output = json.dumps(agent_result, indent=2, ensure_ascii=False)
                        else:
                            st.session_state.fsm_output = str(agent_result)
                    except:
                        st.session_state.fsm_output = str(agent_result)

            # Determine sender for chat history
            sender = "validator_agent" if agent_result else "main_agent"
            st.session_state.chat_history.append({"role": "assistant", "sender": sender, "content": st.session_state.fsm_output})
            st.session_state.last_answered_query = prompt
            st.rerun()

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            error_msg = f"❌ Error generating FSM: {str(e)}\n\nDebug info:\n{error_trace}"
            print(f"[ERROR] {error_msg}")
            st.session_state.chat_history.append({
                "role": "assistant", 
                "sender": "validator_agent",
                "content": f"❌ Error generating FSM: {str(e)}"
            })
            st.rerun()

# --- Floating Chat Button (when chat is collapsed) ---
if not st.session_state.chat_expanded:
    # Create a placeholder for the floating button
    floating_button_placeholder = st.empty()

# Handle the chat toggle from URL parameter
if 'toggle_chat' in st.query_params:
    st.session_state.chat_expanded = not st.session_state.chat_expanded
    # Remove the query parameter without refreshing the page
    st.query_params.clear()
    st.rerun()

# --- Floating Chat Button (when chat is collapsed) ---
if not st.session_state.chat_expanded:
    # Create a placeholder for the floating button
    floating_button_placeholder = st.empty()
    st.rerun()

# --- Floating Chat Button (when chat is collapsed) ---
if not st.session_state.chat_expanded:
    # Create a placeholder for the floating button
    floating_button_placeholder = st.empty()
if not st.session_state.chat_expanded:
    # Create a placeholder for the floating button
    floating_button_placeholder = st.empty()
    st.rerun()

# --- Floating Chat Button (when chat is collapsed) ---
if not st.session_state.chat_expanded:
    # Create a placeholder for the floating button
    floating_button_placeholder = st.empty()

# --- Floating Chat Button (when chat is collapsed) ---
if not st.session_state.chat_expanded:
    # Create a placeholder for the floating button
    floating_button_placeholder = st.empty()

# --- Floating Chat Button (when chat is collapsed) ---
if not st.session_state.chat_expanded:
    # Create a placeholder for the floating button
    floating_button_placeholder = st.empty()

# --- Floating Chat Button (when chat is collapsed) ---
if not st.session_state.chat_expanded:
    # Create a placeholder for the floating button
    floating_button_placeholder = st.empty()
    floating_button_placeholder = st.empty()
    st.rerun()

# --- Floating Chat Button (when chat is collapsed) ---
if not st.session_state.chat_expanded:
    # Create a placeholder for the floating button
    floating_button_placeholder = st.empty()
if not st.session_state.chat_expanded:
    # Create a placeholder for the floating button
    floating_button_placeholder = st.empty()
    st.rerun()

# --- Floating Chat Button (when chat is collapsed) ---
if not st.session_state.chat_expanded:
    # Create a placeholder for the floating button
    floating_button_placeholder = st.empty()

# --- Floating Chat Button (when chat is collapsed) ---
if not st.session_state.chat_expanded:
    # Create a placeholder for the floating button
    floating_button_placeholder = st.empty()
    st.rerun()

# --- Floating Chat Button (when chat is collapsed) ---
if not st.session_state.chat_expanded:
    # Create a placeholder for the floating button
    floating_button_placeholder = st.empty()

# --- Floating Chat Button (when chat is collapsed) ---
if not st.session_state.chat_expanded:
    # Create a placeholder for the floating button
    floating_button_placeholder = st.empty()

# --- Floating Chat Button (when chat is collapsed) ---
if not st.session_state.chat_expanded:
    # Create a placeholder for the floating button
    floating_button_placeholder = st.empty()

# --- Floating Chat Button (when chat is collapsed) ---
if not st.session_state.chat_expanded:
    # Create a placeholder for the floating button
    floating_button_placeholder = st.empty()
    floating_button_placeholder = st.empty()
    st.rerun()

# --- Floating Chat Button (when chat is collapsed) ---
if not st.session_state.chat_expanded:
    # Create a placeholder for the floating button
    floating_button_placeholder = st.empty()
if not st.session_state.chat_expanded:
    # Create a placeholder for the floating button
    floating_button_placeholder = st.empty()
    st.rerun()

# --- Floating Chat Button (when chat is collapsed) ---
if not st.session_state.chat_expanded:
    # Create a placeholder for the floating button
    floating_button_placeholder = st.empty()

# --- Floating Chat Button (when chat is collapsed) ---
if not st.session_state.chat_expanded:
    # Create a placeholder for the floating button
    floating_button_placeholder = st.empty()
    st.rerun()

# --- Floating Chat Button (when chat is collapsed) ---
if not st.session_state.chat_expanded:
    # Create a placeholder for the floating button
    floating_button_placeholder = st.empty()

# --- Floating Chat Button (when chat is collapsed) ---
if not st.session_state.chat_expanded:
    # Create a placeholder for the floating button
    floating_button_placeholder = st.empty()




