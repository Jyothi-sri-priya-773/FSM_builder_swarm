import time
import json
import sqlite3  # Added for DB error handling
from memory import extractor
from State_builder_agent_1 import GPTClient

class QuerySegregator:
    def __init__(self, gpt_client):
        self.gpt_client = gpt_client

    def segregate_query(self, query):
        examples = """
        Examples:
        1. Query: "What is FSM?"
           Type: general_query
        2. Query: "List FSMs available"
           Type: general_query
        3. Query: "Explain FSM with examples."
           Type: general_query
        4. Query: "1. Read a static file named 'LoadRules.txt'.
                   2. Make a GET API call to the URL 'https://www.bing.com/search?q=temperature%20in%20bangalore&form=SWAUA2'."
           Type: fsm_use_case
        5. Query: "1. Add a key-value pair 'status: active' to a token.
                   2. Stop execution after adding a value to the token."
           Type: fsm_use_case
        6. Query: "1. Add the below states to the previous FSM:
                      - Transform a JSON document using a Jolt specification file named 'TransformSpec.json'.
                      - Add a key-value pair 'status: active' to a token."
           Type: add_states_to_previous_fsm
        7. Query: "1. Change the name of the previous FSM to 'FSM_name'.
                   2. Change the description of the above FSM to 'new_fsm'.
                   3. Change the file name to 'py.txt'."
           Type: alter_previous_fsm
        """
        prompt = (
            f"You are an intelligent assistant. Classify the following query as one of the following types: "
            f"'general_query', 'fsm_use_case', 'add_states_to_previous_fsm', or 'alter_previous_fsm'. "
            f"Use the examples below as a reference.\n\n"
            f"{examples}\n\n"
            f"Query: {query}\n\n"
            "Respond with only one of the following: 'general_query', 'fsm_use_case', 'add_states_to_previous_fsm', or 'alter_previous_fsm'.\n\n"
            "Important Instructions:\n"
            "1. If the query contains words or phrases like 'previous FSM', 'above FSM', 'given FSM', 'previously given FSM', or 'above', classify it as 'add_states_to_previous_fsm'.\n"
            "2. If the query does not refer to a previous FSM but describes tasks or operations to be performed, classify it as 'fsm_use_case'.\n"
            "3. If the query is a general question or request for information, classify it as 'general_query'.\n"
            "4. If the query involves modifying the structure, name, or description of an FSM, classify it as 'alter_previous_fsm'."
        )
        start_time = time.time()
        response = self.gpt_client.get_chat_completion([{"role": "user", "content": prompt}])
        if response is None:
            return "general_query"
        try:
            query_type = response.choices[0].message.content.strip().lower()
        except (AttributeError, IndexError):
            return "general_query"
        if query_type not in ["general_query", "fsm_use_case", "add_states_to_previous_fsm", "alter_previous_fsm"]:
            return "general_query"
        return query_type

def extract_fsm_details(query, fsm_memory, gpt_client, file_upload_mode=False, uploaded_fsm_json=None, uploaded_fsm_states=None, **kwargs):
    """
    Extract FSM details from the query and return them in a structured format.
    
    Args:
        query (str): The user's query
        fsm_memory: Instance of FSMMemory for conversation history
        gpt_client: Instance of GPTClient for AI completions
        file_upload_mode (bool): Whether this is a file upload operation
        uploaded_fsm_json: JSON content of uploaded FSM (if any)
        uploaded_fsm_states: States from uploaded FSM (if any)
        **kwargs: Additional arguments that might be passed by the agent framework
        
    Returns:
        dict: A dictionary containing query_type, fsm_details, and other relevant data
    """
    print(f"\n[PLANNING AGENT] Extracting FSM details for query: {query}")
    print(f"[PLANNING AGENT] Additional kwargs: {kwargs}")
    
    import os
    
    # Handle file upload case
    if query.strip().lower().startswith("file:") or (query.strip().endswith(".json") and os.path.exists(query.strip().strip('"'))):
        file_path = query.strip().replace("file:", "").strip().strip('"')
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
                    uploaded_fsm_json = file_content
                    uploaded_fsm_states = json.loads(file_content)
                    if isinstance(uploaded_fsm_states, dict) and "finiteStateMachine" in uploaded_fsm_states:
                        uploaded_fsm_states = uploaded_fsm_states["finiteStateMachine"].get("states", [])
                print("[PLANNING AGENT] Successfully processed file upload")
                return {
                    "query_type": "file_upload", 
                    "fsm_details": [], 
                    "uploaded_fsm_json": uploaded_fsm_json, 
                    "uploaded_fsm_states": uploaded_fsm_states
                }
            except Exception as e:
                error_msg = f"Error processing file: {str(e)}"
                print(f"[PLANNING AGENT] {error_msg}")
                return {"query_type": "file_upload", "fsm_details": [], "error": error_msg}
        else:
            error_msg = f"File not found: {file_path}"
            print(f"[PLANNING AGENT] {error_msg}")
            return {"query_type": "file_upload", "fsm_details": [], "error": error_msg}
    
    # Handle file upload mode
    if file_upload_mode:
        print("[PLANNING AGENT] Processing in file upload mode")
        return {
            "query_type": "file_upload", 
            "fsm_details": [], 
            "uploaded_fsm_json": uploaded_fsm_json, 
            "uploaded_fsm_states": uploaded_fsm_states
        }
    
    # Start processing regular query
    start_time = time.time()

    # Ensure the SQLite 'collections' table exists so downstream extractor/validator code doesn't crash.
    try:
        from memory import ensure_collections_table_exists
        ensure_collections_table_exists("fsm_memory.db")
    except Exception as e:
        print(f"[PLANNING AGENT] Warning: unable to initialise collections table: {e}")

    query_segregator = QuerySegregator(gpt_client)
    
    # Determine query type
    query_type = query_segregator.segregate_query(query)
    print(f"[PLANNING AGENT] Query type: {query_type}")
    
    # Handle general queries
    if query_type == "general_query":
        print("[PLANNING AGENT] Processing general query")
        try:
            response = extractor.query_fsm_data(query)
            fsm_memory.save_conversation(query, response)
            return {"query_type": query_type, "response": response}
        except Exception as e:
            error_msg = f"Error retrieving information for the query: {e}"
            print(f"[PLANNING AGENT] {error_msg}")
            return {"query_type": query_type, "response": error_msg}
    
    # Handle FSM use cases and state additions
    elif query_type in ["fsm_use_case", "add_states_to_previous_fsm"]:
        print(f"[PLANNING AGENT] Processing {query_type} query")
        
        # First try to get relevant FSM names (gracefully handle missing DB)
        try:
            fsm_names = extractor.query_fsm_data(
                f"List FSM names that are useful in performing the following tasks: {query}\n"
                "Return only a comma-separated list of FSM names, no other text."
            )
        except Exception as e:
            if isinstance(e, sqlite3.OperationalError) and "no such table" in str(e).lower():
                print("[PLANNING AGENT] collections table missing – proceeding with empty FSM list")
                fsm_names = ""
            else:
                print(f"[PLANNING AGENT] query_fsm_data error: {e}")
                fsm_names = ""
        
        fsm_details = []
        
        # If no specific FSMs found, try fallback to all templates
        if not fsm_names or fsm_names.strip() == '':
            print("[PLANNING AGENT] No specific FSMs found, falling back to all templates")
            try:
                html_content = extractor.fetch_data_from_url()
                all_templates = extractor.extract_fsm_templates(html_content)
                
                for tpl in all_templates:
                    fsm_details.append({
                        "fsm_name": tpl.get("fsmName", ""),
                        "schema": tpl.get("schema", "Schema not found"),
                        "example": tpl.get("example", "Example not found"),
                        "functionality": tpl.get("functionality", "Functionality not found")
                    })
                print(f"[PLANNING AGENT] Found {len(fsm_details)} FSM templates in fallback")
                
            except Exception as e:
                error_msg = f"Fallback FSM template extraction failed: {e}"
                print(f"[PLANNING AGENT] {error_msg}")
                fsm_details = []
                
            # Return early with fallback details if no specific FSMs found
            return {
                "query_type": query_type, 
                "fsm_details": fsm_details,
                "fallback_used": True
            }
        processed_fsms = set()
        for fsm_name in fsm_names.split(","):
            fsm_name = fsm_name.strip()
            
            # Skip empty names and duplicates
            if not fsm_name or fsm_name in processed_fsms:
                continue
                
            processed_fsms.add(fsm_name)
            print(f"[PLANNING AGENT] Processing FSM: {fsm_name}")
            
            try:
                # Construct query to get FSM details
                combined_query = (
                    f"Provide the schema, example, and functionality for the FSM named exactly '{fsm_name}'. "
                    f"Respond in the following format:\n"
                    f"Schema: <schema>\nExample: <example>\nFunctionality: <functionality>\n"
                    f"IMPORTANT: For any state that uses a 'transition', the schema for 'transition' is:\n"
                    f'"transition": {{\n'
                    f'  "type": "object",\n'
                    f'  "properties": {{\n'
                    f'    "conditions": {{\n'
                    f'      "type": "array",\n'
                    f'      "items": {{ "type": "object" }}\n'
                    f'    }}\n'
                    f'  }},\n'
                    f'  "required": ["conditions"],\n'
                    f'  "description": "Optional - List of transitions applicable for the state if any. If not provided, the next state defined will be executed."\n'
                    f'}}\n'
                )
                
                # Get response from extractor
                print(f"[PLANNING AGENT] Fetching details for FSM: {fsm_name}")
                response = extractor.query_fsm_data(combined_query)
                
                # Parse the response
                schema = None
                example = None
                functionality = None
                current_field = None
                buffer = []
                
                # Split response into lines and process each line
                lines = response.splitlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    if line.lower().startswith("schema:"):
                        if current_field == "example":
                            example = "\n".join(buffer).strip()
                        elif current_field == "functionality":
                            functionality = "\n".join(buffer).strip()
                        buffer = [line[7:].strip()]  # Remove "Schema:" prefix
                        current_field = "schema"
                    elif line.lower().startswith("example:"):
                        if current_field == "schema":
                            schema = "\n".join(buffer).strip()
                        elif current_field == "functionality":
                            functionality = "\n".join(buffer).strip()
                        buffer = [line[8:].strip()]  # Remove "Example:" prefix
                        current_field = "example"
                    elif line.lower().startswith("functionality:"):
                        if current_field == "schema":
                            schema = "\n".join(buffer).strip()
                        elif current_field == "example":
                            example = "\n".join(buffer).strip()
                        buffer = [line[13:].strip()]  # Remove "Functionality:" prefix
                        current_field = "functionality"
                    else:
                        buffer.append(line)
                
                # Process the last section
                if current_field == "schema":
                    schema = "\n".join(buffer).strip()
                elif current_field == "example":
                    example = "\n".join(buffer).strip()
                elif current_field == "functionality":
                    functionality = "\n".join(buffer).strip()
                
                # PATCH: If the example field looks like a full FSM JSON, parse and return it as a dict
                try:
                    example_obj = None
                    if example:
                        example_str = example.strip()
                        # Remove markdown code block if present
                        if example_str.startswith('```json'):
                            example_str = example_str[7:].strip()
                        elif example_str.startswith('```'):
                            example_str = example_str[3:].strip()
                        # Now try to parse as JSON
                        if example_str.startswith('{') and 'finiteStateMachine' in example_str:
                            example_obj = json.loads(example_str)
                    if example_obj and isinstance(example_obj, dict) and 'finiteStateMachine' in example_obj:
                        print(f"[PLANNING AGENT] Detected full FSM JSON in example for {fsm_name}, returning as full FSM JSON.")
                        fsm_details.append(example_obj)
                        continue  # Skip adding the text-based detail
                except Exception as e:
                    print(f"[PLANNING AGENT] Failed to parse example as FSM JSON: {e}\nExample string was: {example}")
                
                # Create FSM detail entry
                fsm_detail = {
                    "fsm_name": fsm_name,
                    "schema": schema if schema else "Schema not found",
                    "example": example if example else "Example not found",
                    "functionality": functionality if functionality else "Functionality not found"
                }
                print(f"[PLANNING AGENT] Extracted FSM detail: {fsm_name}")
                fsm_details.append(fsm_detail)
                
            except Exception as e:
                error_msg = f"Error processing FSM '{fsm_name}': {e}"
                print(f"[PLANNING AGENT] {error_msg}")
                fsm_details.append({
                    "fsm_name": fsm_name,
                    "schema": "Schema not found",
                    "example": "Example not found",
                    "functionality": f"Error: {str(e)}"
                })
        
        # Log the number of FSM details found
        print(f"[PLANNING AGENT] Extracted {len(fsm_details)} FSM details")
        
        # Save conversation to memory if possible
        if hasattr(fsm_memory, 'save_conversation') and callable(getattr(fsm_memory, 'save_conversation', None)):
            try:
                conversation_data = {
                    "query": query,
                    "fsm_details": fsm_details,
                    "query_type": query_type,
                    "timestamp": time.time()
                }
                fsm_memory.save_conversation(query, json.dumps(conversation_data))
                print("[PLANNING AGENT] Saved conversation to memory")
            except Exception as e:
                print(f"[PLANNING AGENT] Error saving conversation to memory: {e}")
        
        # Prepare suggestions for UI review
        review_items = []
        for detail in fsm_details:
            name = detail.get('fsm_name', '')
            review_items.append({'fsm_name': name, 'description': detail.get('functionality', '')})
        print(f"[PLANNING AGENT] Returning FSM suggestions: {review_items}")
        return {
            'query_type': query_type,
            'fsm_suggestions': review_items,
            'success': len(review_items) > 0,
            'timestamp': time.time()
        }
    
    elif query_type == "alter_previous_fsm":
        print("[PLANNING AGENT] Processing alter_previous_fsm query")
        return {
            "query_type": query_type,
            "fsm_details": [],
            "success": True,
            "timestamp": time.time()
        }
    else:
        print(f"[PLANNING AGENT] Unknown query type: {query_type}")
        return {
            "query_type": "unknown",
            "response": "Unknown query type.",
            "success": False,
            "timestamp": time.time()
        }

if __name__ == "__main__":
    # Example: Fetch FSM details for a string manipulation query using the planning agent
    from memory import FSMMemory
    
    # You must provide a real or mock GPTClient here. Replace with your actual GPTClient if needed.
    class DummyGPTClient:
        def get_chat_completion(self, messages):
            # Always classify as 'fsm_use_case' for this test
            class DummyResponse:
                class Choices:
                    class Message:
                        content = 'fsm_use_case'
                    message = Message()
                choices = [Choices()]
            return DummyResponse()
    
    gpt_client = DummyGPTClient()
    fsm_memory = FSMMemory()
    test_query = "create an fsm such that it Takes the input string from json , extract the first 5 characters starting from index 0, construct a response by appending 'The manipulated string is ' to the header and result to the footer."
    result = extract_fsm_details(test_query, fsm_memory, gpt_client)
    print("\n[PLANNING AGENT TEST] FSM Details Result:")
    print(json.dumps(result, indent=2))