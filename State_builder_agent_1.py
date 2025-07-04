import json
import os
from dotenv import load_dotenv, find_dotenv
import httpx
from openai import AzureOpenAI
import time
import re
from concurrent.futures import ThreadPoolExecutor
from memory import get_fsm_schemas

def extract_json_from_markdown(text):
    """
    Extract JSON from markdown code blocks.
    Handles both ```json and ``` blocks.
    """
    if not text or not isinstance(text, str):
        return text
        
    # Try to find JSON in markdown code blocks first
    json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', text)
    if json_match:
        return json_match.group(1)
        
    # If no code blocks, try to find JSON object directly
    json_match = re.search(r'({[\s\S]*})', text)
    if json_match:
        return json_match.group(1)
        
    return text


class GPTClient:
    """
    A client for interacting with Azure OpenAI GPT models.
    """

    def __init__(self):
        """
        Initialize the GPT client for interacting with Azure OpenAI.
        """
        load_dotenv(find_dotenv())
        self.api_key = os.getenv("GPT4O_AZURE_OPENAI_KEY")
        self.endpoint = os.getenv("GPT4O_AZURE_OPENAI_ENDPOINT")
        self.api_version = os.getenv("GPT4O_AZURE_API_VERSION")
        self.deployment_model_name = os.getenv("GPT4O_MODEL_DEPLOYMENT_NAME")

        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
            http_client=httpx.Client(verify=False)
        )

    def get_chat_completion(self, prompts):
        """
        Get a chat completion response from the GPT model.

        Args:
            prompts (list): The list of prompts to send to the GPT model.

        Returns:
            dict: The response from the GPT model.
        """
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.deployment_model_name,
                messages=prompts,
                temperature=0,
                top_p=0.9,
                max_tokens=2000
            )
            print(f"GPT client response completed in {time.time() - start_time:.2f} seconds.")
            return response
        except Exception as e:
            print(f"Error in get_chat_completion: {e}")
            return None


# Add this constant for strict JSON output enforcement
STRICT_JSON_ONLY = (
    "IMPORTANT: Output ONLY valid JSON. Do NOT output any text, markdown, or explanation. If you cannot generate a valid FSM, output an empty JSON object {} only."
)

# Simple prompt for LLM to avoid common JSON mistakes:
LLM_JSON_PROMPT = (
    "You must output ONLY valid JSON. Do not include any explanations, markdown, or comments.\n"
    "Strictly follow these rules:\n"
    "- Do NOT put a comma after the last property in any object or array.\n"
    '- Enclose ALL property names in double quotes (e.g., \"key\": \"value\").\n'
    '- Enclose ALL string values in double quotes.\n'
    "- Do NOT use single quotes for property names or values.\n"
    "- Do NOT include comments or explanations in the output.\n"
    "- Do NOT output Python-style values (use true/false/null, not True/False/None).\n"
    "- Do NOT output partial or incomplete JSON.\n"
    "If you are unsure, output a minimal valid JSON object. Your output must always be valid JSON."
)


class StateBuilderAgent:
    """
    A class for building or extending FSMs based on user queries.
    """

    def __init__(self, client):
        """
        Initialize the StateBuilderAgent.

        Args:
            client (GPTClient): The GPT client for interacting with the language model.
        """
        self.client = client

    def build_fsm(
        self,
        query,
        fsm_details=None,  # Make optional with default None
        context=None,
        query_type=None,
        uploaded_fsm_states=None,
        **kwargs  # Catch any additional kwargs
    ):
        """
        Build or modify an FSM based on the query and FSM details.
        
        Args:
            query (str): The user's query
            fsm_details (list, optional): List of FSM details from planning agent
            context (list, optional): Additional context for the query
            query_type (str, optional): Type of query (e.g., 'file_upload', 'add_states_to_previous_fsm', 'alter_previous_fsm')
            uploaded_fsm_states (list, optional): States from uploaded FSM
            **kwargs: Additional arguments
            
        Returns:
            dict: The built or modified FSM with a 'states' key
        """
        print(f"\n[STATE_BUILDER] ===== Building FSM =====")
        print(f"[STATE_BUILDER] Query: {query}")
        
        # Log received kwargs for debugging
        if kwargs:
            print(f"[STATE_BUILDER] Additional kwargs: {list(kwargs.keys())}")
        
        # Try to get query_type from various sources
        original_query_type = query_type
        
        # 1. Check kwargs first (highest priority)
        if not query_type and 'query_type' in kwargs:
            query_type = kwargs['query_type']
            print(f"[STATE_BUILDER] Using query_type from kwargs: {query_type}")
        
        # 2. Check context (if it's a list of messages)
        if not query_type and context and isinstance(context, list):
            for msg in reversed(context):
                if isinstance(msg, dict) and 'query_type' in msg:
                    query_type = msg['query_type']
                    print(f"[STATE_BUILDER] Using query_type from context message: {query_type}")
                    break
        # If context is a dict
        elif not query_type and isinstance(context, dict) and 'query_type' in context:
            query_type = context['query_type']
            print(f"[STATE_BUILDER] Using query_type from context dict: {query_type}")
        
        # 3. Check kwargs['kwargs']
        if not query_type and 'kwargs' in kwargs and isinstance(kwargs['kwargs'], dict):
            query_type = kwargs['kwargs'].get('query_type')
            if query_type:
                print(f"[STATE_BUILDER] Using query_type from kwargs['kwargs']: {query_type}")
        
        # 4. If still not found, use the original query_type or default to 'fsm_use_case'
        query_type = str(query_type).strip() if query_type else 'fsm_use_case'
        
        # Log the final query_type being used
        if original_query_type != query_type:
            print(f"[STATE_BUILDER] Query type changed from '{original_query_type}' to '{query_type}'")
        
        # Ensure fsm_details is always a list (never None)
        if fsm_details is None:
            fsm_details = []
            print("[STATE_BUILDER] Warning: fsm_details was None, using empty list")
        
        # Ensure fsm_details is a list
        if not isinstance(fsm_details, list):
            print(f"[STATE_BUILDER] Warning: fsm_details is not a list, converting to list")
            fsm_details = [fsm_details] if fsm_details is not None else []
        
        # Validate fsm_details structure with more detailed logging
        valid_fsm_details = []
        print(f"[STATE_BUILDER] Processing {len(fsm_details)} FSM details...")
        
        for i, detail in enumerate(fsm_details, 1):
            try:
                if not isinstance(detail, dict):
                    print(f"[STATE_BUILDER] Warning: fsm_details[{i}] is not a dictionary (got {type(detail).__name__}), skipping")
                    continue
                
                # Log the raw detail for debugging
                print(f"\n[STATE_BUILDER] FSM Detail {i} keys: {list(detail.keys())}")
                
                # Get or generate a name for the FSM
                fsm_name = str(detail.get('fsm_name', f'Unnamed_FSM_{i}')).strip()
                
                # Get schema with fallback to empty object if not provided
                schema = detail.get('schema', '{}')
                if schema in ['No schema provided', '{}', '']:
                    print(f"[STATE_BUILDER] Using default schema for {fsm_name}")
                    schema = '{"type": "object", "properties": {}}'
                
                # Get example with fallback to empty object if not provided
                example = detail.get('example', '{}')
                if example in ['No example provided', '{}', '']:
                    print(f"[STATE_BUILDER] Using default example for {fsm_name}")
                    example = '{"example": "No example provided"}'
                
                # Get functionality with fallback to default if not provided
                functionality = detail.get('functionality', '')
                if not functionality or functionality == 'No functionality provided':
                    functionality = f'Functionality for {fsm_name}'
                    print(f"[STATE_BUILDER] Using default functionality for {fsm_name}")
                
                # PATCH: If this detail looks like a full FSM JSON, log and use it directly
                if (
                    'finiteStateMachine' in detail and
                    isinstance(detail['finiteStateMachine'], dict) and
                    'states' in detail['finiteStateMachine']
                ):
                    print(f"[STATE_BUILDER] Received FULL FSM JSON for {fsm_name}!")
                    print(f"  Top-level keys: {list(detail.keys())}")
                    print(f"  States count: {len(detail['finiteStateMachine']['states'])}")
                    valid_fsm_details.append(detail)
                    print(f"[STATE_BUILDER] Added valid FULL FSM JSON: {fsm_name}")
                    continue
                
                # Create a clean detail dict with all original fields plus cleaned ones
                clean_detail = {
                    **detail,  # Include all original fields
                    'fsm_name': fsm_name,
                    'schema': str(schema).strip(),
                    'example': str(example).strip(),
                    'functionality': str(functionality).strip()
                }
                
                # Log the cleaned detail for debugging
                print(f"[STATE_BUILDER] Cleaned FSM detail for {fsm_name}:")
                for k, v in clean_detail.items():
                    if k in ['schema', 'example'] and len(str(v)) > 100:
                        print(f"  {k}: {str(v)[:100]}...")
                    else:
                        print(f"  {k}: {v}")
                
                valid_fsm_details.append(clean_detail)
                print(f"[STATE_BUILDER] Added valid FSM: {fsm_name}")
                
            except Exception as e:
                print(f"[STATE_BUILDER] Error processing FSM detail {i}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        filtered_count = len(fsm_details) - len(valid_fsm_details)
        if filtered_count > 0:
            print(f"[STATE_BUILDER] Filtered out {filtered_count} invalid FSM details")
        
        fsm_details = valid_fsm_details
        
        # If we have no valid FSM details, create a default one based on the query
        if not fsm_details:
            print("[STATE_BUILDER] WARNING: No valid FSM details found, creating default FSM based on query")
            default_fsm = {
                'fsm_name': 'defaultFSM',
                'schema': '{"type": "object", "properties": {"input": {"type": "object"}, "output": {"type": "object"}}}',
                'example': '{"input": {}, "output": {}}',
                'functionality': f'Default FSM for query: {query[:100]}...' if len(query) > 100 else f'Default FSM for query: {query}'
            }
            fsm_details = [default_fsm]
            print(f"[STATE_BUILDER] Created default FSM: {default_fsm['fsm_name']}")
        else:
            print(f"[STATE_BUILDER] Processing with {len(fsm_details)} valid FSM details")
        
        # Get the valid state types from the schema with better error handling
        try:
            print("\n[STATE_BUILDER] Loading FSM schemas...")
            schemas = get_fsm_schemas()
            if not schemas or not isinstance(schemas, dict):
                print("[STATE_BUILDER] ERROR: get_fsm_schemas() returned empty or invalid result")
                return {"error": "Failed to load FSM schemas", "states": []}
                
            valid_state_types = list(schemas.keys())
            if not valid_state_types:
                print("[STATE_BUILDER] WARNING: No valid state types found in schemas")
            
            valid_types_bullet = '\n'.join(f'- {t}' for t in valid_state_types)
            print(f"[STATE_BUILDER] Loaded {len(valid_state_types)} valid state types")
            
            # Log first few state types for verification
            if valid_state_types:
                sample = ', '.join(valid_state_types[:3])
                if len(valid_state_types) > 3:
                    sample += f" ... and {len(valid_state_types) - 3} more"
                print(f"[STATE_BUILDER] State types: {sample}")
                
        except Exception as e:
            error_msg = f"Error loading FSM schemas: {str(e)}"
            print(f"\n[STATE_BUILDER] {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "states": []
            }
        
        # Build the instruction with FSM details if available
        fsm_context = ""
        if fsm_details:
            fsm_context = "\n\nFSM DETAILS FROM PLANNING AGENT:"
            for i, detail in enumerate(fsm_details, 1):
                fsm_name = detail.get('fsm_name', f'Unnamed_FSM_{i}')
                functionality = detail.get('functionality', 'No functionality provided')
                fsm_context += f"\n\nFSM {i}: {fsm_name}"
                fsm_context += f"\nFunctionality: {functionality}"
                
                # Include schema and example if they're not too large
                schema = detail.get('schema', '')
                if schema and len(str(schema)) < 500:  # Only include if not too large
                    fsm_context += f"\nSchema: {schema}"
                    
                example = detail.get('example', '')
                if example and len(str(example)) < 500:  # Only include if not too large
                    fsm_context += f"\nExample: {example}"
        
        valid_types_instruction = (
            f"IMPORTANT: You are building a Finite State Machine (FSM) based on the following requirements.\n"
            f"You MUST use only the following valid state types for the 'type' field in each state:\n"
            f"{valid_types_bullet}\n\n"
            f"{fsm_context}\n\n"
            "RULES FOR FSM CONSTRUCTION:\n"
            "1. Output MUST be a valid JSON object with a 'states' array containing all states.\n"
            "2. Each state MUST have: 'id', 'type', and 'configuration' fields.\n"
            "3. The 'type' field MUST be one of the valid state types listed above.\n"
            "4. For state transitions, use: {{\"transition\": {{\"conditions\": [{\"if\": ..., \"operation\": ..., \"params\": {{...}}, \"thenState\": ...}]}}}\n"
            "5. When saving output, use 'saveStateOutput' or 'saveStateOutputWithKey' at the same level as 'id' and 'type'.\n"
            "6. NEVER use 'condition' or 'transitionTo' keys - always use 'if', 'thenState', etc.\n"
            "7. All state types and field names MUST match the schema exactly (case-sensitive).\n"
            "8. If the schema requires a field, it MUST be present in the output.\n\n"
            "EXAMPLE of valid FSM structure:\n"
            "{\n"
            "  \"states\": [\n"
            "    {\n"
            "      \"id\": \"StartState\",\n"
            "      \"type\": \"httpRequest\",\n"
            "      \"configuration\": {\n"
            "        \"method\": \"GET\",\n"
            "        \"url\": \"https://api.example.com/data\"\n"
            "      },\n"
            "      \"transition\": {\n"
            "        \"conditions\": [\n"
            "          {\n"
            "            \"if\": \"isEqual\"\n"
            "          }\n"
            "        ]\n"
            "      }\n"
            "    }\n"
            "  ]\n"
            "}\n"
        )

        system_prompt = {
            "role": "system",
            "content": (
                f"{STRICT_JSON_ONLY}\n"
                "You are an expert in building Finite State Machines (FSMs). Your task is to create or modify an FSM "
                "based on the user's requirements and the provided FSM details.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. Output ONLY valid JSON - no markdown, explanations, or other text.\n"
                "2. Follow the schema and examples EXACTLY - including all required fields and correct data types.\n"
                "3. Use only the state types and fields defined in the schema.\n"
                f"{valid_types_instruction}\n"
                f"{STRICT_JSON_ONLY}"
            )
        }

        # Build FSM info from details with better error handling
        fsm_info = []
        for i, fsm in enumerate(fsm_details):
            try:
                if not isinstance(fsm, (dict, str)):
                    print(f"[STATE_BUILDER] Warning: fsm_details[{i}] is not a dictionary or string (got {type(fsm).__name__})")
                    continue
                
                # If fsm is a string, try to parse it as JSON
                if isinstance(fsm, str):
                    try:
                        fsm = json.loads(fsm)
                        if not isinstance(fsm, dict):
                            print(f"[STATE_BUILDER] Warning: Parsed fsm_details[{i}] is not a dictionary")
                            continue
                    except json.JSONDecodeError:
                        print(f"[STATE_BUILDER] Warning: Could not parse fsm_details[{i}] as JSON")
                        continue
                
                # Safely get values with proper type checking
                fsm_name = str(fsm.get('fsm_name', f'Unnamed_FSM_{i}')).strip()
                schema = fsm.get('schema', {})
                example = fsm.get('example', {})
                
                # Convert schema and example to strings if they're not already
                if isinstance(schema, dict):
                    try:
                        schema_str = json.dumps(schema, indent=2)
                    except (TypeError, ValueError):
                        schema_str = str(schema)
                else:
                    schema_str = str(schema)
                
                if isinstance(example, dict):
                    try:
                        example_str = json.dumps(example, indent=2)
                    except (TypeError, ValueError):
                        example_str = str(example)
                else:
                    example_str = str(example)
                
                fsm_info.append(
                    f"FSM Detail {i+1}:"
                    f"\n- Name: {fsm_name}"
                    f"\n- Schema: {schema_str}"
                    f"\n- Example: {example_str}"
                )
                
            except Exception as e:
                print(f"[STATE_BUILDER] Error processing FSM detail {i}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        fsm_info_str = "\n\n".join(fsm_info) if fsm_info else "No valid FSM details provided"
        print(f"[STATE_BUILDER] Processed {len(fsm_info)} valid FSM details")

        # --- Query type handling ---
        if query_type == "alter_previous_fsm":
            print("[STATE_BUILDER] Processing alter_previous_fsm query type")
            
            # First check if previous_fsm_json is in kwargs
            previous_fsm_json = kwargs.get('previous_fsm_json')
            previous_fsm_source = 'kwargs'
            
            # If not in kwargs, try to extract from context
            if not previous_fsm_json and context:
                try:
                    if isinstance(context, list):
                        for entry in reversed(context):
                            if isinstance(entry, dict) and entry.get("role") == "assistant":
                                content = entry.get("content", "")
                                if content and isinstance(content, str):
                                    previous_fsm_json = content
                                    previous_fsm_source = 'context list assistant content'
                                    print("[STATE_BUILDER] Found previous FSM in context list assistant content")
                                    break
                    elif isinstance(context, dict):
                        # If context is a dict, check for common keys that might contain the FSM
                        for key in ['fsm', 'previous_fsm', 'content', 'data', 'previous_fsm_json']:
                            if key in context and context[key]:
                                if isinstance(context[key], str):
                                    previous_fsm_json = context[key]
                                elif isinstance(context[key], dict) and 'states' in context[key]:
                                    # If it's already a dict with states, use it directly
                                    previous_fsm_json = json.dumps(context[key])
                                if previous_fsm_json:
                                    previous_fsm_source = f'context dict key: {key}'
                                    print(f"[STATE_BUILDER] Found previous FSM in context['{key}']")
                                    break
                except Exception as e:
                    print(f"[STATE_BUILDER] Error extracting previous FSM from context: {str(e)}")
            
            # Log if we found a previous FSM and its source
            if previous_fsm_json:
                print(f"[STATE_BUILDER] Using previous FSM from {previous_fsm_source}")
                if len(str(previous_fsm_json)) > 200:
                    print(f"[STATE_BUILDER] Previous FSM (truncated): {str(previous_fsm_json)[:200]}...")
                else:
                    print(f"[STATE_BUILDER] Previous FSM: {previous_fsm_json}")
            else:
                print("[STATE_BUILDER] No previous FSM found in kwargs or context")
            
            # Build conversation context for the prompt with better error handling
            try:
                if isinstance(context, list):
                    conversation_context = [
                        f"{entry.get('role', 'unknown')}: {entry.get('content', '')}" 
                        for entry in context 
                        if isinstance(entry, dict)
                    ]
                elif isinstance(context, dict):
                    # If context is a dict, convert it to a list of strings
                    conversation_context = [f"{k}: {v}" for k, v in context.items()]
                else:
                    conversation_context = []
                
                print(f"[STATE_BUILDER] Processed {len(conversation_context)} context entries")
                
                # Log first few context entries for debugging
                if conversation_context:
                    sample = "\n".join(conversation_context[:3])
                    if len(conversation_context) > 3:
                        sample += "\n..."
                    print(f"[STATE_BUILDER] Context sample:\n{sample}")
                
            except Exception as e:
                error_msg = f"Error processing context: {str(e)}"
                print(f"[STATE_BUILDER] {error_msg}")
                import traceback
                traceback.print_exc()
                conversation_context = [error_msg]
            
            # Create system prompt for FSM modification
            system_prompt = {
                "role": "system",
                "content": (
                    f"{STRICT_JSON_ONLY}\n"
                    "You are an expert in modifying existing FSMs based on user queries.\n\n"
                    "INSTRUCTIONS:\n"
                    "1. Carefully analyze the conversation history to understand the required changes.\n"
                    "2. Make ONLY the specific changes requested by the user.\n"
                    "3. Preserve all existing states and transitions that aren't explicitly modified.\n"
                    "4. Ensure the output is valid JSON that strictly follows the schema.\n"
                    f"{valid_types_instruction}\n"
                    f"{STRICT_JSON_ONLY}"
                )
            }
            
            # Create user prompt with clear instructions
            user_prompt = {
                "role": "user",
                "content": (
                    f"USER REQUEST: {query}\n\n"
                    f"CONVERSATION HISTORY:\n{chr(10).join(conversation_context)}\n\n"
                    f"FSM DETAILS:\n{fsm_info_str}\n\n"
                    "TASK: Modify the FSM according to the user's request. "
                    "Ensure the output is valid JSON that strictly follows the schema.\n"
                    f"{STRICT_JSON_ONLY}"
                )
            }
            
            print(f"[STATE_BUILDER] Sending request to LLM for FSM modification")
            
            try:
                # Send request to LLM
                start_time = time.time()
                response = self.client.get_chat_completion([system_prompt, user_prompt])
                elapsed = time.time() - start_time
                print(f"[STATE_BUILDER] LLM response received in {elapsed:.2f} seconds")
                
                if not response or not response.choices:
                    print("[STATE_BUILDER] Error: Empty or invalid response from LLM")
                    return {"states": []}
                
                # Extract and validate the response
                fsm_output = response.choices[0].message.content.strip()
                print(f"[STATE_BUILDER] Raw LLM response length: {len(fsm_output)} characters")
                
                # Try to extract JSON from the response
                try:
                    # First try to parse the entire output as JSON
                    try:
                        fsm_json = json.loads(fsm_output)
                        print("[STATE_BUILDER] Successfully parsed entire output as JSON")
                        return fsm_json
                    except json.JSONDecodeError:
                        print("[STATE_BUILDER] Could not parse LLM output as direct JSON, trying markdown extraction...")
                    
                    # Try to extract JSON from markdown
                    try:
                        json_str = extract_json_from_markdown(fsm_output)
                        if json_str and json_str != fsm_output:  # Only try to parse if we extracted something
                            fsm_json = json.loads(json_str)
                            print("[STATE_BUILDER] Successfully parsed JSON from markdown")
                            return fsm_json
                    except json.JSONDecodeError as e:
                        print(f"[STATE_BUILDER] Could not parse extracted JSON: {str(e)}")
                    
                    # If we get here, we couldn't parse valid JSON
                    print("[STATE_BUILDER] No valid JSON found in LLM response")
                    print(f"[STATE_BUILDER] Response start: {fsm_output[:200]}...")
                    
                except Exception as e:
                    print(f"[STATE_BUILDER] Error processing LLM response: {str(e)}")
                
                # If we get here, return an empty FSM
                return {"states": []}
                
            except Exception as e:
                print(f"[STATE_BUILDER] Error during LLM API call: {str(e)}")
                return {"states": []}

        elif query_type == "add_states_to_previous_fsm":
            print("\n[STATE_BUILDER] ===== Adding States to Previous FSM =====")
            print(f"[STATE_BUILDER] Query: {query}")
            
            # Extract previous FSM from context if available
            previous_fsm = {}
            try:
                for entry in reversed(context):
                    if isinstance(entry, dict) and entry.get("role") == "assistant":
                        previous_content = entry.get("content", "")
                        if previous_content:
                            try:
                                previous_fsm = json.loads(previous_content)
                                print("[STATE_BUILDER] Found previous FSM in context")
                                break
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                print(f"[STATE_BUILDER] Error extracting previous FSM from context: {str(e)}")
            
            # Build conversation context for the prompt
            conversation_context = []
            try:
                conversation_context = [
                    f"{entry.get('role', 'unknown')}: {entry.get('content', '')}" 
                    for entry in context 
                    if isinstance(entry, dict)
                ]
                print(f"[STATE_BUILDER] Processed {len(conversation_context)} context entries")
            except Exception as e:
                print(f"[STATE_BUILDER] Error processing context: {str(e)}")
                conversation_context = ["Error processing conversation context"]
            
            # Create system prompt for adding states
            system_prompt = {
                "role": "system",
                "content": (
                    f"{STRICT_JSON_ONLY}\n"
                    "You are an expert in extending Finite State Machines (FSMs) by adding new states.\n\n"
                    "YOUR TASK:\n"
                    "1. Analyze the user's request to understand what new states need to be added.\n"
                    "2. Use the provided FSM schema and examples to ensure the new states are valid.\n"
                    "3. Add the new states to the existing FSM while preserving all existing states and transitions.\n"
                    f"{valid_types_instruction}\n"
                    "5. Ensure the output is valid JSON that strictly follows the schema.\n"
                    f"{STRICT_JSON_ONLY}"
                )
            }
            
            # Create user prompt with clear instructions
            user_prompt = {
                "role": "user",
                "content": (
                    f"USER REQUEST: {query}\n\n"
                    f"EXISTING FSM: {json.dumps(previous_fsm, indent=2) if previous_fsm else 'No previous FSM found'}\n\n"
                    f"FSM DETAILS:\n{fsm_info_str}\n\n"
                    f"CONVERSATION HISTORY:\n{chr(10).join(conversation_context)}\n\n"
                    "INSTRUCTIONS FOR NEW STATES:\n"
                    "1. Add only the states explicitly requested by the user.\n"
                    "2. Ensure each state has a unique ID in camelCase.\n"
                    "3. Include only fields mentioned in the user query.\n"
                    "4. For configurations, use only fields from the provided FSM examples.\n"
                    "5. Preserve all existing states and transitions.\n"
                    "6. The output must be a complete, valid FSM JSON.\n\n"
                    "OUTPUT REQUIREMENTS:\n"
                    "1. Output ONLY valid JSON with no additional text.\n"
                    "2. Include all required FSM fields: 'name', 'description', 'version', and 'finiteStateMachine'.\n"
                    "3. Each state must have 'id', 'type', and 'configuration' fields.\n"
                    "4. Use camelCase for all field names.\n"
                    f"{STRICT_JSON_ONLY}"
                )
            }
            
            print("[STATE_BUILDER] Sending request to LLM to add states to FSM")
            
            try:
                # Send request to LLM
                start_time = time.time()
                response = self.client.get_chat_completion([system_prompt, user_prompt])
                elapsed = time.time() - start_time
                print(f"[STATE_BUILDER] LLM response received in {elapsed:.2f} seconds")
                
                if not response or not response.choices:
                    print("[STATE_BUILDER] Error: Empty or invalid response from LLM")
                    return previous_fsm if previous_fsm else {"states": []}
                
                # Extract and validate the response
                fsm_output = response.choices[0].message.content.strip()
                print(f"[STATE_BUILDER] Raw LLM response length: {len(fsm_output)} characters")
                
                # Try to extract and parse JSON from the response
                try:
                    # First try to parse the entire output as JSON
                    try:
                        fsm_json = json.loads(fsm_output)
                        print("[STATE_BUILDER] Successfully parsed entire output as JSON")
                        return self._validate_fsm_structure(fsm_json, previous_fsm)
                    except json.JSONDecodeError:
                        print("[STATE_BUILDER] Could not parse LLM output as direct JSON, trying markdown extraction...")
                    
                    # Try to extract JSON from markdown
                    try:
                        json_str = extract_json_from_markdown(fsm_output)
                        if json_str and json_str != fsm_output:  # Only try to parse if we extracted something
                            fsm_json = json.loads(json_str)
                            print("[STATE_BUILDER] Successfully parsed JSON from markdown")
                            return self._validate_fsm_structure(fsm_json, previous_fsm)
                    except json.JSONDecodeError as e:
                        print(f"[STATE_BUILDER] Could not parse extracted JSON: {str(e)}")
                    
                    # If we get here, we couldn't parse valid JSON
                    print("[STATE_BUILDER] No valid JSON found in LLM response")
                    print(f"[STATE_BUILDER] Response start: {fsm_output[:200]}...")
                    
                except Exception as e:
                    print(f"[STATE_BUILDER] Error processing LLM response: {str(e)}")
                
                # If we get here, return the previous FSM or empty FSM
                return previous_fsm if previous_fsm else {"states": []}
                
            except Exception as e:
                print(f"[STATE_BUILDER] Error during LLM API call: {str(e)}")
                return previous_fsm if previous_fsm else {"states": []}

        if query_type == "file_upload":
            # context is a list of state dicts (states before the insertion point)
            if not isinstance(context, list):
                context = []
            system_prompt = {
                "role": "system",
                "content": (
                    "You are an expert in updating FSMs. Use ONLY the provided FSM schema, examples, and the FSM states from the uploaded file as context. "
                    + valid_types_instruction + "\n"
                    "Do NOT use any conversation history or previous FSMs from chat. "
                    "Generate ONLY the new state(s) described in the user query, as valid JSON (not the full FSM). "
                    "Do not repeat or return any of the existing states. "
                    "Return only the new state(s) as a JSON array or object."
                )
            }
            fsm_info = fsm_info
            context_states_str = json.dumps(context, indent=2)
            user_prompt = {
                "role": "user",
                "content": (
                    f"The user query is: {query}\n\n"
                    f"The following FSM details are provided:\n{fsm_info}\n\n"
                    f"FSM states before the insertion point (from the uploaded file):\n{context_states_str}\n\n"
                    "Return ONLY the new state(s) as valid JSON (not the full FSM)."
                )
            }
            prompts = [system_prompt, user_prompt]
            response = self.client.get_chat_completion(prompts)
            if response and response.choices:
                fsm_output = response.choices[0].message.content.strip()
                try:
                    new_states = json.loads(fsm_output)
                    if isinstance(new_states, dict):
                        new_states = [new_states]
                except Exception:
                    return fsm_output
                merged_fsm = {
                    "name": "GeneratedFSM",
                    "description": "FSM generated from file upload and user query.",
                    "version": 1,
                    "finiteStateMachine": {
                        "states": context + new_states
                    }
                }
                return json.dumps(merged_fsm, indent=4)
            else:
                return "No FSM generated."

        # Default case for 'fsm_use_case' or other types
        system_prompt = {
            "role": "system",
            "content": (
                # --- FSM Generation Instructions ---
                "You are an FSM expert. Your task is to generate a valid FSM JSON object that fulfills the user query intent.\n"
                "CRITICAL: The output MUST be a complete FSM with the following structure:\n"
                "{\n"
                "  \"name\": \"DescriptiveName\",\n"
                "  \"description\": \"What this FSM does\",\n"
                "  \"version\": 1,\n"
                "  \"finiteStateMachine\": {\n"
                "    \"states\": [\n"
                "      {\n"
                "        \"id\": \"StateId\",\n"
                "        \"type\": \"stateType\",\n"
                "        \"configuration\": {\n"
                "          // State-specific configuration\n"
                "        },\n"
                "        \"saveStateOutput\": true,\n"
                "        \"saveStateOutputWithKey\": \"OutputKey\"\n"
                "      }\n"
                "    ]\n"
                "  }\n"
                "}\n\n"
                "For loadStaticFile states, use this exact structure:\n"
                "{\n"
                "  \"id\": \"LoadFileState\",\n"
                "  \"type\": \"loadStaticFile\",\n"
                "  \"configuration\": {\n"
                "    \"fileName\": \"path/to/file.txt\"\n"
                "  },\n"
                "  \"saveStateOutput\": true,\n"
                "  \"saveStateOutputWithKey\": \"FileContent\"\n"
                "}\n\n"
                "RULES:\n"
                "1. The output MUST be a complete FSM with all required top-level fields\n"
                "2. States must be in the 'states' array under 'finiteStateMachine'\n"
                "3. Each state must have 'id', 'type', and 'configuration' fields\n"
                "4. For loadStaticFile, 'fileName' is required in configuration\n"
                "5. Include 'saveStateOutput' and 'saveStateOutputWithKey' for state output handling\n"
                "6. Output ONLY valid JSON with no extra text or explanations\n"
                "STRICT RULE: For every state, the 'type' value must be copied exactly (including case) from the FSM example in fsm_details. Never invent, guess, or change the 'type' value.\n"
                # End concise instructions
            )
        }

        fsm_info = fsm_info

        # Ensure context is a list and handle different input types
        if context is None:
            context = []
        elif not isinstance(context, list):
            if isinstance(context, (str, int, float, bool)):
                context = [{"role": "user", "content": str(context)}]
            elif isinstance(context, dict):
                context = [{"role": "user", "content": str(context)}]
            else:
                try:
                    # Try to convert to string if possible
                    context = [{"role": "user", "content": str(context)}]
                except Exception:
                    context = []
        
        # Safely build conversation context, handling various input formats
        conversation_parts = []
        for entry in context:
            if isinstance(entry, dict):
                role = str(entry.get('role', 'user'))
                content = str(entry.get('content', ''))
                conversation_parts.append(f"{role}: {content}")
            else:
                # Handle non-dict entries by converting to string
                conversation_parts.append(f"user: {str(entry)}")
        
        conversation_context = "\n".join(conversation_parts)

        # --- Extraction instruction for user prompt (fsm_use_case) ---
        extraction_instruction_user = (
            "\nIMPORTANT FSM INSTRUCTIONS:\n"
            "- Always use the exact key names, types, and structure as defined in the schema/example for all states and configurations.\n"
            "- For extracting values from the input JSON, use JSONPath expressions (e.g., '$.fieldName') only in 'addJsonPathValuesToToken' states.\n"
            "- When referencing values produced by an 'addJsonPathValuesToToken' state, use the 'key' you defined in that state for extraction (e.g., '@KEY_NAME::String').\n"
            "- When referencing values produced by any other state, use the 'id' of that state for extraction (e.g., '@STATE_ID::String').\n"
            "- Use '@TOKEN_NAME::TYPE' (e.g., '@INPUT_STRING::String', '@Extract_Substring::String') for extraction from previous states. Do NOT use '$TOKEN_NAME' or '${TOKEN_NAME}' for extraction.\n"
            "- Use '$.field' (e.g., '$.inputString') only for JSONPath expressions to extract from the input JSON, not for referencing previous state outputs.\n"
            "- For 'operationToPerform', use UPPERCASE values (e.g., 'SUBSTRING', not 'substring').\n"
            "- For 'params', use the structure and types as shown in the schema/example (e.g., object, not list, if required).\n"
            "- Do NOT invent or add extra keys or change the structure from the schema/example.\n"
            "- When constructing a response using a value from a previous state (e.g., in 'footerToAppend'), use the correct extraction syntax (e.g., '@STATE_ID::String').\n"
            "- When defining transitions:\n"
            "  * The 'transition' key must be an object with a required 'conditions' array. Each item in 'conditions' must be an object. This is the schema:\n"
            "    \"transition\": {\n"
            "      \"type\": \"object\",\n"
            "      \"properties\": {\n"
            "        \"conditions\": {\n"
            "          \"type\": \"array\",\n"
            "          \"items\": { \"type\": \"object\" }\n"
            "        }\n"
            "      },\n"
            "      \"required\": [\"conditions\"],\n"
            "      \"description\": \"Optional - List of transitions applicable for the state if any. If not provided, the next state defined will be executed.\"\n"
            "    }\n"
            "  * For each error code or condition, create a separate condition object in the 'conditions' array. Do NOT group multiple error codes or checks in a single condition object.\n"
            "  * Use the correct 'if', 'operation', and 'params' structure for each condition as per the schema/example. For 'jsonPathExists', use 'jsonPaths': [ ... ]; for 'valueAtJsonPath' , use 'valueAtJsonPath': [ { 'jsonPath': ..., 'contains': [] } ].\n"
            "  * Always specify both 'thenState' and 'elseState' for every condition in the 'conditions' array, not just the last one. This ensures that the FSM always knows what to do for both error and success paths at every condition.\n"
            "  * If you want a linear chain (e.g., check 401, then 404, then 500, then error fields, then empty response, then success), you must explicitly chain the 'elseState' of each condition to the next check state (e.g., 'CheckFor404', 'CheckFor500', etc.), and only the last 'elseState' should point to the success state.\n"
            "  * If you want all checks to happen in parallel (i.e., any error triggers error state, otherwise success), then each condition should have the same 'elseState' (the success state).\n"
            "  * Follow the schema/example exactly for transitions structure and key names.\n"
            "- For HTTP/API call states, ensure that headers, formData, and other configuration fields match the schema exactly (e.g., use 'headersList' with 'header' and 'value' keys if required by schema).\n"
            "- For error handling in transitions, use the correct 'if' and 'params' for each error type (e.g., 'jsonPathExists' for error fields, 'valueAtJsonPath' for empty checks). Do not use 'path' or 'value' keys for these; always use the plural keys as per schema.\n"
            "- Always validate your FSM output using the provided validator to ensure there are no extra, missing, or misspelled keys, and that all field types match the schema.\n"
            "- Never hardcode secrets or sensitive values in shared or production FSMs.\n"
            "- Add comments or documentation for each state if the FSM is complex, to help future maintainers."
        )

        # General FSM template
        fsm_template = """
        Here's an FSM template for loading files with loadStaticFile state type:
        {
          "name": "LoadFileFSM",
          "description": "Loads files using loadStaticFile state type",
          "version": 1,
          "finiteStateMachine": {
            "states": [
              {
                "id": "LoadFileState1",
                "type": "loadStaticFile",
                "configuration": {
                  "fileName": "path/to/your/file1.txt"
                }
              },
              {
                "id": "LoadFileState2",
                "type": "loadStaticFile",
                "configuration": {
                  "fileName": "path/to/your/file2.txt"
                }
              }
              // Add more loadStaticFile states as needed
            ]
          }
        }"""

        # --- FILTER: Remove FSM details with 'Schema not found' or 'Example not found' ---
        def is_valid_fsm_detail(fsm):
            if not isinstance(fsm, dict):
                return False
            schema = fsm.get('schema', '').strip().lower()
            example = fsm.get('example', '').strip().lower()
            if schema == 'schema not found' or example == 'example not found':
                return False
            if not schema or not example:
                return False
            return True
        fsm_details = [fsm for fsm in fsm_details if is_valid_fsm_detail(fsm)]

        # Only include FSM details in the user prompt if not in correction mode
        user_prompt_content = (
            f"The user query is: {query}\n\n"
            f"The following FSM details are provided:\n{fsm_info}\n\n"
            f"The conversation history is:\n{conversation_context}\n\n"
            f"{fsm_template if query_type == 'fsm_use_case' else ''}\n"
            + (extraction_instruction_user if query_type == "fsm_use_case" else "")
        )
        user_prompt = {
            "role": "user",
            "content": (
                f"{STRICT_JSON_ONLY}\n"
                f"{user_prompt_content}\n"
                f"{STRICT_JSON_ONLY}"
            )
        }

        prompts = [system_prompt, user_prompt]

        try:
            start_time = time.time()
            response = self.client.get_chat_completion(prompts)
            print(f"GPT client response for build_fsm completed in {time.time() - start_time:.2f} seconds.")
            if response and response.choices:
                fsm_output = response.choices[0].message.content.strip()
                print("[DEBUG] Raw LLM output:", fsm_output)  # Log raw output for debugging
                
                return fsm_output

                def extract_json(text):
                    """Extract and clean JSON from text with multiple fallback strategies."""
                    import re
                    import json
                    
                    # Try direct JSON parsing first
                    text = text.strip()
                    try:
                        return json.loads(text), "direct"
                    except json.JSONDecodeError:
                        pass
                    
                    # Try extracting from markdown code blocks
                    markdown_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', text)
                    if markdown_match:
                        try:
                            return json.loads(markdown_match.group(1).strip()), "markdown"
                        except json.JSONDecodeError:
                            pass
                    
                    # Try finding any JSON-like structure
                    json_match = re.search(r'(\{.*\})', text, re.DOTALL)
                    if json_match:
                        try:
                            return json.loads(json_match.group(1).strip()), "regex"
                        except json.JSONDecodeError:
                            pass
                    
                    return None, "no_valid_json"
                
                try:
                    # Try to parse the output using multiple strategies
                    fsm_json, method = extract_json(fsm_output)
                    
                    if fsm_json is None:
                        print("[ERROR] Failed to extract valid JSON from LLM output")
                        print("[DEBUG] Raw output:", fsm_output)
                        # Return a minimal error FSM
                        error_fsm = {
                            "name": "ErrorFSM",
                            "description": "Failed to generate valid FSM JSON",
                            "version": 1,
                            "finiteStateMachine": {
                                "states": [{
                                    "id": "ErrorState",
                                    "type": "fail",
                                    "error": "InvalidFSM",
                                    "cause": "Failed to parse FSM from LLM output"
                                }]
                            }
                        }
                        return json.dumps(error_fsm, indent=4)
                    
                    print(f"[DEBUG] Successfully parsed JSON using method: {method}")
                    return json.dumps(fsm_json, indent=4)
                    
                except Exception as e:
                    print(f"[ERROR] Unexpected error processing FSM: {str(e)}")
                    print(f"[DEBUG] Raw output: {fsm_output}")
                    raise
                # If no valid JSON, return error FSM
                error_fsm = {
                    "name": "ErrorFSM",
                    "description": "No valid FSM JSON generated by LLM.",
                    "version": 1,
                    "finiteStateMachine": {
                        "states": [
                            {
                                "id": "ErrorState",
                                "type": "fail",
                                "error": "NoValidFSMJSON",
                                "cause": fsm_output
                            }
                        ]
                    }
                }
                return json.dumps(error_fsm, indent=4)
            else:
                return "No FSM generated."
        except Exception as e:
            return f"Error: {e}"
            
    def _validate_fsm_structure(self, fsm_json, previous_fsm=None):
        """
        Validate the structure of the parsed FSM JSON.
        
        Args:
            fsm_json (dict): The parsed FSM JSON to validate
            previous_fsm (dict, optional): The previous FSM to fall back to if validation fails
            
        Returns:
            dict: The validated FSM or a fallback FSM if validation fails
        """
        if not isinstance(fsm_json, dict):
            print("[STATE_BUILDER] Error: Expected a dictionary in the response")
            return previous_fsm if previous_fsm else {"states": []}
        
        # Debug print the input structure
        print("[DEBUG] Validating FSM structure:", json.dumps(fsm_json, indent=2))
        
        # Handle different possible structures
        if "finiteStateMachine" in fsm_json and "states" in fsm_json["finiteStateMachine"]:
            # Already in correct format: {"finiteStateMachine": {"states": [...]}}
            print("[STATE_BUILDER] FSM structure is valid")
            return fsm_json
            
        elif "states" in fsm_json:
            # If states is at the top level, move it under finiteStateMachine
            print("[STATE_BUILDER] Moving 'states' under 'finiteStateMachine'")
            return {"finiteStateMachine": {"states": fsm_json["states"]}}
            
        elif "finiteStateMachine" in fsm_json and isinstance(fsm_json["finiteStateMachine"], list):
            # If finiteStateMachine is a list, assume it's the states array
            print("[STATE_BUILDER] Wrapping states array in finiteStateMachine")
            return {"finiteStateMachine": {"states": fsm_json["finiteStateMachine"]}}
            
        elif isinstance(fsm_json.get("finiteStateMachine", {}).get("states"), list):
            # If states is already in the right place but we missed it earlier
            print("[STATE_BUILDER] FSM structure is valid (nested states found)")
            return fsm_json
            
        # If we get here, the structure is invalid
        print("[STATE_BUILDER] Error: Invalid FSM structure")
        print("[DEBUG] Problematic FSM structure:", json.dumps(fsm_json, indent=2))
        
        # Try to extract states from the response if possible
        if "states" in fsm_json:
            return {"finiteStateMachine": {"states": fsm_json["states"]}}
            
        # Fall back to previous FSM or return empty states
        return previous_fsm if previous_fsm else {"finiteStateMachine": {"states": []}}

def build_fsm_parallel(*args, **kwargs):
    """
    Stub for build_fsm_parallel. Not yet implemented.
    """
    return {"error": "build_fsm_parallel is not implemented."}

# Simple prompt for LLM to avoid common JSON mistakes:
LLM_JSON_PROMPT = (
    "You must output ONLY valid JSON. Do not include any explanations, markdown, or comments.\n"
    "Strictly follow these rules:\n"
    "- Do NOT put a comma after the last property in any object or array.\n"
    '- Enclose ALL property names in double quotes (e.g., \"key\": \"value\").\n'
    '- Enclose ALL string values in double quotes.\n'
    "- Do NOT use single quotes for property names or values.\n"
    "- Do NOT include comments or explanations in the output.\n"
    "- Do NOT output Python-style values (use true/false/null, not True/False/None).\n"
    "- Do NOT output partial or incomplete JSON.\n"
    "If you are unsure, output a minimal valid JSON object. Your output must always be valid JSON."
)

def llm_semantic_validate_fsm(query, fsm_json, fsm_details, gpt_client):
    """
    Use a secondary GPT client to semantically validate FSM output.
    Returns (is_semantically_valid, feedback_message)
    """
    # Prepare state templates/examples and schema descriptions from fsm_details
    state_examples = []
    schema_descriptions = []
    for fsm in fsm_details or []:
        if "example" in fsm:
            state_examples.append(fsm["example"])
        if "schema" in fsm:
            try:
                schema_obj = fsm["schema"]
                if isinstance(schema_obj, str):
                    import json as _json
                    schema_obj = _json.loads(schema_obj)
                def collect_descriptions(obj, prefix=""):
                    descs = []
                    if isinstance(obj, dict):
                        if "description" in obj and isinstance(obj["description"], str):
                            descs.append(f"{prefix}description: {obj['description']}")
                        for k, v in obj.items():
                            if isinstance(v, (dict, list)):
                                descs.extend(collect_descriptions(v, prefix + f"{k}."))
                    elif isinstance(obj, list):
                        for idx, v in enumerate(obj):
                            descs.extend(collect_descriptions(v, prefix + f"[{idx}]."))
                    return descs
                schema_descriptions.extend(collect_descriptions(schema_obj))
                # --- PATCH: Also add top-level schema description if present ---
                if "description" in schema_obj and isinstance(schema_obj["description"], str):
                    schema_descriptions.append(f"schema.description: {schema_obj['description']}")
            except Exception:
                pass
    state_examples_str = "\n\n".join(state_examples)
    schema_descriptions_str = "\n".join(schema_descriptions)

    prompt = (
        "You are an expert FSM semantic validator. Your MAIN TASK is to check if the FSM output FULLY FULFILLS the user query intent.\n"
        "Specifically, check:\n"
        "- Are ALL required states present to fulfill the user query intent?\n"
        "- Are ALL keys present and do their values correctly reflect the user query?\n"
        "- Are there any missing, extra, or incorrectly filled states/fields/values?\n"
        "- Are all values and logic derived from the user query, not hallucinated?\n"
        "- Use the provided schema descriptions (including all 'description' fields from the schema templates) to understand the meaning and intent of each key.\n"
        "- If the FSM output is not valid JSON, ignore formatting and focus on SEMANTIC correctness (intent, states, keys, values).\n"
        "- **IMPORTANT:** Also check for incorrect key names (e.g., misspelled, wrong case, or not present in schema/examples) and totally hallucinated key names (keys that do not exist in any schema or example). Suggest corrections for these as well.\n"
        "- **IMPORTANT:** In this FSM system, '${...}' is used for extracting a key from the input JSON (e.g., '${fieldName}' means extract 'fieldName' from the input JSON), while '@...' (e.g., '@temperature::string') is used to extract a value from a previous FSM state or token. Do NOT confuse these two syntaxes. Use '${...}' only for input JSON extraction, and '@STATE_ID::TYPE' or '@TOKEN_NAME::TYPE' only for referencing previous state outputs or tokens.\n"
        "- **IMPORTANT:** There are two ways to provide input to a state: (1) If the user query provides a direct value and the schema has a key to accept it (e.g., 'inputString'), use that key directly with the value. (2) If the user query asks to extract a value from a JSON document, use the extraction syntax (e.g., 'inputString': \"${key1}\"). Always choose the correct method based on the query and schema.\n"
        "- **IMPORTANT:** When a required key (such as 'inputString' for 'manipulateString' state, or 'inputDate' for 'transformDateFormat' state) is missing, but its intent is clear from the schema description or the user query (e.g., 'input string which has to be manipulated', or 'input date to be transformed'), you MUST suggest adding it. For example, if a value is extracted in a previous state (like 'ExtractedDate'), ensure that the next state (like 'transformDateFormat') uses the correct key (e.g., 'inputDate') and maps it to the extracted value (e.g., '@ExtractedDate').\n"
        "- **IMPORTANT:** For every state type, use the schema descriptions and examples to infer the correct required keys and their intent. For example, for 'manipulateString', always check for 'inputString'; for 'transformDateFormat', always check for 'inputDate' or the correct input field as per schema and ensure it is present and correctly mapped as per the user query intent and schema description. This applies to all state types: always check for all required keys as per schema and their correct usage as per the query intent and schema description.\n"
        "- **IMPORTANT:** For output or printing, use 'addHeaderAndFooter' state and extract values from previous states using '@{key}' syntax.\n"
        "If you find any issues, return a JSON object with:\n"
        "{\n"
        "  \"intent_fulfilled\": true/false,\n"
        "  \"missing_states\": [list of missing states],\n"
        "  \"extra_states\": [list of extra states],\n"
        "  \"incorrect_keys\": [list of keys with wrong values, missing, or incorrect names],\n"
        "  \"hallucinated_keys\": [list of keys that do not exist in any schema/example],\n"
        "  \"explanation\": \"Short explanation of what is missing or wrong, or 'PASS' if everything is correct.\",\n"
        "  \"corrections\": [ { \"state\": \"StateId\", \"key\": \"keyName\", \"suggested_value\": \"the correct value or syntax or correct key name\", \"reason\": \"why this is correct\" } ]\n"
        "}\n"
        "If everything is correct, reply with:\n"
        "{\n"
        "  \"intent_fulfilled\": true,\n"
        "  \"missing_states\": [],\n"
        "  \"extra_states\": [],\n"
        "  \"incorrect_keys\": [],\n"
        "  \"hallucinated_keys\": [],\n"
        "  \"explanation\": \"PASS\",\n"
        "  \"corrections\": []\n"
        "}\n"
        "Use the following FSM state examples and schema descriptions for reference:\n"
        f"{state_examples_str}\n"
        "\nSchema Descriptions (from all 'description' fields in the schema templates):\n"
        f"{schema_descriptions_str}\n"
        "\nUser Query:\n"
        f"{query}\n\n"
        "FSM Output:\n"
        f"{json.dumps(fsm_json, indent=2) if isinstance(fsm_json, dict) else fsm_json}\n"
    )
    response = gpt_client.get_chat_completion([{"role": "user", "content": prompt}])
    print("\n--- LLM SEMANTIC VALIDATION PROMPT ---")
    print(prompt)
    print("--- END OF PROMPT ---\n")
    if response and response.choices:
        # Get the raw response content
        raw_content = response.choices[0].message.content
        
        try:
            # Try to parse the content as JSON
            fsm_data = json.loads(raw_content)
            
            # If it's already a complete FSM, return as is
            if all(key in fsm_data for key in ['name', 'description', 'version', 'finiteStateMachine']):
                return raw_content
                
            # If it's just a state, wrap it in a complete FSM structure
            if isinstance(fsm_data, dict) and 'type' in fsm_data:
                complete_fsm = {
                    "name": f"GeneratedFSM_{fsm_data.get('id', '1')}",
                    "description": f"Auto-generated FSM for {fsm_data.get('id', 'operation')}",
                    "version": 1,
                    "finiteStateMachine": {
                        "states": [fsm_data]
                    }
                }
                return json.dumps(complete_fsm, indent=2)
                
        except json.JSONDecodeError:
            pass
            
        # If we can't parse it or it's not a valid state, return a minimal valid FSM
        default_fsm = {
            "name": "DefaultFSM",
            "description": "Automatically generated FSM",
            "version": 1,
            "finiteStateMachine": {
                "states": [
                    {
                        "id": "DefaultState",
                        "type": "addHeaderAndFooter",
                        "configuration": {
                            "headerToAppend": "Error: Could not generate valid FSM. ",
                            "footerToAppend": ""
                        }
                    }
                ]
            }
        }
        return json.dumps(default_fsm, indent=2).strip()

    return False, "No response from LLM semantic validator."

def build_fsm(
    query,
    fsm_details=None,
    context=None,
    query_type=None,
    uploaded_fsm_states=None,
    **kwargs  # Catch any additional kwargs to prevent argument errors
):
    """
    Build or modify an FSM based on the query and FSM details.
    
    Args:
        query (str): The user's query
        fsm_details (list, optional): List of FSM details from planning agent
        context (list, optional): Additional context for the query
        query_type (str, optional): Type of query (e.g., 'file_upload', 'add_states_to_previous_fsm')
        uploaded_fsm_states (list, optional): States from uploaded FSM
        **kwargs: Additional arguments that might be passed by the agent framework
        
    Returns:
        dict: The built or modified FSM with a 'states' array
    """
    # Debug logging with enhanced FSM details
    print(f"\n[BUILD_FSM] ===== Starting FSM Build =====")
    print(f"[BUILD_FSM] Query: {query}")
    print(f"[BUILD_FSM] Query Type: {query_type or 'fsm_use_case'}")
    
    # Log context keys and basic info
    if context:
        print(f"[BUILD_FSM] Context keys: {list(context.keys())}")
        if 'fsm_details' in context:
            print(f"[BUILD_FSM] FSM Details in context: {len(context['fsm_details']) if context['fsm_details'] else 0} items")
    
    # Log FSM details with more structure
    if fsm_details:
        print(f"[BUILD_FSM] Received {len(fsm_details)} FSM details:")
        for i, detail in enumerate(fsm_details, 1):
            if not isinstance(detail, dict):
                print(f"  {i}. Invalid FSM detail (not a dict): {detail}")
                continue
            name = detail.get('fsm_name', f'Unnamed FSM {i}')
            func = detail.get('functionality', 'No functionality provided')
            print(f"  {i}. {name} - {func}")
    else:
        print("[BUILD_FSM] No FSM details provided")
    
    if uploaded_fsm_states:
        print(f"[BUILD_FSM] Uploaded States: {len(uploaded_fsm_states)} states")
    else:
        print("[BUILD_FSM] No uploaded states")
    
    # Initialize GPT client and state builder with error handling
    try:
        print("[BUILD_FSM] Initializing GPT client...")
        gpt_client = GPTClient()
        print("[BUILD_FSM] Initializing StateBuilderAgent...")
        state_builder = StateBuilderAgent(client=gpt_client)
        print("[BUILD_FSM] Successfully initialized state builder")
    except Exception as e:
        error_msg = f"Error initializing GPT client or state builder: {str(e)}"
        print(f"[BUILD_FSM] {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "error": error_msg,
            "traceback": traceback.format_exc(),
            "states": []
        }
    
    # Ensure fsm_details is a list of valid FSM details
    if fsm_details is None:
        # Try to get fsm_details from context if not provided directly
        if context and 'fsm_details' in context:
            fsm_details = context['fsm_details']
            print("[BUILD_FSM] Using FSM details from context")
        else:
            fsm_details = []
            print("[BUILD_FSM] No FSM details provided and none found in context")
    
    # Ensure it's a list
    if not isinstance(fsm_details, list):
        print(f"[BUILD_FSM] Converting non-list FSM details to list")
        fsm_details = [fsm_details] if fsm_details is not None else []
    
    print(f"[BUILD_FSM] Received {len(fsm_details)} FSM details for processing")
    
    # Filter out invalid FSM details with detailed logging
    valid_fsm_details = []
    
    print(f"[BUILD_FSM] Validating {len(fsm_details)} FSM details...")
    
    # Deduplicate FSM details by fsm_name, keeping the first occurrence
    seen_fsms = {}
    deduplicated_fsms = []
    
    for i, fsm in enumerate(fsm_details, 1):
        if not isinstance(fsm, dict):
            print(f"[BUILD_FSM] Warning: FSM detail {i} is not a dictionary, got {type(fsm).__name__}")
            continue
            
        fsm_name = str(fsm.get('fsm_name', f'fsm_{i}')).strip()
        if fsm_name in seen_fsms:
            print(f"[BUILD_FSM] Skipping duplicate FSM: {fsm_name}")
            continue
            
        seen_fsms[fsm_name] = True
        deduplicated_fsms.append(fsm)
    
    print(f"[BUILD_FSM] After deduplication: {len(deduplicated_fsms)} unique FSMs")
    
    for i, fsm in enumerate(deduplicated_fsms, 1):
        try:
            if not isinstance(fsm, dict):
                print(f"[BUILD_FSM] Warning: FSM detail {i} is not a dictionary, got {type(fsm).__name__}")
                continue
            
            # Debug print the raw FSM detail
            print(f"\n[BUILD_FSM] Raw FSM detail {i}:")
            for k, v in fsm.items():
                print(f"  {k}: {str(v)[:100]}{'...' if len(str(v)) > 100 else ''}")
                
            # Ensure all required fields exist and are strings
            fsm_name = str(fsm.get('fsm_name', f'fsm_{i}')).strip()
            fsm_schema = str(fsm.get('schema', '{}')).strip()
            fsm_example = str(fsm.get('example', '{}')).strip()
            fsm_functionality = str(fsm.get('functionality', 'No functionality provided')).strip()
            
            print(f"[BUILD_FSM] Processing FSM {i}: {fsm_name}")
            
            # Create a clean FSM detail dictionary with all original fields plus cleaned ones
            clean_fsm = {
                **fsm,  # Include all original fields
                'fsm_name': fsm_name,
                'schema': fsm_schema,
                'example': fsm_example,
                'functionality': fsm_functionality
            }
            
            # Validate required fields - be lenient with placeholders
            missing_fields = []
            
            # Only check if schema is completely missing, not if it's a placeholder
            if not fsm_schema:
                missing_fields.append('schema')
                # Add a default schema if missing
                fsm_schema = '{"type": "object", "properties": {}}'
                clean_fsm['schema'] = fsm_schema
                
            # Only check if example is completely missing, not if it's a placeholder
            if not fsm_example:
                missing_fields.append('example')
                # Add a default example if missing
                fsm_example = '{"example": "No example provided"}'
                clean_fsm['example'] = fsm_example
            
            # Only skip if both schema and example are completely missing
            if missing_fields and (not fsm_schema or not fsm_example):
                print(f"[BUILD_FSM] Warning: FSM '{fsm_name}' is missing required fields: {', '.join(missing_fields)}")
                continue
                
            # Log if we're using placeholders
            if 'No schema provided' in fsm_schema or 'No example provided' in fsm_example:
                print(f"[BUILD_FSM] Using placeholder values for FSM: {fsm_name}")
                print(f"  - Schema: {'Using placeholder' if 'No schema provided' in fsm_schema else 'Provided'}")
                print(f"  - Example: {'Using placeholder' if 'No example provided' in fsm_example else 'Provided'}")
                
            valid_fsm_details.append(clean_fsm)
            print(f"[BUILD_FSM] Added valid FSM: {fsm_name}")
            
        except Exception as e:
            print(f"[BUILD_FSM] Error processing FSM detail {i}: {str(e)}")
            continue
    
    filtered_count = len(fsm_details) - len(valid_fsm_details)
    if filtered_count > 0:
        print(f"[BUILD_FSM] Filtered out {filtered_count} invalid FSM details")
    
    # Ensure context is a list of message dictionaries
    if context is None:
        context = []
    elif isinstance(context, dict):
        context = [{"role": "user", "content": str(context)}]
    
    # Set default query type if not provided
    if not query_type:
        query_type = "fsm_use_case"
    
    # Log the first valid FSM detail for debugging before processing
    if valid_fsm_details:
        print("\n[BUILD_FSM] First valid FSM detail before processing:")
        for k, v in valid_fsm_details[0].items():
            print(f"  {k}: {str(v)[:200]}{'...' if len(str(v)) > 200 else ''}")
    
    try:
        # Prepare common arguments for build_fsm
        build_args = {
            'query': query,
            'fsm_details': valid_fsm_details,
            'context': context or {},
            'query_type': query_type
        }
        
        # Add uploaded_fsm_states if provided
        if uploaded_fsm_states is not None:
            build_args['uploaded_fsm_states'] = uploaded_fsm_states
            print(f"[BUILD_FSM] Processing {query_type} query with {len(uploaded_fsm_states)} uploaded states")
        else:
            print(f"[BUILD_FSM] Processing {query_type} query without uploaded states")
        
        # Call the state builder
        result = state_builder.build_fsm(**build_args)
        
        # # Validate the result
        # if not isinstance(result, dict):
        #     print("[BUILD_FSM] Warning: Result is not a dictionary, creating empty FSM")
        #     result = {"states": []}
        # elif "states" not in result:
        #     print("[BUILD_FSM] Warning: Result missing 'states' key, adding empty states array")
        #     result["states"] = []
        
        # print(f"[BUILD_FSM] Successfully built FSM with {len(result.get('states', []))} states")
        return result
        
    except Exception as e:
        error_msg = f"Error building FSM: {str(e)}"
        print(f"\n[BUILD_FSM] {error_msg}")
        import traceback
        traceback.print_exc()
        # Return empty FSM on error but include error details for debugging
        return {
            "error": error_msg,
            "traceback": traceback.format_exc(),
            "states": []
        }

# In StateBuilderAgent.build_fsm, the following query types are handled:

# 1. "alter_previous_fsm"
#    - Modifies an existing FSM based on the user query and conversation context.
#    - Uses previous FSM JSON from context.
#    - Builds a prompt for the LLM to alter the FSM as per user instructions.

# 2. "add_states_to_previous_fsm"
#    - Adds new states to an existing FSM.
#    - Extracts the previous FSM from context.
#    - Builds a prompt for the LLM to add new states while preserving existing ones.

# 3. "file_upload"
#    - Handles uploaded FSM files.
#    - Uses uploaded FSM states as context.
#    - Builds a prompt for the LLM to generate only the new state(s) as JSON, then merges them with uploaded states.

# 4. Default/"fsm_use_case" (and any other types)
#    - Standard FSM generation from scratch based on the user query and FSM details.
#    - Builds a prompt for the LLM to generate a complete FSM JSON.

# Each type has its own prompt logic and output handling.
# The function also filters out FSM details with "Schema not found" or "Example not found" for all types.