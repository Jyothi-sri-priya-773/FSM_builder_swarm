import json
import re
from memory import get_fsm_schemas
from validator import validate_fsm  # Import manual validator

def format_faults_as_suggestions(faults):
    """
    Format the faults from validate_fsm as a list of suggestions for the LLM prompt.
    """
    if not faults:
        return "No schema violations detected."
    suggestions = []
    for fault in faults:
        loc = fault.get("location", "?")
        state = fault.get("state", "?")
        problem = fault.get("problem", "?")
        key = fault.get("key", "?")
        expected = fault.get("expected", "?")
        actual = fault.get("actual", "?")
        reason = fault.get("reason", "?")
        suggestions.append(f"- In {state} at {loc}: {problem} key '{key}' (expected: {expected}, actual: {actual}). Reason: {reason}")
    return "\n".join(suggestions)

def llm_semantic_validate_fsm(query, fsm_output, gpt_client=None, fsm_details=None, *args, **kwargs):
    """
    Use a secondary GPT client to semantically correct FSM output.
    Strictly validates and corrects each key in the FSM states to match the schema exactly.
    Returns ONLY the semantically corrected FSM (dict or JSON string).
    """
    # Ensure gpt_client is a valid object, not a string
    if gpt_client is None or isinstance(gpt_client, str) or not hasattr(gpt_client, "get_chat_completion"):
        from State_builder_agent_1 import GPTClient
        gpt_client = GPTClient()
    
    # Get all available schemas
    schemas = get_fsm_schemas()
    # --- Prepare detailed schema information for the prompt ---
    schema_details = {}
    for state_type, schema in schemas.items():
        # Store the entire schema for each state type
        schema_details[state_type] = schema
    
    # --- Run manual validator BEFORE LLM ---
    fsm_output = strip_code_fences_and_json(fsm_output)
    
    # Ensure fsm_output is a dictionary
    if isinstance(fsm_output, str):
        try:
            fsm_output = json.loads(fsm_output)
        except json.JSONDecodeError as e:
            print(f"[WARNING] Failed to parse FSM JSON: {e}")
            # If parsing fails, try to create a basic FSM structure
            fsm_output = {
                "name": "",
                "description": "",
                "version": "1.0",
                "finiteStateMachine": {
                    "states": []
                }
            }
    
    is_valid, faults = validate_fsm(fsm_output, schemas)
    suggestions = format_faults_as_suggestions(faults)
    print(f"[DEBUG] Manual validator suggestions (pre-LLM):\n{suggestions}")

    # --- Only include relevant schemas and state-level examples from fsm_details ---
    relevant_state_types = set()
    state_examples = {}
    if fsm_details:
        for fsm in fsm_details:
            example = fsm.get('example', '')
            if example:
                try:
                    # Remove code fences if present
                    if example.strip().startswith('```json'):
                        example = example.strip().split('```json', 1)[-1].strip('`\n')
                    example_json = json.loads(example)
                    # Find states in the example FSM
                    states = []
                    if 'finiteStateMachine' in example_json and 'states' in example_json['finiteStateMachine']:
                        states = example_json['finiteStateMachine']['states']
                    elif 'states' in example_json:
                        states = example_json['states']
                    for st in states:
                        stype = st.get('type')
                        if stype:
                            relevant_state_types.add(stype)
                            # Only keep the first example for each type
                            if stype not in state_examples:
                                state_examples[stype] = st
                except Exception:
                    pass
    # Fallback: if no relevant state types found, use all schemas
    if not relevant_state_types:
        print("[DEBUG] No relevant state types found in fsm_details. Using all schemas as fallback.")
        relevant_state_types = set(schema_details.keys())
    # Build relevant schema_details using state types
    relevant_schema_details = {k: v for k, v in schema_details.items() if k in relevant_state_types}
    print("[DEBUG] Relevant state types for semantic validation:", relevant_state_types)


    print("[DEBUG] Extracted state examples from fsm_details:")
    for stype, stex in state_examples.items():
        print(f"  Type: {stype}\n  Example: {json.dumps(stex, indent=2)}")
    # --- Correction prompt (STRICT JSON) ---
    prompt = (
        "You are an FSM schema validator that STRICTLY enforces the provided schemas.\n\n"
        "IMPORTANT: If no schema details or examples are provided, you must still check the FSM against the USER QUERY and do your best to correct any mistakes or inconsistencies.\n"
        "If the FSM is already correct and matches the query, return it unchanged. If you find mistakes, correct them using your best judgment based on the query and any available schema.\n"
        "Never return an empty object unless the FSM is completely invalid and cannot be fixed.\n\n"
        "YOUR TASK (STRICT):\n"
        "1. For EACH state in the FSM, check EVERY key against its schema.\n"
        "2. For each key in the state (including nested configuration):\n"
        "   - If the key doesn't exist in the schema, REMOVE IT.\n"
        "   - If the key is misspelled or has wrong case, CORRECT IT.\n"
        "   - If a required key is missing, ADD IT with an appropriate value (never leave required fields empty or missing).\n"
        "3. For configuration objects, ensure ALL keys match the schema exactly.\n"
        "4. When correcting values, use the provided STATE EXAMPLES as a reference for what valid values look like for each key and configuration.\n"
        "5. If a value is missing or invalid, infer a correct value by strictly following the schema and using the STATE EXAMPLES as a guide.\n\n"
        "RELEVANT SCHEMA DETAILS (use only these):\n"
        f"{json.dumps(relevant_schema_details, indent=2)}\n\n"
        "STATE EXAMPLES (for each relevant type):\n"
        + "\n\n".join([
            f"Type: {stype}\nExample: {json.dumps(stex, indent=2)}" for stype, stex in state_examples.items()
        ]) + "\n\n"
        "USER QUERY:\n"
        f"{query}\n\n"
        "CURRENT FSM OUTPUT (to correct):\n"
        f"{json.dumps(fsm_output, indent=2) if isinstance(fsm_output, dict) else fsm_output}\n\n"
        "SCHEMA VALIDATION SUGGESTIONS (from manual validator):\n"
        f"{suggestions}\n\n"
        "INSTRUCTIONS (STRICT):\n"
        "1. For EACH state, check EVERY key against the schema.\n"
        "2. Remove or correct any keys that don't match the schema.\n"
        "3. Add any missing required keys with appropriate values (never leave required fields empty or missing).\n"
        "4. Ensure all configuration keys match the schema exactly.\n"
        "5. If no schema or example is available, use your best judgment to correct the FSM so it matches the query.\n"
        "6. If the FSM is already correct and matches the query, return it unchanged.\n"
        "7. Never return an empty object unless the FSM is completely invalid and cannot be fixed.\n"
        "8. Return ONLY the corrected FSM as STRICT, VALID JSON (no markdown, no code fences, no extra text).\n"
        "9. If the input is not valid JSON, return the original FSM output.\n"
    )
    validation_rules = """
    STRICT VALIDATION RULES:
    1. For EACH state, check EVERY key against its schema:
       - Remove any keys not explicitly defined in the schema
       - Replace hallucinated keys with correct ones (e.g., 'httpGet' → 'performHttpCall')
       - Ensure all required fields are present with correct types
    2. For configuration objects:
       - Only include keys that exist in the schema
       - Convert keys to match schema exactly (case-sensitive)
       - Add missing required configuration keys with empty values
    3. Special handling for common patterns:
       - 'httpGet'/'httpPost' → 'performHttpCall' with 'methodType' set
       - 'xmlDataExtraction' → 'extractXPathValues' with proper configuration
       - 'url' → 'providerEndPointURL' for HTTP calls
       - 'headers' → 'requestHeaders' for HTTP calls
    4. Key transformations (case-sensitive):
       - 'xpath' → 'xpaths' (as array) for XPath extraction
       - 'method' → 'methodType' for HTTP calls
       - 'body' → 'requestBody' for HTTP calls
    5. Return ONLY the corrected FSM JSON with no additional text or markdown
    """
    prompt += f"\n\n{validation_rules}"

    # --- First LLM correction ---
    response = gpt_client.get_chat_completion([{"role": "user", "content": prompt}])
    corrected = None
    if response and response.choices:
        corrected = response.choices[0].message.content.strip()
        print("\n[DEBUG] LLM semantic validator output (before auto-correction):\n", corrected)
        # --- Post-process: strip code fences if present ---
        if corrected.startswith('```'):
            corrected = corrected.strip('`')
            corrected = corrected.replace('json', '', 1).strip()
        try:
            corrected_fsm = json.loads(corrected)
            print("[DEBUG] LLM semantic validator output (parsed as JSON):\n", json.dumps(corrected_fsm, indent=2))
        except Exception as e:
            print(f"[DEBUG] LLM semantic validator output is not valid JSON. Exception: {e}")
            print(f"[DEBUG] Raw LLM output (unparsed):\n{corrected}")
            corrected_fsm = corrected
    else:
        print("[DEBUG] LLM semantic validator returned nothing, returning original FSM output.")
        corrected_fsm = fsm_output

    # --- Run manual validator AGAIN on LLM output ---
    corrected_fsm = strip_code_fences_and_json(corrected_fsm)
    is_valid2, faults2 = validate_fsm(corrected_fsm, schemas)
    suggestions2 = format_faults_as_suggestions(faults2)
    print(f"[DEBUG] Manual validator suggestions (post-LLM):\n{suggestions2}")

    if is_valid2 or not faults2:
        return corrected_fsm

    # --- Second LLM correction with updated suggestions ---
    prompt2 = (
        prompt +
        "\n\nADDITIONAL SCHEMA VALIDATION SUGGESTIONS (after first correction):\n" +
        suggestions2 +
        "\n\nPlease correct the FSM again, strictly following the updated suggestions above."
    )
    response2 = gpt_client.get_chat_completion([{"role": "user", "content": prompt2}])
    corrected2 = None
    if response2 and response2.choices:
        corrected2 = response2.choices[0].message.content.strip()
        print("\n[DEBUG] LLM semantic validator output (second round):\n", corrected2)
        # --- Post-process: strip code fences if present ---
        if corrected2.startswith('```'):
            corrected2 = corrected2.strip('`')
            corrected2 = corrected2.replace('json', '', 1).strip()
        try:
            corrected_fsm2 = json.loads(corrected2)
            print("[DEBUG] LLM semantic validator output (second round, parsed as JSON):\n", json.dumps(corrected_fsm2, indent=2))
        except Exception as e:
            print(f"[DEBUG] LLM semantic validator output (second round) is not valid JSON. Exception: {e}")
            print(f"[DEBUG] Raw LLM output (second round, unparsed):\n{corrected2}")
            corrected_fsm2 = corrected2
    else:
        print("[DEBUG] LLM semantic validator (second round) returned nothing, returning previous FSM output.")
        corrected_fsm2 = corrected_fsm

    # --- Final manual validation ---
    corrected_fsm2 = strip_code_fences_and_json(corrected_fsm2)
    is_valid3, faults3 = validate_fsm(corrected_fsm2, schemas)
    # If the last output is empty or not a dict/list, return the last non-empty LLM output
    def is_empty_fsm(fsm):
        if not fsm:
            return True
        if isinstance(fsm, dict):
            # Consider FSM empty if it has no states or only empty states
            if 'finiteStateMachine' in fsm:
                states = fsm['finiteStateMachine'].get('states', [])
                return not states
            elif 'states' in fsm:
                return not fsm['states']
            return not fsm
        if isinstance(fsm, list):
            return len(fsm) == 0
        return False
    # If the last output is empty, return the last non-empty LLM output
    if not is_valid3 and is_empty_fsm(corrected_fsm2):
        print("[DEBUG] Last LLM output is empty FSM. Returning previous non-empty LLM output.")
        if not is_empty_fsm(corrected_fsm):
            return ensure_fsm_structure(corrected_fsm)
        return ensure_fsm_structure(fsm_output)
    if is_valid3 or not faults3:
        return ensure_fsm_structure(corrected_fsm2)
    print("[DEBUG] FSM is still not schema-compliant after two LLM corrections. Returning last output.")
    return ensure_fsm_structure(corrected_fsm2)

def extract_state_types_from_examples(fsm_details):
    """
    Extract all unique state 'type' values from the 'example' field of each FSM detail.
    Handles both stringified JSON (with or without code fences) and dict examples.
    """
    state_types = set()
    for detail in fsm_details:
        example = detail.get('example')
        if not example:
            continue
        # Remove code fences if present
        if isinstance(example, str):
            example_str = example.strip()
            if example_str.startswith('```'):
                example_str = re.sub(r'^```[a-zA-Z0-9]*', '', example_str).strip()
                example_str = re.sub(r'```$', '', example_str).strip()
            try:
                example_json = json.loads(example_str)
            except Exception:
                continue
        elif isinstance(example, dict):
            example_json = example
        else:
            continue
        # Traverse to states
        fsm = example_json.get('finiteStateMachine')
        if not fsm:
            continue
        states = fsm.get('states', [])
        for state in states:
            t = state.get('type')
            if t:
                state_types.add(t)
    return state_types

def extract_state_types_from_fsm_examples(fsm_details):
    """
    Extract all unique state types from the 'finiteStateMachine' examples in fsm_details.
    Returns a set of state type strings.
    """
    state_types = set()
    for fsm in fsm_details:
        fsm_example = fsm.get('example', '')
        if not fsm_example:
            continue
        # Remove code fences if present
        if fsm_example.strip().startswith('```'):
            fsm_example = fsm_example.strip().split('\n', 1)[-1]
            if fsm_example.endswith('```'):
                fsm_example = fsm_example[:-3]
        try:
            import json
            example_json = json.loads(fsm_example)
            states = example_json.get('finiteStateMachine', {}).get('states', [])
            for state in states:
                stype = state.get('type')
                if stype:
                    state_types.add(stype)
        except Exception as e:
            print(f"[DEBUG] Failed to parse FSM example for state type extraction: {e}")
    return state_types

def strip_code_fences_and_json(fsm):
    if isinstance(fsm, str):
        s = fsm.strip()
        if s.startswith('```'):
            s = s.lstrip('`')
            if s.startswith('json'):
                s = s[4:].lstrip('\n')
            if s.endswith('```'):
                s = s[:-3]
        return s
    return fsm

def ensure_fsm_structure(fsm):
    # Add default top-level keys if missing
    if not isinstance(fsm, dict):
        return fsm
    if 'name' not in fsm:
        fsm['name'] = 'AutoGeneratedFSM'
    if 'description' not in fsm:
        fsm['description'] = 'Auto-generated FSM'
    if 'version' not in fsm:
        fsm['version'] = '1.0'
    if 'finiteStateMachine' not in fsm:
        # Try to move 'states' to 'finiteStateMachine'
        if 'states' in fsm:
            fsm['finiteStateMachine'] = {'states': fsm.pop('states')}
        else:
            fsm['finiteStateMachine'] = {'states': []}
    return fsm

# In the main validation logic, after receiving fsm_details:
#
#   1. Extract state types from examples
#   2. Filter schemas from get_fsm_schemas() to only those matching the extracted types
#   3. Use only those schemas and the state-level examples for prompt construction
#
# Example integration (replace the relevant section):
#
#   state_types = extract_state_types_from_examples(fsm_details)
#   all_schemas = get_fsm_schemas()
#   filtered_schemas = [s for s in all_schemas if s.get('type') in state_types]
#   # Use filtered_schemas and state-level examples for prompt
