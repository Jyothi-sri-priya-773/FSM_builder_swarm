import json
import difflib





def suggest_key(key, valid_keys):
    matches = difflib.get_close_matches(key, valid_keys, n=1, cutoff=0.6)
    return matches[0] if matches else None

def validate_fsm(fsm_json, schemas):
    """
    Manual validation: Validate an FSM against provided schemas (full JSONSchema-like, with type, required, properties, item_schema, etc).
    Returns (is_valid, statewise_faults)
    Each error is a dict:
    {
        "state": "StateId (Type)",
        "location": "configuration/inputFrom/transition/...",
        "problem": "missing/wrong/extra/wrong_case/wrong_value",
        "key": "keyName",
        "expected": "expected value or key",
        "actual": "actual value or key",
        "reason": "why this is wrong"
    }
    """
    statewise_faults = []

    required_top_fields = ["name", "description", "version", "finiteStateMachine"]
    # PATCH: If fsm_json is a list, treat as states list, skip top-level checks
    if isinstance(fsm_json, list):
        states = fsm_json
    elif isinstance(fsm_json, dict):
        for field in required_top_fields:
            if field not in fsm_json:
                statewise_faults.append({
                    "state": "FSM (top-level)",
                    "location": "top-level",
                    "problem": "missing",
                    "key": field,
                    "expected": "present",
                    "actual": "missing",
                    "reason": f"Missing required top-level key '{field}'"
                })
        extra_top_keys = set(fsm_json.keys()) - set(required_top_fields)
        for extra_key in extra_top_keys:
            suggestion = suggest_key(extra_key, required_top_fields)
            statewise_faults.append({
                "state": "FSM (top-level)",
                "location": "top-level",
                "problem": "extra",
                "key": extra_key,
                "expected": f"one of {required_top_fields}",
                "actual": "unexpected",
                "reason": f"Extra top-level key '{extra_key}'"
            })
        # PATCH: If "finiteStateMachine" is a list, treat as states
        fsm_states = fsm_json.get("finiteStateMachine", {})
        if isinstance(fsm_states, list):
            states = fsm_states
        elif isinstance(fsm_states, dict):
            states = fsm_states.get("states", [])
        else:
            states = []
    else:
        statewise_faults.append({
            "state": "FSM (top-level)",
            "location": "top-level",
            "problem": "wrong_value",
            "key": "fsm_json",
            "expected": "dict or list",
            "actual": type(fsm_json).__name__,
            "reason": "FSM must be a dict or list"
        })
        return False, statewise_faults

    if not isinstance(states, list):
        statewise_faults.append({
            "state": "FSM (top-level)",
            "location": "finiteStateMachine.states",
            "problem": "wrong_value",
            "key": "states",
            "expected": "list",
            "actual": type(states).__name__,
            "reason": "states must be a list"
        })
        return False, statewise_faults

    def validate_obj(obj, schema, path, state_name=None, state_type=None):
        sch_type = schema.get("type")
        if sch_type == "object":
            required_keys = schema.get("required", [])
            if isinstance(required_keys, bool):
                required_keys = []
            elif not isinstance(required_keys, list):
                required_keys = list(required_keys) if required_keys else []
            properties = schema.get("properties", {})
            for req in required_keys:
                prop = properties.get(req, {})
                if req not in obj and (prop.get("required") is True or req in required_keys):
                    statewise_faults.append({
                        "state": state_name,
                        "location": path,
                        "problem": "missing",
                        "key": req,
                        "expected": "present",
                        "actual": "missing",
                        "reason": f"Missing required key '{req}'"
                    })
            allowed_keys = set(properties.keys())
            if isinstance(obj, dict):
                extra_keys = set(obj.keys()) - allowed_keys
                for extra_key in extra_keys:
                    if extra_key not in allowed_keys:
                        suggestion = suggest_key(extra_key, allowed_keys)
                        statewise_faults.append({
                            "state": state_name,
                            "location": path,
                            "problem": "extra",
                            "key": extra_key,
                            "expected": f"one of {list(allowed_keys)}",
                            "actual": "unexpected",
                            "reason": f"Extra key '{extra_key}'"
                        })
                for key, val in obj.items():
                    # --- PATCH: Remove keys with empty list or empty dict if not required ---
                    if (
                        (isinstance(val, list) and not val)
                        or (isinstance(val, dict) and not val)
                    ) and key in properties and not properties[key].get("required"):
                        # Not required and empty, can be deleted
                        statewise_faults.append({
                            "state": state_name,
                            "location": f"{path}.{key}",
                            "problem": "extra",
                            "key": key,
                            "expected": "not present (empty and optional)",
                            "actual": "empty",
                            "reason": f"Optional key '{key}' is empty and can be removed"
                        })
                    else:
                        if key in properties:
                            validate_obj(val, properties[key], f"{path}.{key}", state_name, state_type)
            else:
                statewise_faults.append({
                    "state": state_name,
                    "location": path,
                    "problem": "wrong_value",
                    "key": path.split(".")[-1],
                    "expected": "object",
                    "actual": type(obj).__name__,
                    "reason": "Expected object"
                })
        elif sch_type == "array":
            if not isinstance(obj, list):
                statewise_faults.append({
                    "state": state_name,
                    "location": path,
                    "problem": "wrong_value",
                    "key": path.split(".")[-1],
                    "expected": "array",
                    "actual": type(obj).__name__,
                    "reason": "Expected array"
                })
                return
            item_schema = schema.get("item_schema") or schema.get("items")
            if item_schema:
                for idx, item in enumerate(obj):
                    validate_obj(item, item_schema, f"{path}[{idx}]", state_name, state_type)
        else:
            expected_type = sch_type
            if expected_type:
                py_type = {
                    "string": str,
                    "integer": int,
                    "number": (int, float),
                    "boolean": bool,
                    "object": dict,
                    "array": list
                }.get(expected_type)
                if py_type and not isinstance(obj, py_type):
                    statewise_faults.append({
                        "state": state_name,
                        "location": path,
                        "problem": "wrong_value",
                        "key": path.split(".")[-1],
                        "expected": expected_type,
                        "actual": type(obj).__name__,
                        "reason": f"Expected type {expected_type}"
                    })

    # Forbid 'initialState' and 'finalState' as keys anywhere in FSM
    forbidden_keys = {"initialState", "finalState"}
    def check_forbidden_keys(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in forbidden_keys:
                    statewise_faults.append({
                        "state": "FSM (any state)",
                        "location": path,
                        "problem": "extra",
                        "key": k,
                        "expected": "forbidden",
                        "actual": "present",
                        "reason": f"Key '{k}' is forbidden"
                    })
                check_forbidden_keys(v, path + "." + k if path else k)
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                check_forbidden_keys(item, f"{path}[{idx}]")
    check_forbidden_keys(fsm_json)

    for idx, state in enumerate(states):
        state_type = state.get("type")
        state_id = state.get("id", idx)
        state_name = f"{state_id} ({state_type})" if state_type else str(state_id)
        state_type_for_error = state_type if state_type else None
        if not state_type:
            statewise_faults.append({
                "state": state_name,
                "location": f"states[{idx}]",
                "problem": "missing",
                "key": "type",
                "expected": "present",
                "actual": "missing",
                "reason": "Missing required 'type' key"
            })
            continue
        schema = schemas.get(state_type)
        if not schema:
            statewise_faults.append({
                "state": state_name,
                "location": f"states[{idx}]",
                "problem": "wrong_value",
                "key": "type",
                "expected": f"one of {list(schemas.keys())}",
                "actual": state_type,
                "reason": "Invalid state type"
            })
            continue
        required_fields = schema.get("required", [])
        # Enforce required keys and case sensitivity
        for req in required_fields:
            if req not in state:
                statewise_faults.append({
                    "state": state_name,
                    "location": f"state '{state_id}' ({state_type})",
                    "problem": "missing",
                    "key": req,
                    "expected": "present (case-sensitive, as in schema example)",
                    "actual": "missing",
                    "reason": f"Missing required key '{req}'"
                })
        # Check for case-insensitive matches (wrong case)
        for key in state.keys():
            if key not in required_fields and key.lower() in [r.lower() for r in required_fields]:
                statewise_faults.append({
                    "state": state_name,
                    "location": f"state '{state_id}' ({state_type})",
                    "problem": "wrong_case",
                    "key": key,
                    "expected": f"case-sensitive match: {required_fields}",
                    "actual": "wrong case",
                    "reason": f"Key '{key}' is wrong case, should be as in schema"
                })
        allowed_keys = set(required_fields)
        for k, v in schema.items():
            if isinstance(v, dict) and v.get("optional"):
                allowed_keys.add(k)
        extra_keys = set(state.keys()) - allowed_keys
        # --- PATCH: Do not flag keys inside 'configuration' as extra at the state level ---
        if "configuration" in state and isinstance(state["configuration"], dict):
            config_keys = set(state["configuration"].keys())
        else:
            config_keys = set()
        for extra_key in extra_keys:
            # Only flag as extra if not 'configuration' and not a key inside configuration
            if extra_key != "configuration":
                suggestion = suggest_key(extra_key, allowed_keys)
                statewise_faults.append({
                    "state": state_name,
                    "location": f"state '{state_id}' ({state_type})",
                    "problem": "extra",
                    "key": extra_key,
                    "expected": f"one of {list(allowed_keys)}",
                    "actual": "unexpected",
                    "reason": f"Key '{extra_key}' is not allowed by schema"
                })
        config_schema = schema.get("configuration")
        config = state.get("configuration", {})
        if config_schema:
            if isinstance(config_schema, dict) and config_schema.get("type") == "object":
                validate_obj(config, config_schema, f"state '{state_id}' ({state_type}) configuration", state_name, state_type_for_error)
            else:
                # --- PATCH: Only check for extra keys inside configuration, not at state level ---
                for conf_key, conf_val in config_schema.items():
                    if conf_val.get("required") is True and conf_key not in config:
                        statewise_faults.append({
                            "state": state_name,
                            "location": f"state '{state_id}' ({state_type}) configuration",
                            "problem": "missing",
                            "key": conf_key,
                            "expected": "present",
                            "actual": "missing",
                            "reason": f"Missing required configuration key '{conf_key}'"
                        })
                    if conf_key in config:
                        if conf_val.get("type") == "array" and isinstance(config[conf_key], list):
                            item_schema = conf_val.get("item_schema") or conf_val.get("items")
                            for idx2, item in enumerate(config[conf_key]):
                                if isinstance(item, dict) and item_schema:
                                    allowed_item_keys = set(item_schema.get("required", [])) | set(item_schema.get("properties", {}).keys())
                                    extra_item_keys = set(item.keys()) - allowed_item_keys
                                    for extra_item_key in extra_item_keys:
                                        statewise_faults.append({
                                            "state": state_name,
                                            "location": f"state '{state_id}' ({state_type}) configuration.{conf_key}[{idx2}]",
                                            "problem": "extra",
                                            "key": extra_item_key,
                                            "expected": f"one of {list(allowed_item_keys)}",
                                            "actual": "unexpected",
                                            "reason": f"Key '{extra_item_key}' is not allowed in item schema"
                                        })
                        validate_obj(config[conf_key], conf_val, f"state '{state_id}' ({state_type}) configuration.{conf_key}", state_name, state_type_for_error)
                allowed_config_keys = set(conf_key for conf_key, conf_val in config_schema.items())
                extra_config_keys = set(config.keys()) - allowed_config_keys
                for extra_key in extra_config_keys:
                    suggestion = suggest_key(extra_key, allowed_config_keys)
                    statewise_faults.append({
                        "state": state_name,
                        "location": f"state '{state_id}' ({state_type}) configuration",
                        "problem": "extra",
                        "key": extra_key,
                        "expected": f"one of {list(allowed_config_keys)}",
                        "actual": "unexpected",
                        "reason": f"Key '{extra_key}' is not allowed in configuration"
                    })
        elif config:
            statewise_faults.append({
                "state": state_name,
                "location": f"state '{state_id}' ({state_type}) configuration",
                "problem": "extra",
                "key": list(config.keys()),
                "expected": "no keys (schema defines none)",
                "actual": "extra keys present",
                "reason": "Configuration keys present but not defined in schema"
            })

    is_valid = (len(statewise_faults) == 0)
    return is_valid, statewise_faults

def manual_validator_correction(fsm, schemas):
    import copy
    # Validate FSM and apply simple corrections for missing/extra keys
    is_valid, faults = validate_fsm(copy.deepcopy(fsm), schemas)
    if is_valid or not faults:
        return fsm
    # Try to auto-fix common issues (missing required keys, remove extra keys)
    if isinstance(fsm, str):
        import json
        fsm = json.loads(fsm)
    # Remove extra top-level keys
    required_top_fields = ["name", "description", "version", "finiteStateMachine"]
    for fault in faults:
        if fault.get("problem") == "extra" and fault.get("location") == "top-level":
            key = fault.get("key")
            if key in fsm:
                del fsm[key]
        if fault.get("problem") == "missing" and fault.get("location") == "top-level":
            key = fault.get("key")
            if key not in fsm:
                fsm[key] = "" if key != "finiteStateMachine" else {"states": []}
    # Remove extra keys in states/configuration (not exhaustive)
    if "finiteStateMachine" in fsm and "states" in fsm["finiteStateMachine"]:
        for state in fsm["finiteStateMachine"]["states"]:
            if not isinstance(state, dict):
                continue
            allowed_keys = set(["id", "type", "configuration", "saveStateOutput", "saveStateOutputWithKey", "transition", "inputFrom", "stopExecution"])
            for key in list(state.keys()):
                if key not in allowed_keys:
                    del state[key]
            # Patch configuration
            if "configuration" in state and isinstance(state["configuration"], dict):
                config = state["configuration"]
                for key in list(config.keys()):
                    if not isinstance(key, str):
                        del config[key]
    # Add missing required keys in states
    for fault in faults:
        if fault.get("problem") == "missing" and "state" in fault and "key" in fault:
            state_name = fault["state"]
            key = fault["key"]
            # Find the state by id (state_name may be 'id (type)')
            state_id = state_name.split(" (")[0]
            for state in fsm["finiteStateMachine"]["states"]:
                if str(state.get("id")) == state_id and key not in state:
                    state[key] = ""  # Default value for missing key
    return fsm
