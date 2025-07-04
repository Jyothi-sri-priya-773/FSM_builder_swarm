# fsm_agents.py
import json
from swarm import Agent
from swarm.types import Result
from memory import extractor
from State_builder_agent_1 import GPTClient, extract_json_from_markdown


def route_to_planning(query: str) -> Result:
    """
    Routes user input: if FSM-related or not a daily conversation, delegate to planning_agent;
    otherwise, prompt for an FSM description.
    """
    text = query.strip().lower()
    daily_keywords = [
        "hello", "hi", "how are you", "good morning", "good evening", "good night",
        "what's up", "how's it going", "thank you", "thanks", "bye"
    ]
    # If not a daily conversation, transfer to planning_agent
    if not any(k in text for k in daily_keywords):
        return Result(value=query, agent=planning_agent)
    # Instead of prompting for a description, return a polite message but still transfer to planning_agent
    return Result(value=query, agent=planning_agent)


def planning_tool(query: str, **kwargs) -> Result:
    """
    Given a user FSM-related query, generate an ordered plan for FSM construction as a Markdown table.

    - Use extractor to fetch relevant FSM examples (name, description, functionality).
    - Analyze the user query and examples to identify the required FSM states, what needs to be filled in each state, and transitions between states.
    - For each step, output:
        | State Type | Description (what to fill/how to form according to user query) | Transition Notes |
      The State Type **must exactly match the 'type' field** used in FSM examples (e.g., addJsonPathValuesToToken, performHttpCall, addHeaderAndFooter, etc).
      Do NOT invent new state types; always use the exact 'type' string from the FSM examples for each step.
    - The table should be clear and actionable for the user.
    - If the user provides corrections, update the steps accordingly and return the corrected table.
    - If the user is satisfied, stop and do not generate further steps.
    - Use GPTClient for analysis, but all orchestration must be done via Swarm.
    - Output only the Markdown table (no extra text).
    """
    # 1. Retrieve relevant FSM examples
    docs = extractor.query_fsm_data(query)
    # Ensure docs is a list of dicts, not strings
    examples = []
    allowed_types = set()
    if isinstance(docs, str):
        try:
            docs_json = json.loads(docs)
            if isinstance(docs_json, list):
                docs = docs_json
            else:
                docs = []
        except Exception:
            docs = []
    if isinstance(docs, list):
        for d in docs:
            if isinstance(d, dict):
                examples.append({
                    "name": d.get("name", ""),
                    "description": d.get("description", ""),
                    "functionality": d.get("functionality", ""),
                    "type": d.get("type", "")
                })
                # Extract all unique types from the examples
                if "type" in d and d["type"]:
                    allowed_types.add(d["type"])
                # Also extract types from states if present (for FSM JSON examples)
                if "states" in d:
                    for state in d["states"]:
                        if isinstance(state, dict) and "type" in state and state["type"]:
                            allowed_types.add(state["type"])

    allowed_types_list = sorted(list(allowed_types))

    # --- ADD THIS CHECK ---
    if not allowed_types_list:
        print("ERROR: No FSM state types could be retrieved from examples.")
        print("DEBUG: Raw docs returned from extractor.query_fsm_data(query):")
        print(docs)
        print("DEBUG: Parsed examples (should contain 'type' fields):")
        for ex in examples:
            print(json.dumps(ex, indent=2))
        print("DEBUG: If docs is empty or examples have no 'type', check your FSM data source (Confluence, DB, etc.) and extractor logic.")
        table_md = (
            "| State Type | Description | Transition Notes |\n"
            "|---|---|---|\n"
            "|  | No FSM state types could be retrieved from examples. Please check your FSM examples data source or database. |  |"
        )
        return Result(value=table_md, agent=None)

    # --- DEBUG LOGGING: Show allowed_types_list and examples for troubleshooting ---
    print("DEBUG: Allowed FSM state types extracted for planning agent:", allowed_types_list)
    print("DEBUG: Example FSMs passed to LLM prompt:")
    for ex in examples:
        print(json.dumps(ex, indent=2))

    # Steps and example for reference (query, steps, then FSM JSON)
    reference_steps = '''
Query:
Extract the value of "inputDate" from the input JSON. 
Transform the extracted date from format "yyyy-MM-dd'T'HH:mm:ss" in UTC to "dd-MM-yyyy HH:mm:ss" in Asia/Kolkata timezone.
Add the header "The Result of given input is" and footer with the transformed date to the input string.

Steps for the above query:
| State Type                  | Description                                                                                   | Transition Notes |
|---------------------------- |----------------------------------------------------------------------------------------------|------------------|
| addJsonPathValuesToToken    | Extract the value of 'inputDate' from the input JSON and save it as 'inputdate' in the token | Next: transformDateAndTime |
| transformDateAndTime        | Transform 'inputdate' from UTC 'yyyy-MM-dd\\'T\\'HH:mm:ss' to 'dd-MM-yyyy HH:mm:ss' in Asia/Kolkata and save as 'transformedDate' | Next: addHeaderAndFooter |
| addHeaderAndFooter          | Add header 'The Result of given input is' and footer with 'transformedDate' to the input string | End             |

FSM Example:
{
  "name": "Test_TransformJsonUsingJolt",
  "version": 1,
  "finiteStateMachine": {
    "states": [
      {
        "id": "Add_Request_Params_To_Token",
        "type": "addJsonPathValuesToToken",
        "configuration": {
          "keyJsonPathList": [
            {
              "key": "inputdate",
              "jsonPath": "$.inputDate"
            }
          ]
        },
        "saveStateOutput" : true
      },
      {
        "id": "DateTransformationExample",
        "type": "transformDateAndTime",
        "configuration": {
          "dateFormatConversionInputs": [
            {
              "inputDate": "@inputdate::String",
              "inputDateFormat": "yyyy-MM-dd'T'HH:mm:ss",
              "outputDateFormat": "dd-MM-yyyy HH:mm:ss",
              "inputTimeZone": "UTC",
              "outputTimeZone": "Asia/Kolkata",
              "saveTransformedDateTo": "transformedDate"
            }
          ]
        }
      },
      {
        "id": "randomid1",
        "type": "addHeaderAndFooter",
        "configuration": {
          "headerToAppend": "The Result of given input is",
          "footerToAppend": "@transformedDate::String"
        }
      }
    ]
  }
}
'''

    # 2. Prepare GPT prompts
    system_prompt = (
        "You are an FSM planning assistant. Given a user goal and FSM examples, "
        "produce an ordered JSON array of steps. Each step must be a JSON object with "
        "\"type\" (the exact FSM state type as in the FSM examples, e.g., addJsonPathValuesToToken, performHttpCall, addHeaderAndFooter, etc), "
        "\"description\", and \"transition_notes\" fields. "
        "Then, format the steps as a Markdown table with columns: State Type, Description, Transition Notes. "
        "The State Type column MUST use one of the following allowed types ONLY: "
        f"{allowed_types_list}. "
        "Do NOT invent or paraphrase state types; always use the exact 'type' string from the FSM examples for each step. "
        "If the user provides corrections, update the steps accordingly. "
        "If the user is satisfied, do not generate further steps. "
        "Output only the Markdown table."
    )
    user_prompt = (
        f"User query: {query}\n\n"
        f"{reference_steps}\n\n"
        "Relevant FSMs and their Examples (pay attention to the 'type' field for each state):\n" +
        "\n".join([
            f"- Name: {e['name']}\n  Description: {e['description']}\n  Functionality: {e['functionality']}" for e in examples
        ]) +
        f"\n\nAllowed FSM state types (use only these for State Type column): {allowed_types_list}\n"
        "Generate an ordered JSON array of steps, each with 'type' (matching the FSM state type from examples, do not invent new types), 'description', and 'transition_notes'. "
        "Then, format the steps as a Markdown table with columns: State Type, Description, Transition Notes. "
        "The State Type column MUST use one of the allowed types above for each step. "
        "Do NOT invent or paraphrase state types; always use the exact 'type' string from the FSM examples for each step. "
        "Output only the Markdown table."
    )

    client = GPTClient()
    response = client.get_chat_completion([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

    content = response.choices[0].message.content

    # --- Always convert to Markdown table, even if model returns only description or prose ---
    import re
    def extract_json_steps(text):
        # Try to extract JSON array from the response
        match = re.search(r"\[\s*{[\s\S]*?}\s*\]", text)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
        return None

    steps = None
    if content.strip().startswith("["):
        try:
            steps = json.loads(content)
        except Exception:
            steps = None
    else:
        steps = extract_json_steps(content)

    if steps and isinstance(steps, list):
        table_lines = ["| State Type | Description | Transition Notes |", "|---|---|---|"]
        for step in steps:
            state_type = step.get("type", "") or step.get("state_type", "") or step.get("state_name", "")
            desc = step.get("description", "").replace("\n", "<br>")
            notes = step.get("transition_notes", "").replace("\n", "<br>")
            table_lines.append(f"| {state_type} | {desc} | {notes} |")
        table_md = "\n".join(table_lines)
        return Result(value=table_md, agent=None)
    else:
        # Fallback: try to extract table from content, else wrap as a single row
        table_match = re.search(r"\|.*\|", content)
        if table_match:
            # Already a table, return as is
            return Result(value=content, agent=None)
        else:
            # Wrap description/prose as a single table row
            table_md = (
                "| State Type | Description | Transition Notes |\n"
                "|---|---|---|\n"
                f"|  | {content.replace(chr(10), '<br>')} |  |"
            )
            return Result(value=table_md, agent=None)


user_interaction_agent = Agent(
    name="user_interaction_agent",
    instructions=(
        "If the user asks an FSM-related query, route to the planning_agent using route_to_planning. "
        "Otherwise, prompt the user to specify an FSM description."
    ),
    functions=[route_to_planning]
)

planning_agent = Agent(
    name="planning_agent",
    instructions=(
        "When invoked, call planning_tool with the user query. "
        "The planning_tool will use extractor to fetch FSM examples and GPT to analyze the query. "
        "It will generate an ordered Markdown table of FSM-building steps, with columns: State Name, Description (what to fill/how to form), and Transition Notes. "
        "If the user provides corrections, update the steps and return the corrected table. "
        "If the user is satisfied, do not generate further steps. "
        "Output only the Markdown table."
    ),
    functions=[planning_tool]
)

# NOTE: Swarm Agent objects do not have a .run() method directly.
# To execute an agent, you should use the Swarm class to orchestrate agent execution.
# Example usage in your main.py:
#
# from swarm.core import Swarm
# swarm = Swarm()
# result = swarm.run(agent=user_interaction_agent, messages=[{"role": "user", "content": user_query}])
#
# Do NOT call user_interaction_agent.run(...), as this will cause an AttributeError.

