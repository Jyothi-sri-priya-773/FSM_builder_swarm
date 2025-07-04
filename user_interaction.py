import os
from dotenv import load_dotenv, find_dotenv
from memory import ConfluenceDataExtractor

class UserInteractionClient:
    def __init__(self, fsm_info_path=None):
        # Ensure credentials are loaded for Confluence access
        load_dotenv(find_dotenv())
        self.fsm_info_path = fsm_info_path or os.path.join(os.path.dirname(__file__), "fsm_info.txt")
        self.llama_extractor = None
        self.confluence_url = "https://amadeus.atlassian.net/wiki/spaces/CATS/pages/2461188677/FSM-Documentation"
        self.small_talk_greetings = [
            "hi", "hello", "hey", "good morning", "good afternoon", "good evening"
        ]
        self.small_talk_responses = [
            "Hello! How can I help you with FSMs today?",
            "Hi there! Ask me anything about FSMs.",
            "Hey! Ready to answer your FSM questions.",
            "Hello! How are you today?",
            "Hi! How can I assist you?"
        ]

    def is_small_talk(self, user_question):
        q = user_question.strip().lower()
        if any(q.startswith(greet) for greet in self.small_talk_greetings) and len(q.split()) <= 4:
            return True
        if q in ["how are you", "how are you?", "how are you doing", "how are you doing?"]:
            return True
        return False

    def get_small_talk_response(self, user_question):
        q = user_question.strip().lower()
        if any(q.startswith(greet) for greet in self.small_talk_greetings):
            import random
            return random.choice(self.small_talk_responses)
        if "how are you" in q:
            return "I'm an AI, but I'm here to help you with FSMs! How can I assist you?"
        return "Hello! How can I help you with FSMs today?"

    def is_general_fsm_question(self, user_question):
        # Only treat as general FSM question if it's about FSM definition, list, or schema/example for a state
        q = user_question.lower().strip()
        # 1. what is fsm
        if q in ["what is fsm", "fsm full form", "what is the full form of fsm"]:
            return True
        # 2. list available fsms
        if q in ["list fsms", "list of fsms", "available fsms", "show fsms"]:
            return True
        # 3. give example of <fsm_name>
        if q.startswith("give example of ") or q.startswith("show example of ") or q.startswith("example of "):
            return True
        # 4. give schema of <fsm_name>
        if q.startswith("give schema of ") or q.startswith("show schema of ") or q.startswith("schema of "):
            return True
        return False

    def get_fsm_info(self):
        try:
            with open(self.fsm_info_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def get_llama_index_answer(self, user_question):
        """
        Instead of using Confluence, use the content of fsm_info.txt as the document for LlamaIndex.
        """
        try:
            # Read FSM info from file
            fsm_info = self.get_fsm_info()
            if not fsm_info:
                return "Sorry, I do not have information on that (FSM info file missing)."
            # Prepare a single document for LlamaIndex
            from llama_index.core.schema import Document
            documents = [Document(text=fsm_info, metadata={"source": self.fsm_info_path})]
            # Create a new extractor each time to avoid stale index
            self.llama_extractor = None
            class DummyExtractor:
                def __init__(self, docs):
                    from llama_index.core import VectorStoreIndex
                    from llama_index.vector_stores.chroma import ChromaVectorStore
                    from llama_index.core import StorageContext, Settings
                    import chromadb
                    chroma_client = chromadb.EphemeralClient()
                    chroma_collection = chroma_client.get_or_create_collection("fsm_chroma")
                    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    self.index = VectorStoreIndex.from_documents(
                        docs,
                        storage_context=storage_context,
                        embed_model=Settings.embed_model
                    )
                def query_fsm_data(self, query):
                    query_engine = self.index.as_query_engine()
                    answer = query_engine.query(query)
                    return str(answer)
            self.llama_extractor = DummyExtractor(documents)
            answer = self.llama_extractor.query_fsm_data(user_question)
            if answer and answer.strip() and answer.strip().lower() not in ["empty response", user_question.strip().lower()]:
                return answer.strip()
            return "Sorry, I do not have information on that."
        except Exception as e:
            print(f"LlamaIndex query error: {e}")
            return f"Sorry, I do not have information on that. (LlamaIndex error: {e})"

    def chat(self, user_question):
        if self.is_small_talk(user_question):
            return self.get_small_talk_response(user_question)
        # Only answer general FSM definition/list/schema/example queries here
        if self.is_general_fsm_question(user_question):
            fsm_info = self.get_fsm_info()
            q = user_question.lower().strip()
            import re, json

            # 1. what is fsm
            if q in ["what is fsm", "fsm full form", "what is the full form of fsm"]:
                return "FSM stands for Finite State Machine."

            # 2. list available fsms
            if q in ["list fsms", "list of fsms", "available fsms", "show fsms"]:
                lines = fsm_info.splitlines()
                fsm_names = []
                for line in lines:
                    if line.strip() and not line.strip().startswith("{") and not line.strip().startswith('"') and not ":" in line and len(line.strip().split()) == 1:
                        fsm_names.append(line.strip())
                if fsm_names:
                    return "Available FSMs:\n" + "\n".join(f"- {n}" for n in fsm_names)
                return "No FSMs found in the documentation."

            # 3. give example of <fsm_name>
            match_example = re.match(r"(give|show)?\s*example of\s+([a-zA-Z0-9_]+)", q)
            if match_example:
                state_name = match_example.group(2).strip().lower()
                lines = fsm_info.splitlines()
                idx = -1
                for i, line in enumerate(lines):
                    if line.strip().lower() == state_name:
                        idx = i
                        break
                if idx != -1:
                    # Find the next JSON block (example)
                    example = None
                    for j in range(idx+1, len(lines)):
                        if lines[j].strip().startswith("{"):
                            block = [lines[j]]
                            for k in range(j+1, len(lines)):
                                block.append(lines[k])
                                if lines[k].strip().endswith("}"):
                                    break
                            try:
                                json.loads("\n".join(block))
                                example = "\n".join(block)
                                break
                            except Exception:
                                continue
                    if example:
                        return f"```json\n{example}\n```"
                    return f"Sorry, could not find example for '{state_name}'."

            # 4. give schema of <fsm_name>
            match_schema = re.match(r"(give|show)?\s*schema of\s+([a-zA-Z0-9_]+)", q)
            if match_schema:
                state_name = match_schema.group(2).strip().lower()
                lines = fsm_info.splitlines()
                idx = -1
                found_example = False
                for i, line in enumerate(lines):
                    if line.strip().lower() == state_name:
                        idx = i
                        break
                if idx != -1:
                    # Find the first JSON block (example), then the next JSON block (schema)
                    schema = None
                    for j in range(idx+1, len(lines)):
                        if lines[j].strip().startswith("{"):
                            if not found_example:
                                # Skip the example block
                                found_example = True
                                block = [lines[j]]
                                for k in range(j+1, len(lines)):
                                    block.append(lines[k])
                                    if lines[k].strip().endswith("}"):
                                        break
                                continue
                            else:
                                # This is the schema block
                                block2 = [lines[j]]
                                for n in range(j+1, len(lines)):
                                    block2.append(lines[n])
                                    if lines[n].strip().endswith("}"):
                                        break
                                try:
                                    json.loads("\n".join(block2))
                                    schema = "\n".join(block2)
                                    break
                                except Exception:
                                    continue
                    if schema:
                        return f"```json\n{schema}\n```"
                    return f"Sorry, could not find schema for '{state_name}'."

            return "Sorry, I do not have information on that."
        # All other queries (FSM user cases, etc.) should flow to agentic FSM logic
        return None

if __name__ == "__main__":
    # Example usage of UserInteractionClient for interactive queries

    # Make sure your .env file is set up with the required credentials for Confluence access
    # and that fsm_info.txt exists in the same directory as this script.

    client = UserInteractionClient()
    print("Welcome to the FSM Interactive Client!")
    print("Type your question (type 'exit' to quit):")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            print("Goodbye!")
            break
        response = client.chat(user_input)
        if response is not None:
            print("Bot:", response)
        else:
            print("Bot: This looks like an FSM creation/modification request. Please use the main FSM agent flow for this query.")
    client = UserInteractionClient()
    print("Welcome to the FSM Interactive Client!")
    print("Type your question (type 'exit' to quit):")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            print("Goodbye!")
            break
        response = client.chat(user_input)
        if response is not None:
            print("Bot:", response)
        else:
            print("Bot: This looks like an FSM creation/modification request. Please use the main FSM agent flow for this query.")
            print("Bot: This looks like an FSM creation/modification request. Please use the main FSM agent flow for this query.")
