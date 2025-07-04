import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv
from llama_index.llms.azure_openai import AzureOpenAI as LlamaAzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, Settings
from llama_index.core.schema import Document
import chromadb
import httpx
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
from openai import APIConnectionError

class FSMMemory:
    """
    FSM Memory Manager for storing and retrieving the most recent conversation.
    """

    def __init__(self):
        self.memory = {}

    def get(self, key):
        return self.memory.get(key)

    def set(self, key, value):
        self.memory[key] = value

    def save_conversation(self, query, response):
        """
        Save a conversation (query and response) to memory.
        """
        if "conversations" not in self.memory:
            self.memory["conversations"] = []
        self.memory["conversations"].append({"query": query, "response": response})

    def get_conversations(self):
        """
        Retrieve all saved conversations.
        """
        return self.memory.get("conversations", [])


class ConfluenceDataExtractor:
    """
    Extract FSM data from a Confluence page and configure a Llama Index database.
    """

    def __init__(self, url):
        load_dotenv(find_dotenv())
        self.url = url
        self.username = os.getenv("ATLASSIAN_BOT_USERNAME")
        self.api_token = os.getenv("ATLASSIAN_BOT_SECRET")
        try:
            self.llm = self.init_llm()
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            self.llm = None
        try:
            self.embed_model = self.init_embedding()
        except Exception as e:
            print(f"Error initializing embedding model: {e}")
            self.embed_model = None
        self.settings()
        self.llm_index = None  # Initialize the Llama Index instance

    def init_llm(self):
        """
        Initialize the LLM (Azure OpenAI) for querying.
        """
        try:
            return LlamaAzureOpenAI(
                engine="gpt-4o",
                model="gpt-4o",
                api_key=os.getenv("GPT4O_AZURE_OPENAI_KEY"),
                azure_endpoint=os.getenv("GPT4O_AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("GPT4O_AZURE_API_VERSION"),
                http_client=httpx.Client(verify=False)
            )
        except Exception as e:
            print(f"Error initializing AzureOpenAI LLM: {e}")
            raise

    def init_embedding(self):
        """
        Initialize the embedding model for Llama Index.
        """
        try:
            return AzureOpenAIEmbedding(
                api_key=os.getenv("ADA_EMBEDDING_AZURE_OPENAI_KEY"),
                azure_endpoint=os.getenv("ADA_EMBEDDING_AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("ADA_EMBEDDING_AZURE_API_VERSION"),
                deployment_model_name=os.getenv("ADA_EMBEDDING_MODEL_DEPLOYMENT_NAME")
            )
        except Exception as e:
            print(f"Error initializing AzureOpenAIEmbedding: {e}")
            raise

    def settings(self):
        """
        Configure global settings for Llama Index.
        """
        if self.llm:
            Settings.llm = self.llm
        if self.embed_model:
            Settings.embed_model = self.embed_model

    def fetch_data_from_url(self):
        """
        Fetch data from the given Confluence URL and return the HTML content.
        """
        start_time = time.time()  # Start timer
        try:
            response = requests.get(
                self.url,
                auth=(self.username, self.api_token)
            )
            response.raise_for_status()
            print(f"Data fetched successfully in {time.time() - start_time:.2f} seconds.")
            return response.text
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to fetch data from URL: {e}")

    def extract_fsm_data(self, html_content):
        """
        Extract FSM data (name, example, description/functionality) from the Confluence table.
        Only FSM name, example, and functionality are needed for planning agent.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        documents = []

        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                cols = row.find_all("td")
                if len(cols) >= 4:
                    fsm_name = cols[0].text.strip()
                    fsm_example = cols[1].text.strip()
                    # fsm_schema = cols[2].text.strip()  # Not needed for planning agent
                    fsm_functionality = cols[3].text.strip()

                    if fsm_name and fsm_functionality:
                        document_text = f"FSM Name: {fsm_name}\nFunctionality: {fsm_functionality}\nExample: {fsm_example}"
                        document = Document(text=document_text, metadata={
                            "fsm_name": fsm_name,
                            "example": fsm_example,
                            "functionality": fsm_functionality
                        })
                        documents.append(document)

        return documents

    def configure_database(self, documents):
        """
        Configure the Llama Index database with the extracted FSM data.
        """
        start_time = time.time()  # Start timer
        # Use EphemeralClient and ensure NO tenant argument is passed (fixes tenant error)
        chroma_client = chromadb.EphemeralClient()
        # Remove any tenant/default_tenant logic, just use collection name
        chroma_collection = chroma_client.get_or_create_collection("fsm_chroma")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=Settings.embed_model
        )
        print(f"Llama Index database configured successfully in {time.time() - start_time:.2f} seconds.")
        return index

    # def query_fsm_data(self, query):
    #     """
    #     Query the Llama Index to find FSMs that match the given functionality.

    #     Args:
    #         query (str): The functionality query.

    #     Returns:
    #         str: The query result from the Llama Index.
    #     """
    #     if not self.llm_index:
    #         raise ValueError("Llama Index is not configured. Please configure the database first.")

    #     query_engine = self.llm_index.as_query_engine()
    #     answer = query_engine.query(query)
    #     return str(answer)
    def query_fsm_data(self, query):
        """
        Query the Llama Index to find FSMs that match the given functionality.
        """
        if not self.llm_index:
            html_content = self.fetch_data_from_url()
            documents = self.extract_fsm_data(html_content)
            try:
                self.llm_index = self.configure_database(documents)
            except Exception as e:
                print(f"Error configuring Llama Index database: {e}")
                return "Error: Could not configure Llama Index database."
        start_time = time.time()  # Start timer
        try:
            query_engine = self.llm_index.as_query_engine()
            answer = query_engine.query(query)
            print(f"Query completed in {time.time() - start_time:.2f} seconds.")
            return str(answer)
        except APIConnectionError as e:
            print(f"OpenAI API connection error: {e}")
            return "Error: Could not connect to OpenAI API. Please check your network or API settings."
        except Exception as e:
            print(f"Error during LLM query: {e}")
            return f"Error: {e}"

    def extract_fsm_templates(self, html_content):
        """
        Extract FSM templates (name, example, functionality) from the Confluence table.
        Returns a list of dicts, each representing a template.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        templates = []
        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                cols = row.find_all("td")
                if len(cols) >= 4:
                    fsm_name = cols[0].text.strip()
                    fsm_example = cols[1].text.strip()
                    # fsm_schema = cols[2].text.strip()  # Not needed for planning agent
                    fsm_functionality = cols[3].text.strip()
                    if fsm_name and fsm_functionality:
                        templates.append({
                            "fsmName": fsm_name,
                            "example": fsm_example,
                            "functionality": fsm_functionality
                        })
        return templates

def ensure_collections_table_exists(db_path):
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS collections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            data TEXT
        )
    """)
    conn.commit()
    conn.close()

# Generalized transition schema for all FSM blueprints
GENERALIZED_TRANSITION_SCHEMA = {
    "type": "object",
    "properties": {
        "conditions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "if": {"type": "string"},
                    "operation": {"type": "string"},
                    "params": {"type": "object"},
                    "thenState": {"type": "string"},
                    "elseState": {"type": "string"}
                },
                "required": ["if", "operation", "params", "thenState"],
                "description": "Each condition object represents a single transition check. Do not group multiple error codes or checks in one object."
            }
        }
    },
    "required": ["conditions"],
    "description": "Optional - List of transitions applicable for the state if any. If not provided, the next state defined will be executed."
}

def patch_schema_with_transition(schema):
    # Add or update the 'transition' property in the schema if not present or incomplete
    if "transition" not in schema or not isinstance(schema["transition"], dict):
        schema["transition"] = GENERALIZED_TRANSITION_SCHEMA
    else:
        # Ensure the schema for transition matches the generalized schema
        schema["transition"].update(GENERALIZED_TRANSITION_SCHEMA)
    return schema

def get_fsm_schemas():
    """
    Return a dictionary mapping state type to its schema.
    This is a stub. Replace with logic to fetch schemas from Confluence or your DB as needed.
    Example return format:
    {
        "performHttpCall": {
            "required": ["id", "type", "configuration"],
            "configuration": {
                "providerEndPointURL": {"required": True},
                "methodType": {"required": True}
                # ...other config fields...
            }
        },
        # ...add other state types...
    }
    """
    # Example hardcoded schemas for demonstration
    schemas = {
        "PerformHttpCall": {
    "required": ["id", "type", "configuration"],
    "configuration": {
        "retryCount": {
            "required": True,
            "type": "integer",
            "description": "The number of times to retry the call."
        },
        "backOffDuration": {
            "required": True,
            "type": "integer",
            "description": "The duration in milliseconds for backoff between retries."
        },
        "timeOutDuration": {
            "required": True,
            "type": "integer",
            "description": "The timeout duration in milliseconds for the call."
        },
        "methodType": {
            "required": True,
            "type": "string",
            "description": "Indicates the HTTP method to use (GET/POST/PUT/DELETE)."
        },
        "providerEndPointURL": {
            "optional": True,
            "type": "string",
            "description": "Endpoint URL for the call."
        },
        "headersList": {
            "optional": True,
            "type": "array",
            "description": "An array of HTTP headers to pass to the call.",
            "item_schema": {
                "type": "object",
                "required": ["header", "value"],
                "properties": {
                    "header": {
                        "type": "string",
                        "description": "The header name."
                    },
                    "value": {
                        "type": "string",
                        "description": "The header value."
                    }
                }
            }
        },
        "formData": {
            "optional": True,
            "type": "array",
            "description": "An array of Form Data to pass to the call.",
            "item_schema": {
                "type": "object",
                "required": ["key", "value"],
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The form data key."
                    },
                    "value": {
                        "type": "string",
                        "description": "The form data value."
                    }
                }
            }
        },
        "siConfiguration": {
            "optional": True,
            "type": "object",
            "description": "SI Configuration definition for the call.",
            "required": ["connectorHost", "connectorPort", "siInstance", "sap"],
            "properties": {
                "connectorHost": {
                    "type": "string",
                    "description": "Name of the SI connector to use for the call."
                },
                "connectorPort": {
                    "type": "string",
                    "description": "Port of the SI connector to use for the call."
                },
                "siInstance": {
                    "type": "string",
                    "description": "SI instance to use for the call."
                },
                "sap": {
                    "type": "string",
                    "description": "SAP to target for the call."
                },
                "requiresEWA": {
                    "type": "boolean",
                    "description": "Indicates if integration with EWA is required."
                },
                "soapAction": {
                    "type": "string",
                    "description": "Indicates if SOAP Action is required, not required for REST calls."
                }
            }
        },
        "emulatedClientProfile": {
            "optional": True,
            "type": "string",
            "description": "Emulated client profile for the call - requires definition in MapClientProfileToSIConf.json."
        }
    },
    "inputFrom": {
        "optional": True,
        "type": "string",
        "description": "Input of the state, can be a literal or token reference."
    },
    "saveStateOutput": {
        "optional": True,
        "type": "boolean",
        "description": "If provided, stores the state output in the context."
    },
    "saveStateOutputWithKey": {
        "optional": True,
        "type": "string",
        "description": "Overrides the key with which the output is stored in the context."
    },
    "stopExecution": {
        "optional": True,
        "type": "boolean",
        "description": "If true, FSM execution stops after this state."
    },
    "transition": {
        "optional": True,  # <-- transitions are optional
        "type": "object",
        "description": "Optional - List of transitions applicable for the state if any.",
        "properties": {
            "conditions": {
                "type": "array",
                "items": {
                    "type": "object"
                }
            }
        }
    }
},
        "loadStaticFile": {
    "required": ["id", "type", "configuration"],
    "configuration": {
        "fileName": {
            "required": True,
            "type": "string",
            "description": "The file name to load."
        }
    },
    "inputFrom": {
        "optional": True,
        "type": "string",
        "description": "The input of the state, can be a value or a reference to a token key."
    },
    "saveStateOutput": {
        "optional": True,
        "type": "boolean",
        "description": "If provided, will store the state output in the context."
    },
    "saveStateOutputWithKey": {
        "optional": True,
        "type": "string",
        "description": "Allows overriding the key with which the output is stored in the context."
    },
    "stopExecution": {
        "optional": True,
        "type": "boolean",
        "description": "If set to true, execution stops after this state."
    },
    "transition": {
        "optional": True,  # <-- transitions are optional
        "type": "object",
        "description": "Optional - List of transitions applicable for the state if any.",
        "properties": {
            "conditions": {
                "type": "array",
                "items": {
                    "type": "object"
                }
            }
        }
    }
},
        "addXPathValuesToToken": {
    "required": ["id", "type", "configuration"],
    "configuration": {
        "keyXPathsList": {
            "required": True,
            "type": "array",
            "description": "List of key and xPath to extract the value from the input and save it in the token.",
            "item_schema": {
                "type": "object",
                "required": ["key", "xPath"],
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key in the token where the value will be saved."
                    },
                    "xPath": {
                        "type": "string",
                        "description": "The value to extract the xPath from the input."
                    },
                    "namespaces": {
                        "type": "object",
                        "description": "The namespace of the xPath."
                    }
                }
            }
        }
    },
    "inputFrom": {
        "optional": True,
        "type": "string",
        "description": "The input of the state, can be a value or a reference to a token key."
    },
    "saveStateOutput": {
        "optional": True,
        "type": "boolean",
        "description": "If provided, will store the state output in the context."
    },
    "saveStateOutputWithKey": {
        "optional": True,
        "type": "string",
        "description": "Allow to override the key with which the output is stored in the context."
    },
    "stopExecution": {
        "optional": True,
        "type": "boolean",
        "description": "If set to true, the execution will stop after the state is executed."
    },
    "transition": {
        "optional": True,
        "type": "object",
        "description": "Optional - List of transitions applicable for the state if any.",
        "properties": {
            "conditions": {
                "type": "array",
                "items": {
                    "type": "object"
                }
            }
        }
    }
},
        "addJsonPathValuesToToken": {
    "required": ["id", "type", "configuration"],
    "configuration": {
        "keyJsonPathList": {
            "required": True,
            "type": "array",
            "description": "List of key and jsonPath to extract the value from the input and save it in the token.",
            "item_schema": {
                "type": "object",
                "required": ["jsonPath", "key"],
                "properties": {
                    "jsonPath": {
                        "type": "string",
                        "description": "The value to extract the jsonPath from the input."
                    },
                    "jsonPathParams": {
                        "type": "array",
                        "description": "A list of values that will be replaced dynamically in the jsonPath using {pathParam[i]}.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "key": {
                        "type": "string",
                        "description": "The key in the token where the value will be saved."
                    },
                    "saveAsPrimitive": {
                        "type": "boolean",
                        "description": "If the value is a single value in a JSONArray, save the first element as the value of the key."
                    }
                }
            }
        }
    },
    "inputFrom": {
        "optional": True,
        "type": "string",
        "description": "Input of the state, value or token reference. Defaults to previous state output."
    },
    "saveStateOutput": {
        "optional": True,
        "type": "boolean",
        "description": "If provided, stores the output in the context."
    },
    "saveStateOutputWithKey": {
        "optional": True,
        "type": "string",
        "description": "Overrides the key used to store output in the context."
    },
    "stopExecution": {
        "optional": True,
        "type": "boolean",
        "description": "If true, FSM execution stops after this state."
    },
    "transition": {
        "optional": True,
        "type": "object",
        "description": "Optional transitions after this state.",
        "properties": {
            "conditions": {
                "type": "array",
                "items": {
                    "type": "object"
                }
            }
        }
    }
},
        "transformJsonUsingJolt": {
    "required": ["id", "type", "configuration"],
    "configuration": {
        "name": {
            "required": True,
            "type": "string",
            "description": "Name of the Jolt Spec file"
        }
    },
    "inputFrom": {
        "optional": True,
        "type": "string",
        "description": "Optional - The input of the state, can be a value or a reference to a token key."
    },
    "saveStateOutput": {
        "optional": True,
        "type": "boolean",
        "description": "Optional - If provided, will store the state output in the context."
    },
    "saveStateOutputWithKey": {
        "optional": True,
        "type": "string",
        "description": "Optional - Allows overriding the key with which the output is stored in the context."
    },
    "stopExecution": {
        "optional": True,
        "type": "boolean",
        "description": "Optional - If set to true, the execution will stop after the state is executed."
    },
    "transition": {
        "optional": True,
        "type": "object",
        "description": "Optional - List of transitions applicable for the state if any.",
        "properties": {
            "conditions": {
                "type": "array",
                "items": {
                    "type": "object"
                }
            }
        }
    }
},
        "transformDateFormat": {
    "required": ["id", "type"],
    "configuration": {
        "dateFormatConversionInputs": {
            "required": True,
            "type": "array",
            "description": "List of date conversion input objects.",
            "item_schema": {
                "type": "object",
                "properties": {
                    "inputDate": {
                        "type": "string",
                        "description": "Input date to be transformed"
                    },
                    "inputDateFormat": {
                        "type": "string",
                        "description": "Format to resolve input date"
                    },
                    "inputTimeZone": {
                        "type": "string",
                        "description": "Input time zone"
                    },
                    "outputDateFormat": {
                        "type": "string",
                        "description": "Format to resolve output date"
                    },
                    "outputTimeZone": {
                        "type": "string",
                        "description": "Output time zone"
                    },
                    "saveTransformedDateTo": {
                        "type": "string",
                        "description": "Unique key to save transformed date to token map"
                    }
                }
            }
        }
    },
    "inputFrom": {
        "optional": True,
        "type": "string",
        "description": "Optional - The input of the state, can be a value or a reference to a token key."
    },
    "saveStateOutput": {
        "optional": True,
        "type": "boolean",
        "description": "Optional - If provided, will store the state output in the context."
    },
    "saveStateOutputWithKey": {
        "optional": True,
        "type": "string",
        "description": "Optional - Allows overriding the key with which the output is stored in the context."
    },
    "stopExecution": {
        "optional": True,
        "type": "boolean",
        "description": "Optional - If set to true, the execution will stop after the state is executed."
    },
    "transition": {
        "optional": True,
        "type": "object",
        "description": "Optional - List of transitions applicable for the state if any.",
        "properties": {
            "conditions": {
                "type": "array",
                "description": "Conditions to evaluate for transitioning to the next state.",
                "items": {
                    "type": "object"
                }
            }
        }
    }
}
,
        "transformDateFormat": {
            "required": ["id", "type", "configuration"],
            "configuration": {
                "dateFormatConversionInputs": {
                    "required": True,
                    "item_schema": {
                        "required": [
                            "inputDate",
                            "inputDateFormat",
                            "outputDateFormat",
                            "inputTimeZone",
                            "outputTimeZone"
                        ]
                    }
                }
                # Each item in dateFormatConversionInputs must have the above fields
            },
                "inputFrom": {"optional": True},
                "saveStateOutput": {"optional": True},
                "saveStateOutputWithKey": {"optional": True},
                "stopExecution": {"optional": True},
                "transition": {"optional": True}
        },
        "addHeaderAndFooter": {
    "required": ["id", "type", "configuration"],
    "configuration": {
        "footerToAppend": {
            "required": True,
            "type": "string",
            "description": "Suffix to append to the input."
        },
        "headerToAppend": {
            "required": True,
            "type": "string",
            "description": "Prefix to append to the input."
        }
    },
    "inputFrom": {
        "optional": True,
        "type": "string",
        "description": "Optional - The input of the state, can be a value or a reference to a token key."
    },
    "saveStateOutput": {
        "optional": True,
        "type": "boolean",
        "description": "Optional - If provided, will store the state output in the context."
    },
    "saveStateOutputWithKey": {
        "optional": True,
        "type": "string",
        "description": "Optional - Allows overriding the key with which the output is stored in the context."
    },
    "stopExecution": {
        "optional": True,
        "type": "boolean",
        "description": "Optional - If set to true, the execution will stop after the state is executed."
    },
    "transition": {
        "optional": True,
        "type": "object",
        "description": "Optional - List of transitions applicable for the state if any.",
        "properties": {
            "conditions": {
                "type": "array",
                "description": "Conditions to evaluate for transitioning to the next state.",
                "items": {
                    "type": "object"
                }
            }
        }
    }
}
,
        "evaluateRule": {
    "required": ["id", "type", "configuration"],
    "configuration": {
        "input": {
            "required": True,
            "type": "array",
            "description": "List of inputs to evaluate the rule. The order of the inputs should match with the inputs mentioned in the CSV rule file.",
            "item_schema": {
                "type": "string"
            }
        },
        "ruleFile": {
            "required": True,
            "type": "string",
            "description": "Rule file to evaluate the inputs."
        }
    },
    "inputFrom": {
        "optional": True,
        "type": "string",
        "description": "The input of the state, can be a value or a reference to a token key."
    },
    "saveStateOutput": {
        "optional": True,
        "type": "boolean",
        "description": "If provided, will store the state output in the context."
    },
    "saveStateOutputWithKey": {
        "optional": True,
        "type": "string",
        "description": "Allows overriding the key with which the output is stored in the context."
    },
    "stopExecution": {
        "optional": True,
        "type": "boolean",
        "description": "If set to true, the execution will stop after the state is executed."
    },
    "transition": {
        "optional": True,
        "type": "object",
        "description": "List of transitions applicable for the state if any.",
        "properties": {
            "conditions": {
                "type": "array",
                "description": "Conditions to evaluate for transitioning to the next state.",
                "items": {
                    "type": "object"
                }
            }
        }
    }
},
        "manipulateString": {
    "required": ["id", "type"],
    "configuration": {
        "inputString": {
            "optional": True,
            "type": "string",
            "description": "String which has to be manipulated"
        },
        "operationToPerform": {
            "optional": True,
            "type": "string",
            "description": "Operation to be performed on the input string"
        },
        "params": {
            "optional": True,
            "type": "array",
            "description": "List of parameters required for the operation",
            "item_schema": {
                "type": "string"
            }
        }
    },
    "inputFrom": {
        "optional": True,
        "type": "string",
        "description": "The input of the state, can be a value or a reference to a token key."
    },
    "saveStateOutput": {
        "optional": True,
        "type": "boolean",
        "description": "If Provided, will store the state output in the context and could be reused later."
    },
    "saveStateOutputWithKey": {
        "optional": True,
        "type": "string",
        "description": "Allow to override the key with which the output is stored in the context."
    },
    "stopExecution": {
        "optional": True,
        "type": "boolean",
        "description": "If set to true, the execution will stop after the state is executed."
    },
    "transition": {
        "optional": True,
        "type": "object",
        "description": "List of transitions applicable for the state if any.",
        "properties": {
            "conditions": {
                "type": "array",
                "description": "Conditions to evaluate for transitioning to the next state.",
                "items": {
                    "type": "object"
                }
            }
        }
    }
},
        "addNodesToJsonUsingJsonPath": {
    "required": ["id", "type", "configuration"],
    "configuration": {
        "jsonPathParamsList": {
            "required": True,
            "type": "array",
            "description": "List of nodes to add to the input using JsonPath.",
            "item_schema": {
                "type": "object",
                "required": ["jsonPath", "value"],
                "properties": {
                    "jsonPath": {
                        "type": "string",
                        "description": "Path where the node should be added in the json document."
                    },
                    "jsonPathParams": {
                        "type": "array",
                        "description": "A list of values that will be replaced dynamically in the jsonPath using {pathParam[i]}.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to be added to the input."
                    }
                }
            }
        }
    },
    "inputFrom": {
        "optional": True,
        "type": "string",
        "description": "The input of the state, can be a value or a reference to a token key."
    },
    "saveStateOutput": {
        "optional": True,
        "type": "boolean",
        "description": "If Provided, will store the state output in the context and could be reused later."
    },
    "saveStateOutputWithKey": {
        "optional": True,
        "type": "string",
        "description": "Allow to override the key with which the output is stored in the context."
    },
    "stopExecution": {
        "optional": True,
        "type": "boolean",
        "description": "If set to true, the execution will stop after the state is executed."
    },
    "transition": {
        "optional": True,
        "type": "object",
        "description": "List of transitions applicable for the state if any.",
        "properties": {
            "conditions": {
                "type": "array",
                "description": "Conditions to evaluate for transitioning to the next state.",
                "items": {
                    "type": "object"
                }
            }
        }
    }
}
,
        "removeNodesFromJsonUsingJsonPath": {
            "required": ["id", "type", "configuration"],
            "configuration": {
                "jsonPathParamsList": {"required": True}
            },
                "inputFrom": {"optional": True},
                "saveStateOutput": {"optional": True},
                "saveStateOutputWithKey": {"optional": True},
                "stopExecution": {"optional": True},
                "transition": {"optional": True}
        },
        "ValidateJsonSchema": {
    "required": ["id", "type", "configuration"],
    "configuration": {
        "jsonSchemaFile": {
            "required": True,
            "type": "string",
            "description": "The name of the file containing the JSON schema."
        }
    },
    "inputFrom": {
        "optional": True,
        "type": "string",
        "description": "The input of the state, can be a value or a reference to a token key."
    },
    "saveStateOutput": {
        "optional": True,
        "type": "boolean",
        "description": "If provided, will store the state output in the context and could be reused later."
    },
    "saveStateOutputWithKey": {
        "optional": True,
        "type": "string",
        "description": "Allows overriding the key with which the output is stored in the context."
    },
    "stopExecution": {
        "optional": True,
        "type": "boolean",
        "description": "If set to true, the execution will stop after the state is executed."
    },
    "transition": {
        "optional": True,
        "type": "object",
        "description": "List of transitions applicable for the state if any.",
        "properties": {
            "conditions": {
                "type": "array",
                "description": "Conditions to evaluate for transitioning to the next state.",
                "items": {
                    "type": "object"
                }
            }
        }
    }
}
,
        "addValuesToToken": {
    "required": ["id", "type", "configuration"],
    "configuration": {
        "keyValuesList": {
            "required": True,
            "type": "array",
            "description": "List of key and value to save in the token",
            "item_schema": {
                "type": "object",
                "required": ["key", "value"],
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key in the token where the value will be saved."
                    },
                    "value": {
                        "type": "string",
                        "description": "The value to set in the token"
                    }
                }
            }
        }
    },
    "inputFrom": {
        "optional": True,
        "type": "string",
        "description": "The input of the state, can be a value or a reference to a token key."
    },
    "saveStateOutput": {
        "optional": True,
        "type": "boolean",
        "description": "If Provided, will store the state output in the context and could be reused later."
    },
    "saveStateOutputWithKey": {
        "optional": True,
        "type": "string",
        "description": "Allow to override the key with which the output is stored in the context."
    },
    "stopExecution": {
        "optional": True,
        "type": "boolean",
        "description": "If set to true, the execution will stop after the state is executed."
    },
    "transition": {
        "optional": True,
        "type": "object",
        "description": "List of transitions applicable for the state if any.",
        "properties": {
            "conditions": {
                "type": "array",
                "description": "Conditions to evaluate for transitioning to the next state.",
                "items": {
                    "type": "object"
                }
            }
        }
    }
}
,
        "ValidateXMLSchema": {
    "required": ["id", "type", "configuration"],
    "configuration": {
        "xsdFile": {
            "required": True,
            "type": "string",
            "description": "The name of the file containing the XSD schema."
        }
    },
    "inputFrom": {
        "optional": True,
        "type": "string",
        "description": "The input of the state, can be a value or a reference to a token key."
    },
    "saveStateOutput": {
        "optional": True,
        "type": "boolean",
        "description": "If provided, will store the state output in the context and could be reused later."
    },
    "saveStateOutputWithKey": {
        "optional": True,
        "type": "string",
        "description": "Allows overriding the key with which the output is stored in the context."
    },
    "stopExecution": {
        "optional": True,
        "type": "boolean",
        "description": "If set to true, the execution will stop after the state is executed."
    },
    "transition": {
        "optional": True,
        "type": "object",
        "description": "Optional - List of transitions applicable for the state if any.",
        "properties": {
            "conditions": {
                "type": "array",
                "description": "Conditions to evaluate for transitioning to the next state.",
                "items": {
                    "type": "object"
                }
            }
        }
    }
},
        "transformXmlUsingXPath": {
    "required": ["id", "type", "configuration"],
    "configuration": {
        "params": {
            "required": True,
            "type": "object",
            "description": "List of key-value pairs where the key is the XPath and the value is the element to update."
        }
    },
    "inputFrom": {
        "optional": True,
        "type": "string",
        "description": "The input of the state, can be a value or a reference to a token key."
    },
    "saveStateOutput": {
        "optional": True,
        "type": "boolean",
        "description": "If provided, the state output will be stored in the context."
    },
    "saveStateOutputWithKey": {
        "optional": True,
        "type": "string",
        "description": "Override the key with which the output is stored in the context."
    },
    "stopExecution": {
        "optional": True,
        "type": "boolean",
        "description": "If set to true, FSM execution will stop after this state."
    },
    "transition": {
        "optional": True,
        "type": "object",
        "description": "List of transitions applicable for the state if any.",
        "properties": {
            "conditions": {
                "type": "array",
                "description": "Conditions to evaluate for transitioning to the next state.",
                "items": {
                    "type": "object"
                }
            }
        }
    }
}
,
        "transformXmlToJson": {
    "required": ["id", "type", "configuration"],
    "configuration": {
        "transformElementToArray": {
            "optional": True,
            "type": "array",
            "description": "A list of elements (as string path arrays) that should be transformed to arrays even if they contain a single element.",
            "items": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            }
        }
    },
    "inputFrom": {
        "optional": True,
        "type": "string",
        "description": "The input of the state, can be a value or a reference to a token key."
    },
    "saveStateOutput": {
        "optional": True,
        "type": "boolean",
        "description": "If provided, the state output will be stored in the context."
    },
    "saveStateOutputWithKey": {
        "optional": True,
        "type": "string",
        "description": "Override the key with which the output is stored in the context."
    },
    "stopExecution": {
        "optional": True,
        "type": "boolean",
        "description": "If set to true, FSM execution will stop after this state."
    },
    "transition": {
        "optional": True,
        "type": "object",
        "description": "List of transitions applicable for the state if any.",
        "properties": {
            "conditions": {
                "type": "array",
                "description": "Conditions to evaluate for transitioning to the next state.",
                "items": {
                    "type": "object"
                }
            }
        }
    }
}
,
        "transformXmlUsingXslt": {
    "required": ["id", "type", "configuration"],
    "configuration": {
        "name": {
            "required": True,
            "type": "string",
            "description": "The name of the file containing the XSLT transformation."
        },
        "params": {
            "optional": True,
            "type": "array",
            "description": "A list of parameters from the token that could be supplied to the XSLT Transformation.",
            "items": {
                "type": "string"
            }
        }
    },
    "inputFrom": {
        "optional": True,
        "type": "string",
        "description": "The input of the state, can be a value or a reference to a token key."
    },
    "saveStateOutput": {
        "optional": True,
        "type": "boolean",
        "description": "If provided, the state output will be stored in the context."
    },
    "saveStateOutputWithKey": {
        "optional": True,
        "type": "string",
        "description": "Override the key with which the output is stored in the context."
    },
    "stopExecution": {
        "optional": True,
        "type": "boolean",
        "description": "If set to true, FSM execution will stop after this state."
    },
    "transition": {
        "optional": True,
        "type": "object",
        "description": "List of transitions applicable for the state if any.",
        "properties": {
            "conditions": {
                "type": "array",
                "description": "Conditions to evaluate for transitioning to the next state.",
                "items": {
                    "type": "object"
                }
            }
        }
    }
}
,
        "transformJsonToXml": {
    "required": ["id", "type"],
    "inputFrom": {
        "optional": True,
        "type": "string",
        "description": "The input of the state, can be a value or a reference to a token key."
    },
    "saveStateOutput": {
        "optional": True,
        "type": "boolean",
        "description": "If provided, the state output will be saved in the context."
    },
    "saveStateOutputWithKey": {
        "optional": True,
        "type": "string",
        "description": "Override the key with which the output is stored in the context."
    },
    "stopExecution": {
        "optional": True,
        "type": "boolean",
        "description": "If set to true, FSM execution will stop after this state."
    },
    "transition": {
        "optional": True,
        "type": "object",
        "description": "List of transitions applicable for the state if any.",
        "properties": {
            "conditions": {
                "type": "array",
                "description": "Conditions to evaluate for transitioning to the next state.",
                "items": {
                    "type": "object"
                }
            }
        }
    }
}
,
        "generateTransactionId": {
            "required": ["id", "type", "configuration"],
            "configuration": {
                # "UUID_Params_List" and "uuidParamsList" are both accepted, both optional
                # Do not mark as required
            },
                "inputFrom": {"optional": True},
                "saveStateOutput": {"optional": True},
                "saveStateOutputWithKey": {"optional": True},
                "stopExecution": {"optional": True},
                "transition": {"optional": True}
        },
        "forEachLoopState": {
    "required": ["id", "type", "configuration"],
    "configuration": {
        "loopOverOn": {
            "required": True,
            "type": "string",
            "description": "The elements on which we want to loop, generally an array in the token."
        },
        "loopOverStates": {
            "required": True,
            "type": "array",
            "description": "The list of states to execute in the loop.",
            "items": {
                "type": "object"
            }
        },
        "passOnToLoopOverStatesAs": {
            "required": True,
            "type": "string",
            "description": "The name of the variable that can be used in the states defined in loopOverStates."
        },
        "executeInParallel": {
            "optional": True,
            "type": "boolean",
            "description": "If the loop should be executed in parallel. Default is false."
        }
    },
    "inputFrom": {
        "optional": True,
        "type": "string",
        "description": "The input of the state, can be a value or a reference to a token key."
    },
    "saveStateOutput": {
        "optional": True,
        "type": "boolean",
        "description": "If provided, will store the state output in the context and could be reused later."
    },
    "saveStateOutputWithKey": {
        "optional": True,
        "type": "string",
        "description": "Override the key with which the output is stored in the context."
    },
    "stopExecution": {
        "optional": True,
        "type": "boolean",
        "description": "If set to true, FSM execution will stop after this state."
    },
    "transition": {
        "optional": True,
        "type": "object",
        "description": "List of transitions applicable for the state if any.",
        "properties": {
            "conditions": {
                "type": "array",
                "description": "Conditions to evaluate for transitioning to the next state.",
                "items": {
                    "type": "object"
                }
            }
        }
    }
}
,
        "customizeErrorState": {
            "required": ["id", "type", "configuration"],
            "configuration": {
                "inputError": {"required": True}
                # "overrideInputErrorFromFile" is optional
            },
                "inputFrom": {"optional": True},
                "saveStateOutput": {"optional": True},
                "saveStateOutputWithKey": {"optional": True},
                "stopExecution": {"optional": True},
                "transition": {"optional": True}
        },
        "performSetOperations": {
            "required": ["id", "type", "configuration"],
            "configuration": {
                "parentSetKey": {"required": True},
                "childSetKey": {"required": True},
                "operationToPerform": {"required": True}
            },
                "inputFrom": {"optional": True},
                "saveStateOutput": {"optional": True},
                "saveStateOutputWithKey": {"optional": True},
                "stopExecution": {"optional": True},
                "transition": {"optional": True}
        },
        "BlockState": {
    "required": ["id", "type", "configuration"],
    "configuration": {
        "states": {
            "required": True,
            "type": "array",
            "description": "The list of states to execute in a single block.",
            "items": {
                "type": "object"
            }
        }
    },
    "inputFrom": {
        "optional": True,
        "type": "string",
        "description": "The input of the state, can be a value or a reference to a token key."
    },
    "saveStateOutput": {
        "optional": True,
        "type": "boolean",
        "description": "If provided, will store the state output in the context and could be reused later."
    },
    "saveStateOutputWithKey": {
        "optional": True,
        "type": "string",
        "description": "Override the key with which the output is stored in the context."
    },
    "stopExecution": {
        "optional": True,
        "type": "boolean",
        "description": "If set to true, FSM execution will stop after this state."
    },
    "transition": {
        "optional": True,
        "type": "object",
        "description": "List of transitions applicable for the state if any.",
        "properties": {
            "conditions": {
                "type": "array",
                "description": "Conditions to evaluate for transitioning to the next state.",
                "items": {
                    "type": "object"
                }
            }
        }
    }
}
,
        "parallelExecutionState": {
            "required": ["id", "type", "configuration"],
            "configuration": {
                "stateList": {"required": True}
            },
                "inputFrom": {"optional": True},
                "saveStateOutput": {"optional": True},
                "saveStateOutputWithKey": {"optional": True},
                "stopExecution": {"optional": True},
                "transition": {"optional": True}
        },
        "compareTwoJsonsAndRemoveNodeFromParentJson": {
            "required": ["id", "type", "configuration"],
            "configuration": {
                "parentJsonArray": {"required": True},
                "childJsonArray": {"required": True}
            },
                "inputFrom": {"optional": True},
                "saveStateOutput": {"optional": True},
                "saveStateOutputWithKey": {"optional": True},
                "stopExecution": {"optional": True},
                "transition": {"optional": True}
        },
        "compareTwoListsAndRemoveFaultEntriesFromOriginal": {
            "required": ["id", "type", "configuration"],
            "configuration": {
                "parentJsonArray": {"required": True},
                "childJsonArray": {"required": True}
            },
                "inputFrom": {"optional": True},
                "saveStateOutput": {"optional": True},
                "saveStateOutputWithKey": {"optional": True},
                "stopExecution": {"optional": True},
                "transition": {"optional": True}
        },
        "addLastExecutedStateStatusToToken": {
            "required": ["id", "type", "configuration"],
            "configuration": {
                "key": {"required": True},
                # "saveAsList" is optional
            },
                "inputFrom": {"optional": True},
                "saveStateOutput": {"optional": True},
                "saveStateOutputWithKey": {"optional": True},
                "stopExecution": {"optional": True},
                "transition": {"optional": True}
        },
        "addErrorValuesToToken": {
            "required": ["id", "type", "configuration"],
            "configuration": {
                "keyValuesList": {"required": True}
                # "saveAsSet" is optional
            },
                "inputFrom": {"optional": True},
                "saveStateOutput": {"optional": True},
                "saveStateOutputWithKey": {"optional": True},
                "stopExecution": {"optional": True},
                "transition": {"optional": True}
        },
        "collectFailuresFromLoopState": {
            "required": ["id", "type", "configuration"],
            "configuration": {
                "key": {"required": True},
                "value": {"required": True}
            },
                "inputFrom": {"optional": True},
                "saveStateOutput": {"optional": True},
                "saveStateOutputWithKey": {"optional": True},
                "stopExecution": {"optional": True},
                "transition": {"optional": True}
        },
        "ValidateXMLSchema": {
    "required": ["id", "type", "configuration"],
    "configuration": {
        "xsdFile": {
            "required": True,
            "type": "string",
            "description": "The name of the file containing the XSD schema."
        }
    },
    "inputFrom": {
        "optional": True,
        "type": "string",
        "description": "The input of the state, can be a value or a reference to a token key."
    },
    "saveStateOutput": {
        "optional": True,
        "type": "boolean",
        "description": "If provided, will store the state output in the context and could be reused later."
    },
    "saveStateOutputWithKey": {
        "optional": True,
        "type": "string",
        "description": "Allows overriding the key with which the output is stored in the context."
    },
    "stopExecution": {
        "optional": True,
        "type": "boolean",
        "description": "If set to true, the execution will stop after the state is executed."
    },
    "transition": {
        "optional": True,
        "type": "object",
        "description": "List of transitions applicable for the state if any.",
        "properties": {
            "conditions": {
                "type": "array",
                "description": "Conditions to evaluate for transitioning to the next state.",
                "items": {
                    "type": "object"
                }
            }
        }
    }
}
,
        "importFsm": {
    "required": ["id", "type", "configuration"],
    "configuration": {
        "fsmName": {
            "required": True,
            "type": "string",
            "description": "Importable FSM name is required to import to states"
        }
    },
    "inputFrom": {
        "optional": True,
        "type": "string",
        "description": "The input of the state, can be a value or a reference to a token key."
    },
    "saveStateOutput": {
        "optional": True,
        "type": "boolean",
        "description": "If provided, will store the state output in the context and could be reused later."
    },
    "saveStateOutputWithKey": {
        "optional": True,
        "type": "string",
        "description": "Allow to override the key with which the output is stored in the context."
    },
    "stopExecution": {
        "optional": True,
        "type": "boolean",
        "description": "If set to true, the execution will stop after the state is executed."
    },
    "transition": {
        "optional": True,
        "type": "object",
        "description": "List of transitions applicable for the state if any.",
        "properties": {
            "conditions": {
                "type": "array",
                "description": "Conditions to evaluate for transitioning to the next state.",
                "items": {
                    "type": "object"
                }
            }
        }
    }
}
,
        "transformXmlUsingXQuery": {
    "required": ["id", "type", "configuration"],
    "configuration": {
        "name": {
            "required": True,
            "type": "string",
            "description": "The name of the file containing the XQuery transformation."
        }
    },
    "inputFrom": {
        "optional": True,
        "type": "string",
        "description": "The input of the state, can be a value or a reference to a token key."
    },
    "saveStateOutput": {
        "optional": True,
        "type": "boolean",
        "description": "If provided, will store the state output in the context and could be reused later."
    },
    "saveStateOutputWithKey": {
        "optional": True,
        "type": "string",
        "description": "Allows overriding the key with which the output is stored in the context."
    },
    "stopExecution": {
        "optional": True,
        "type": "boolean",
        "description": "If set to true, the execution will stop after the state is executed."
    },
    "transition": {
        "optional": True,
        "type": "object",
        "description": "List of transitions applicable for the state if any.",
        "properties": {
            "conditions": {
                "type": "array",
                "description": "Conditions to evaluate for transitioning to the next state.",
                "items": {
                    "type": "object"
                }
            }
        }
    }
}
,
        # Add more schemas as needed
    }
    # Patch every schema with the generalized transition schema if it supports transitions
    for state_type, schema in schemas.items():
        # If the schema is for a state that can have transitions, patch it
        if (
            "transition" in schema.get("required", [])
            or "transition" in schema
            or state_type in [
                "performHttpCall",
                "forEachLoopState",
                "addJsonPathValuesToToken",
                "addNodesToJsonUsingJsonPath",
                "transformJsonUsingJolt",
                "customizeErrorState",
                "loadStaticFile"
            ]
        ):
            patch_schema_with_transition(schema)
    return schemas


# confluence_url = "https://amadeus.atlassian.net/wiki/spaces/CATS/pages/2509677865/FSM-Examples"


confluence_url = "https://amadeus.atlassian.net/wiki/spaces/CATS/pages/2739174275/states"


extractor = ConfluenceDataExtractor(confluence_url)
def get_fsm_templates_for_ui():
    """
    Returns a list of FSM templates for use in the UI sidebar tab 'fsm templates'.
    Each template contains the FSM name and its example, formatted as:
    template <fsm_name>
    
    <fsm_example>
    """
    try:
        html_content = extractor.fetch_data_from_url()
        templates = []
        for tpl in extractor.extract_fsm_templates(html_content):
            # Only add if both name and example are present
            if tpl.get("fsmName") and tpl.get("example"):
                templates.append({
                    "name": tpl["fsmName"],
                    "template": f"template {tpl['fsmName']}\n\n{tpl['example']}"
                })
        return templates
    except Exception as e:
        print(f"Error fetching/extracting FSM templates: {e}")
        return []

app = FastAPI()

# Allow CORS for local development (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/fsm-templates")
def api_get_fsm_templates():
    """
    API endpoint to get FSM templates for the UI sidebar tab.
    Extracts from Confluence only.
    """
    templates = get_fsm_templates_for_ui()
    return JSONResponse(content=templates)

# You can comment out or remove this function if not needed:
# def load_fsm_templates_from_file(filepath="fsm_info.txt"):
#     """
#     Loads FSM templates from a file. Supports JSON array or JSON Lines format.
#     Returns a list of FSM templates.
#     """
#     templates = []
#     if not os.path.exists(filepath):
#         return templates
#     with open(filepath, "r", encoding="utf-8") as f:
#         content = f.read().strip()
#         try:
#             # Try parsing as a JSON array
#             data = json.loads(content)
#             if isinstance(data, list):
#                 templates = data
#             else:
#                 templates = [data]
#         except json.JSONDecodeError:
#             # Try parsing as JSON Lines (one JSON object per line)
#             templates = []
#             for line in content.splitlines():
#                 line = line.strip()
#                 if line:
#                     try:
#                         templates.append(json.loads(line))
#                     except Exception:
#                         continue
#     return templates

if __name__ == "__main__":
    # Example test case for query_fsm_data
    memory = FSMMemory()
    # This is a sample query that should match a string manipulation FSM if present in your DB or Confluence
    test_query = "create an fsm such that it Takes the input string from json , extract the first 5 characters starting from index 0, construct a response by appending 'The manipulated string is ' to the header and result to the footer."
    try:
     
        extractor = ConfluenceDataExtractor(url="https://amadeus.atlassian.net/wiki/spaces/CATS/pages/2739174275/states")
        result = extractor.query_fsm_data(test_query)
        print("FSM Query Result:\n", result)
        print("[TEST] No ConfluenceDataExtractor configured. Please add your extractor and test.")
    except Exception as e:
        print(f"[TEST] Error during FSM query: {e}")
