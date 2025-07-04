import os
from dotenv import load_dotenv, find_dotenv
import httpx
from openai import AzureOpenAI

class GPTClient:
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
            response = self.client.chat.completions.create(
                model=self.deployment_model_name,
                messages=prompts,
                temperature=0.7,
                max_tokens=2000
            )
            return response
        except Exception as e:
            print(f"Error getting chat completion: {str(e)}")
            raise
