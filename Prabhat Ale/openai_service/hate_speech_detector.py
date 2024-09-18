import json
import openai
from json.decoder import JSONDecodeError
from config import (
    role,
    top_p,
    api_key,
    max_tokens,
    model_name,
    temperature,
    presence_penalty,
    frequency_penalty
)

class HateSpeechDetector:

    openai.api_key = api_key

    def __init__(self):
        """
        Initialize the ProductSentimentAnalyzer with model settings and API key.
        """

        self.model_name = model_name
        self.role = role
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty


    def extract_json_content(self, text):
        """
        Extracts JSON content from a given text.

        Args:
            text (str): The input text containing JSON content.

        Returns:
            str: The extracted JSON content as a string.
        """
        start_pos = text.find("{")
        end_pos = text.rfind("}") + 1
        json_string = text[start_pos:end_pos]
        return json_string

    

    def identify_hatespeech_target(self, prompt_template: str):
        """
        Analyze a product query email and extract relevant information.

        Args:
            query (str): The input query email text.

        Returns:
            dict: A dictionary containing the extracted information, including product sentiment, titles, and categories.
        """
        self.prompt_template = prompt_template
        completion = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": self.role,
                "content": self.prompt_template
                }
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty
        )
        response = completion['choices'][0]['message']['content']
        tokens_used = completion['usage']['total_tokens']
        try:
            json_string = self.extract_json_content(response)
            json_response = json.loads(json_string)
            final_json_response = {key: value for key, value in json_response.items() if value is not None and value != ""}
            return final_json_response
        except Exception as e:
            pass

