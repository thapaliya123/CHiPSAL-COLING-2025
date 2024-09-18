import json
from typing import List
from utils import read_json_file


class PromptHandler:
    """
    A class to handle the generation and structuring of prompts for extracting values
    from real estate documents using an LLM (Large Language Model) API.
    """

    def __init__(self):
        """
        Initializes the PromptHandler instance with page content and questions.

        Args:
            page_content (str): The text extracted from the real estate PDF document.
            questions (List[str]): A list of questions to be answered based on the page content.
        """
        pass

    def gen_prompt_template(self, sentence: str) -> str:
        """
        Generates the system prompt template for extracting values from a real estate document.

        Args:
            sentence (str): The input sentence to be classified.

        Returns:
            str: The system prompt template with placeholders for page content and questions.
        """
        PROMPT = """You are a Devnagari expert skilled in understanding and analyzing the Devanagari script used in languages like Nepali and Hindi. 
Your job is to identify hate speech and determine its target—whether it is directed at an individual, an organization, or a community.
        
Please classify the following sentence into one of the three categories of hate speech:

1. **Individual**: Hate speech directed at a specific person or public figure. Most of hatespeech is directed to political person or famou public figure.
2. **Organization**: Hate speech targeting an organization, institution, or company. Name of political parties, political organization, political or financial institutions or companies.
3. **Community**: Hate speech directed at a specific group of people based on their ethnicity, religion, gender, general public such as voters,  other community-related attributes.

**Input Sentence:**

"{sentence}"

**Instructions:**
- Identify if the hate speech is directed towards an individual, an organization, or a community.
- If a particular person, political group, or organization is not mentioned in the sentence, and the statement does not target a specific individual or organization, it should be classified as directed towards the community.
- Provide the classification in JSON format as shown below.
- You must choose output among one of these three categories.

**Output JSON Format:**
{{
  "target": "{{category}}"
}}

**Example:**
"""
        return PROMPT.format(sentence=sentence)

    def generate_prompt_examples(self, file_path):
        # Read the JSON file
        try:
            prompt_examples = read_json_file(file_path)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading JSON file: {e}")
            return ""

        # Initialize the total prompt examples string and index number
        total_prompt_examples = ""
        index_no = 0

        # Iterate over each target and its examples
        for target, examples in prompt_examples.items():
            for example in examples:
                index_no += 1
                example_sentence = example.get("sentence", "")
                example_target = example.get("classification", "")

                # Format the example and add to the total prompt examples
                prompt_example = f"""{index_no}. Input Sentence: "{example_sentence}"
Output JSON: 
{{
  "target": "{example_target}"
}}
"""
                total_prompt_examples += prompt_example + "\n"

        return total_prompt_examples


    @staticmethod
    def correct_json_template(json_str: str) -> str:
        """
        Generates a prompt template that assists in correcting a malformed JSON string by removing
        any non-key-value text and ensuring the structure is valid.

        Args:
            json_str (str): The malformed JSON string that needs correction.

        Returns:
            str: A prompt template with the provided JSON string embedded,
                 ready to be used for correcting the JSON format.
        """
        JSON_CORRECTION_PROMPT = """"
"You are an expert in writing valid JSON formats. Your task is to correct the malformed JSON string provided and return the valid JSON format anyway possible.
You must remove unnecessary text which makes the JSON invalid. You must only return a single valid key-value pair in a well-formed JSON format without any explanation and you must remove extra text content that makes the JSON invalid.
- Here is the JSON string you need to correct:

# Malformed JSON String:
{json_string}

Please do not generate any additional information except the valid JSON output."
"""
        return JSON_CORRECTION_PROMPT.format(json_string=json_str)
