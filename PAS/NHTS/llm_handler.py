import json
import torch
import ollama
import requests
from openai import OpenAI
from transformers import pipeline, logging


class LLMHandler:
    def __init__(self, llm, api_keys, use_api=False):
        self.llm = llm
        self.api_keys = api_keys
        self.use_api = use_api
        self.client = self.create_client()  # Initialize client here

        # Check for unsupported local models
        if self.llm == "llama3.1:405b" and not self.use_api:
            raise ValueError("Running 'llama3.1:405b' locally without using an API is not allowed due to hardware limitations.")

        # Mapping for model names
        self.model_mapping = {
            'llama3:8b': 'Meta-Llama-3-8B-Instruct',
            'llama3.1:8b': 'Meta-Llama-3.1-8B-Instruct',
            'llama3.1:70b': 'Meta-Llama-3.1-70B-Instruct',
            'llama3.1:405b': 'Meta-Llama-3.1-405B-Instruct',
            'llama-3-8b-lexi': 'Orenguteng/Llama-3-8B-Lexi-Uncensored',
            'gpt-3.5-turbo': 'gpt-3.5-turbo',
            'gpt-4-turbo': 'gpt-4-turbo'
            # Add other mappings here if needed
        }

        # Determine model name for API or local usage
        self.api_model_name = self.model_mapping.get(llm, llm)
    
    def create_client(self):
        # Initialize the LLM client based on the API or local setup
        if self.use_api:
            if 'gpt' in self.llm:
                return OpenAI(api_key=self.api_keys['openai'])
            elif 'llama' in self.llm:
                # SambaNova API
                return None  # Will handle this in the chat function
        else:
            if self.llm == 'llama-3-8b-lexi':
                device = 2 if torch.cuda.is_available() else -1 # -1 represents CPU, 2 represents GPU
                print(f"Using device: {device}")
                return pipeline("text-generation", model="Orenguteng/Llama-3-8B-Lexi-Uncensored", device=device)
            elif self.llm in ['llama3:8b', 'llama3.1:8b', 'llama3.1:70b','llama3.3:70b','deepseek-r1:70b']:  # local llama via Ollama
                return ollama
            else:
                raise ValueError(f"Unsupported local model: {self.llm}")

    # def reconfigure(self, llm, api_keys, use_api):
    #     self.cleanup()  # Clean up before reconfiguring the new client
    #     self.llm = llm
    #     self.api_keys = api_keys
    #     self.use_api = use_api
    #     self.client = self.create_client()  # Reinitialize client

    def chat(self, messages):  # 可以是同一个LLM的多轮对话
        # Ensure the client is initialized
        if not self.client:
            raise RuntimeError("LLM client is not initialized.")

        if self.use_api:
            if 'gpt' in self.llm:
                response = self.client.chat.completions.create(
                    model=self.llm,
                    messages=messages,
                    temperature=0,
                    top_p=0,
                    seed=0
                )
                reply = response.choices[0].message.content
            elif self.llm in ['llama3.1:8b', 'llama3.1:70b', 'llama3.1:405b','llama3.3:70b']:
                # Handle SambaNova API request
                url = "https://fast-api.snova.ai/v1/chat/completions"
                headers = {
                    "Authorization": f"Basic {self.api_keys['sambanova']}",
                    "Content-Type": "application/json"
                }
                data = {
                    "messages": messages,
                    "stop": ["<|eot_id|>"],
                    "model": self.api_model_name,
                    "stream": True,
                    "stream_options": {"include_usage": True}
                }
                response = requests.post(url, headers=headers, json=data, stream=True)
                reply = self._parse_sambanova_response(response)
            elif self.llm == 'deepseek-r1:70b':
                # Handle DeepSeek API request
                url = "http://localhost:5000/api/v1/chat"  # 替换为DeepSeek的实际API地址
                headers = {
                    "Authorization": f"Bearer {self.api_keys['deepseek']}",  # 替换为DeepSeek的API密钥
                    "Content-Type": "application/json"
                }
                data = {
                    "messages": messages,
                    "model": "deepseek-r1:70b",
                    "temperature": 0,
                    "top_p": 0,
                    "seed": 0
                }
                response = requests.post(url, headers=headers, json=data)
                reply = response.json()['choices'][0]['message']['content']
            # Add logic for other API-based LLMs here

        else:  # local LLM
            if self.llm == 'llama-3-8b-lexi':
                response = self.client(messages, max_new_tokens=8192)
                reply = response[0]['generated_text'][-1]['content']
            elif self.llm in ['llama3:8b', 'llama3.1:8b', 'llama3.1:70b', 'llama3.3:70b']:
                response = self.client.chat(model=self.llm, messages=messages)
                reply = response['message']['content']
            elif self.llm == 'deepseek-r1:70b':
                # Handle local DeepSeek model
                response = self.client.chat(model='deepseek-r1:70b', messages=messages)
                reply = response['message']['content']
            # Add logic for other local LLMs here
            else:
                raise ValueError(f"Unsupported local model: {self.llm}")
            # print(reply)
        return reply
    
    def _parse_sambanova_response(self, response):
        full_reply = ""
        if response.status_code == 200:
            for chunk in response.iter_lines():
                if chunk:
                    chunk_data = chunk.decode('utf-8').strip()
                    if chunk_data == "data: [DONE]":
                        continue  # Skip [DONE] marker
                    if chunk_data.startswith("data: "):
                        chunk_json = chunk_data[len("data: "):]
                        try:
                            chunk_dict = json.loads(chunk_json)
                            # Check if choices list is not empty and content exists
                            if "choices" in chunk_dict and chunk_dict["choices"]:
                                content = chunk_dict["choices"][0]["delta"].get("content", "")
                                full_reply += content
                        except json.JSONDecodeError:
                            print(f"Failed to parse chunk: {chunk_json}")
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(f"Response content: {response.text}")
        return full_reply
    
    def handle_one_inquiry(self, prompt, enable_secondary_inquiries=False):
        """
        Handles one LLM inquiry, including optional secondary inquiries.
        Ensures GPU memory is cleared after each inquiry.
        
        Parameters:
        - prompt (str): The input prompt for the LLM.
        - enable_secondary_inquiries (bool): Whether to perform secondary inquiries to refine the LLM's response.

        Returns:
        - list: LLM response.
        """
        try:
            # use new client
            conversation_history = [{"role": "user", "content": prompt}]
            gpt_response = self.chat(conversation_history)
            conversation_history.append({"role": "assistant", "content": gpt_response})
            
            # Secondary inquiries if enabled
            if enable_secondary_inquiries:
                # print("Secondary inquiries enabled.")
                conversation_history.append({"role": "user", "content": "So, your answer is: (e.g., number1, number2)"})
                gpt_response = self.chat(conversation_history)
                conversation_history.append({"role": "assistant", "content": gpt_response})
            # print(conversation_history)
            # print(gpt_response)
            return gpt_response
        finally:
            # After handling the inquiry, release GPU memory
            torch.cuda.empty_cache()
    
    # def cleanup(self):
    #     """
    #     Cleanup method to clear GPU memory after each inquiry.
    #     """
    #     # if self.client and hasattr(self.client, 'model'):
    #     #     del self.client.model  # Delete the model
    #     torch.cuda.empty_cache()  # Clear GPU cache

if __name__ == "__main__":
    prompts = ["Who are you?", "1+1=?", "What animal do you like?", "What is your favorite color?", "What is your favorite food?"]
    # handler = LLMHandler(llm="llama3.1:70b", api_keys={"sambanova": "your-api-key"}, use_api=False)
    import utils
    api_keys = utils.load_api_keys("config/api_keys.json")
    llm_handler = LLMHandler(llm="llama3.1:70b", api_keys=api_keys,use_api=False)
    responses = llm_handler.handle_multiple_inquiries(prompts, enable_secondary_inquiries=True)

    for response in responses:
        print(response)
