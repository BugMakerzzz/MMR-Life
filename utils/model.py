from .config import *
from openai import OpenAI
import anthropic

class Model():

    def __init__(self, model_name, model_url, model_port=14396):   
        self.api_model = True if model_name in API_MODEL_MAP_DICT.keys() else False
        if self.api_model:
            self.client = OpenAI(
                base_url=OPENAI_BASE_URL,
                api_key=OPENAI_API_KEY,
                default_headers={"x-foo": "true"}
            )
            self.model_path = API_MODEL_MAP_DICT[model_name]
        else:    
            self.client = OpenAI(
                base_url=VLLM_URL_DICT[model_url].format(port=model_port),
                api_key=VLLM_API_KEY
            )
            self.model_path = model_name
    
    def generate(self, message, generation_config):
        try:
            response = self.client.chat.completions.create(
                    model= self.model_path,
                    messages=message,
                    max_tokens=generation_config['max_tokens'],
                    temperature=generation_config['temperature'],
                    n=generation_config['n'],
                    top_p=generation_config['top_p']
                    
                )
            output =  [response.choices[i].message.content for i in range(generation_config['n'])]
            usage =  response.usage
        except Exception as e:
            # 处理其他可能的错误
            print(f"Error: {e}")
            output = [""]
        if 'usage' in generation_config:
            return output, usage
        else:
            return output 