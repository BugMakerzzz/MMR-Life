from abc import ABC, abstractmethod
from typing import List
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from utils.config import  skywork_rm_path
from utils.load_data import extract_answer
import re
from trl import AutoModelForCausalLMWithValueHead
from qwen_vl_utils import process_vision_info
from transformers.utils import cached_file
from safetensors import safe_open


class Reward(ABC):
    def __init__(self, model_path: str):
        self.model_path = model_path.format(dir='usercache')
        self.model = None 
        self.tokenizer = None 
        self._initialize()

    @abstractmethod
    def _initialize(self):
        pass

    @abstractmethod
    def score(self, question:str, responses:List[str], **kwargs) -> List[float]:
        """计算分数的抽象方法"""
        pass
    
    
    def find_bestn_answer(self, question, completions: List[str]):
        """Returns the most confident answer, its completion, its id in the input list, and its confidence."""
        if completions is None or len(completions) == 0:
            return None, None, None, None
        
        answers = extract_answer(completions)
        # print(answers)
        scores = self.score(question, completions)
        confidence = max(scores)
        max_index = scores.index(confidence)
        # max_index, confidence = max(enumerate(scores), key=lambda x: x[1])
        # print(max(enumerate(scores), key=lambda x: x[1]))
        most_confident_answer = answers[max_index]
        most_confident_completion = completions[max_index]
        # print(most_confident_answer)
        # print(completions)
        # assert confidence > 0
        return (
            most_confident_answer,
            most_confident_completion,
            scores,
        )
    

# # 示例子类1：基于特定名称类型进行初始化
# class InternReward(Reward):
#     def _initialize(self):
#         self.model = AutoModelForSequenceClassification.from_pretrained(
#             self.model_path
#             torch_dtype=torch.float16,
#             device_map='auto',
#         )
#         tokenizer = AutoTokenizer.from_pretrained(self.model_path)
#         self.model.tokenizer = tokenizer
        
#     def score(self, question, responses, image):
#         scores = []
#         for res in responses:
#             content = [{"role": "user", "content": question}, {"role": "assistant", "content": res}]
#             score = self.model.get_score(content, image, hd_num=9)
#             scores.append(score)
#         return scores 
    
class SkyworkReward(Reward):
    def _initialize(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        vhead_file = cached_file(
            path_or_repo_id=self.model_path, filename="value_head.safetensors"
        )
        with safe_open(vhead_file, framework="pt", device="cpu") as f:
            vhead_params = {key: f.get_tensor(key) for key in f.keys()}
        self.model.load_state_dict(vhead_params, strict=False)
        self.model.requires_grad_(False)
        self.model.eval()
        
    def score(self, question, responses):
        results = []
        for res in responses:
            messages = [{"role": "user", "content": question}, {"role": "assistant", "content": res}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            values = self.model(**inputs, return_dict=True, use_cache=False)[-1]
            scores = values.gather(
                dim=-1, index=(inputs["attention_mask"].sum(dim=-1, keepdim=True) - 1)
            )
            score = scores[0].item()
            results.append(score)
        return results 


# 工厂方法，根据name动态返回具体子类的实例
def reward_factory(name: str) -> Reward:
    if name.startswith('Skywork'):
        return SkyworkReward(skywork_rm_path)
    else:
        return None 