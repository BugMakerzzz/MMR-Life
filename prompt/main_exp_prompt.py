system_cot_prompt = "Please act as a reasoning expert to choose the correct option of the given multimodal reasoning questions based on multiple images. The last line of your response should be of the following format: # Answer: A/B/C/D/E. Let's think step by step before answering."
system_direct_prompt = "Please act as a reasoning expert to choose the correct option of the given multimodal reasoning questions based on multiple images. Your response should be of the following format: # Answer: A/B/C/D/E. You should directly give the answer without thinking."

system_prompt_dic = {
    'cot': system_cot_prompt,
    'direct': system_direct_prompt
}