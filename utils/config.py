# openai config
OPENAI_BASE_URL = ''
# OPENAI_API_KEY = 'sk-cfOd3L4nm4v2kDKK7f8932Ed841349699cFaC8175a0dC0Ad'
OPENAI_API_KEY = ''

# VLLM_BASE_URL = 'http://175.102.130.120:14396/v1'
# VLLM_BASE_URL = 'http://210.75.240.156:14396/v1'
VLLM_API_KEY = 'ljc'


API_MODEL_MAP_DICT = {
    'gpt-4.1': 'gpt-4.1-2025-04-14', # $0.002/K $0.008/K
    'gpt-5': 'gpt-5-2025-08-07', # $0.00125/K $0.01/K
    'gpt-5-mini': 'gpt-5-mini-2025-08-07', # $0.00025/K $0.002/K
    'gpt-4o': 'gpt-4o-2024-11-20', # $0.0025/K $0.01/K
    'gpt-4o-mini': 'gpt-4o-mini-2024-07-18',
    'o1': 'o1-2024-12-17', # 0.015
    'o1-mini': 'o1-mini-2024-09-12', # 0.003
    'o4-mini': 'o4-mini-2025-04-16', # $0.0011/K  $0.0044/K
    'claude-sonnet-4': 'claude-sonnet-4-20250514', # $0.009/K  $0.045/K
    'claude-sonnet-4-thinking': 'claude-sonnet-4-20250514-thinking',
    'claude-3.7-sonnet': 'claude-3-7-sonnet-20250219',
    'claude-3.7-sonnet-thinking': 'claude-3-7-sonnet-20250219-thinking',# $0.0084/K, $0.042/K
    'gemini-2.5-flash': 'gemini-2.5-flash', # $0.0003/K  $0.0025/K
    'gemini-2.5-pro': 'gemini-2.5-pro', # $0.001/K $0.008/K
    'doubao-1.5-vision-pro': 'doubao-1-5-vision-pro-32k',
    'grok-3-reasoning': 'grok-3-reasoning', #0.004
    'gpt-5-mini-high': 'gpt-5-mini-high',
    'gpt-5-mini-minimal': 'gpt-5-mini-minimal',
    'gpt-5-high': 'gpt-5-high',
    'gpt-5-minimal': 'gpt-5-minimal',
    'gpt-4.1-mini': 'gpt-4.1-mini-2025-04-14',
    'gemini-2.5-flash-thinking': 'gemini-2.5-flash-thinking',
    'gemini-2.5-flash-nothinking': 'gemini-2.5-flash-nothinking',
    'doubao-1.5-vision': 'doubao-1-5-vision-pro-32k'
}

VLLM_URL_DICT = {
    '120': 'http://175.102.130.120:{port}/v1',
    '139': 'http://210.75.240.139:{port}/v1',
    '153': 'http://210.75.240.153:{port}/v1',
    '154': 'http://210.75.240.154:{port}/v1',
    '155': 'http://210.75.240.155:{port}/v1',
    '156': 'http://210.75.240.156:{port}/v1'
}
 
TYPE_TASK_MAP = {
    'abductive': ['HAA', 'CIA', 'MCA'],
    'analogical': ['ARI', 'PSI', 'ASI'],
    'causal': ['CIP', 'MCP', 'CFP'],
    'deductive': ['MCD', 'CWD', 'RSD'],
    'inductive': ['BMI', 'PDI', 'SFI'],
    'spatial': ['RPE', 'CRE', 'NRP'],
    'temporal': ['CTR', 'DSP', 'HAL']
}

TASK_PATH_MAP = {
    'HAA': 'abductive/Human Activity Attribution/',
    'CIA': 'abductive/Character Interaction Attribution/',
    'MCA': 'abductive/Multi-Hop Collision Attribution/',
    'ARI': 'analogical/Animal Relation Inference/',
    'PSI': 'analogical/Product Similarity Inference/',
    'ASI': 'analogical/Artwork Style Inference/',
    'CIP': 'causal/Character Interaction Prediction/',
    'MCP': 'causal/Multi-Hop Collision Prediction/',
    'CFP': 'causal/Counterfactual Fluid Prediction/',
    'MCD': 'deductive/Material Composition Deduction/',
    'CWD': 'deductive/Card Winner Deduction/',
    'RSD': 'deductive/Recipe Step Deduction/',
    'BMI': 'inductive/Bird Migration Induction/',
    'PDI': 'inductive/Plant Disease Induction/',
    'SFI': 'inductive/Sport Feature Induction/',
    'RPE': 'spatial/Relative Position Estimation/',
    'CRE': 'spatial/Camera Rotation Estimation/',
    'NRP': 'spatial/Navigation Route Planning/',
    'CTR': 'temporal/Crowd Timeline Reconstruction/',
    'DSP': 'temporal/Driving Sequence Prediction/',
    'HAL': 'temporal/Human Activity Localization/'
}

intern_rm_path = '/mnt/{dir}/zhaosuifeng/model/internlm-xcomposer2d5-7b-reward'
r1_rm_path = '/mnt/{dir}/zhaosuifeng/model/R1-Reward'
skywork_rm_path = '/mnt/{dir}/zhaosuifeng/model/Skywork-VL-Reward-7B'
unified_rm_path = '/mnt/{dir}/zhaosuifeng/model/UnifiedReward-7b-v1.5'
visual_rm_path = '/mnt/{dir}/zhaosuifeng/model/VisualPRM-8B-v1.1'