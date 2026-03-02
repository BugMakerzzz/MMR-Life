system_prompt = "Please act as a reasoning expert to answer the given multimodal reasoning questions based on multiple images. The last line of your response should be of the following format: {}. Let's think step by step before answering."

general_prompt = system_prompt.format("'\nAnswer: '")
pair_prompt = system_prompt.format("'\nAnswer: (Event 1, Event 2)'")
order_prompt = system_prompt.format("'\nAnswer: x-x-x-x-x...', where 'x' is the number corresponding to the image.")
rotate_prompt = system_prompt.format("'\nAnswer: Rotate clockwise/counterclockwise about x degrees(, then clockwise/counterclockwise about x degrees)(, and finally counterclockwise about x degrees)', where x represents the degrees.")
direction_prompt = system_prompt.format("'\nAnswer: DIRECTION, where DIRECTION represents the eight common directions.")
plan_prompt = system_prompt.format("'\nAnswer: 1. ACTION 2. ACTION ...', where ACTION is limited to the following actions: Turn Left, Turn Right, Turn Back, Go forward until the xxx. You should add numbering (e.g., 1., 2., 3. ...) before each action in the answer.")
object_prompt = system_prompt.format("'\nAnswer: COLOR stick/container(, COLOR stick/container)(, COLOR stick/container)', where COLOR represents different colors.")

system_prompt_dic = {
    "abductive": {
       "cartoon": general_prompt,
       "cleverer": general_prompt,
       "TVbench": general_prompt
       
    },
    "causal": {
        "cartoon": general_prompt,
        "cleverer_pair": pair_prompt,
        "cleverer_pred": general_prompt,
        "fluid": general_prompt
    },
    "deductive": {
        "material": general_prompt,
        "recipe": order_prompt
    },
    "temporal": {
        "crowd": order_prompt
    },
    "spatial": {
        "direction": direction_prompt,
        "navi": rotate_prompt,
        'path': plan_prompt
    }
}