
dif_shoes_question_prompts = [
    "My friend already owns these pairs of shoes, but he prefers other styles. Which of the following shoes would be the best gift for him?",
    "My friend already has these shoes, but he’s into different ones. Which pair should I give him from the options?",
    "Since my friend already owns these pairs but favors other shoes, please suggest which of the following would make the best gift.",
    "My friend already has these shoes, but likes a different kind. Which one of these should I get him?",
    "My friend owns these pairs of shoes, but he prefers other designs. From the list below, which should I consider buying for him?"
]

sim_shoes_question_prompts = [
    "My friend already owns these shoes and would like something similar. Which of the following pairs should I consider as a gift?",
    "My friend already has these pairs and is looking for a similar style. Which option should I choose for him?",
    "Since my friend already owns these shoes but wants a comparable pair, which of the following should I give him?",
    "My friend already has these shoes and wants another pair like them. Which one from the list would be a good choice?",
    "My friend owns these shoes and prefers a similar design. From the following options, which should I consider buying for him?"
]

art_question_prompts = [
    "The provided images consist of several paintings attributed to the same artist. From the options below, please identify the one that is most likely created by this artist.",
    "Here are some paintings from one artist. Choose the option that you think was painted by the same artist.",
    "You are given a set of paintings from a particular artist. Based on the following choices, select the artwork that best matches this artist’s style.",
    "These images all come from the same artist. Among the options, pick the one that looks like it was also made by this artist.",
    "The input contains paintings by a single artist. From the listed options, determine which work most likely belongs to the same creator."
]

animal_question_prompts = [
    "Given three images of animals, your task is to choose a fourth animal image such that the analogy between the first two images corresponds to the analogy between the last two images.",
    "You are presented with three animal pictures. Select the most appropriate fourth animal so that the relationship between the first pair is analogous to the relationship between the second pair.",
    "Consider a sequence of three animal images. Identify which animal should appear as the fourth image to maintain the same relational pattern observed between the first two and the last two images.",
    "From the set of three animal images provided, determine the correct fourth animal that preserves the analogy structure, ensuring that the relation of the first two images mirrors that of the last two.",
    "Imagine you are solving a visual analogy task with animals: choose the fourth animal image that best completes the analogy, where the connection between image1 and image2 parallels the connection between image3 and the missing image."
]



leaf_disease_question_prompts = [
    "The images display leaves affected by the same disease I have seen before. Which of the option leaves are infected with this same disease?",
    "The shown leaves present the same disease I previously observed. Please identify which of the option leaves are infected with this type of disease.",
    "The images include leaves that show the same disease I’ve noticed earlier. From the options, which leaf is infected by the same disease?",
    "The leaves in the images are affected by a disease I already recognize. Which of the option leaves share the same infection?",
    "The provided leaves illustrate the same disease observed before. Which of the option leaves are also infected with this disease?"
]

leaf_type_question_prompts = [
    "I have collected leaves as shown in the images and want to keep finding new ones in the same category. Which of the option leaves should I select?",
    "The images show leaves I’ve already collected, and I’d like to continue searching for more in the same category. From the options, which leaf do I need?",
    "I’ve gathered leaves illustrated in the images and would like to add new ones from the same type. Which option should I pick?",
    "These leaves in the images are already collected, and I want to keep collecting more from the same category. Which leaf in the options fits?",
    "I have collected some leaves similar to those shown in the images. To continue building the same collection, which of the option leaves should I choose?"
]

bird_question_prompts = [
    "The images display the observed shifts in bird distribution over the years. Based on these, please predict the most probable distribution map for the upcoming year.",
    "Given the historical changes in bird distribution shown in the images, which distribution map is most likely for the next year?",
    "The figures illustrate how bird distribution has changed across the years. Please identify the map that best represents the expected distribution for the following year.",
    "Looking at the observed yearly changes in bird distribution shown in the images, predict the most likely distribution pattern for next year.",
    "These images depict the progression of bird distribution over time. From this, determine which map would most likely illustrate the distribution for the coming year."
]

sport_question_prompts = [
    "Based on the sports illustrated in the figure, select the correct sport from the options to complete the next figure.",
    "Considering the sports shown, which of the following should be chosen as the next figure?",
    "From the given figure of sports, identify the appropriate choice for the next sport among the options.",
    "Looking at the sequence of sports in the figure, pick the one that best fits as the next from the listed options.",
    "Given the sports displayed, determine which sport from the provided options should appear in the next position."
]

driving_question_prompts = [
    "Which image in the options is most likely to appear in the next moment?",
    "From the given options, which image is most likely to occur next?",
    "Please choose the image that is most likely to appear at the next moment from the options.",
    "Which of the images in the options is most likely to be the next one to occur?",
    "Among the options provided, which image is most likely to happen in the next moment?"
]

crowd_question_prompts = [
    "What is the correct chronological sequence of these images?",
    "Please arrange these images in the correct order of events.",
    "Can you determine the chronological order of these images?",
    "In what order do these images appear chronologically?",
    "What is the proper sequence of these images in time?"
]

navi_question_prompts = [
    "Given the continuous images, what were the successive rotation angles of the camera?",
    "Based on these images in sequence, what degrees of rotation did the camera undergo?",
    "What were the rotation angles of the camera between these continuous images?",
    "From the continuous sequence of images, what were the degrees of rotation the camera made?",
    "Looking at the series of images, what successive rotations did the camera perform?"
]




user_prompt_dic = {
    "abductive": {
       "cartoon": None,
       "cleverer": None,
       "TVbench": None
    },
    "causal": {
        "cartoon": None,
        "cleverer_pair": None,
        "cleverer_pred": None,
        "fluid": None
    },
    "analogical": {
        "animal": animal_question_prompts,
        "art": art_question_prompts,
        "dif_shoes": dif_shoes_question_prompts,
        "sim_shoes": sim_shoes_question_prompts  
    },
    "inductive": {
        "plant": leaf_disease_question_prompts,
        "bird": bird_question_prompts,
        "sport":  sport_question_prompts
    },
    "spatial": {
        "direction": None,
        "navi": navi_question_prompts,
        "path": None
    },
    "deductive": {
        "material": None,
        "poke": None,
        "recipe": None
    },
    "temporal": {
        "crowd": crowd_question_prompts,
        "driving": driving_question_prompts
    },
}