import re
from qwen_vl_utils import process_vision_info
import json
from PIL import Image
import base64

def get_action_atlas(model, processor, obs):
    ATLAS_PROMPT =  """
    You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

    1. Basic Actions
    Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
    Basic Action 1: CLICK 
        - purpose: Click at the specified position.
        - format: CLICK <point>[[x-axis, y-axis]]</point>
        - example usage: CLICK <point>[[101, 872]]</point>
        
    Basic Action 2: TYPE
        - purpose: Enter specified text at the designated location.
        - format: TYPE [input text]
        - example usage: TYPE [Shanghai shopping mall]

    Basic Action 3: SCROLL
        - purpose: SCROLL in the specified direction.
        - format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
        - example usage: SCROLL [UP]
        
    2. Custom Actions
    Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.
    Custom Action 1: LONG_PRESS 
        - purpose: Long press at the specified position.
        - format: LONG_PRESS <point>[[x-axis, y-axis]]</point>
        - example usage: LONG_PRESS <point>[[101, 872]]</point>
        
    Custom Action 2: OPEN_APP
        - purpose: Open the specified application.
        - format: OPEN_APP [app_name]
        - example usage: OPEN_APP [Google Chrome]

    Custom Action 3: PRESS_BACK
        - purpose: Press a back button to navigate to the previous screen.
        - format: PRESS_BACK
        - example usage: PRESS_BACK

    Custom Action 4: PRESS_HOME
        - purpose: Press a home button to navigate to the home page.
        - format: PRESS_HOME
        - example usage: PRESS_HOME

    Custom Action 5: PRESS_RECENT
        - purpose: Press the recent button to view or switch between recently used applications.
        - format: PRESS_RECENT
        - example usage: PRESS_RECENT

    Custom Action 6: ENTER
        - purpose: Press the enter button.
        - format: ENTER
        - example usage: ENTER

    Custom Action 7: WAIT
        - purpose: Wait for the screen to load.
        - format: WAIT
        - example usage: WAIT

    Custom Action 8: COMPLETE
        - purpose: Indicate the task is finished.
        - format: COMPLETE
        - example usage: COMPLETE

    In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. Ensure you strictly generate two sections: Thoughts and Actions.
    Thoughts: Clearly outline your reasoning process for current step.
    Actions: Specify the actual actions you will take based on your reasoning. You should follow action format above when generating. 

    Your current task instruction, action history, and associated screenshot are as follows:
    Screenshot: 
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", "text": ATLAS_PROMPT,
                },
                {
                    "type": "image",
                    "image": obs['image_path'],
                },
                {"type": "text", "text": f"Task instruction: {obs['task']}" },
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    # pattern = r"Actions:\n(.*?)<\|im_end\|>"
    pattern = r":\n*(.*?)<\|im_end\|>"
    match = re.search(pattern, output_text[0], re.DOTALL)
    if match:
        action = match.group(1)
    else:
        pattern = r"actions:\n(.*?)<\|im_end\|>"
        match = re.search(pattern, output_text[0], re.DOTALL)
        if match:
            action = match.group(1)
        else:
            print("No matching content found.")
            
    print(output_text[0])
    return action

def get_action_qwen3vl(model, processor, obs):
    prompt = "You are a smartphone assistant to help users complete tasks by interacting with apps. I will give you a screenshot of the current phone screen."
    prompt += "\n### Background ###\n"
    prompt += f"The user's instruction is: {obs['task']}"
    prompt += "\n\n"
    prompt += "### Response requirements ###\n"
    prompt += (
        "Now you need to combine all of the above to decide just one action on the current page. "
        "You must choose one of the actions below:\n"
    )
    prompt += (
        '"SWIPE[UP]": Swipe the screen up.\n'
        '"SWIPE[DOWN]": Swipe the screen down.\n'
        '"SWIPE[LEFT]": Swipe the screen left.\n'
        '"SWIPE[RIGHT]": Swipe the screen right.\n'
    )
    prompt += '"CLICK[x,y]": Click the screen at the coordinates (x, y). x is the pixel from left to right and y is the pixel from top to bottom\n'
    prompt += '"TYPE[text]": Type the given text in the current input field.\n'
    prompt += '"LONG_PRESS[x,y]": Long press the screen at the coordinates (x, y). x is the pixel from left to right and y is the pixel from top to bottom\n'
    prompt += (
        '"PRESS_BACK": Press the back button.\n'
        '"PRESS_HOME": Press the home button.\n'
        '"WAIT": Wait for the screen to load.\n'
        '"TASK_COMPLETE[answer]": Mark the task as complete. If the instruction requires answering a question, provide the answer inside the brackets. If no answer is needed, use empty brackets "TASK_COMPLETE[]".\n'
    )
    prompt += "\n\n"
    prompt += "### Response Example ###\n"
    prompt += (
        "Your output should be a string and nothing else, containing only the action type you choose from the list above.\n"
        "For example:\n"
        'SWIPE[UP]\n'
        'CLICK[156,867]\n'
        'TYPE[Rome]\n'
        'LONG_PRESS[156,867]\n'
        'PRESS_BACK\n'
        'PRESS_HOME\n'
        'WAIT\n'
        'TASK_COMPLETE[1h30m]\n'
        'TASK_COMPLETE[]\n'
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", "text": prompt,
                },
                {
                    "type": "image",
                    "image": obs['image_path'],
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, 
        image_patch_size=16, 
        return_video_kwargs=True, 
        return_video_metadata=True
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_kwargs.get("video_metadata", None),
        padding=True,
        return_tensors="pt",
        do_resize=False 
    )
    
    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=256 
    )
    
    generated_ids_trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    
    token_count = len(generated_ids_trimmed[0])

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    print(output_text)
    return output_text,token_count
