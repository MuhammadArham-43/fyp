import pytest
import yaml

with open('test_config.yaml', 'r') as f:
    config = yaml.safe_load(f)


def test_sd_prompter():
    from Promptist.inferencePromptist import Promptist
    
    test_prompt = "digital art painting"    
    promptist = Promptist()
    optimized_prompt = promptist.generate(test_prompt)
    
    del promptist
    assert type(optimized_prompt) == str and len(optimized_prompt) != 0


def test_hard_prompts():
    from HardPrompts.inferenceHardPromptist import InferenceHardPrompt
    from PIL import Image
    
    hard_prompt_config = config['hard_prompts']
    pez = InferenceHardPrompt(hard_prompt_config)
    test_img_path = "image.png"
    prompt = pez.discoverPrompt(Image.open(test_img_path))
    
    del pez
    print(prompt)
    print(type(prompt))
    assert type(prompt) == str and len(prompt) != 0
    

def test_llm_prompter():
    from inferencePrompter import InferenceLLMPrompter

    test_prompt = "how to curate introduction of a research paper?"
    llm_prompter_config = config['llm_prompter']
    llm_prompter = InferenceLLMPrompter(
        lora_weights_path=llm_prompter_config['weights_path']
    )
    optimized_prompt = llm_prompter.optimize_prompt(test_prompt, max_new_tokens=100)
    
    del llm_prompter
    assert type(optimized_prompt) == str and len(optimized_prompt) != 0
    

    
    
    
    
    
    
    