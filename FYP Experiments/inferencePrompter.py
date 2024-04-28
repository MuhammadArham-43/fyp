import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class InferenceLLMPrompter:
    def __init__(self, lm_model_path: str = "microsoft/phi-2", lora_weights_path: str = None):    
        # torch.set_default_device("cuda")
        self.device = torch.device("cuda")
        self.model = AutoModelForCausalLM.from_pretrained(lm_model_path, torch_dtype="auto", trust_remote_code=True).to(self.device)
        if lora_weights_path:
            PeftModel.from_pretrained(self.model, lora_weights_path)
        self.tokenizer = AutoTokenizer.from_pretrained(lm_model_path, trust_remote_code=True)

    def optimize_prompt(self, prompt:str):        
        input = "Prompt: " + prompt
        inputs = self.tokenizer(input, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, do_sample=True, max_length=100)
        text = self.tokenizer.batch_decode(outputs)[0]
        text = self.postprocess(text)
        return text
    
    def postprocess(self, text:str):
        # Split the text into lines
        lines = text.split("\n")
        # print(lines)
        # Filter lines that start with "### Optimized Prompt:"
        optimized_prompts = [line.strip() for line in lines if line.startswith(" ### Optimized Prompt:")]
        optimized_prompts = [line.replace("### Optimized Prompt:", "") for line in optimized_prompts]
        # print(optimized_prompts)
        return optimized_prompts[0]
    
    
def main():
    model = InferenceLLMPrompter()
    prompt = input("Enter your Prompt: ")
    optimizedPrompt = model.optimize_prompt(prompt)
    print("Optimized Prompt is:", optimizedPrompt)
            
if __name__ == "__main__":
    main()

# print(outputs)
# text = tokenizer.batch_decode(outputs)[0]
# print(text)