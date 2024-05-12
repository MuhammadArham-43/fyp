import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class InferenceLLMPrompter:
    def __init__(self, lm_model_path: str = "microsoft/phi-2", lora_weights_path: str = None):
        # torch.set_default_device("cuda")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(lm_model_path, torch_dtype=dtype, trust_remote_code=True).to(self.device)
        if lora_weights_path:
            PeftModel.from_pretrained(self.model, lora_weights_path)
        self.tokenizer = AutoTokenizer.from_pretrained(lm_model_path, trust_remote_code=True)

    def optimize_prompt(self, prompt:str, max_new_tokens:int=100):        
        input = "Prompt: " + prompt
        inputs = self.tokenizer(input, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, do_sample=True, max_new_tokens=max_new_tokens)
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