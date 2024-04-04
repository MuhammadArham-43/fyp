##Imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image
import time

class StableDiffusion:
    def __init__(self):
        self.model_id = "Lykon/dreamshaper-xl-1-0"
        self.device = "cuda"
        self.pipe = AutoPipelineForText2Image.from_pretrained(self.model_id, torch_dtype=torch.float32)
        self.pipe = self.pipe.to(self.device)

    def generateImage(self, prompt):
        image = self.pipe(prompt).images[0]  
        image.save("result001.jpeg")
        print(type(image))
        return image

def main():
    model = StableDiffusion()
    prompt = input("Please give prompt:")
    model.generateImage(prompt)

if __name__ == "__main__":
    main ()