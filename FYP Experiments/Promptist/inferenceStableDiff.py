##Imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image
import time

class StableDiffusion:
    def __init__(self, config):
        self.model_id = config["sd_model"] if "sd_model" in config else "Lykon/dreamshaper-xl-1-0"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = AutoPipelineForText2Image.from_pretrained(self.model_id, torch_dtype=torch.float32)
        self.pipe = self.pipe.to(self.device)
        self.num_infer_steps = config["num_inference_steps"]

    def generateImage(self, prompt):
        image = self.pipe(prompt, num_inference_steps=self.num_infer_steps).images[0]  
        image.save("result001.jpeg")
        print(type(image))
        return image

def main():
    model = StableDiffusion()
    prompt = input("Please give prompt:")
    model.generateImage(prompt)

if __name__ == "__main__":
    main ()