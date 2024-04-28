import torch
import argparse
from PIL import Image
import HardPrompts.open_clip_model as open_clip_model
from .optim_utils import *

class InferenceHardPrompt:
    def __init__(self, config):
        self.config_path = config["config_path"]
        self.args = argparse.Namespace()
        self.args.__dict__.update(read_json(self.config_path))
        self.args.iter = config["num_iterations"]
        print(self.args)
        self.args.print_new_best = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self._, self.preprocess = open_clip_model.create_model_and_transforms(self.args.clip_model, pretrained=self.args.clip_pretrain, device=self.device)
        # print(type(self.model)) 
        # print(type(self.model.forward_text_embedding))
        
         
    def discoverPrompt(self, image):
        learned_prompt = optimize_prompt(self.model, self.preprocess, self.args, self.device, target_images=[image])
        return learned_prompt
    
def main():
    inference = InferenceHardPrompt()
    image = Image.open("/home/evobits/arham/fyp/FYP Experiments/image.png")
    disocveredPrompt = inference.discoverPrompt(image)
    print("Disocvered Prompt", disocveredPrompt)

if __name__ == "__main__":
    main()