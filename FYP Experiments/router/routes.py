from fastapi import APIRouter, Query, UploadFile, File, HTTPException, Body, Response
from fastapi.responses import FileResponse
from Promptist.inferencePromptist import Promptist
from Promptist.inferenceStableDiff import StableDiffusion
from HardPrompts.inferenceHardPromptist import InferenceHardPrompt
from PIL import Image, ImageDraw, ImageFont
from inferencePrompter import InferenceLLMPrompter
import io
import os
import shutil
import uuid
import numpy as np
from PIL import Image
import cv2
import base64

#Router for accessing promptist
class PromptistRouter(APIRouter):
    def __init__(self):
        super().__init__();
        self.router:APIRouter = APIRouter()
        self.router.add_api_route("/optimize-prompt-stable-diffusion/", self.optimize_prompt, methods = ["POST"])  
        self.promptist = Promptist()
    
    def optimize_prompt(self, inputData = Body()):
        # Here you would perform your optimization logic
        print(inputData)
        prompt = inputData["prompt"]
        optimized_prompt = self.promptist.generate(prompt)  # For demonstration, just converting to uppercase
        return {"optimized_prompt": optimized_prompt}


#Router for accessing Stable Diffusion
class StableDiffusionRouter(APIRouter):
    def __init__(self):
        super().__init__();
        self.router:APIRouter = APIRouter()
        self.router.add_api_route("/generate-image/", self.generateImage, methods = ["POST"])  
        self.stableDiffusion = StableDiffusion()
    
    async def generateImage(self, inputData = Body()):
        # Here you would perform your optimization logic
        # print(inputData)
        prompt = inputData["prompt"]
        image = self.stableDiffusion.generateImage(prompt)  # For demonstration, just converting to uppercase
        # BytesIO is a file-like buffer stored in memory
        imgByteArr = io.BytesIO()
        # image.save expects a file-like as a argument
        image.save(imgByteArr, format="png")
        # Turn the BytesIO object back into a bytes object
        imgByteArr = imgByteArr.getvalue()
        print(type(imgByteArr))
        b64_img = base64.b64encode(imgByteArr)
        print(type(b64_img))
        return {"image_bytes" : b64_img}


#Router for Prompt Discovery
class PromptDiscoveryRouter(APIRouter):
    def __init__(self):
        super().__init__();
        self.router:APIRouter = APIRouter()
        self.router.add_api_route("/discover-prompt/", self.findPrompt, methods = ["POST"])  
        self.model = InferenceHardPrompt()
    
    async def findPrompt(self, image: UploadFile = File(...)):
        image_content = await image.read()
        img_pil = Image.open(io.BytesIO(image_content))
        # img_pil.save("image.jpeg")
        discoveredPrompt = self.model.discoverPrompt(img_pil)
        # buf = io.BytesIO()
        # generated_image.save(buf, format='IMAGE/PNG')
        # byte_im = buf.getvalue()
        # print("Image Bytes", byte_im)
        return {"discoveredPrompt" : discoveredPrompt}

#Router for Prompt Optimizer for LLM
class PrompterRouter(APIRouter):
    def __init__(self):
        super().__init__();
        self.router:APIRouter = APIRouter()
        self.router.add_api_route("/optimize-prompt-llm/", self.optimize_prompt, methods = ["POST"])  
        self.model = InferenceLLMPrompter()
    
    async def optimize_prompt(self, inputData = Body()):
        # Here you would perform your optimization logic
        print(inputData)
        prompt = inputData["prompt"]
        optimizedPrompt = self.model.optimize_prompt(prompt)  # For demonstration, just converting to uppercase
        return {"optimizedPrompt": optimizedPrompt}