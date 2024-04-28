from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router.routes import PromptistRouter, StableDiffusionRouter, PromptDiscoveryRouter, PrompterRouter
import uvicorn
import yaml




def read_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = read_config("configuration.yaml")

# promptistRouter = PromptistRouter()
# stableDiffusionRouter = StableDiffusionRouter()
# promptDiscoveryRouter = PromptDiscoveryRouter()
# llmPrompter = PrompterRouter()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router=PromptistRouter(config["promptist"]).router)
app.include_router(router=StableDiffusionRouter(config["promptist"]).router)
# app.include_router(router=PromptDiscoveryRouter(config["hard_prompts"]).router)
# app.include_router(router=PrompterRouter(config["llm_prompter"]["weights_path"]).router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
