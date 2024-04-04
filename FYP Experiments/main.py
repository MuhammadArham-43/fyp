from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router.routes import PromptistRouter, StableDiffusionRouter, PromptDiscoveryRouter, PrompterRouter

# promptistRouter = PromptistRouter()
# stableDiffusionRouter = StableDiffusionRouter()
# promptDiscoveryRouter = PromptDiscoveryRouter()
llmPrompter = PrompterRouter()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.include_router(router=promptistRouter.router)
# app.include_router(router=stableDiffusionRouter.router)
# app.include_router(router=promptDiscoveryRouter.router)
app.include_router(router=llmPrompter.router)
