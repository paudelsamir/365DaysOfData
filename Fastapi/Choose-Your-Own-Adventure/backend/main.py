# usually contains fastapi app, includes routers, starts uvicorn server etc


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.configs import settings

app = FastAPI(
    title = "Choose Your Own Adventure",
    description = "api to generate cool stories",
    version = "0.1.0",
    docs_url = "/docs",
    redoc_url = "/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app = "main:app", host = "0.0.0.0", port = 8000, reload = True)