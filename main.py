from sensor.pipeline.training_pipeline import TrainingPipeline
from fastapi import FastAPI
from uvicorn import run as app_run
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from sensor.constant.application import APP_HOST, APP_PORT
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
origins = ["*"]


app.add_middleware(
    CORSMiddleware
    , allow_origins=origins
    , allow_credentials=True
    , allow_methods=["*"]
    , allow_headers=["*"]
)


@app.get("/", tags=["authentication"])
def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
def train_route():
    try:
        train_pipeline = TrainingPipeline()
        if train_pipeline.is_pipeline_running:
            return Response("Training pipeline is already running.")
        train_pipeline.run_pipeline()

        return Response("Trining Success..!")
    except Exception as e:
        return Response(f"Error: [{e}]")
    
    
if __name__=="__main__":
    app_run(app=app, host=APP_HOST, port=APP_PORT)

