from fastapi import FastAPI
from fastapi.responses import JSONResponse
import joblib
from pydantic import BaseModel

from person_stat import get_user_stat, get_posts_features
import yaml


ei_model=joblib.load("./ei_model.joblib")

credentials = {}
with open("credentials.yaml") as stream:
    try:
        credentials = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

app = FastAPI()

class GetUserPersonalityRequest(BaseModel):
    user_id: str

class GetUserPersonalityResponse(BaseModel):
    personality: str
    bad_word_flg: bool
    deviant_score: float

class ErrorResponse(BaseModel):
    error: str

@app.post("/getUserPersonality", responses={418: {"model": ErrorResponse}})
def getUserPersonality(req: GetUserPersonalityRequest) -> GetUserPersonalityResponse:
    print(req.user_id)
    try:
        stats = get_user_stat(req.user_id, credentials['vk_token'])
    except Exception as e:
        return JSONResponse(status_code=418, content={
            'error': e.message
        })
    if stats['is_closed']:
        return JSONResponse(status_code=418, content={
            'error': "User profile closed"
        })
    try:
        post_features = get_posts_features(req.user_id, credentials['vk_token'])
    except Exception as e:
        return JSONResponse(status_code=418, content={
            'error': e.message
        })

    return {
        "personality": "aaa",
        "bad_word_flg": False,
        "deviant_score": 0.,
    }
