from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json

app = FastAPI()


class Request(BaseModel):
    input_text: str


def get_complete_response(url, data):
    response = requests.post(url, json=data)
    response.raise_for_status()  # Check for HTTP request errors

    complete_response = ""
    done = False
    start_index = 0

    while not done:
        end_index = response.text.find("}", start_index) + 1
        if end_index <= start_index:
            break  # No more JSON objects to process

        json_text = response.text[start_index:end_index]
        try:
            json_data = json.loads(json_text)
            complete_response += json_data.get("response", "")
            done = json_data.get("done", False)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            break  # Exit if we encounter a JSON decoding error

        start_index = (
            end_index  # Move the start index to the end of the last JSON object
        )

    return complete_response


@app.post("/predict")
async def predict(request: Request):
    try:
        model_api_url = "http://localhost:11434/api/generate"
        data_to_send = {
            "model": "meditron",
            "prompt": request.input_text,
        }

        final_response = get_complete_response(model_api_url, data_to_send)
        print("Complete response:", final_response)

        return {"response": final_response}
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=str(e))
