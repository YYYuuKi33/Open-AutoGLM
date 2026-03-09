from typing import Any, Dict, Optional
import json
import os
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import the project's PhoneAgent. This module exists in the repo.
from phone_agent import PhoneAgent
from phone_agent.model import ModelConfig

app = FastAPI(title="Open-AutoGLM AI Service", version="0.1")


class RunRequest(BaseModel):
    task: str
    # optional model config override
    model: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None


class RunResponse(BaseModel):
    plan_id: str
    todo_id: Optional[str] = None
    plan: Any


def _build_model_config(body_model: Optional[Dict[str, Any]]) -> ModelConfig:
    """Build ModelConfig either from request body or environment defaults.
    This helper tries to be permissive: if the repo provides a ModelConfig
    class compatible constructor (as in README example), we create it.
    """
    # Try to read defaults from environment
    base_url = None
    model_name = None
    api_key = None
    try:
        base_url = os.environ.get("PHONE_AGENT_BASE_URL")
        model_name = os.environ.get("PHONE_AGENT_MODEL")
        api_key = os.environ.get("PHONE_AGENT_API_KEY")
    except Exception:
        pass

    if body_model:
        # accept keys: base_url, api_key, model_name, temperature, max_tokens
        base_url = body_model.get("base_url", base_url)
        api_key = body_model.get("api_key", api_key)
        model_name = body_model.get("model_name", model_name)
        # other params passed through
    # Construct ModelConfig using available keys; remaining args default inside class
    kwargs = {}
    if base_url is not None:
        kwargs["base_url"] = base_url
    if api_key is not None:
        kwargs["api_key"] = api_key
    if model_name is not None:
        kwargs["model_name"] = model_name
    # allow other fields
    if body_model:
        for k, v in body_model.items():
            if k not in ("base_url", "api_key", "model_name"):
                kwargs[k] = v

    # If ModelConfig accepts these keywords, this will succeed; otherwise it will raise
    return ModelConfig(**kwargs)


def _convert_agent_output_to_action_plan(agent_output: Any) -> Any:
    """Try to convert different agent.run outputs into a structured ActionPlan.

    - If agent_output is a dict/list already, return it.
    - If it's a JSON string, parse it.
    - Otherwise wrap the textual output into {'text': agent_output}.
    """
    if agent_output is None:
        return {"steps": []}
    if isinstance(agent_output, (dict, list)):
        return agent_output
    if isinstance(agent_output, str):
        # try parse JSON
        try:
            parsed = json.loads(agent_output)
            return parsed
        except Exception:
            # not JSON — wrap into textual plan
            return {"text": agent_output}
    # fallback
    return {"text": str(agent_output)}


@app.post("/run", response_model=RunResponse)
def run(request: RunRequest):
    """Run the PhoneAgent to generate an ActionPlan for the given task.

    This endpoint synchronously constructs a PhoneAgent (with optional model
    settings), calls `agent.run(task)`, converts result to a ActionPlan JSON and
    returns it with a generated plan_id.
    """
    if not request.task or not request.task.strip():
        raise HTTPException(status_code=400, detail="task is required")

    # build model config
    try:
        model_config = _build_model_config(request.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"invalid model config: {e}")

    # instantiate agent
    try:
        agent = PhoneAgent(model_config=model_config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to init PhoneAgent: {e}")

    # call agent.run synchronously
    try:
        result = agent.run(request.task)
    except Exception as e:
        # propagate as 500
        raise HTTPException(status_code=500, detail=f"agent.run error: {e}")

    plan = _convert_agent_output_to_action_plan(result)
    plan_id = str(uuid.uuid4())

    response = RunResponse(plan_id=plan_id, plan=plan)
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.ai_service:app", host="127.0.0.1", port=8001, log_level="info")

