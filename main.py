import os
import uuid
import time
import json
from typing import Annotated, List, Dict, Optional
from typing_extensions import TypedDict
from jose import JWTError, jwt
from datetime import datetime, timedelta
from pydantic import BaseModel
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph import START, END

# Set OpenAI API key
SECRET_KEY = "your-secret-key"
OPENAI_API_KEY = "Your API KEY"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# === Auth Helpers ===
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# === State & Agent Setup ===
class State(TypedDict):
    messages: Annotated[list, add_messages]
    tool_result: str

# Initialize LLM (Language Model)
llm = ChatOpenAI(model="gpt-4o-mini", stream_usage=True)

# Define chatbot function (calling LLM)
async def chatbot(state: State):
    # Call and return message
    response = await llm.ainvoke(state["messages"])
    usage = response.response_metadata.get("token_usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    # Tool call result (assumed example)
    tool_result = "Tool call information"

    return {
        "messages": [response],
        "tool_result": tool_result,
    }

# Create state graph
graph_builder = StateGraph(State)

# Add chatbot node
graph_builder.add_node("chatbot", chatbot)

# Add edges between nodes: START -> chatbot -> END
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile graph (final agent creation)
agent = graph_builder.compile()

# Define Message model
class Message(BaseModel):
    role: str  # 'system', 'user', 'assistant'
    content: str

# Define ChatCompletionRequest model
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None

# Format data for SSE (Server-Sent Events)
def format_sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"

# Handle streaming response
async def stream_chat_response(request: ChatCompletionRequest):
    state: State = {
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "tool_result": ""
    }

    # Streaming events from the agent
    async for event in agent.astream(state, stream_mode=["messages", "updates"]):
        if isinstance(event, tuple):
            stream_type, value = event
        else:
            stream_type, value = "messages", event

        if stream_type == "messages":
            msg, metadata = value
            token = msg.content            
            if token:
                chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "delta": {"content": token},
                        "index": 0,
                        "finish_reason": None
                    }]
                }
                yield format_sse(chunk)

        elif stream_type == "updates":
            # tool_result might be here
            tool_result = value.get('chatbot')['tool_result']

        # Send tool_result as a separate chunk (optional, can be removed)
    if tool_result:
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "delta": {
                    "function_call": {
                        "name": "source",
                        "arguments": tool_result
                    }
                },
                "index": 0,
                "finish_reason": "stop"
            }]
        }        
        yield format_sse(chunk)
    else:
        # End signal
        done_chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "delta": {},
                "index": 0,
                "finish_reason": "stop"
            }]
        }
        yield format_sse(done_chunk)
    yield "data: [DONE]\n\n"

# Handle OpenAI format response (non-streaming)
async def get_chat_response(request: ChatCompletionRequest):
    state: State = {
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "tool_result": ""
    }

    result = await agent.ainvoke(state)
    final_message = result["messages"][-1]
    resp = final_message.content
    usage = final_message.response_metadata.get("token_usage")
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = prompt_tokens + completion_tokens

    tool_result = result.get("tool_result", "")
    # Default message
    message = {
        "role": "assistant",
        "content": resp,
    }

    # Add function_call if tool_result is present
    if tool_result:
        message["function_call"] = {
            "name": "source",
            "arguments": tool_result
        }
    
    # Return response formatted according to OpenAI API
    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    })

# Define FastAPI endpoints
app = FastAPI()

@app.post("/token")
def issue_token(username: str, password: str):
    if username != "testuser" or password != "secret":
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/v1/chat/completions")
async def chat_completions(
    request_data: ChatCompletionRequest,
    authorization: Optional[str] = Header(default=None)
):
    # Handle streaming request
    if request_data.stream:
        return StreamingResponse(
            stream_chat_response(request_data),
            media_type="text/event-stream"
        )
    # Handle non-streaming request
    else:
        return await get_chat_response(request_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
