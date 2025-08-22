# python imports
import os
import json
import time
import traceback
from typing import Dict, Any, Tuple

# installed imports
import openai
from openai import OpenAI
from packaging import version
from flask import Blueprint, request, jsonify
from marshmallow import Schema, fields, ValidationError

# local imports
from . import logger
from .functions import create_assistant, get_conversation, FUNCTIONS

TIMEZONE = os.getenv("TIMEZONE", "America/New_York")

# Check OpenAI version compatibility
required_version = version.parse("1.1.1")
current_version = version.parse(openai.__version__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

if current_version < required_version:
    raise ValueError(
        f"Error: OpenAI version {openai.__version__} is less than the required version 1.1.1"
    )
else:
    logger.info("OpenAI version is compatible.")

chatbot = Blueprint("chatbot", __name__)

# Dict for logging SMS response cooldown
wait = {}
COOLDOWN = 10

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Create or load assistant
assistant_id = create_assistant(client)


# Request validation schemas
class ChatRequestSchema(Schema):
    thread_id = fields.String(required=False, allow_none=True)
    message = fields.String(required=True)
    stream = fields.Boolean(required=False)  # v2 streaming support


chat_schema = ChatRequestSchema()


def handle_api_error(func):
    """Decorator for consistent error handling"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            logger.error(f"Validation error in {func.__name__}: {e.messages}")
            return (
                jsonify({"error": "Invalid request data", "details": e.messages}),
                400,
            )
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {traceback.format_exc()}")
            return jsonify({"error": "Internal server error"}), 500

    wrapper.__name__ = func.__name__
    return wrapper


# Generate response
@chatbot.route("/chat", methods=["POST"])
@handle_api_error
def chat() -> Tuple[Dict[str, Any], int]:
    data = chat_schema.load(request.get_json())

    thread_id = data.get("thread_id")
    user_input = data.get("message", "")

    if not thread_id:
        # Create thread
        logger.info("Creating thread ID")
        thread = client.beta.threads.create()
        thread_id = thread.id
        logger.info(f"New thread created with ID: {thread.id}")
        return jsonify({"error": "Missing thread_id"}), 400

    logger.info(f"Received message: {user_input} for thread ID: {thread_id}")

    # Add message to thread
    client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=user_input
    )

    # Run the Assistant with v2 features
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        max_completion_tokens=1000,  # Token control for cost management
        temperature=0.1,  # Consistent responses
    )

    # Poll for completion with timeout
    max_attempts = 60  # 60 seconds timeout
    attempt = 0

    while attempt < max_attempts:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread_id, run_id=run.id
        )
        logger.info(f"Run status: {run_status.status}")

        if run_status.status in ["completed", "failed", "expired"]:
            break
        elif run_status.status == "requires_action":
            # Handle function calls
            tool_outputs = []
            for tool_call in run_status.required_action.submit_tool_outputs.tool_calls:
                logger.info(f"Processing function call: {tool_call.function.name}")

                function_name = tool_call.function.name
                if function_name in FUNCTIONS:
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                        output = FUNCTIONS[function_name](**arguments)
                        if output:
                            logger.info(f"Function output: {output}")
                            tool_outputs.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "output": json.dumps(output),
                                }
                            )
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.error(f"Error processing function call: {e}")
                        tool_outputs.append(
                            {
                                "tool_call_id": tool_call.id,
                                "output": json.dumps(
                                    {"error": "Invalid function arguments"}
                                ),
                            }
                        )
                else:
                    logger.warning(f"Unknown function: {function_name}")
                    tool_outputs.append(
                        {
                            "tool_call_id": tool_call.id,
                            "output": json.dumps({"error": "Function not found"}),
                        }
                    )

            # Submit tool outputs
            if tool_outputs:
                client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run.id,
                    tool_outputs=tool_outputs,
                )

        time.sleep(1)
        attempt += 1

    if attempt >= max_attempts:
        logger.error(f"Run timed out for thread {thread_id}")
        return jsonify({"error": "Request timed out"}), 408

    # Retrieve the latest message from the assistant
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    if not messages.data:
        logger.error("No messages found in thread")
        return jsonify({"error": "No response generated"}), 500

    response = messages.data[0].content[0].text.value
    logger.info(f"Assistant response: {response}")

    return jsonify({"response": response}), 200


@chatbot.get("/get-conversation/<thread_id>")
@handle_api_error
def retrieve_conversation(thread_id: str) -> Tuple[Dict[str, Any], int]:
    if not thread_id or not thread_id.strip():
        return jsonify({"error": "Invalid thread_id"}), 400

    logger.info(f"Getting conversation context for {thread_id}")
    conversation = get_conversation(thread_id)
    return jsonify({"conversation": conversation}), 200
