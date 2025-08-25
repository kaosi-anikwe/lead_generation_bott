# python imports
import os
import io
import json
import uuid
import traceback
import requests
from typing import Tuple, List, Dict, Any, Optional

# installed imports
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader

# local imports
from . import logger
from .config import Config


load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Init OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)


def load_instructions_from_file(instructions_file_path: str) -> str:
    """Load assistant instructions from a text file or PDF"""
    try:
        if not os.path.exists(instructions_file_path):
            logger.warning(
                f"Instructions file {instructions_file_path} not found, using default"
            )
            return "You are a helpful AI assistant."

        # First, try to detect if it's a PDF by reading the first few bytes
        with open(instructions_file_path, "rb") as file:
            header = file.read(5)
            file.seek(0)
            
        if header.startswith(b'%PDF-'):
            # It's a PDF file
            try:
                reader = PdfReader(instructions_file_path)
                instructions = ""
                for page in reader.pages:
                    instructions += page.extract_text() + "\n"
                
                instructions = instructions.strip()
                if not instructions:
                    logger.warning(
                        f"PDF file {instructions_file_path} contains no readable text, using default"
                    )
                    return "You are a helpful AI assistant."
                
                logger.info(f"Loaded instructions from PDF {instructions_file_path}")
                return instructions
                
            except Exception as e:
                logger.error(f"Error reading PDF file {instructions_file_path}: {e}")
                return "You are a helpful AI assistant."
        else:
            # Assume it's a text file
            try:
                with open(instructions_file_path, "r", encoding="utf-8") as file:
                    instructions = file.read().strip()
                    if not instructions:
                        logger.warning(
                            f"Instructions file {instructions_file_path} is empty, using default"
                        )
                        return "You are a helpful AI assistant."

                    logger.info(f"Loaded instructions from text file {instructions_file_path}")
                    return instructions
                    
            except UnicodeDecodeError:
                logger.error(f"File {instructions_file_path} is not a valid text file or PDF")
                return "You are a helpful AI assistant."

    except Exception as e:
        logger.error(f"Error reading instructions file {instructions_file_path}: {e}")
        return "You are a helpful AI assistant."


def setup_knowledge_base(
    client: OpenAI, knowledge_file_path: str
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Setup knowledge base and return tools and tool_resources"""
    if not os.path.exists(knowledge_file_path):
        logger.warning(f"Knowledge file {knowledge_file_path} not found")
        return FUNCTION_DESCRIPTIONS, None

    try:
        # Detect file type and prepare content
        with open(knowledge_file_path, "rb") as file:
            header = file.read(5)
            file.seek(0)  # Reset file pointer
            
        if header.startswith(b'%PDF-'):
            # It's a PDF file - upload directly as binary
            with open(knowledge_file_path, "rb") as knowledge_file:
                knowledge_doc = io.BytesIO(knowledge_file.read())
                knowledge_doc.name = f"{uuid.uuid4().hex}_{os.path.basename(knowledge_file_path)}.pdf"
            logger.info(f"Preparing PDF knowledge base: {knowledge_file_path}")
            
        else:
            # It's a text file - convert to proper format for upload
            try:
                with open(knowledge_file_path, "r", encoding="utf-8") as text_file:
                    text_content = text_file.read()
                
                # Create a BytesIO object with the text content
                knowledge_doc = io.BytesIO(text_content.encode('utf-8'))
                knowledge_doc.name = f"{uuid.uuid4().hex}_{os.path.basename(knowledge_file_path)}.txt"
                logger.info(f"Preparing text knowledge base: {knowledge_file_path}")
                
            except UnicodeDecodeError:
                # If it's not a valid text file and not a PDF, treat as binary
                with open(knowledge_file_path, "rb") as knowledge_file:
                    knowledge_doc = io.BytesIO(knowledge_file.read())
                    knowledge_doc.name = f"{uuid.uuid4().hex}_{os.path.basename(knowledge_file_path)}"
                logger.info(f"Preparing binary knowledge base: {knowledge_file_path}")

        # Upload knowledge document
        file = client.files.create(
            file=knowledge_doc,
            purpose="assistants",
        )

        vector_store = client.beta.vector_stores.create(file_ids=[file.id])
        tool_resources = {"file_search": {"vector_store_ids": [vector_store.id]}}
        tools = [{"type": "file_search"}, *FUNCTION_DESCRIPTIONS]

        logger.info(f"Successfully uploaded knowledge base: {knowledge_file_path}")
        return tools, tool_resources

    except Exception as e:
        logger.error(f"Error setting up knowledge base: {e}")
        return FUNCTION_DESCRIPTIONS, None


def create_or_update_assistant(
    client: OpenAI,
    assistant_id: str,
    instructions: str,
    tools: List[Dict[str, Any]],
    tool_resources: Optional[Dict[str, Any]],
) -> str:
    """Create new assistant or update existing one"""
    model = Config.OPENAI_MODEL

    assistant_config = {
        "instructions": instructions,
        "model": model,
        "tools": tools,
        "tool_resources": tool_resources,
        "temperature": 0.1,
        "response_format": {"type": "text"},
    }

    if assistant_id:
        # Try to update existing assistant
        try:
            existing_assistant = client.beta.assistants.retrieve(assistant_id)
            logger.info(f"Found existing assistant: {assistant_id}")

            # Update the assistant
            updated_assistant = client.beta.assistants.update(
                assistant_id=assistant_id, **assistant_config
            )

            logger.info(f"Successfully updated assistant: {assistant_id}")
            return updated_assistant.id

        except Exception as e:
            logger.warning(f"Failed to update assistant {assistant_id}: {e}")
            logger.info("Creating new assistant instead")

    # Create new assistant
    try:
        assistant = client.beta.assistants.create(**assistant_config)
        logger.info(f"Created new assistant with ID: {assistant.id}")
        return assistant.id

    except Exception as e:
        logger.error(f"Error creating assistant: {e}")
        raise


def create_assistant(client: OpenAI) -> str:
    """
    Create or update an OpenAI assistant based on environment variables.

    Environment variables used:
    - ASSISTANT_ID: If provided, attempts to update this assistant
    - KNOWLEDGE_BASE_FILE: Path to knowledge base file (default: knowledge.pdf)
    - INSTRUCTIONS_FILE: Path to instructions text file (default: instructions.txt)
    """
    try:
        # Get configuration from environment
        assistant_id = Config.ASSISTANT_ID
        knowledge_file_path = Config.KNOWLEDGE_BASE_FILE
        instructions_file_path = Config.INSTRUCTIONS_FILE

        logger.info(f"Setting up assistant with:")
        logger.info(f"  Assistant ID: {assistant_id or 'None (will create new)'}")
        logger.info(f"  Knowledge file: {knowledge_file_path}")
        logger.info(f"  Instructions file: {instructions_file_path}")

        # Load instructions from file
        instructions = load_instructions_from_file(instructions_file_path)

        # Setup knowledge base
        tools, tool_resources = setup_knowledge_base(client, knowledge_file_path)

        # Create or update assistant
        final_assistant_id = create_or_update_assistant(
            client, assistant_id, instructions, tools, tool_resources
        )

        # Save assistant ID to file for backward compatibility
        assistant_file_path = "assistant.json"
        try:
            with open(assistant_file_path, "w") as file:
                json.dump({"assistant_id": final_assistant_id}, file)
            logger.info(f"Saved assistant ID to {assistant_file_path}")
        except Exception as e:
            logger.warning(f"Failed to save assistant ID to file: {e}")

        return final_assistant_id

    except Exception as e:
        logger.error(f"Error in create_assistant: {traceback.format_exc()}")
        raise


def get_conversation(thread_id: str, client: OpenAI = client) -> str:
    try:
        if not thread_id:
            return "Error: Thread ID is required"

        logger.info(f"Getting conversation for {thread_id}")
        messages = list(client.beta.threads.messages.list(thread_id=thread_id))
        logger.info(f"Retrieved {len(messages)} messages")

        if not messages:
            return "No messages found in conversation"

        conversation_parts = []
        for message in reversed(messages):
            if message.content and len(message.content) > 0:
                content = message.content[0].text.value
                conversation_parts.append(f"{message.role.upper()}: {content}")

        return "\n".join(conversation_parts)

    except Exception as e:
        logger.error(f"Error retrieving conversation: {traceback.format_exc()}")
        return f"Error retrieving conversation: {str(e)}"


def extract_user_info(user_info: str) -> dict:
    """Extract and process user information for appointment booking."""
    try:
        info = json.loads(user_info)
        
        # Send webhook to LEAD_WEBHOOK
        webhook_url = os.getenv("LEAD_WEBHOOK")
        if webhook_url:
            try:
                response = requests.post(webhook_url, json=info)
                logger.info(f"Lead webhook sent successfully: {response.status_code}")
            except Exception as e:
                logger.error(f"Failed to send lead webhook: {e}")
        
        return {
            "success": True,
            "message": "User information extracted and processed successfully",
            "data": info
        }
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in user_info: {user_info}")
        return {
            "success": False,
            "message": "Invalid user information format"
        }
    except Exception as e:
        logger.error(f"Error processing user info: {e}")
        return {
            "success": False,
            "message": "Failed to process user information"
        }


def contact_support(user_info: str) -> dict:
    """Forward user complaint/request to support team."""
    try:
        info = json.loads(user_info)

        info["email"] = os.getenv("EMAIL_RECIPIENT")
        
        # Send webhook to NOLEAD_WEBHOOK
        webhook_url = os.getenv("NOLEAD_WEBHOOK")
        if webhook_url:
            try:
                response = requests.post(webhook_url, json=info)
                logger.info(f"Support webhook sent successfully: {response.status_code}")
            except Exception as e:
                logger.error(f"Failed to send support webhook: {e}")
        
        return {
            "success": True,
            "message": "Your request has been forwarded to our support team. They will contact you shortly."
        }
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in user_info: {user_info}")
        return {
            "success": False,
            "message": "Invalid user information format"
        }
    except Exception as e:
        logger.error(f"Error contacting support: {e}")
        return {
            "success": False,
            "message": "Failed to contact support"
        }


def image_upload(upload: bool, prompt: str) -> dict:
    """Prompt user to upload an image if beneficial for their situation."""
    return {
        "upload_needed": upload,
        "prompt": prompt,
        "message": prompt if upload else "No image upload needed for this situation."
    }


def embed_video(embed: bool, prompt: str) -> dict:
    """Display testimonial videos when user requests reviews or referrals."""
    return {
        "embed_video": embed,
        "prompt": prompt,
        "message": prompt if embed else "No video testimonials needed for this request."
    }


FUNCTION_DESCRIPTIONS = [
    {
        "type": "function",
        "function": {
            "name": "extract_user_info",
            "description": "Extracts and processes complete user information for appointment booking. ONLY call this function ONCE when ALL required information has been collected: name, email, phone number, website, and appointment summary. Do not call multiple times in the same conversation. The function sends the lead data to a webhook and returns confirmation of successful processing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_info": {
                        "type": "string",
                        "description": "JSON string containing complete user information with required fields: name, email, phone_number, website, and info (appointment summary). Example: {\"name\": \"John Doe\", \"email\": \"john@example.com\", \"phone_number\": \"123-456-7890\", \"website\": \"example.com\", \"info\": \"interested in mental health course consultation\"}",
                    }
                },
                "required": ["user_info"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "contact_support",
            "description": "Forwards user complaints or requests that cannot be handled directly to the support team. Use when user needs manager escalation, receipt retrieval, or other support-only tasks. Requires complete user information before submission.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_info": {
                        "type": "string",
                        "description": "JSON string containing user information with fields: name, email, phone_number, and info (complaint/request details). Example: {\"name\": \"Jane Smith\", \"email\": \"jane@example.com\", \"phone_number\": \"987-654-3210\", \"info\": \"need to speak with manager about billing issue\"}",
                    }
                },
                "required": ["user_info"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "image_upload",
            "description": "Determines if the user's situation would benefit from uploading an image for better assistance. Only call when visual context would genuinely help resolve their issue. Does not handle actual upload - only prompts the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "upload": {
                        "type": "boolean",
                        "description": "Whether the current situation would benefit from an image upload.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "User-friendly message explaining why an image would be helpful and requesting upload.",
                    },
                },
                "required": ["upload", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "embed_video",
            "description": "Shows testimonial videos when users request reviews, referrals, or want to see other clients the company has worked with. Only call when user specifically asks about testimonials, reviews, or proof of past work.",
            "parameters": {
                "type": "object",
                "properties": {
                    "embed": {
                        "type": "boolean",
                        "description": "Whether to display testimonial videos based on user request.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "User-friendly message introducing the testimonial content.",
                    },
                },
                "required": ["embed", "prompt"],
            },
        },
    },
]

FUNCTIONS = {
    "extract_user_info": extract_user_info,
    "contact_support": contact_support,
    "image_upload": image_upload,
    "embed_video": embed_video,
}
