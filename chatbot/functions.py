# python imports
import os
import io
import json
import uuid
import traceback
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


FUNCTION_DESCRIPTIONS = []

FUNCTIONS = {}
