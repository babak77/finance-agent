from __future__ import annotations
from typing import List, AsyncGenerator, Any
import asyncio
import os
from datetime import datetime
from enum import Enum
import logging
from pathlib import Path

import streamlit as st
import logfire
from supabase import create_client
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart
)
from financial_agent import finance_ai_expert as ai_expert, PydanticAIDeps
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageRole(str, Enum):
    """Enumeration of possible message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class ChatMessage(BaseModel):
    """Enhanced format of messages with additional metadata."""
    role: MessageRole
    timestamp: datetime = Field(default_factory=datetime.now)
    content: str
    metadata: dict = Field(default_factory=dict)

class ChatConfig(BaseModel):
    """Configuration settings for the chat application."""
    title: str = "AI Agentic RAG"
    subtitle: str = "Ask any question about Finance and trading."
    input_placeholder: str = "How can I help you?"
    max_message_history: int = 50
    stream_chunk_size: int = 1024

class ChatState:
    """Class to manage chat state and session variables."""
    def __init__(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def messages(self) -> List[ModelMessage]:
        return st.session_state.messages

    def add_message(self, message: ModelMessage) -> None:
        """Add a message to the chat history."""
        if len(self.messages) >= ChatConfig().max_message_history:
            self.messages.pop(0)
        self.messages.append(message)

class APIClients:
    """Class to manage API clients and connections."""
    def __init__(self):
        self._load_environment()
        self.openai = AsyncOpenAI(api_key=self.openai_api_key)
        self.supabase = create_client(self.supabase_url, self.supabase_key)
        
        # Configure logfire
        logfire.configure(send_to_logfire='never')

    def _load_environment(self) -> None:
        """Load environment variables from .env file."""
        env_path = Path('.env')
        if env_path.exists():
            from dotenv import load_dotenv
            load_dotenv()
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not all([self.openai_api_key, self.supabase_url, self.supabase_key]):
            raise EnvironmentError("Missing required environment variables")

class MessageDisplay:
    """Class to handle message display logic."""
    @staticmethod
    def display_message_part(part: Any) -> None:
        """Display a single message part in the Streamlit UI."""
        try:
            if part.part_kind == 'system-prompt':
                with st.chat_message(MessageRole.SYSTEM.value):
                    st.markdown(f"**System**: {part.content}")
            
            elif part.part_kind == 'user-prompt':
                with st.chat_message(MessageRole.USER.value):
                    st.markdown(part.content)
            
            elif part.part_kind == 'text':
                with st.chat_message(MessageRole.ASSISTANT.value):
                    st.markdown(part.content)
            
            elif part.part_kind == 'tool-call':
                with st.chat_message(MessageRole.TOOL.value):
                    st.markdown(f"**Tool Call**: {part.content}")
        
        except Exception as e:
            logger.error(f"Error displaying message part: {e}")
            st.error("Error displaying message")

class ChatAgent:
    """Class to handle chat agent operations."""
    def __init__(self, api_clients: APIClients):
        self.api_clients = api_clients
        self.chat_state = ChatState()

    async def stream_response(self, user_input: str) -> AsyncGenerator[str, None]:
        """Stream the agent's response for a given user input."""
        deps = PydanticAIDeps(
            supabase=self.api_clients.supabase,
            openai_client=self.api_clients.openai
        )

        try:
            async with ai_expert.run_stream(
                user_input,
                deps=deps,
                message_history=self.chat_state.messages[:-1],
            ) as result:
                partial_text = ""
                async for chunk in result.stream_text(delta=True):
                    partial_text += chunk
                    yield partial_text

                # Process and store new messages
                filtered_messages = [
                    msg for msg in result.new_messages()
                    if not (hasattr(msg, 'parts') and
                           any(part.part_kind == 'user-prompt' for part in msg.parts))
                ]
                
                for message in filtered_messages:
                    self.chat_state.add_message(message)

                self.chat_state.add_message(
                    ModelResponse(parts=[TextPart(content=partial_text)])
                )

        except Exception as e:
            logger.error(f"Error in stream_response: {e}")
            yield f"Error: {str(e)}"

async def main():
    """Main application function."""
    try:
        # Initialize components
        config = ChatConfig()
        api_clients = APIClients()
        chat_agent = ChatAgent(api_clients)
        message_display = MessageDisplay()

        # Set up UI
        st.title(config.title)
        st.write(config.subtitle)

        # Display existing messages
        for msg in chat_agent.chat_state.messages:
            if isinstance(msg, (ModelRequest, ModelResponse)):
                for part in msg.parts:
                    message_display.display_message_part(part)

        # Handle user input
        user_input = st.chat_input(config.input_placeholder)
        if user_input:
            # Add user message to state
            chat_agent.chat_state.add_message(
                ModelRequest(parts=[UserPromptPart(content=user_input)])
            )
            
            # Display user message
            with st.chat_message(MessageRole.USER.value):
                st.markdown(user_input)

            # Display assistant response
            with st.chat_message(MessageRole.ASSISTANT.value):
                message_placeholder = st.empty()
                async for partial_response in chat_agent.stream_response(user_input):
                    message_placeholder.markdown(partial_response)

    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An error occurred. Please try again later.")

if __name__ == "__main__":
    asyncio.run(main())
