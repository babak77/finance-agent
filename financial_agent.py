from __future__ import annotations
from typing import List, Dict, Optional
from dotenv import load_dotenv
import os

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from dataclasses import dataclass
from supabase import Client, create_client

from ai_agent import pydantic_ai_expert, PydanticAIDeps
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List
from embedding_function import get_openai_embedding

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt = """
You are a knowledgeable financial assistant that answers questions based on the provided document chunks.
Follow these guidelines:
1. Use only information from the provided chunks to answer questions
2. If the chunks don't contain enough information, say so clearly
3. Cite specific documents when using their information
4. Be concise but thorough in your explanations
5. Focus on factual information rather than opinions
6. If asked about investment advice, remind users to consult financial advisors
"""
finance_ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=1
)

@finance_ai_expert.tool
async def retrieve_relevant_data(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant data chunks based on the query with RAG.

    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query

    Returns:
        A formatted string containing the top 5 most relevant data chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_openai_embedding(ctx.deps.openai_client, user_query)
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_book_documents',
            {
                'query_embedding': query_embedding,
                'match_threshold': 0.0,
                'match_count': 5
            }
        ).execute()

        if not result.data:
            return "No relevant documentation found."

        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
                        {doc['heading']}
                        {doc['content']}
                        """
            formatted_chunks.append(chunk_text)

        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@finance_ai_expert.tool
async def get_summary(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of summary field to help find the relevant content.

    Returns:
        List[str]: List of unique filenames
    """
    try:
        # Query Supabase for filenames
        result = ctx.deps.supabase.from_('book_embeddings') \
            .select('summary') \
            .execute()

        if not result.data:
            return []

        # Extract unique filenames
        filenames = sorted(set(doc['filename'] for doc in result.data))
        return filenames

    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@finance_ai_expert.tool
async def list_filename(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all different filenames.

    Returns:
        List[str]: List of unique filenames
    """
    try:
        # Query Supabase for filenames
        result = ctx.deps.supabase.from_('book_embeddings') \
            .select('filename') \
            .execute()

        if not result.data:
            return []

        # Extract unique filenames
        filenames = sorted(set(doc['filename'] for doc in result.data))
        return filenames

    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []