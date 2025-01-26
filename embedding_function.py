from typing import List

from openai import AsyncOpenAI

async def get_openai_embedding(client: AsyncOpenAI, text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

