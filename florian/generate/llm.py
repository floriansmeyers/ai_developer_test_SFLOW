import re

import config


def generate_answer(system_prompt: str, user_prompt: str) -> str:
    """Generate an answer using GPT-4o."""
    client = config.get_openai_client()

    def _call():
        return client.chat.completions.create(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

    response = config.retry_api_call(_call)
    return response.choices[0].message.content


def strip_sources_block(answer: str) -> str:
    """Remove any trailing Sources: section the LLM may have generated.

    The CLI displays a deterministic "Retrieved Sources" panel, so the
    LLM-generated block is redundant and potentially inaccurate.
    """
    return re.sub(r"\n*\s*Sources:\s*.*", "", answer, flags=re.DOTALL).strip()
