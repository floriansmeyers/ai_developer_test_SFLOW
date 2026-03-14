import config


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = config.get_openai_client()
    response = config.retry_api_call(
        lambda: client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    )
    return [item.embedding for item in response.data]


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]
