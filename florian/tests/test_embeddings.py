from unittest.mock import patch, MagicMock

from index.embeddings import embed_texts, embed_query


def test_embed_texts_returns_list():
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2]),
        MagicMock(embedding=[0.3, 0.4]),
    ]
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response

    with patch("config.get_openai_client", return_value=mock_client), \
         patch("config.retry_api_call", side_effect=lambda fn: fn()):
        result = embed_texts(["hello", "world"])

    assert len(result) == 2
    assert result[0] == [0.1, 0.2]
    assert result[1] == [0.3, 0.4]


def test_embed_query_returns_single_vector():
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.5, 0.6])]
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response

    with patch("config.get_openai_client", return_value=mock_client), \
         patch("config.retry_api_call", side_effect=lambda fn: fn()):
        result = embed_query("test")

    assert result == [0.5, 0.6]
