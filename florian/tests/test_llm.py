"""Tests for generate/llm.py."""

from unittest.mock import patch, MagicMock

from generate.llm import strip_sources_block


def test_strip_trailing_sources_block():
    """Strips trailing 'Sources: ...' line."""
    answer = "Invoices are created per unit.\n\nSources: invoicing.md"
    result = strip_sources_block(answer)
    assert result == "Invoices are created per unit."


def test_strip_sources_with_bullet_list():
    """Strips Sources: block that contains a bullet list."""
    answer = (
        "Invoices are created per unit.\n\n"
        "Sources:\n- invoicing.md\n- projects.md"
    )
    result = strip_sources_block(answer)
    assert result == "Invoices are created per unit."


def test_no_op_when_no_sources_block():
    """Returns answer unchanged when there is no Sources: block."""
    answer = "Invoices are created per unit."
    result = strip_sources_block(answer)
    assert result == answer


def test_preserves_content_before_sources():
    """Content before the Sources: block is fully preserved."""
    answer = (
        "First paragraph.\n\n"
        "Second paragraph with details.\n\n"
        "Sources: invoicing.md, projects.md"
    )
    result = strip_sources_block(answer)
    assert result == "First paragraph.\n\nSecond paragraph with details."


def test_generate_answer_with_mock():
    """generate_answer should call the OpenAI client and return content."""
    mock_choice = MagicMock()
    mock_choice.message.content = "Mocked answer"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("generate.llm.config") as mock_config:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_config.get_openai_client.return_value = mock_client
        mock_config.LLM_MODEL = "gpt-4o"
        mock_config.LLM_TEMPERATURE = 0.1
        mock_config.LLM_MAX_TOKENS = 1024
        mock_config.retry_api_call = lambda fn: fn()

        from generate.llm import generate_answer
        result = generate_answer("system", "user")
        assert result == "Mocked answer"
        mock_client.chat.completions.create.assert_called_once()
