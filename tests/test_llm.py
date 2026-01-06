"""
Tests for rag/llm/model.py
"""

import pytest
from unittest.mock import patch, Mock, MagicMock
from rag.llm.model import LLMInterface


@pytest.mark.unit
class TestLLMInterface:
    """Test cases for LLMInterface class"""

    def test_init_default(self):
        """Test initialization with default parameters"""
        llm = LLMInterface()

        assert llm.model_name == "llama3.2"
        assert llm.temperature == 0.7
        assert llm.max_tokens == 512
        assert llm.base_url == "http://localhost:11434"
        assert llm.client is None

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters"""
        llm = LLMInterface(
            model_name="mistral",
            temperature=0.5,
            max_tokens=256,
            base_url="http://custom:8080"
        )

        assert llm.model_name == "mistral"
        assert llm.temperature == 0.5
        assert llm.max_tokens == 256
        assert llm.base_url == "http://custom:8080"

    @patch('rag.llm.model.ollama')
    def test_load_model_success(self, mock_ollama):
        """Test successful model loading"""
        llm = LLMInterface()
        llm.load_model()

        assert llm.client == mock_ollama

    @patch('rag.llm.model.ollama')
    def test_load_model_only_once(self, mock_ollama):
        """Test model is loaded only once (lazy loading)"""
        llm = LLMInterface()
        llm.load_model()
        llm.load_model()  # Call again

        # Client should be set once
        assert llm.client is not None

    def test_load_model_import_error(self):
        """Test error handling when ollama not installed"""
        with patch.dict('sys.modules', {'ollama': None}):
            llm = LLMInterface()
            with pytest.raises(ImportError, match="ollama-python not installed"):
                llm.load_model()

    @patch('rag.llm.model.ollama')
    def test_generate_basic(self, mock_ollama):
        """Test basic text generation"""
        mock_ollama.chat.return_value = {
            'message': {'content': 'Generated response'}
        }

        llm = LLMInterface()
        response = llm.generate("Test prompt")

        assert response == 'Generated response'
        mock_ollama.chat.assert_called_once()

    @patch('rag.llm.model.ollama')
    def test_generate_with_system_prompt(self, mock_ollama):
        """Test generation with system prompt"""
        mock_ollama.chat.return_value = {
            'message': {'content': 'Response with system prompt'}
        }

        llm = LLMInterface()
        response = llm.generate(
            "User prompt",
            system_prompt="You are a helpful assistant"
        )

        # Check that system message was included
        call_args = mock_ollama.chat.call_args
        messages = call_args[1]['messages']

        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
        assert messages[0]['content'] == "You are a helpful assistant"
        assert messages[1]['role'] == 'user'
        assert response == 'Response with system prompt'

    @patch('rag.llm.model.ollama')
    def test_generate_without_system_prompt(self, mock_ollama):
        """Test generation without system prompt"""
        mock_ollama.chat.return_value = {
            'message': {'content': 'Response'}
        }

        llm = LLMInterface()
        response = llm.generate("User prompt")

        # Check that only user message was included
        call_args = mock_ollama.chat.call_args
        messages = call_args[1]['messages']

        assert len(messages) == 1
        assert messages[0]['role'] == 'user'

    @patch('rag.llm.model.ollama')
    def test_generate_custom_temperature(self, mock_ollama):
        """Test generation with custom temperature"""
        mock_ollama.chat.return_value = {
            'message': {'content': 'Response'}
        }

        llm = LLMInterface(temperature=0.7)
        llm.generate("Test", temperature=0.3)

        # Check that custom temperature was used
        call_args = mock_ollama.chat.call_args
        options = call_args[1]['options']

        assert options['temperature'] == 0.3

    @patch('rag.llm.model.ollama')
    def test_generate_custom_max_tokens(self, mock_ollama):
        """Test generation with custom max tokens"""
        mock_ollama.chat.return_value = {
            'message': {'content': 'Response'}
        }

        llm = LLMInterface(max_tokens=512)
        llm.generate("Test", max_tokens=256)

        # Check that custom max_tokens was used
        call_args = mock_ollama.chat.call_args
        options = call_args[1]['options']

        assert options['num_predict'] == 256

    @patch('rag.llm.model.ollama')
    def test_generate_uses_default_params(self, mock_ollama):
        """Test that default parameters are used when not overridden"""
        mock_ollama.chat.return_value = {
            'message': {'content': 'Response'}
        }

        llm = LLMInterface(temperature=0.8, max_tokens=1024)
        llm.generate("Test")

        call_args = mock_ollama.chat.call_args
        options = call_args[1]['options']

        assert options['temperature'] == 0.8
        assert options['num_predict'] == 1024

    @patch('rag.llm.model.ollama')
    def test_generate_with_context_basic(self, mock_ollama):
        """Test RAG generation with context"""
        mock_ollama.chat.return_value = {
            'message': {'content': 'Answer based on context'}
        }

        llm = LLMInterface()
        context = "Document 1: Information about AI\nDocument 2: Machine learning basics"
        response = llm.generate_with_context(
            query="What is AI?",
            context=context
        )

        assert response == 'Answer based on context'

        # Check that prompt includes context
        call_args = mock_ollama.chat.call_args
        messages = call_args[1]['messages']
        user_message = messages[-1]['content']

        assert "Context:" in user_message
        assert context in user_message
        assert "What is AI?" in user_message

    @patch('rag.llm.model.ollama')
    def test_generate_with_context_custom_system(self, mock_ollama):
        """Test RAG generation with custom system prompt"""
        mock_ollama.chat.return_value = {
            'message': {'content': 'Custom response'}
        }

        llm = LLMInterface()
        custom_system = "You are a specialized assistant"
        response = llm.generate_with_context(
            query="Test query",
            context="Test context",
            system_prompt=custom_system
        )

        # Check that custom system prompt was used
        call_args = mock_ollama.chat.call_args
        messages = call_args[1]['messages']

        assert messages[0]['role'] == 'system'
        assert messages[0]['content'] == custom_system

    @patch('rag.llm.model.ollama')
    def test_generate_with_context_default_system(self, mock_ollama):
        """Test that default system prompt is used in RAG generation"""
        mock_ollama.chat.return_value = {
            'message': {'content': 'Response'}
        }

        llm = LLMInterface()
        llm.generate_with_context(
            query="Test",
            context="Context"
        )

        call_args = mock_ollama.chat.call_args
        messages = call_args[1]['messages']

        assert messages[0]['role'] == 'system'
        assert 'helpful AI assistant' in messages[0]['content']

    @patch('rag.llm.model.ollama')
    def test_check_availability_success(self, mock_ollama):
        """Test successful availability check"""
        mock_ollama.list.return_value = {
            'models': [
                {'name': 'llama3.2'},
                {'name': 'mistral'}
            ]
        }

        llm = LLMInterface(model_name='llama3.2')
        available = llm.check_availability()

        assert available is True
        mock_ollama.list.assert_called_once()

    @patch('rag.llm.model.ollama')
    def test_check_availability_model_not_found(self, mock_ollama, capsys):
        """Test availability check when model not installed"""
        mock_ollama.list.return_value = {
            'models': [
                {'name': 'mistral'}
            ]
        }

        llm = LLMInterface(model_name='llama3.2')
        available = llm.check_availability()

        assert available is False

        # Check that helpful message was printed
        captured = capsys.readouterr()
        assert "not found" in captured.out
        assert "ollama pull" in captured.out

    @patch('rag.llm.model.ollama')
    def test_check_availability_connection_error(self, mock_ollama, capsys):
        """Test availability check when Ollama not running"""
        mock_ollama.list.side_effect = Exception("Connection refused")

        llm = LLMInterface()
        available = llm.check_availability()

        assert available is False

        # Check that helpful error message was printed
        captured = capsys.readouterr()
        assert "Error connecting" in captured.out
        assert "ollama" in captured.out.lower()

    @patch('rag.llm.model.ollama')
    def test_model_name_passed_to_chat(self, mock_ollama):
        """Test that model name is correctly passed to Ollama"""
        mock_ollama.chat.return_value = {
            'message': {'content': 'Response'}
        }

        llm = LLMInterface(model_name='mistral')
        llm.generate("Test")

        call_args = mock_ollama.chat.call_args
        assert call_args[1]['model'] == 'mistral'

    @patch('rag.llm.model.ollama')
    def test_empty_prompt(self, mock_ollama):
        """Test handling of empty prompt"""
        mock_ollama.chat.return_value = {
            'message': {'content': 'Response to empty'}
        }

        llm = LLMInterface()
        response = llm.generate("")

        assert isinstance(response, str)

    @patch('rag.llm.model.ollama')
    def test_empty_context(self, mock_ollama):
        """Test RAG generation with empty context"""
        mock_ollama.chat.return_value = {
            'message': {'content': 'Response'}
        }

        llm = LLMInterface()
        response = llm.generate_with_context(
            query="Test query",
            context=""
        )

        assert isinstance(response, str)

    @patch('rag.llm.model.ollama')
    def test_long_context(self, mock_ollama):
        """Test handling of very long context"""
        mock_ollama.chat.return_value = {
            'message': {'content': 'Response'}
        }

        llm = LLMInterface()
        long_context = "Document content. " * 1000
        response = llm.generate_with_context(
            query="Summarize",
            context=long_context
        )

        assert isinstance(response, str)


@pytest.mark.integration
@pytest.mark.requires_ollama
class TestLLMIntegration:
    """Integration tests requiring Ollama to be running"""

    def test_real_ollama_connection(self):
        """Test with real Ollama instance (if available)"""
        try:
            llm = LLMInterface()
            if llm.check_availability():
                response = llm.generate("Say 'test' if you can hear me")
                assert isinstance(response, str)
                assert len(response) > 0
            else:
                pytest.skip("Ollama not available")
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")

    def test_real_rag_generation(self):
        """Test RAG generation with real Ollama"""
        try:
            llm = LLMInterface()
            if llm.check_availability():
                context = "Python is a programming language. It is widely used for data science."
                response = llm.generate_with_context(
                    query="What is Python used for?",
                    context=context
                )
                assert isinstance(response, str)
                assert len(response) > 0
            else:
                pytest.skip("Ollama not available")
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")
