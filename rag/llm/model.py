"""
LLM interface for local models using Ollama
"""

from typing import Optional, Dict, Any


class LLMInterface:
    """
    Interface for interacting with local language models via Ollama.

    Supports any model available in Ollama (llama3.2, mistral, etc.)
    """

    def __init__(
        self,
        model_name: str = "llama3.2",
        temperature: float = 0.7,
        max_tokens: int = 512,
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the LLM interface.

        Args:
            model_name: Name of the Ollama model to use
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            base_url: Ollama server URL
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.client = None

    def load_model(self):
        """Initialize connection to Ollama (lazy loading)"""
        if self.client is None:
            try:
                import ollama
                self.client = ollama
                print(f"Connected to Ollama with model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "ollama-python not installed. "
                    "Install with: pip install ollama"
                )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt to set context
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Generated text response
        """
        self.load_model()

        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        # Generate response
        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            options={
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens
            }
        )

        return response['message']['content']

    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response using retrieved context (RAG).

        Args:
            query: User query
            context: Retrieved context from documents
            system_prompt: Optional system prompt

        Returns:
            Generated answer
        """
        default_system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.
Use the context below to answer the user's question accurately.
If the context doesn't contain relevant information, say so clearly.
Be concise and informative."""

        rag_prompt = f"""Context:
{context}

Question: {query}

Answer based on the context above:"""

        return self.generate(
            prompt=rag_prompt,
            system_prompt=system_prompt or default_system_prompt
        )

    def check_availability(self) -> bool:
        """
        Check if Ollama is available and the model is installed.

        Returns:
            True if model is available, False otherwise
        """
        try:
            self.load_model()
            # Try to list models
            models = self.client.list()
            model_names = [m['name'] for m in models.get('models', [])]

            if self.model_name in model_names:
                return True
            else:
                print(f"Model '{self.model_name}' not found in Ollama.")
                print(f"Available models: {', '.join(model_names)}")
                print(f"\nTo install, run: ollama pull {self.model_name}")
                return False

        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            print("\nMake sure Ollama is running. Install from: https://ollama.ai")
            return False
