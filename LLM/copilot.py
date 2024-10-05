from LLM.gpt import DeepSeekClient
class LLMCopilot:
    def __init__(self, client: DeepSeekClient):
        self.client = client

    def get_advice(self, user_query: str) -> str:
        """
        Allows the user to ask for advice directly from the LLM.
        """
        prompt = f"You are an experienced mathematician and optimization expert. {user_query}"
        response = self.client.get_llm_suggestion(prompt)
        return response
