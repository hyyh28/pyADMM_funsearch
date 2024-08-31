import http.client
import json

class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key
        self.base_url = base_url

    def _make_request(self, messages, model="deepseek-coder", max_tokens=2048, temperature=1.0):
        payload = json.dumps({
            "messages": messages,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature
        })

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }

        conn = http.client.HTTPSConnection(self.base_url.replace("https://", ""))
        conn.request("POST", "/chat/completions", payload, headers)
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        data = json.loads(data)
        return data

    def get_llm_suggestion(self, background: str, prompt: str) -> str:
        messages = [
            {"role": "system", "content": background},
            {"role": "user", "content": prompt},
        ]
        try:
            response_data = self._make_request(messages)
            response = response_data['choices'][0]['message']['content']
            return response
        except Exception as e:
            print(f"Error during LLM interaction: {e}")
            return None

def suggest_admm_parameters(previous_reward=None):
    prompt = "Suggest parameters lam, rho, and alpha for the ADMM optimizer to deal with LASSO problem. Only output the parameters such as lam=x, rho=y, alpha=z (where x, y, z are float) without any descriptions."
    if previous_reward is not None:
        prompt += f" The previous reward was {previous_reward:.4f}. You need to improve the performance by giving better parameters."

    client = DeepSeekClient(api_key="sk-66575172e83e40b2bbcaa1cf6b9f0ae8")
    suggestion = client.get_llm_suggestion(prompt)

    if suggestion:
        params = {}
        for param in suggestion.split(","):
            key, value = param.split("=")
            params[key.strip()] = float(value.strip())

        return params['lam'], params['rho'], params['alpha']
    else:
        return None, None, None
