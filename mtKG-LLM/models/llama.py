from groq import Groq
import json

class LLaMA:
    def __init__(self):
        self.client = Groq(
            api_key=''
        )
    def execute(self, system_prompts, user_prompts):
        messages = list()
        for prompt in system_prompts:
            messages.append({"role": "system", "content": prompt})
            messages.append({"role": "system", "content": 'The answer should be within 50 words".'})
        for prompt in user_prompts:
            if not isinstance(prompt, str):
                messages.append({"role": "user", "content": [prompt]})
            else:
                messages.append({"role": "user", "content": prompt})
        max_attempt = 5
        now_attempt = 0
        while now_attempt < max_attempt:
            try:
                completion = self.client.chat.completions.create(
                    model="",
                    messages=messages,
                    temperature = 0.1,
                    top_p= 0.1
                )
                content = completion.choices[0].message.content    
                break
            except Exception as e:
                print(str(e))
                now_attempt += 1
        return content.strip()