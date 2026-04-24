from openai import OpenAI
import json

class Qwen:
    def __init__(self):
        self.client = OpenAI(
            base_url="",
            api_key=''
        )

    def execute(self, system_prompts, user_prompts):
        messages = list()
        for prompt in system_prompts:
            messages.append({"role": "system", "content": prompt})
            messages.append({"role": "system", "content": 'Please reply with the required information only and do not add comments or thoughts".'})
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
                    stream=False,
                    temperature = 0.1,
                    top_p= 0.1
                )
                content = completion.choices[0].message.content
                content = content[:next(idx for idx, c in reversed(list(enumerate(content))) if c in "}") + 1]
                content = content[next(idx for idx, c in reversed(list(enumerate(content))) if c in "{"):]
                json.loads(content.replace('```json', '').replace('`', ''))
                break
            except Exception as e:
                print(content)
                print(str(e))
                messages.append({"role": "user", "content": 'There is an error in your JSON format.'})
                now_attempt += 1
        # print(completion.choices[0].message.content)
        return content