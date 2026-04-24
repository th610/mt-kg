from openai import OpenAI
import traceback
import json
import time

class Deepseek:
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
        content = None
        while now_attempt < max_attempt:
            try:
                completion = self.client.chat.completions.create(
                    model="",
                    messages=messages,
                    stream=False,
                    temperature = 0.1,
                    top_p= 0.1
                )
                content = completion.choices[0].message.content.replace('[OUTPUT]', '').replace('```json', '').replace('`', '').strip()
                json.loads(content)
                break
            except Exception as e:
                print(str(e))
                if content:
                    print(content)
                else:
                    time.sleep(60)
                # print(traceback.format_exc())
                # print(completion)
                now_attempt += 1
        if now_attempt == 5:
            content = json.dumps({
                "reasoning": "none",
                "relation": "unknown"
            })
        # print(completion.choices[0].message.content)
        return content