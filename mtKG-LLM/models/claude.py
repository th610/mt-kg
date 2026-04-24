from openai import OpenAI

class Claude:
    def __init__(self):
        self.client = OpenAI(
            base_url='',
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
                    temperature = 0.1,
                    top_p= 0.1
                )
                break
            except Exception as e:
                print(str(e))
                now_attempt += 1
        # print(completion.choices[0].message.content)
        return completion.choices[0].message.content