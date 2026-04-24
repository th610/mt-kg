import json


def construct_payload(system_prompts, user_prompts, model_name, max_tokens=1000, temperature=0.8):
    payload = dict()
    payload['messages'] = list()
    for prompt in system_prompts:
        payload['messages'].append({"role": "system", "content": prompt})
    for prompt in user_prompts:
        if not isinstance(prompt, str):
            payload['messages'].append({"role": "user", "content": [prompt]})
        else:
            payload['messages'].append({"role": "user", "content": prompt})
    payload['model'] = model_name
    payload[max_tokens] = max_tokens
    payload['stream'] = False
    payload['temperature'] = temperature

    return json.dumps(payload)
