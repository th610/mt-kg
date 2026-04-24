import json
def background_summarize(whole_image, llm):
    # return ''
    system_prompts = ['You are a helpful assistant to summarize the background of the image.']
    user_prompts = list()
    user_prompts.append('What is the background of the image?')
    user_prompts.append({'type':'image_url', 'image_url': {'url':f'data:image/jpeg;base64,{whole_image}'}})
    response = llm.execute(system_prompts, user_prompts)
    return response

def interaction_summarize(whole_image, person1_image, person2_image, conversation, llm):
    # return ''
    system_prompts = ['''
        [TASK]
        As a perceptive assistant, you are expected to synthesize both visual and textual information to provide a concise and insightful summary of the interaction between two people depicted in the dataset. Evaluate the setting, the appearance and positioning of individuals, and their conversation to deduce their relationship, emotional state, and the purpose of their interaction. Use context clues from the whole image, detailed images of each person, and the conversation snippet provided. In the absence of a piece of information, use informed assumptions based on available data to complete your summary effectively. Make sure your summary encapsulates the essence of their interaction, elucidating on the underlying dynamics, possible intentions, and roles of the people involved. Your narrative should be brief yet comprehensive, providing clear insight into the nature of their interaction. The output should follow the JSON format below.
        ---

        [FORMAT]
        Follow the following format:

        [INPUT]
        whole_image: the entire image showing both people
        image_of_first_person: image focusing on the first person
        image_of_second_person: image focusing on the second person
        conversation: recorded or text-based conversation between the two people
        [OUTPUT]
        {
            "my_reasoning": "Your careful and step-by-step reasoning before you return the desired outputs for the given inputs",
            "interaction_summary": "summary of the interaction between the two people based on the given images and conversation"
        }



        ---

        [EXAMPLES]

        [Example 1]
        [INPUT]
        whole_image: Two people at a coffee shop, both working on laptops.
        image_of_first_person: A woman with glasses, typing on her laptop.
        image_of_second_person: A man with a beard, looking at his screen.
        conversation: Woman: 'How's your project going?' Man: 'It's going well, thanks.'
        [OUTPUT]
        {
            "my_reasoning": "The two individuals are working togather and talking about each other's project.",
            "interaction_summary": "The woman and man are likely friends or colleagues working together on their respective projects."
        }
        ---
        [Example 2]
        [INPUT]
        whole_image: Two people standing at a store counter, both smiling.
        image_of_first_person: A young woman with brown hair and blue eyes.
        image_of_second_person: An older man with gray hair and a mustache.
        conversation: Woman: 'Hello, how can I help you today?' Man: 'Hi, I'm looking for a new phone.'
        [OUTPUT]
        {
            "my_reasoning": "The woman is asking to help the man ina store, possibly selling phones.",
            "interaction_summary": "The woman, likely a store employee, is assisting the man in purchasing a new phone."
        }
        ---
        [Example 3]
        [INPUT]
        whole_image: Two people standing near a whiteboard, both looking at it.
        image_of_first_person: A woman with a marker, explaining something.
        image_of_second_person: A man with a notebook, taking notes.
        conversation: Woman: 'So, this is the main idea...' Man: 'Can you repeat that?'
        {
            "my_reasoning": "The two individuals are talking and writing something. They seem to be discussing or explaining something.",
            "interaction_summary": "The woman is explaining a concept to the man, likely in a teaching or mentoring capacity."
        }

        ---
        [Example 4]
        [INPUT]
        whole_image: Two people at a restaurant, looking at menus.
        image_of_first_person: A woman with a red scarf, looking at her menu.
        image_of_second_person: A man with a mustache, asking the waiter a question.
        conversation: Woman: 'What's good here?' Man: 'The seafood is great.'
        {
            "my_reasoning": "In a restaurant environment, the two individuals talk about the meals. So that, thet should be deciding what to have.",
            "interaction_summary": "The woman and man are deciding what to order at a restaurant, with the man making a recommendation."
        }
        ---



        For the given inputs, first generate your reasoning and then generate the outputs.
    ''']
    user_prompts = list()
    user_prompts.append('''
        [INPUT]
    ''')
    user_prompts.append('whole_image:')
    user_prompts.append({'type':'image_url', 'image_url': {'url':f'data:image/jpeg;base64,{whole_image}'}})
    if person1_image == '':
        user_prompts.append('image_of_the_first_person: No image for the first person.')
    else:
        user_prompts.append('image_of_the_first_person:')
        user_prompts.append({'type':'image_url', 'image_url': {'url':f'data:image/jpeg;base64,{person1_image}'}})

    if person2_image == '':
        user_prompts.append('image_of_the_second_person: No image for the second person.')
    else:
        user_prompts.append('image_of_the_second_person:')
        user_prompts.append({'type':'image_url', 'image_url': {'url':f'data:image/jpeg;base64,{person2_image}'}})
    if conversation == '':
        user_prompts.append('conversation: No conversation between the two individuals.')
    else:
        user_prompts.append(f'conversation: {conversation}')
    response = llm.execute(system_prompts, user_prompts).replace('```json', '').replace('`', '')
    response = json.loads(response)['interaction_summary']    
    return response

def individual_summarize(whole_image, person_image, spoken, llm):
    # return ''
    system_prompts = ['''
        [TASK]
        As an insightful assistant, craft a concise summary about a person depicted in a given multimedia context. Use the visual and spoken cues provided to determine the individual’s likely role, actions, and interaction within the scene. In your summary, explicitly mention discernible demographics like approximate age and attire, and interpret the relationship or activity the individual is involved in based on their apparel, age, and spoken words. Ensure your descriptions remain unbiased, incorporating only observable or inferrable information to enhance contextual awareness particularly for purposes such as aiding visually impaired users or security monitoring. The output should follow the JSON format shown below.
        ---

        [FORMAT]
        Follow the following format:

        [INPUT]
        whole_image: The complete image containing the environment and multiple individuals possibly
        image_of_person: A closer or isolated image of the specific individual to be summarized
        words_spoken_by_the_person: Transcript or list of words that the individual has spoken
        [OUTPUT]
        {
            "my_reasoning": "Your careful and step-by-step reasoning before you return the desired outputs for the given inputs",
            "summary": "A concise description of the individual, including possible relations to others, based on visual cues and spoken words. Details such as sex, age, and outfit are included."
        }

        ---

        [EXAMPLES]

        [Example 1]
        [INPUT]
        whole_image: Classroom with students and a teacher
        image_of_person: The teacher, a woman in her 50s with glasses
        words_spoken_by_the_person: This mathematical concept is very important in Calculus
        [OUTPUT]
        {
            "my_reasoning": "The background and her age indicates that she is a teacher. Her words indicates that she is have a math class.",
            "summary": "The woman, likely the teacher, is in her 50s, wearing glasses, and is instructing a class of students."
        }
        ---
        [Example 2]
        [INPUT]
        whole_image: Gym with people exercising
        image_of_person: A man in his 20s with a towel
        words_spoken_by_the_person: Hey buddy! You should lower your hips to avoid injury
        [OUTPUT]
        {
            "my_reasoning": "The background and appearance shows that the person is having an exercise. The spoken words shows that he is training with a friend.",
            "summary": "The man, likely a fitness enthusiast, is in his 20s, wearing a towel, and is working out with his buddy at the gym."
        }
        ---
        [Example 3]
        [INPUT]
        whole_image: Outdoor hiking trail with a person and a dog
        image_of_person: A man in his 40s with a backpack
        words_spoken_by_the_person: Wow! What a great view here
        [OUTPUT]
        {
            "my_reasoning": "The theme of the image is a person hiking with a dog.",
            "summary": "The man, likely an outdoorsy individual, is in his 40s, wearing a backpack, and is hiking with his dog in a natural setting."
        }
        =---
        [Example 4]
        [INPUT]
        whole_image: Wedding ceremony with a bride and groom
        image_of_person: The groom, a man in his 30s with a black tuxedo
        words_spoken_by_the_person: Yes, I do!
        [OUTPUT]
        {
            "my_reasoning": "The background indicates a wedding. The word of the man suggests that he is the bridegroom.",
            "summary": "The man, the groom, is in his 30s, wearing a black tuxedo, and is getting married to his partner in a wedding ceremony."
        }
        ---



        For the given inputs, first generate your reasoning and then generate the outputs.
    ''']
    user_prompts = list()
    user_prompts.append('Can you give me a summary of the person in the image?')
    user_prompts.append('whole_image:')
    user_prompts.append({'type':'image_url', 'image_url': {'url':f'data:image/jpeg;base64,{whole_image}'}})
    if person_image == '':
        user_prompts.append('No image for the person.')
    else:
        user_prompts.append('image_of_person:')
        user_prompts.append({'type':'image_url', 'image_url': {'url':f'data:image/jpeg;base64,{person_image}'}})
    if spoken == '':
        user_prompts.append('words_spoken_by_the_person: The person spoke nothing.')
    else:
        user_prompts.append(f'words_spoken_by_the_person: {spoken}')
    response = llm.execute(system_prompts, user_prompts).replace('```json', '').replace('`', '')
    response = json.loads(response)['summary']
    return response