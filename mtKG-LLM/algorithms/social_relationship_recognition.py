import json
from utils import image_utils

def open_set_recognition(graph, query, context_summary, llm):
    # return ''
    system_prompts = ['''
        [TASK]
        Analyze the relationship information provided, which includes details about how the individuals are connected ('edge_info'), personal data of both individuals ('individual_info_1', 'individual_info_2'), and a brief context of their interaction ('context_summary'). From this, determine the social relationship between the individuals. Your response should clearly state the social relationship as it directly correlates to the context and details provided. Skip any personal comments, focusing only on identifying and delivering the exact social relationship type. You should try your best to recognize the social relationship. If there is little available information for you to make an inference, you can response the relationship with unknown. The output should be in the JSON format shown below.
        ---

        [FORMAT]
        Follow the following format:

        [INPUT]
        edge_info: The information regarding the connection or interaction between the two individuals
        individual_info_1: The information of the first individual
        individual_info_2: The information of the second individual
        context_summary: A brief summary providing the context of the relationship between the two individuals
        [OUTPUT]
        {
        "reasoning": "Your careful and step-by-step reasoning before you return the desired outputs for the given inputs. "
        "relation": "The social relationship infered from the information."
        }



        ---

        [EXAMPLES]

        [Example 1]
        [INPUT]
        edge_info: blood relation
        individual_info_1: {"name":"Jane","age":40,"occupation":"doctor"}
        individual_info_2: {"name":"Tom","age":35,"occupation":"engineer"}
        context_summary: same parents
        [OUTPUT]
        {
        "reasoning": "There is blood relation between the individuals and they have the same parents. "
        "relation": "sibling"
        }
        ---
        [Example 2]
        [INPUT]
        edge_info: marriage
        individual_info_1: {"name":"Olivia","age":30,"occupation":"lawyer"}
        individual_info_2: {"name":"Michael","age":35,"occupation":"doctor"}
        context_summary: going to a store for groceries
        [OUTPUT]
        {
        "reasoning": "The two individuals have married with each other and is possibly buying groceries for the family."
        "relation": "spouse"
        }
        ---
        [Example 3]
        [INPUT]
        edge_info: attended same university
        individual_info_1: {"name":"John","age":25,"occupation":"software engineer"}
        individual_info_2: {"name":"Alice","age":25,"occupation":"data scientist"}
        context_summary: College friends
        [OUTPUT]
        {
        "reasoning": "The two individuals study in the same university and they are friends. They are possibly classmates."
        "relation": "classmate"
        }
        ---
        [Example 4]
        [INPUT]
        edge_info:  providing a guidance about the first day in the company
        individual_info_1: {"name":"David","age":50,"occupation":"Manager"}
        individual_info_2: {"name":"Sophia","age":25,"occupation":"graduate student"}
        context_summary: office background with employees workings
        [OUTPUT]
        {
        "reasoning": "The two individuals study in the same university and they are friends. They are possibly classmates."
        "relation": "classmate"
        }
        ---


        For the given inputs, first generate your reasoning and then generate the outputs.
    ''']
    user_prompts = list()
    user_prompts.append('''        
        [INPUT]
    ''')
    source_id = query[0]
    target_id = query[1]

    edge_info = graph.get_edge(source_id, target_id)['description']
    edge_info = f"History info: {edge_info['history']}. Current info: {edge_info['current']}"
    individual_info_1 = graph.get_node(source_id)['description']
    individual_info_1 = f"History info: {individual_info_1['history']}. Current info: {individual_info_1['current']}"
    individual_info_2 = graph.get_node(source_id)['description']
    individual_info_2 = f"History info: {individual_info_2['history']}. Current info: {individual_info_2['current']}"

    user_prompts.append(f'edge_info: {edge_info}')
    user_prompts.append(f'individual_info_1: {individual_info_1}')
    user_prompts.append(f'individual_info_2: {individual_info_2}')
    user_prompts.append(f'context_summary: {context_summary}')
    response = llm.execute(system_prompts, user_prompts).replace('```json', '').replace('`', '')
    response = json.loads(response)['relation']
    return response

def close_set_recognition(graph, query, context_summary, relation_dict, llm):
    # return ''
    system_prompts = ['''
        [TASK]
        Analyze the relationship information provided, which includes details about how the individuals are connected ('edge_info'), personal data of both individuals ('individual_info_1', 'individual_info_2'), and a brief context of their interaction ('context_summary'). From this, determine and select the most accurate social relationship category, provided in the list under ('relation_list'). Your response should clearly state the chosen category as it directly correlates to the context and details provided. Skip any personal comments, focusing only on identifying and delivering the exact social relationship type. The output should be in the JSON format shown below.
        ---

        [FORMAT]
        Follow the following format:

        [INPUT]
        edge_info: The information regarding the connection or interaction between the two individuals
        individual_info_1: The information of the first individual
        individual_info_2: The information of the second individual
        context_summary: A brief summary providing the context of the relationship between the two individuals
        relation_list: List of potential social relationships that can be used to describe the relationship between the individuals
        [OUTPUT]
        {
            "reasoning": "Your careful and step-by-step reasoning before you return the desired outputs for the given inputs. "
            "relation": "The social relationship chosen from the relation_list."
        }



        ---

        [EXAMPLES]

        [Example 1]
        [INPUT]
        edge_info: blood relation
        individual_info_1: {"name":"Jane","age":40,"occupation":"doctor"}
        individual_info_2: {"name":"Tom","age":35,"occupation":"engineer"}
        context_summary: same parents
        relation_list: ["friend","cousins member","partner","opponent","sibling"]
        [OUTPUT]
        {
            "reasoning": "There is blood relation between the individuals and they have the same parents. "
            "relation": "sibling"
        }
        ---
        [Example 2]
        [INPUT]
        edge_info: marriage
        individual_info_1: {"name":"Olivia","age":30,"occupation":"lawyer"}
        individual_info_2: {"name":"Michael","age":35,"occupation":"doctor"}
        context_summary: going to a store for groceries
        relation_list: ["friend","family member","partner","spouse","in-law"]
        [OUTPUT]
        {
            "reasoning": "The two individuals have married with each other and is possibly buying groceries for the family."
            "relation": "spouse"
        }
        ---
        [Example 3]
        [INPUT]
        edge_info: attended same university
        individual_info_1: {"name":"John","age":25,"occupation":"software engineer"}
        individual_info_2: {"name":"Alice","age":25,"occupation":"data scientist"}
        context_summary: College friends
        relation_list: ["family member","classmate","colleague","partner"]
        [OUTPUT]
        {
            "reasoning": "The two individuals study in the same university and they are friends. They are possibly classmates."
            "relation": "classmate"
        }
        ---
        [Example 4]
        [INPUT]
        edge_info:  providing a guidance about the first day in the company
        individual_info_1: {"name":"David","age":50,"occupation":"Manager"}
        individual_info_2: {"name":"Sophia","age":25,"occupation":"graduate student"}
        context_summary: office background with employees workings
        relation_list: ["friend","family member","colleague","partner","mentor"]
        [OUTPUT]
        {
            "reasoning": "The two individuals study in the same university and they are friends. They are possibly classmates."
            "relation": "classmate"
        }
        ---


        For the given inputs, first generate your reasoning and then generate the outputs.
    ''']
    # system_prompts = ['You are a helpful assistant designed for recognizing the social relationship for the two individuals.', 'You should only output the social relationship chosen from the relation_dict and must not add other content such as comments or thoughts.']

    user_prompts = list()
    user_prompts.append('''        
        [INPUT]
    ''')
    source_id = query[0]
    target_id = query[1]

    edge_info = graph.get_edge(source_id, target_id)['description']
    edge_info = f"History info: {edge_info['history']}. Current info: {edge_info['current']}"
    individual_info_1 = graph.get_node(source_id)['description']
    individual_info_1 = f"History info: {individual_info_1['history']}. Current info: {individual_info_1['current']}"
    individual_info_2 = graph.get_node(source_id)['description']
    individual_info_2 = f"History info: {individual_info_2['history']}. Current info: {individual_info_2['current']}"

    user_prompts.append(f'edge_info: {edge_info}')
    user_prompts.append(f'individual_info_1: {individual_info_1}')
    user_prompts.append(f'individual_info_2: {individual_info_2}')
    user_prompts.append(f'context_summary: {context_summary}')
    user_prompts.append(f'relation_dict {relation_dict.keys()}')
    response = llm.execute(system_prompts, user_prompts).replace('```json', '').replace('`', '')
    response = json.loads(response)['relation']
    
    return response

def test(background, face, query, context_summary, relation_dict, llm):
    # return ''
    system_prompts = ['''
        [TASK]
        Analyze the relationship information provided, which includes details about how the individuals are connected ('edge_info'), personal data of both individuals ('individual_info_1', 'individual_info_2'), and a brief context of their interaction ('context_summary'). From this, determine and select the most accurate social relationship category, provided in the list under ('relation_list'). Your response should clearly state the chosen category as it directly correlates to the context and details provided. Skip any personal comments, focusing only on identifying and delivering the exact social relationship type. The output should be in the JSON format shown below.
        ---

        [FORMAT]
        Follow the following format:

        [INPUT]
        edge_info: The whole picture
        individual_info_1: The information of the first individual
        individual_info_2: The information of the second individual
        relation_list: List of potential social relationships that can be used to describe the relationship between the individuals
        [OUTPUT]
        {
            "reasoning": "Your careful and step-by-step reasoning before you return the desired outputs for the given inputs. "
            "relation": "The social relationship chosen from the relation_list."
        }

        For the given inputs, first generate your reasoning and then generate the outputs.
    ''']
    # system_prompts = ['You are a helpful assistant designed for recognizing the social relationship for the two individuals.', 'You should only output the social relationship chosen from the relation_dict and must not add other content such as comments or thoughts.']

    user_prompts = list()
    user_prompts.append('''        
        [INPUT]
    ''')
    source_id = query[0]
    target_id = query[1]


    background = image_utils.encode_numpy_image(background)
    user_prompts.append('edge info:')
    user_prompts.append({'type':'image_url', 'image_url': {'url':f'data:image/jpeg;base64,{background}'}})
    user_prompts.append(f'individual_info_1:')
    if face[0] == '':
        user_prompts.append(f'No face image for individual 1, please summarize the relationfor the whole image.')
    else:
        face[0] = image_utils.encode_numpy_image(face[0])
        user_prompts.append({'type':'image_url', 'image_url': {'url':f'data:image/jpeg;base64,{face[0]}'}})
    user_prompts.append(f'individual_info_2:')
    if face[1] == '':
        user_prompts.append(f'No face image for individual 2, please summarize the relationfor the whole image.')
    else:
        face[1] = image_utils.encode_numpy_image(face[1])
        user_prompts.append({'type':'image_url', 'image_url': {'url':f'data:image/jpeg;base64,{face[1]}'}})
    user_prompts.append(f'relation_dict {relation_dict.keys()}')
    response = llm.execute(system_prompts, user_prompts).replace('```json', '').replace('`', '')
    response = json.loads(response)['relation']
    
    return response