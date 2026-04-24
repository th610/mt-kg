import json

def community_summarise(graph, node_set, edge_set, llm):
    system_prompts = ['''
        [TASK]
        Your task is to synthesize and summarize the provided character profiles and their interpersonal relationships into a concise, coherent key information summary. Focus on integrating both the professional roles and the relationship dynamics in your summary. Ensure the clarity and breadth of contextual details present, highlighting how each individual is related to others and any professional roles or attributions that provide additional understanding of the group dynamics. The output should follow the JSON format shown below.

        For each input:
        1. Begin by identifying each character's role or profession from the character information.
        2. Extract and simplify the relationship data from the relations description.
        3. Integrate both sets of information into a clear and succinct summary that outlines both the relationships and professional roles, emphasizing interactions that provide insight into the group’s social or professional structure. Ensure the output is well-organized and easy to comprehend.
        ---

        [FORMAT]
        Follow the following format:

        [INPUT]
        character_information: The list of descriptions providing details about individual characters within the social group
        relation_information: The list of descriptions that detail the relationships between the characters in the social group
        [OUTPUT]
        {
            "my_reasoning": "Your careful and step-by-step reasoning before you return the desired outputs for the given inputs",
            "key_information_summary": "A summary that highlights the key information relevant for recognizing relationships within the social group"
        }


        ---

        [EXAMPLES]

        [Example 1]
        [INPUT]
        character_information: ["Frank is a husband.","Mary is a wife.","Tim is a son."]
        relation_information: ["Frank is married to Mary.","Tim is the son of Frank and Mary."]
        [OUTPUT]
        {
            "my_reasoning": "Frank and Marry are spouse and Tim is their son.",
            "key_information_summary": "Frank and Mary are married, and they have a son named Tim."
        }
        ---
        [Example 2]
        [INPUT]
        character_information: ["David is a manager.","Alex is an intern.","Rachel is a team lead."]
        relation_information: ["David supervises Rachel.","Rachel mentors Alex."]
        [OUTPUT]
        {
            "my_reasoning": "David is the manager of Rachel. Rachel mentors Alex.",
            "key_information_summary": "David is the manager, Rachel is the team lead and mentors Alex, and Alex is an intern."
        }
        ---
        [Example 3]
        [INPUT]
        character_information: ["Lily is a sister.","Oliver is a brother.","Kate is a cousin."]
        relation_information: ["Lily and Oliver are siblings.","Kate is the cousin of Lily and Oliver."]
        [OUTPUT]
        {
            "my_reasoning": "Lily and Oliver are siblings. They have a cousin named Kate.",
            "key_information_summary": "Lily and Oliver are siblings, and Kate is their cousin."
        }
        ---
        [Example 4]
        [INPUT]
        character_information: ["Eva is a colleague.","Patrick is a manager.","Sophia is an intern."]
        relation_information: ["Eva reports to Patrick.","Patrick supervises Sophia."]
        [OUTPUT]
        {
            "my_reasoning": "Patrick is the manager of Eva since Eva reports to Patrick. Patrick mentors Sophia.",
            "key_information_summary": "Eva reports to Patrick, and Patrick supervises Sophia."
        }
        ---



        For the given inputs, first generate your reasoning and then generate the outputs.

    ''']
    user_prompts = ['''
        [INPUT]
    ''']
    
    character_information = list()
    for node_id in node_set:
        node = graph.get_node(node_id)
        character_information.append(f'The information of {node_id}: {node["description"]}')
    
    relation_information = list()
    for edge_id in edge_set:
        edge = graph.get_edge(edge_id[0], edge_id[1])
        relation_information.append(f'The information between {edge["source"]} and {edge["target"]}: {edge["description"]}')
    
    user_prompts.append(f'character_information: {json.dumps(character_information)}')
    user_prompts.append(f'relation_information: {json.dumps(relation_information)}')
    response = llm.execute(system_prompts, user_prompts).replace('```json', '').replace('`', '')
    response = json.loads(response)['key_information_summary']
    return response