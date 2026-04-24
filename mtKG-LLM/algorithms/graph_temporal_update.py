import json

def kg_history_summarize(history, current, llm):
    if current == '':
        return ''
    system_prompts = ['''
        [TASK]
        As a narrative synthesis specialist, your task is to create cohesive summaries that seamlessly bridge past conditions with current updates, focusing on progression and resolutions. For each input:

        1. Analyze the historical and current information provided.
        2. Combine these details to illustrate a clear transition from the initial problem to the subsequent improvements or solutions implemented.
        3. Write a summary that integrates both the history and current updates, emphasizing the effectiveness and results of the actions taken.
        4. Provide your reasoning that led to this summary. Your narrative should serve as a benchmark for clear communication in fields requiring regular updates, such as customer relationships, academic research, and healthcare.

        Each response should aim to engage and inform stakeholders by clearly demonstrating how past challenges have been addressed, ensuring the narrative is valuable for real-world applications and decision-making. Ensure your outputs adhere to the JSON structure provided, featuring both your reasoning and the synthesized history. Your output should follow the JSON format shown below.
        ---

        [FORMAT]

        [INPUT]
        History_information: Previous contextual data provided by an expert
        Current_information: Latest data or updates provided by an expert
        [OUTPUT]
        {
            "my_reasoning": "Careful and step-by-step reasoning before returning the outputs for the given inputs"
            "New_history_information": "A summary that combines both the historical and current information while emphasizing progress, maintaining chronological integrity and relevance for decision-making"
        }



        ---

        [EXAMPLES]

        [Example 1]
        [INPUT]
        History_information: Peter bought a bag of chips from a local store.
        Current_information: Peter find the chips are wet so he goes back to the store and returns it.
        [OUTPUT]
        {
            "my_reasoning": "Peter first bought the chips and found it to be bad later. Peter returned the chips eventually."
            "New_history_information": "Peter bought a bag of stale chips and returns it to the store."
        } 
        ---
        [Example 2]
        [INPUT]
        History_information: Tompson was doing badly at school.
        Current_information: Mr.Johnson teaches him how to learn from mistakes and Tompson finds it easier to make progress.
        [OUTPUT]
        {
            "my_reasoning": "Mr.Johnson helps Tompson to imporve Tompson's learning skills."
            "New_history_information": "Tompson makes pregress with the help of Mr.Johnson."
        } 
        my_reasoning: 
        New_history_information: 
        ---
        [Example 3]
        [INPUT]
        History_information: Harry and Charles had a fight because of the teamwork.
        Current_information: Harry and Charles apologizes to each other for the fight yesterday.
        [OUTPUT]
        {
            "my_reasoning": "The two individuals was angry with each other yesterday. They apologizes a day after., indicating a reverse of relation."
            "New_history_information": "Harry and Charles reconciled after the fight that took place yesterday."
        } 
        ---
        [Example 4]
        [INPUT]
        History_information: The young man and the young girl decide to marry each other.
        Current_information: They held a wedding and married each other.
        [OUTPUT]
        {
            "my_reasoning": "The relation between the two individuals progressed from lovers to couple with the wedding being held."
            "New_history_information": "The yound lovers married to each other, becoming a couple."
        } 
        ---



        For the given inputs, first generate your reasoning and then generate the outputs.
    ''']
    user_prompts = ['''
        [INPUT]
    ''']
    user_prompts.append(f'History information: {history}')
    user_prompts.append(f'Current information: {current}')
    response = llm.execute(system_prompts, user_prompts).replace('```json', '').replace('`', '')
    response = json.loads(response)['New_history_information']
    return response

def temporal_kg_update(graph, background_summaries, interaction_summaries, individual_summaries, queries, llm):
    for i, query in enumerate(queries):
        source_id = query[0]
        target_id = query[1]

        for individual in range(2):
            id = query[individual]
            node_summary = individual_summaries[i][individual]
            history_node = graph.get_node(id)['description']['history']
            current_node = graph.get_node(id)['description']['current']
            graph.set_node(id, {
                'description': {
                    "history": kg_history_summarize(history_node, current_node, llm),
                    'current': node_summary
                }
            })

        edge_summary = background_summaries[i] + interaction_summaries[i]
        history_edge = graph.get_edge(source_id, target_id)['description']['history']
        current_edge = graph.get_edge(source_id, target_id)['description']['current']
        graph.set_edge((source_id, target_id), {
            'description': {
                "history": kg_history_summarize(history_edge, current_edge, llm),
                'current': edge_summary
            }
        })