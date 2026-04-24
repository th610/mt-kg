import pickle
from data.moviegraphs_data import MovieGraphsDataset
from data.hlvu_data import HLVUDataset
from models.hai_jing_llm import Hai_Jing_LLM
from models.gpt4o import GPT4
from models.claude import Claude
from models.doubao import Doubao
from models.gemini import Gemini
from models.llama import LLaMA
from models.qwen import Qwen
from models.deepseek import Deepseek
import random
from models.graph import Graph
from algorithms import graph_temporal_update, social_relationship_recognition, community_summary_generation
from evaluation.close_set_evaluator import CloseSetEvaluator
from configs.moviegraphs import data_cfg
import time

random.seed()

llm = Deepseek()
graph = Graph()
dataset = MovieGraphsDataset(train=True, llm=llm)
# dataset = HLVUDataset(llm=llm)
dataset_len = len(dataset)
evaluator = CloseSetEvaluator()

predicted_file = open(data_cfg.DATA.PREDICTED_PATH, 'rb')
predicted = pickle.load(predicted_file)
predicted_file.close()
for scene in predicted:
	for prediction in scene:
		evaluator.add_data(prediction[0], prediction[1])

# predicted = list()

start = time.time()
for i in range(len(predicted), dataset_len):
	background_summaries, interaction_summaries, individual_summaries, queries, labels, end, video_name, background_images, face_images = dataset.__getitem__(i)
	graph_temporal_update.temporal_kg_update(graph, background_summaries, interaction_summaries, individual_summaries, queries, llm)
	predicted.append(list())
	for query_index, query in enumerate(queries):
		print(f'data {video_name}-{i}-{query} *****************************************')
		community_to_nodes, community_to_edges = graph.get_graph_communities()
		for community_id, node_set in community_to_nodes.items():
			community_infos += community_summary_generation.community_summarise(graph, node_set, community_to_edges[community_id], llm)

		# social_relationship = social_relationship_recognition.open_set_recognition(graph, query, community_infos, llm)
		social_relationship = social_relationship_recognition.close_set_recognition(graph, query, community_infos, dataset.get_relation_dict(),llm)
		print(social_relationship)
		print(labels[query_index])
		evaluator.add_data(social_relationship, labels[query_index])
		predicted[-1].append((social_relationship, labels[query_index]))
	
	predicted_file = open(data_cfg.DATA.PREDICTED_PATH, 'wb')
	pickle.dump(predicted, predicted_file, protocol=pickle.HIGHEST_PROTOCOL)
	predicted_file.close()

	if i % 10 == 0:
		predicted_file = open(data_cfg.DATA.PREDICTED_BACKUP_PATH, 'wb')
		pickle.dump(predicted, predicted_file, protocol=pickle.HIGHEST_PROTOCOL)
		predicted_file.close()

	precision = evaluator.get_accuracy()
	if end:
		print(f'finished {video_name}+++++++++++++++++++++++++++++++++++++++++++++++++')
	print(f'***************** Precision {precision}, using {time.time() - start} seconds.')
