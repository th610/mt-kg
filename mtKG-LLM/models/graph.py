import networkx as nx
from graspologic.partition import leiden
import random

class Graph:
	def __init__(self):
		self.graph = nx.Graph()
		self.node_to_community = dict()

	def add_node(self, id, description):
		self.graph.add_node(
			id,
			type="",
			description={
				'history': '',
				'current': description
			}
		)

	def add_edge(self, source_id, target_id, description):
		self.graph.add_edge(
			source_id,
			target_id,
			source = source_id,
			target = target_id,
			description={
				'history': '',
				'current': description
			}		
		)

	def get_node(self, id):
		if id not in self.graph.nodes:
			self.add_node(id, '')
		return self.graph.nodes._nodes[id]
	
	def set_node(self, node_id, data):
		nx.set_node_attributes(self.graph, {
			node_id: data
		})
	
	def get_edge(self, source_id, target_id):
		if (source_id, target_id) not in self.graph.edges:
			self.add_edge(source_id, target_id, '')
		return self.graph.edges._adjdict[source_id][target_id]
	
	def set_edge(self, edge_id, data):
		nx.set_edge_attributes(self.graph, {
			edge_id: data
		})

	def edge_update(self, source_id, target_id, description):
		self.graph.remove_edge(source_id, target_id)
		self.add_edge(source_id, target_id, description)

	def get_graph_communities(self):
		community_mapping = leiden(
        	self.graph, resolution=0.001
		)
		community_to_nodes = dict()
		for node, community_id in community_mapping.items():
			if community_id not in community_to_nodes:
				community_to_nodes[community_id] = set()
			community_to_nodes[community_id].add(node)

		community_to_edges = dict()
		for community_id, node_set in community_to_nodes.items():
			if community_id not in community_to_edges:
				community_to_edges[community_id] = set()
			for node in node_set:
				for edge in self.graph.edges(node):
					if edge not in community_to_edges[community_id]:
						community_to_edges[community_id].add(edge)

		return community_to_nodes, community_to_edges