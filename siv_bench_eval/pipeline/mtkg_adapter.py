"""
pipeline/mtkg_adapter.py

SIV-Bench 비디오를 mtKG 파이프라인으로 처리하는 어댑터.

흐름:
  프레임 16장 → 4 구간으로 분할 → 구간별 배경/상호작용 요약
  → 시간적 그래프 업데이트 → 커뮤니티 요약 → context 문자열 반환

제약:
  - 얼굴 크롭 없음 → person1_image / person2_image = '' (mtKG가 내부에서 처리)
  - 자막 텍스트 없음 → conversation = '' (마찬가지)
  - 등장인물 ID는 generic ("person_A", "person_B") 사용
"""

import logging
import sys
from pathlib import Path
from typing import List

# mtKG-LLM 모듈 경로 추가
_MTKG_ROOT = str(Path(__file__).parent.parent.parent / "mtKG-LLM")
if _MTKG_ROOT not in sys.path:
    sys.path.insert(0, _MTKG_ROOT)

import networkx as nx

from algorithms.multimodel_info_extraction import background_summarize, interaction_summarize
from algorithms.graph_temporal_update import temporal_kg_update
from algorithms.community_summary_generation import community_summarise


class Graph:
    """mtKG Graph 클래스의 경량 재구현.

    원본(models/graph.py)은 graspologic(Leiden)에 의존하는데
    Python 3.13에서 설치 불가이므로, networkx greedy_modularity_communities로 대체.
    인터페이스는 original과 동일하게 유지.
    """

    def __init__(self):
        self.graph = nx.Graph()

    def add_node(self, id, description):
        self.graph.add_node(id, type="", description={"history": "", "current": description})

    def add_edge(self, source_id, target_id, description):
        self.graph.add_edge(
            source_id, target_id,
            source=source_id, target=target_id,
            description={"history": "", "current": description},
        )

    def get_node(self, id):
        if id not in self.graph.nodes:
            self.add_node(id, "")
        return self.graph.nodes[id]

    def set_node(self, node_id, data):
        nx.set_node_attributes(self.graph, {node_id: data})

    def get_edge(self, source_id, target_id):
        if not self.graph.has_edge(source_id, target_id):
            self.add_edge(source_id, target_id, "")
        return self.graph.edges[source_id, target_id]

    def set_edge(self, edge_id, data):
        nx.set_edge_attributes(self.graph, {edge_id: data})

    def edge_update(self, source_id, target_id, description):
        self.graph.remove_edge(source_id, target_id)
        self.add_edge(source_id, target_id, description)

    def get_graph_communities(self):
        # graspologic Leiden 대신 networkx greedy modularity 사용
        if len(self.graph.nodes) == 0:
            return {}, {}
        communities = nx.algorithms.community.greedy_modularity_communities(self.graph)
        community_to_nodes = {i: set(c) for i, c in enumerate(communities)}
        community_to_edges = {}
        for community_id, node_set in community_to_nodes.items():
            edges = set()
            for node in node_set:
                for edge in self.graph.edges(node):
                    edges.add(edge)
            community_to_edges[community_id] = edges
        return community_to_nodes, community_to_edges

logger = logging.getLogger(__name__)

DEFAULT_NUM_SEGMENTS = 4

_SOURCE_ID = "person_A"
_TARGET_ID = "person_B"


class O4MiniMtKGBridge:
    """O4Mini 래퍼를 mtKG의 llm.execute() 인터페이스에 맞게 연결하는 브리지."""

    def __init__(self, o4mini_model):
        self._model = o4mini_model

    def execute(self, system_prompts: List, user_prompts: List) -> str:
        messages = []

        for prompt in system_prompts:
            messages.append({"role": "system", "content": str(prompt)})

        for prompt in user_prompts:
            if isinstance(prompt, str):
                messages.append({"role": "user", "content": prompt})
            else:
                messages.append({"role": "user", "content": [prompt]})

        return self._model.chat(messages)


def build_mtkg_context(
    frames_b64: List[str],
    llm_bridge: O4MiniMtKGBridge,
    num_segments: int = DEFAULT_NUM_SEGMENTS,
) -> str:
    """프레임 리스트를 받아 mtKG 파이프라인을 실행하고 context 문자열을 반환한다."""
    total = len(frames_b64)

    if total >= num_segments:
        step = total // num_segments
        segment_frames = [frames_b64[i * step] for i in range(num_segments)]
    else:
        segment_frames = [frames_b64[i % total] for i in range(num_segments)]

    background_summaries = []
    interaction_summaries = []
    individual_summaries = []
    queries = []

    for idx, frame_b64 in enumerate(segment_frames):
        try:
            bg = background_summarize(frame_b64, llm_bridge)
        except Exception as e:
            logger.warning(f"[구간 {idx}] background_summarize 실패: {e}")
            bg = ""

        try:
            ia = interaction_summarize(
                whole_image=frame_b64,
                person1_image="",
                person2_image="",
                conversation="",
                llm=llm_bridge,
            )
        except Exception as e:
            logger.warning(f"[구간 {idx}] interaction_summarize 실패: {e}")
            ia = ""

        background_summaries.append(bg)
        interaction_summaries.append(ia)
        individual_summaries.append(["", ""])
        queries.append((_SOURCE_ID, _TARGET_ID))

    graph = Graph()
    try:
        temporal_kg_update(
            graph,
            background_summaries,
            interaction_summaries,
            individual_summaries,
            queries,
            llm_bridge,
        )
    except Exception as e:
        logger.warning(f"temporal_kg_update 실패: {e}")
        return ""

    try:
        community_to_nodes, community_to_edges = graph.get_graph_communities()
        context = ""
        for community_id, node_set in community_to_nodes.items():
            summary = community_summarise(
                graph, node_set, community_to_edges[community_id], llm_bridge
            )
            context += summary + " "
        return context.strip()
    except Exception as e:
        logger.warning(f"community_summarise 실패: {e}")
        return ""


def build_mtkg_qa_prompt(
    context: str,
    question: str,
    options: List[str],
) -> List[dict]:
    """그래프 context + 질문 + 선택지로 OpenAI messages를 구성한다."""
    system = (
        "You are given a structured summary of social interactions between people in a video. "
        "Use this context to answer the multiple choice question. "
        "Respond with ONLY the letter of the correct answer (e.g. A, B, C, ...)."
    )

    options_text = "\n".join(
        f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
    )

    if context:
        user_text = (
            f"[Interaction Context]\n{context}\n\n"
            f"[Question]\n{question}\n\n"
            f"[Options]\n{options_text}"
        )
    else:
        user_text = (
            f"[Question]\n{question}\n\n"
            f"[Options]\n{options_text}"
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]
