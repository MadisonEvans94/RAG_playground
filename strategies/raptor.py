from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage

from .base import BaseRAGStrategy, RAGResponse


class Node:
    def __init__(self, text: str, embedding: np.ndarray, depth: int):
        self.text: str = text
        self.embedding: np.ndarray = embedding
        self.depth: int = depth
        self.children: List[Node] = []


class RAPTORStrategy(BaseRAGStrategy):
    """RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) implementation"""
    
    def _setup(self):
        """Initialize RAPTOR-specific components"""
        # Extract config values with defaults
        self.chunk_size = self.config.get('chunk_size', 500)
        self.chunk_overlap = self.config.get('chunk_overlap', 75)
        self.model_name = self.config.get('model_name', 'gpt-3.5-turbo')
        self.model_temperature = self.config.get('model_temperature', 0)
        self.max_clusters = self.config.get('max_clusters', 50)
        self.recursive_depth = self.config.get('recursive_depth', 3)
        self.cluster_threshold = self.config.get('cluster_threshold', 0.5)
        self.random_state = self.config.get('random_state', 1234)
        
        # Initialize models
        self.embedding_model = OpenAIEmbeddings()
        self.llm = ChatOpenAI(
            temperature=self.model_temperature,
            model=self.model_name
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        
        # Tree will be built during ingestion
        self.tree: Optional[List[Node]] = None
        
        # Summary template
        self.summary_template = """You are an assistant to create a summary of the text input provided. 
        It should be human-readable. It should contain a minimum of 1 word and a maximum of 4 words
        Text:
        {text}
        """
    
    def ingest_documents(self, source_path: str) -> None:
        """Ingest documents and build RAPTOR tree"""
        # Import your existing utility functions here
        from utils import create_nodes_from_documents, markdown_to_text
        
        # Convert markdown to text if needed
        destination_folder = self.config.get('destination_folder', 'text-files')
        phrases_to_remove = self.config.get('phrases_to_remove', [])
        
        markdown_to_text(source_path, destination_folder, phrases_to_remove)
        
        # Create initial nodes
        nodes = create_nodes_from_documents(
            destination_folder, 
            self.embedding_model, 
            self.config
        )
        
        # Build recursive tree
        self.tree = self._create_recursive_tree(nodes, depth=0)
    
    def _create_recursive_tree(self, nodes: List[Node], depth: int = 0) -> List[Node]:
        """Build the recursive RAPTOR tree"""
        if depth >= self.recursive_depth:
            return nodes
        
        embeddings = [node.embedding for node in nodes]
        gmm = GaussianMixture(
            n_components=min(self.max_clusters, len(nodes)),
            random_state=self.random_state
        )
        clusters = gmm.fit_predict(embeddings)
        
        clustered_nodes: Dict[int, List[Node]] = {}
        for cluster_id in set(clusters):
            clustered_nodes[cluster_id] = []
        
        for i, cluster_id in enumerate(clusters):
            clustered_nodes[cluster_id].append(nodes[i])
        
        for cluster_id, cluster_nodes in clustered_nodes.items():
            if len(cluster_nodes) > 1:
                # Create summary for cluster
                summary = self._summarize_cluster(cluster_nodes)
                parent_node = self._create_node(summary, depth + 1)
                parent_node.children = self._create_recursive_tree(
                    cluster_nodes, depth + 1
                )
                clustered_nodes[cluster_id] = [parent_node]
            else:
                clustered_nodes[cluster_id] = cluster_nodes
        
        return [node for nodes in clustered_nodes.values() for node in nodes]
    
    def _create_node(self, text: str, depth: int) -> Node:
        """Create a new node with embedding"""
        embedding = self.embedding_model.embed_query(text)
        return Node(text=text, embedding=embedding, depth=depth)
    
    def _summarize_cluster(self, nodes: List[Node]) -> str:
        """Summarize a cluster of nodes"""
        combined_text = " ".join([node.text for node in nodes])
        prompt = self.summary_template.format(text=combined_text)
        human_message = HumanMessage(content=prompt)
        response = self.llm.invoke([human_message])
        return response.content.strip()
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Node]:
        """Retrieve relevant nodes from the tree"""
        if not self.tree:
            raise ValueError("No documents have been ingested yet")
        
        query_embedding = self.embedding_model.embed_query(query)
        relevant_nodes = []
        
        def traverse_and_collect(node: Node):
            similarity = cosine_similarity(
                [query_embedding], [node.embedding]
            )[0][0]
            if similarity > self.cluster_threshold:
                relevant_nodes.append((node, similarity))
            for child in node.children:
                traverse_and_collect(child)
        
        for node in self.tree:
            traverse_and_collect(node)
        
        relevant_nodes.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in relevant_nodes[:top_k]]
    
    def generate_response(self, query: str, context: List[Node]) -> RAGResponse:
        """Generate response using retrieved nodes"""
        context_texts = [node.text for node in context]
        combined_context = "\n\n".join(context_texts)
        
        prompt_template = """Answer the question, and utilize the context to help guide your answer:
        Context:
        {context}

        Question: {question}
        """
        
        prompt = prompt_template.format(context=combined_context, question=query)
        human_message = HumanMessage(content=prompt)
        response_message = self.llm.invoke([human_message])
        
        return RAGResponse(
            answer=response_message.content.strip(),
            sources=context_texts[:3],  # Return top 3 sources
            metadata={
                'num_nodes_retrieved': len(context),
                'strategy': 'RAPTOR'
            }
        )