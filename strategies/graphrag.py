from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
import networkx as nx
from dataclasses import dataclass
import community as community_louvain
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import json

from .base import BaseRAGStrategy, RAGResponse


@dataclass
class Entity:
    """Represents an entity in the knowledge graph"""
    name: str
    type: str
    description: str
    embedding: Optional[np.ndarray] = None


@dataclass
class Relationship:
    """Represents a relationship between entities"""
    source: str
    target: str
    relationship_type: str
    description: str
    weight: float = 1.0


@dataclass
class Community:
    """Represents a community of entities"""
    id: int
    level: int
    entities: List[str]
    summary: str
    parent_id: Optional[int] = None
    embedding: Optional[np.ndarray] = None


class GraphRAGStrategy(BaseRAGStrategy):
    """GraphRAG implementation using knowledge graphs for enhanced retrieval"""
    
    def _setup(self):
        """Initialize GraphRAG-specific components"""
        # Extract config values
        self.chunk_size = self.config.get('chunk_size', 500)
        self.chunk_overlap = self.config.get('chunk_overlap', 75)
        self.model_name = self.config.get('model_name', 'gpt-3.5-turbo')
        self.model_temperature = self.config.get('model_temperature', 0)
        self.entity_extraction_max_tokens = self.config.get('entity_extraction_max_tokens', 500)
        self.community_summary_max_tokens = self.config.get('community_summary_max_tokens', 300)
        self.min_community_size = self.config.get('min_community_size', 2)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.5)
        
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
        
        # Graph components
        self.graph = nx.DiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.communities: Dict[int, Community] = {}
        self.documents: List[Document] = []
        self.vectorstore: Optional[FAISS] = None
        
        # Prompts
        self.entity_extraction_prompt = """Extract entities and relationships from the following text.
        
Format your response as JSON with the following structure:
{{
    "entities": [
        {{"name": "entity name", "type": "entity type", "description": "brief description"}}
    ],
    "relationships": [
        {{"source": "source entity", "target": "target entity", "type": "relationship type", "description": "relationship description"}}
    ]
}}

Text:
{text}

Extracted JSON:"""

        self.community_summary_prompt = """Summarize the following community of related entities and their relationships.
Focus on the main themes, key entities, and important connections.

Entities in this community:
{entities}

Relationships:
{relationships}

Create a concise summary (2-3 sentences):"""

        self.global_query_prompt = """Answer the following question using the provided community summaries.
Synthesize information across multiple communities to provide a comprehensive answer.

Community Summaries:
{summaries}

Question: {question}

Answer:"""

        self.local_query_prompt = """Answer the following question using the provided entity information and relationships.

Central Entity: {entity}
Entity Description: {entity_description}

Related Entities and Relationships:
{context}

Question: {question}

Answer:"""
    
    def ingest_documents(self, source_path: str) -> None:
        """Ingest documents and build knowledge graph"""
        from utils import markdown_to_text
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        
        # Convert markdown to text
        destination_folder = self.config.get('destination_folder', 'text-files')
        phrases_to_remove = self.config.get('phrases_to_remove', [])
        
        markdown_to_text(source_path, destination_folder, phrases_to_remove)
        
        # Load documents
        loader = DirectoryLoader(
            destination_folder,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        documents = loader.load()
        
        # Split documents into chunks
        self.documents = []
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            self.documents.extend(chunks)
        
        # Extract entities and relationships from each chunk
        print("Extracting entities and relationships...")
        for i, doc in enumerate(self.documents):
            print(f"Processing document {i+1}/{len(self.documents)}")
            self._extract_entities_relationships(doc.page_content)
        
        # Build the graph
        self._build_graph()
        
        # Detect communities
        print("Detecting communities...")
        self._detect_communities()
        
        # Generate community summaries
        print("Generating community summaries...")
        self._generate_community_summaries()
        
        # Create vector store for community summaries
        self._create_vectorstore()
        
        print(f"Graph built with {len(self.entities)} entities and {len(self.relationships)} relationships")
        print(f"Detected {len(self.communities)} communities")
    
    def _extract_entities_relationships(self, text: str) -> None:
        """Extract entities and relationships from text using LLM"""
        prompt = self.entity_extraction_prompt.format(text=text)
        message = HumanMessage(content=prompt)
        
        try:
            response = self.llm.invoke([message])
            
            # Parse JSON response
            result = json.loads(response.content)
            
            # Process entities
            for entity_data in result.get('entities', []):
                entity_name = entity_data['name'].lower().strip()
                if entity_name not in self.entities:
                    entity = Entity(
                        name=entity_name,
                        type=entity_data.get('type', 'unknown'),
                        description=entity_data.get('description', '')
                    )
                    entity.embedding = self.embedding_model.embed_query(
                        f"{entity.name} {entity.type} {entity.description}"
                    )
                    self.entities[entity_name] = entity
            
            # Process relationships
            for rel_data in result.get('relationships', []):
                source = rel_data['source'].lower().strip()
                target = rel_data['target'].lower().strip()
                
                # Only add relationship if both entities exist
                if source in self.entities and target in self.entities:
                    relationship = Relationship(
                        source=source,
                        target=target,
                        relationship_type=rel_data.get('type', 'related_to'),
                        description=rel_data.get('description', '')
                    )
                    self.relationships.append(relationship)
                    
        except Exception as e:
            print(f"Error extracting entities/relationships: {e}")
    
    def _build_graph(self) -> None:
        """Build NetworkX graph from entities and relationships"""
        # Add nodes
        for entity_name, entity in self.entities.items():
            self.graph.add_node(
                entity_name,
                type=entity.type,
                description=entity.description,
                embedding=entity.embedding
            )
        
        # Add edges
        for rel in self.relationships:
            self.graph.add_edge(
                rel.source,
                rel.target,
                type=rel.relationship_type,
                description=rel.description,
                weight=rel.weight
            )
    
    def _detect_communities(self) -> None:
        """Detect communities using Louvain algorithm (simpler than Leiden)"""
        if len(self.graph.nodes()) == 0:
            return
        
        # Convert to undirected graph for community detection
        undirected_graph = self.graph.to_undirected()
        
        # Detect communities
        partition = community_louvain.best_partition(undirected_graph)
        
        # Group entities by community
        communities_dict = defaultdict(list)
        for entity, community_id in partition.items():
            communities_dict[community_id].append(entity)
        
        # Create Community objects
        for comm_id, entities in communities_dict.items():
            if len(entities) >= self.min_community_size:
                self.communities[comm_id] = Community(
                    id=comm_id,
                    level=0,  # Single level for simplicity
                    entities=entities,
                    summary=""  # Will be generated next
                )
    
    def _generate_community_summaries(self) -> None:
        """Generate summaries for each community"""
        for comm_id, community in self.communities.items():
            # Gather entity descriptions
            entity_descriptions = []
            for entity_name in community.entities:
                entity = self.entities[entity_name]
                entity_descriptions.append(f"- {entity.name} ({entity.type}): {entity.description}")
            
            # Gather relationships within community
            community_relationships = []
            for rel in self.relationships:
                if rel.source in community.entities and rel.target in community.entities:
                    community_relationships.append(
                        f"- {rel.source} --[{rel.relationship_type}]--> {rel.target}: {rel.description}"
                    )
            
            # Generate summary
            prompt = self.community_summary_prompt.format(
                entities="\n".join(entity_descriptions[:10]),  # Limit for context
                relationships="\n".join(community_relationships[:10])
            )
            
            message = HumanMessage(content=prompt)
            response = self.llm.invoke([message])
            
            community.summary = response.content.strip()
            community.embedding = self.embedding_model.embed_query(community.summary)
    
    def _create_vectorstore(self) -> None:
        """Create vector store from community summaries"""
        if not self.communities:
            return
        
        # Create documents from community summaries
        community_docs = []
        for comm_id, community in self.communities.items():
            doc = Document(
                page_content=community.summary,
                metadata={
                    'community_id': comm_id,
                    'entities': community.entities,
                    'level': community.level
                }
            )
            community_docs.append(doc)
        
        # Create FAISS vector store
        if community_docs:
            self.vectorstore = FAISS.from_documents(
                documents=community_docs,
                embedding=self.embedding_model
            )
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Any]:
        """Retrieve relevant information based on query type"""
        # Determine if query is entity-specific (local) or broad (global)
        query_type = self._classify_query(query)
        
        if query_type == "local":
            return self._local_retrieve(query, top_k)
        else:
            return self._global_retrieve(query, top_k)
    
    def _classify_query(self, query: str) -> str:
        """Classify query as local (entity-specific) or global (broad)"""
        # Simple heuristic: check if query mentions specific entities
        query_lower = query.lower()
        
        # Check if any known entities are mentioned
        for entity_name in self.entities.keys():
            if entity_name in query_lower:
                return "local"
        
        # Keywords that suggest global queries
        global_keywords = ['overall', 'main themes', 'summarize', 'overview', 'general', 'all', 'entire']
        if any(keyword in query_lower for keyword in global_keywords):
            return "global"
        
        # Default to global for broad questions
        return "global"
    
    def _local_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Local retrieval focusing on specific entities"""
        query_lower = query.lower()
        
        # Find mentioned entities
        mentioned_entities = []
        for entity_name in self.entities.keys():
            if entity_name in query_lower:
                mentioned_entities.append(entity_name)
        
        if not mentioned_entities:
            # Fall back to finding most similar entities
            query_embedding = self.embedding_model.embed_query(query)
            entity_similarities = []
            
            for entity_name, entity in self.entities.items():
                if entity.embedding is not None:
                    sim = cosine_similarity([query_embedding], [entity.embedding])[0][0]
                    entity_similarities.append((entity_name, sim))
            
            entity_similarities.sort(key=lambda x: x[1], reverse=True)
            mentioned_entities = [name for name, _ in entity_similarities[:3]]
        
        # Gather context around mentioned entities
        context_items = []
        for entity_name in mentioned_entities:

            # Get entity info
            entity = self.entities[entity_name]
            
            # Get neighbors and relationships
            neighbors = []
            if entity_name in self.graph:

                # Outgoing edges
                for target in self.graph.successors(entity_name):
                    edge_data = self.graph[entity_name][target]
                    neighbors.append({
                        'entity': target,
                        'relationship': edge_data.get('type', 'related_to'),
                        'description': edge_data.get('description', '')
                    })
                
                # Incoming edges
                for source in self.graph.predecessors(entity_name):
                    edge_data = self.graph[source][entity_name]
                    neighbors.append({
                        'entity': source,
                        'relationship': f"inverse of {edge_data.get('type', 'related_to')}",
                        'description': edge_data.get('description', '')
                    })
            
            context_items.append({
                'entity': entity,
                'neighbors': neighbors[:top_k]  # Limit neighbors
            })
        
        return context_items
    
    def _global_retrieve(self, query: str, top_k: int) -> List[Community]:
        """Global retrieval using community summaries"""
        if not self.vectorstore:
            return []
        
        # Search for relevant community summaries
        similar_docs = self.vectorstore.similarity_search(query, k=top_k)
        
        # Get corresponding communities
        relevant_communities = []
        for doc in similar_docs:
            comm_id = doc.metadata['community_id']
            if comm_id in self.communities:
                relevant_communities.append(self.communities[comm_id])
        
        return relevant_communities
    
    def generate_response(self, query: str, context: List[Any]) -> RAGResponse:
        """Generate response based on retrieved context"""
        query_type = self._classify_query(query)
        
        if query_type == "local":
            return self._generate_local_response(query, context)
        else:
            return self._generate_global_response(query, context)
    
    def _generate_local_response(self, query: str, context: List[Dict[str, Any]]) -> RAGResponse:
        """Generate response for entity-specific queries"""
        if not context:
            return RAGResponse(
                answer="I couldn't find specific information about the entities mentioned in your query.",
                sources=[],
                metadata={'strategy': 'GraphRAG', 'query_type': 'local'}
            )
        
        # Format context for the first entity (primary focus)
        primary_context = context[0]
        entity = primary_context['entity']
        
        # Format neighbor information
        neighbor_info = []
        for neighbor in primary_context['neighbors']:
            neighbor_entity = self.entities.get(neighbor['entity'])
            if neighbor_entity:
                neighbor_info.append(
                    f"- {neighbor['entity']} ({neighbor['relationship']}): {neighbor['description']}"
                )
        
        prompt = self.local_query_prompt.format(
            entity=entity.name,
            entity_description=entity.description,
            context="\n".join(neighbor_info),
            question=query
        )
        
        message = HumanMessage(content=prompt)
        response = self.llm.invoke([message])
        
        sources = [f"{entity.name}: {entity.description}"]
        sources.extend([f"{n['entity']} ({n['relationship']})" for n in primary_context['neighbors'][:2]])
        
        return RAGResponse(
            answer=response.content.strip(),
            sources=sources,
            metadata={
                'strategy': 'GraphRAG',
                'query_type': 'local',
                'primary_entity': entity.name,
                'num_neighbors': len(primary_context['neighbors'])
            }
        )
    
    def _generate_global_response(self, query: str, context: List[Community]) -> RAGResponse:
        """Generate response for broad queries using community summaries"""
        if not context:
            return RAGResponse(
                answer="I couldn't find relevant information to answer your query.",
                sources=[],
                metadata={'strategy': 'GraphRAG', 'query_type': 'global'}
            )
        
        # Format community summaries
        summaries = []
        for i, community in enumerate(context):
            summaries.append(f"Community {i+1} (contains {len(community.entities)} entities):\n{community.summary}")
        
        prompt = self.global_query_prompt.format(
            summaries="\n\n".join(summaries),
            question=query
        )
        
        message = HumanMessage(content=prompt)
        response = self.llm.invoke([message])
        
        # Gather sources (community summaries)
        sources = [f"Community {i+1}: {comm.summary[:100]}..." for i, comm in enumerate(context[:3])]
        
        return RAGResponse(
            answer=response.content.strip(),
            sources=sources,
            metadata={
                'strategy': 'GraphRAG',
                'query_type': 'global',
                'num_communities': len(context),
                'total_entities': sum(len(c.entities) for c in context)
            }
        )