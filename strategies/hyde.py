from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from .base import BaseRAGStrategy, RAGResponse


class HyDEStrategy(BaseRAGStrategy):
    """HyDE (Hypothetical Document Embeddings) implementation for improved retrieval"""
    
    def _setup(self):
        """Initialize HyDE-specific components"""
        # Extract config values with defaults
        self.chunk_size = self.config.get('chunk_size', 500)
        self.chunk_overlap = self.config.get('chunk_overlap', 75)
        self.model_name = self.config.get('model_name', 'gpt-3.5-turbo')
        self.model_temperature = self.config.get('model_temperature', 0.7)  # Higher temp for creative generation
        self.num_hypothetical_docs = self.config.get('num_hypothetical_docs', 3)
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
        
        # Vector store will be created during ingestion
        self.vectorstore: Optional[FAISS] = None
        self.documents: List[Document] = []
        
        # Template for generating hypothetical documents
        self.hypothetical_doc_template = """Please write a detailed, informative paragraph that would answer the following question. 
        Write as if you are creating a reference document or textbook entry about this topic.
        Make it factual and comprehensive.
        
        Question: {question}
        
        Detailed paragraph:"""
        
        # Template for final answer generation
        self.answer_template = """Answer the question based on the provided context. 
        Be specific and cite information from the context when relevant.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
    
    def ingest_documents(self, source_path: str) -> None:
        """Ingest documents and create vector store"""
        from utils import markdown_to_text
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        
        # Convert markdown to text if needed
        destination_folder = self.config.get('destination_folder', 'text-files')
        phrases_to_remove = self.config.get('phrases_to_remove', [])
        
        markdown_to_text(source_path, destination_folder, phrases_to_remove)
        
        # Load all text documents
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
        
        # Create vector store
        if self.documents:
            self.vectorstore = FAISS.from_documents(
                documents=self.documents,
                embedding=self.embedding_model
            )
        else:
            raise ValueError("No documents found to ingest")
    
    def _generate_hypothetical_documents(self, query: str) -> List[str]:
        """Generate hypothetical documents that might answer the query"""
        hypothetical_docs = []
        
        for i in range(self.num_hypothetical_docs):
            # Add variation to the prompt for diversity
            variation_prompt = ""
            if i == 1:
                variation_prompt = " Focus on technical details and implementation."
            elif i == 2:
                variation_prompt = " Focus on practical applications and examples."
            
            prompt = self.hypothetical_doc_template.format(question=query) + variation_prompt
            human_message = HumanMessage(content=prompt)
            response = self.llm.invoke([human_message])
            hypothetical_docs.append(response.content.strip())
        
        return hypothetical_docs
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve relevant documents using HyDE approach"""
        if not self.vectorstore:
            raise ValueError("No documents have been ingested yet")
        
        # Step 1: Generate hypothetical documents
        hypothetical_docs = self._generate_hypothetical_documents(query)
        
        # Step 2: Embed hypothetical documents
        hypothetical_embeddings = [
            self.embedding_model.embed_query(doc) 
            for doc in hypothetical_docs
        ]
        
        # Step 3: Retrieve similar real documents for each hypothetical doc
        all_retrieved_docs = []
        seen_content = set()  # To avoid duplicates
        
        for hyp_embedding in hypothetical_embeddings:
            # Use the vector store's similarity search
            similar_docs = self.vectorstore.similarity_search_by_vector(
                embedding=hyp_embedding,
                k=top_k
            )
            
            # Add unique documents
            for doc in similar_docs:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    all_retrieved_docs.append(doc)
        
        # Step 4: Re-rank all retrieved documents by similarity to original query
        query_embedding = self.embedding_model.embed_query(query)
        doc_embeddings = [
            self.embedding_model.embed_query(doc.page_content) 
            for doc in all_retrieved_docs
        ]
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        # Sort by similarity and filter by threshold
        doc_similarity_pairs = [
            (doc, sim) for doc, sim in zip(all_retrieved_docs, similarities)
            if sim >= self.similarity_threshold
        ]
        doc_similarity_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k documents
        return [doc for doc, _ in doc_similarity_pairs[:top_k]]
    
    def generate_response(self, query: str, context: List[Document]) -> RAGResponse:
        """Generate response using retrieved documents"""
        # Extract text from documents
        context_texts = [doc.page_content for doc in context]
        combined_context = "\n\n".join(context_texts)
        
        # Generate final answer
        prompt = self.answer_template.format(
            context=combined_context, 
            question=query
        )
        human_message = HumanMessage(content=prompt)
        response_message = self.llm.invoke([human_message])
        
        # Extract source information
        sources = []
        for doc in context[:3]:  # Top 3 sources
            source_text = doc.page_content[:200] + "..."
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                source_text = f"[{doc.metadata['source']}] {source_text}"
            sources.append(source_text)
        
        return RAGResponse(
            answer=response_message.content.strip(),
            sources=sources,
            metadata={
                'num_docs_retrieved': len(context),
                'num_hypothetical_docs': self.num_hypothetical_docs,
                'strategy': 'HyDE'
            }
        )
    
    def query(self, query: str, top_k: int = 5) -> RAGResponse:
        """Override to provide HyDE-specific query handling"""
        # For debugging/transparency, we could optionally return the hypothetical docs
        retrieved_docs = self.retrieve(query, top_k)
        response = self.generate_response(query, retrieved_docs)
        
        # Optionally add hypothetical documents to metadata for transparency
        if self.config.get('include_hypothetical_docs', False):
            hypothetical_docs = self._generate_hypothetical_documents(query)
            response.metadata['hypothetical_documents'] = hypothetical_docs
        
        return response