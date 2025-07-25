from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()




@dataclass
class RAGResponse:
    """Standard response format for all RAG strategies"""
    answer: str
    sources: List[str]
    metadata: Dict[str, Any]


class BaseRAGStrategy(ABC):
    """Abstract base class for all RAG strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._setup()
    
    @abstractmethod
    def _setup(self):
        """Initialize strategy-specific components"""
        pass
    
    @abstractmethod
    def ingest_documents(self, source_path: str) -> None:
        """Ingest and process documents according to the strategy"""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Any]:
        """Retrieve relevant information for the query"""
        pass
    
    @abstractmethod
    def generate_response(self, query: str, context: List[Any]) -> RAGResponse:
        """Generate response using retrieved context"""
        pass
    
    def query(self, query: str, top_k: int = 5) -> RAGResponse:
        """Main entry point for querying - can be overridden if needed"""
        context = self.retrieve(query, top_k)
        return self.generate_response(query, context)