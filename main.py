import argparse
import json
from typing import Dict, Any
from strategies.raptor import RAPTORStrategy
from strategies.base import BaseRAGStrategy
from strategies.hyde import HyDEStrategy
from strategies.graphrag import GraphRAGStrategy

# Registry of available strategies
STRATEGIES = {
    'raptor': RAPTORStrategy,
    'graphrag': GraphRAGStrategy,
    'hyde': HyDEStrategy,
}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_strategy(strategy_name: str, config: Dict[str, Any]) -> BaseRAGStrategy:
    """Factory function to create strategy instances"""
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGIES.keys())}")
    
    strategy_class = STRATEGIES[strategy_name]
    return strategy_class(config)


def interactive_chat(strategy: BaseRAGStrategy):
    """Run an interactive chat session"""
    print(f"\n🤖 RAG Chat using {strategy.__class__.__name__}")
    print("Type 'quit' or 'exit' to end the session\n")
    
    while True:
        query = input("You: ").strip()
        
        if query.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        try:
            response = strategy.query(query)
            print(f"\nAssistant: {response.answer}")
            
            if response.sources:
                print("\n Sources:")
                for i, source in enumerate(response.sources, 1):
                    print(f"{i}. {source[:100]}...")
            
            print()  # Empty line for readability
            
        except Exception as e:
            print(f"\n Error: {str(e)}\n")


def main():
    parser = argparse.ArgumentParser(description='Test RAG strategies')
    parser.add_argument(
        '--strategy', 
        type=str, 
        default='raptor',
        choices=list(STRATEGIES.keys()),
        help='RAG strategy to use'
    )
    parser.add_argument(
        '--config', 
        type=str,
        default='config/raptor_config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--source', 
        type=str,
        default='knowledge-vault/nodes',
        help='Path to source documents'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Single query to run (if not provided, starts interactive mode)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create strategy
    print(f"🚀 Initializing {args.strategy.upper()} strategy...")
    strategy = create_strategy(args.strategy, config)
    
    # Ingest documents
    print(f"📄 Ingesting documents from {args.source}...")
    strategy.ingest_documents(args.source)
    
    # Run query or start interactive mode
    if args.query:
        response = strategy.query(args.query)
        print(f"\nQuery: {args.query}")
        print(f"Answer: {response.answer}")
        if response.sources:
            print("\nSources:")
            for i, source in enumerate(response.sources, 1):
                print(f"{i}. {source[:100]}...")
    else:
        interactive_chat(strategy)


if __name__ == "__main__":
    main()