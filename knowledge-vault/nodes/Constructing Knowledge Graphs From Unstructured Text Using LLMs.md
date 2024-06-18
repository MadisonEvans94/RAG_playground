> 2024-05-01

#seed
**links**: https://neo4j.com/developer-blog/construct-knowledge-graphs-unstructured-text/#extract, [GitHub](https://github.com/neo4j/NaLLM)
**brain-dump**:

---

1. **Extracting Nodes and Relationships**: The text is input to the LLM, which identifies and extracts entities (nodes) and their relationships. The text is divided into manageable chunks to fit within the LLM’s context window, enhancing the model's ability to process the data effectively.
2. **Entity Disambiguation**: This step involves using the LLM to merge duplicate entities that may have been identified multiple times across different text chunks. The LLM groups entities by type and merges duplicates to maintain a clean and accurate dataset.
3. **Importing into Neo4j**: Finally, the extracted data is formatted into CSV files and imported into Neo4j, a graph database. This allows for the structured data to be stored, analyzed, and visualized effectively.
