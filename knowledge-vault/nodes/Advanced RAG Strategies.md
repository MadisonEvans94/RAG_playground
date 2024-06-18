
> 2024-05-01

#seed 
**links**: 
**brain-dump**: 

---

_[Step-back approach to prompting](https://arxiv.org/abs/2310.06117)_ [emerged as a way to tackle the limitations of basic vector search in RAG.](https://arxiv.org/abs/2310.06117) The step-back approach to prompting emphasizes the importance of taking a step back from the details of a task to focus on the broader conceptual framework

![[RAG_stepback_approach.png]]

>Step-back prompting. Image from [research paper](https://arxiv.org/abs/2310.06117) licensed under CC BY 4.0.


**_Parent document retrievers_** have emerged as the solution based on the hypothesis that directly using a document’s vector might be inefficient.

Large documents can be **split** into smaller chunks, which are converted to vectors, improving indexing for similarity searches. Although these smaller vectors better represent specific concepts, the original large document is retrieved as it provides better **context** for answers.

![[RAG_parent_document_retrieval_strategies.png]]

**Available Strategies**

- Typical **RAG**: Traditional method where the exact data indexed is the data retrieved.
- **Parent Retriever**: Instead of indexing entire documents, data is divided into smaller chunks, referred to as “parent” and “child” documents. Child documents are indexed for better representation of specific concepts, while parent documents are retrieved to ensure context retention.
- **Hypothetical Questions**: Documents are processed to generate potential questions they might answer. These questions are then indexed for better representation of specific concepts, while parent documents are retrieved to ensure context retention.
- **Summaries**: Instead of indexing the entire document, a summary of the document is created and indexed. Similarly, the parent document is retrieved in a RAG application.


![[Advanced_RAG_graph_diagram.png]]

In the above diagram, The purple nodes are the parent documents, which have a length of 512 tokens. Each parent document has multiple child nodes (orange) that contain a subsection of the parent document. Additionally, the parent nodes have potential questions represented as blue nodes and a single summary node in red.

![[advanced_RAG_example.png]]

In today’s RAG applications, the ability to retrieve **accurate** and contextual information from a large text corpus is crucial. The traditional approach to vector similarity search, while powerful, sometimes **overlooks** the specific context when longer text is embedded.

By splitting longer documents into smaller vectors and indexing them for similarity, we increase retrieval accuracy while retaining the contextual information of parent documents to generate the answers. Similarly, we can use the LLM to generate hypothetical questions or summaries of text. Then, the LLM indexes those questions and summaries for a better representation of specific concepts while still returning the information from the parent document.

Sometimes, you’ll need to combine graph and textual data to find relevant information. For example, consider this question:

> **What’s the latest news about the founders of Prosper Robotics?**

In this example, you’d want the LLM to identify the Prosper Robotics founders using the knowledge graph structure and then retrieve recent articles that mention them.

![[Pasted image 20240501150613.png]]

A knowledge graph represents structured information about entities and their relationships, as well as unstructured text as **node properties**. You can also use natural language techniques like named entity recognition to connect unstructured information to relevant entities in the knowledge graph, as shown by the **MENTIONS** relationship.

When a knowledge graph contains **structured** and **unstructured** data, the smart search tool can use Cypher queries or vector similarity search to retrieve relevant information. In some cases, you can also use a **combination** of the two.

For example, you can start with a Cypher query to identify relevant documents and then apply vector similarity search to find specific information within those documents.

Another fascinating development around LLMs is the [chain-of-thought question answering](https://cobusgreyling.medium.com/chain-of-thought-prompting-llm-reasoning-147a6cdb312b), especially with [LLM agents](https://python.langchain.com/docs/modules/agents.html).

LLM agents can separate questions into **multiple steps**, define a plan, and draw from any of the provided tools to generate an answer.

Let’s again consider the same question:

> **What’s the latest news about the founders of Prosper Robotics?**

![[Pasted image 20240501151041.png]]

Suppose you don’t have explicit connections between articles and the entities they mention, or the articles and entities are in different databases. An LLM agent using a **chain-of-thought** flow would be very helpful in this case. First, the agent would separate the question into sub-questions:

- Who is the founder of Prosper Robotics?
- What’s the latest news about the founder?

Now the agent can decide which tool to use. Let’s say it’s **grounded** on a knowledge graph, which means it can retrieve structured information, like the name of the founder of Prosper Robotics.

The agent discovers that the founder of Prosper Robotics is Shariq Hashme. Now the agent can rewrite the second question with the information from the first question:

- What’s the latest news about Shariq Hashme?

While chain-of-thought demonstrates the reasoning capabilities of LLMs, it’s not the most user-friendly technique since response **latency** can be high due to **multiple** LLM calls. But it's still exciting to understand more about integrating knowledge graphs into chain-of-thought flows for many use cases.