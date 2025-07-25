from typing import Dict, List, Optional, Tuple
import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.mixture import GaussianMixture
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class Node:
    def __init__(self, text: str, embedding: np.ndarray, depth: int):
        self.text: str = text
        self.embedding: np.ndarray = embedding
        self.depth: int = depth
        self.children: List[Node] = []


def text_cleanup(content: str, phrases_to_remove: List[str]) -> str:
    """
    Cleans up the given text content by removing specified markdown elements and phrases.

    Parameters:
        content (str): The markdown content to clean.
        phrases_to_remove (List[str]): Phrases that should be removed from the content.

    Returns:
        str: The cleaned text.
    """
    lines = content.splitlines()
    filtered_content = []
    for line in lines:
        if any(phrase in line for phrase in phrases_to_remove) or re.search(r'> \*\w+ on \d{4}-\d{2}-\d{2}', line) or line.strip().startswith('#'):
            continue
        line = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', line)
        line = re.sub(r'\*\*(.*?)\*\*|\*(.*?)\*|> *(.*?)', r'\1', line)
        line = re.sub(r'https?://\S+|\[\[.*?\]\]', '', line)
        line = re.sub(r'^#+\s*|---|\*\*links\*\*: ', '', line)
        filtered_content.append(line + '\n')
    return ''.join(filtered_content)


def create_nodes_from_documents(folder_path: str, embedding_model: OpenAIEmbeddings, config) -> List[Node]:
    """
    Create nodes from documents with embeddings.
    
    Parameters:
        folder_path (str): Path to folder containing text documents
        embedding_model (OpenAIEmbeddings): Model to create embeddings
        config: Configuration object or dict with text_splitter settings
        
    Returns:
        List[Node]: List of Node objects with embeddings
    """
    nodes = []
    
    # Handle both dict and object configs
    if isinstance(config, dict):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get('chunk_size', 500),
            chunk_overlap=config.get('chunk_overlap', 75),
            length_function=len,
            is_separator_regex=False
        )
    else:
        # Original behavior for Config objects
        text_splitter = config.text_splitter
    
    # Get all text files
    text_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    for filename in tqdm(text_files, desc="Processing documents"):
        file_path = os.path.join(folder_path, filename)
        
        # Load document
        loader = UnstructuredFileLoader(file_path, mode="single", strategy="fast")
        docs = loader.load()
        
        # Split documents into chunks
        for doc in docs:
            chunks = text_splitter.split_documents([doc])
            
            # Create nodes with embeddings for each chunk
            for chunk in chunks:
                chunk_text = chunk.page_content
                chunk_embedding = embedding_model.embed_query(chunk_text)
                node = Node(
                    text=chunk_text,
                    embedding=np.array(chunk_embedding),
                    depth=0  
                )
                nodes.append(node)
    
    return nodes


def cluster_nodes(nodes: List[Node], config) -> pd.DataFrame:
    """
    Clusters nodes based on their embeddings and returns a DataFrame with cluster labels and other details.

    Parameters:
        nodes (List[Node]): The list of nodes to cluster.
        config: Configuration instance containing clustering parameters.

    Returns:
        pd.DataFrame: DataFrame containing the text of each node, their embeddings, and assigned cluster labels.
    """
    # Handle dict config
    if isinstance(config, dict):
        embedding_dim = config.get('embedding_dim', 2)
        max_clusters = config.get('max_clusters', 50)
        cluster_threshold = config.get('cluster_threshold', 0.5)
        random_state = config.get('random_state', 1234)
    else:
        embedding_dim = config.embedding_dim
        max_clusters = config.max_clusters
        cluster_threshold = config.cluster_threshold
        random_state = config.random_state
    
    embeddings = np.array([node.embedding for node in nodes])
    reduced_embeddings = reduce_cluster_embeddings(
        embeddings, embedding_dim, config)
    labels, _ = gmm_clustering(reduced_embeddings, config)
    
    return pd.DataFrame({
        'Text': [node.text for node in nodes],
        'Embedding': list(reduced_embeddings),
        'Cluster': [label[0] if len(label) > 0 else -1 for label in labels]
    })


def visualize_clusters(df: pd.DataFrame) -> None:
    """
    Visualizes the clustering of nodes using their embeddings.

    Parameters:
        df (pd.DataFrame): DataFrame containing embedding data and cluster labels for visualization.

    Returns:
        None: This function does not return any value; it shows a plot directly.
    """
    embeddings = np.stack(df['Embedding'].to_list())
    labels = df['Cluster'].to_numpy()
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1],
                    color=color, label=f'Cluster {label}', alpha=0.5)
    plt.title("Cluster Visualization of Global Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.show()


def markdown_to_text(source_folder: str, destination_folder: str, phrases_to_remove: List[str], keyword: str = "#sprout") -> None:
    """
    Converts markdown files from a source directory to cleaned text files in a destination directory,
    but only for files that contain a specified keyword.

    Parameters:
        source_folder (str): Directory containing the markdown files to process.
        destination_folder (str): Target directory where the cleaned text files will be saved.
        phrases_to_remove (List[str]): List of phrases that should be removed from the text.
        keyword (str): Keyword to look for in the markdown files; only files containing this keyword are processed.

    Returns:
        None: This function does not return a value; it writes to files directly.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    processed_count = 0 
    for filename in os.listdir(source_folder):
        if filename.endswith('.md'):
            file_path = os.path.join(source_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            # Check if the content contains the keyword
            if keyword in content:
                cleaned_content = text_cleanup(content, phrases_to_remove)
                output_path = os.path.join(
                    destination_folder, filename.replace('.md', '.txt'))
                with open(output_path, 'w', encoding='utf-8') as file:
                    file.write(cleaned_content)
                processed_count += 1  

    print(f"{processed_count} files were processed and added to the '{destination_folder}' folder.")


def reduce_cluster_embeddings(embeddings: np.ndarray, dim: int, config) -> np.ndarray:
    """
    Applies dimensionality reduction on the embeddings using UMAP.

    Parameters:
        embeddings (np.ndarray): Array of embedding vectors to be reduced.
        dim (int): Target dimensionality for the embeddings.
        config: Configuration object containing algorithm parameters.

    Returns:
        np.ndarray: Array of reduced embeddings.
    """
    return umap.UMAP(
        n_neighbors=int((len(embeddings) - 1) ** 0.5),
        n_components=dim,
        metric="cosine"
    ).fit_transform(embeddings)


def get_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 10, random_state: int = 1234) -> int:
    """
    Determines the optimal number of clusters for Gaussian Mixture Model clustering using the Bayesian Information Criterion (BIC).

    Parameters:
        embeddings (np.ndarray): Array of embeddings to cluster.
        max_clusters (int): Maximum number of clusters to consider.
        random_state (int): Seed for the random number generator.

    Returns:
        int: The optimal number of clusters.
    """
    max_clusters = min(max_clusters, len(embeddings))
    bics = [GaussianMixture(n_components=n, random_state=random_state).fit(embeddings).bic(embeddings)
            for n in range(1, max_clusters)]
    return np.argmin(bics) + 1


def gmm_clustering(embeddings: np.ndarray, config) -> Tuple[List[List[int]], int]:
    """
    Applies Gaussian Mixture Modeling to cluster the given embeddings and returns cluster labels.

    Parameters:
        embeddings (np.ndarray): Array of embeddings to cluster.
        config: Configuration object containing clustering settings.

    Returns:
        Tuple[List[List[int]], int]: A tuple containing a list of cluster labels for each point and the number of clusters used.
    """
    # Handle dict config
    if isinstance(config, dict):
        max_clusters = config.get('max_clusters', 50)
        cluster_threshold = config.get('cluster_threshold', 0.5)
        random_state = config.get('random_state', 1234)
    else:
        max_clusters = config.max_clusters
        cluster_threshold = config.cluster_threshold
        random_state = config.random_state
    
    n_clusters = get_optimal_clusters(embeddings, max_clusters, random_state)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state).fit(embeddings)
    probs = gm.predict_proba(embeddings)
    return [np.where(prob > cluster_threshold)[0] for prob in probs], n_clusters


def format_cluster_texts(df: pd.DataFrame) -> Dict[int, str]:
    """
    Formats the texts of each cluster into a single string per cluster.

    Parameters:
        df (pd.DataFrame): DataFrame containing the text and cluster labels.

    Returns:
        Dict[int, str]: Dictionary with cluster IDs as keys and concatenated string of texts as values.
    """
    clustered_texts = {}
    for cluster in df['Cluster'].unique():
        cluster_texts = df[df['Cluster'] == cluster]['Text'].tolist()
        clustered_texts[cluster] = " --- ".join(cluster_texts)
    return clustered_texts


def summarize_clusters(df: pd.DataFrame, model: ChatOpenAI, prompt_template: str) -> Dict[int, str]:
    """
    Summarizes the texts within each cluster using a configured language model.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the cluster texts.
        model (ChatOpenAI): The language model used for generating summaries.
        prompt_template (str): Template string used to format the input to the model.

    Returns:
        Dict[int, str]: Dictionary with cluster IDs as keys and summaries as values.
    """
    clustered_texts = format_cluster_texts(df)
    template = ChatPromptTemplate.from_template(prompt_template)
    chain = template | model | StrOutputParser()
    return {cluster: chain.invoke({"text": text}) for cluster, text in clustered_texts.items()}