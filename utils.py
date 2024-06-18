from typing import Dict, List, Optional, Tuple
import os
import re
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.mixture import GaussianMixture
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


class Node:
    def __init__(self, embedding: float, text: str, parent: Optional['Node']):
        self.embedding = embedding
        self.text = text
        self.parent = parent
        self.children = []


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
        # Replace markdown links
        line = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', line)
        # Remove bold, italics, and block quotes
        line = re.sub(r'\*\*(.*?)\*\*|\*(.*?)\*|> *(.*?)', r'\1', line)
        # Remove URLs and wikilinks
        line = re.sub(r'https?://\S+|\[\[.*?\]\]', '', line)
        # Clean headers and separators
        line = re.sub(r'^#+\s*|---|\*\*links\*\*: ', '', line)
        filtered_content.append(line + '\n')
    return ''.join(filtered_content)


def create_nodes_from_documents(destination_folder: str, embedding_model: OpenAIEmbeddings, config) -> List[Node]:
    """
    Processes documents from the specified folder and creates Node objects with embeddings.

    Parameters:
        destination_folder (str): The folder where text files are located.
        embedding_model (OpenAIEmbeddings): The model used to generate embeddings for text chunks.
        config (Config)uration instance containing settings like text_splitter.

    Returns:
        List[Node]: A list of Node objects with embeddings and text data.
    """
    loader = DirectoryLoader(
        destination_folder, glob="**/*.txt", show_progress=True)
    docs = loader.load()
    nodes = []
    for doc in docs:
        chunks = config.text_splitter.split_documents([doc])
        for chunk in chunks:
            embedding = embedding_model.embed_query(chunk.page_content)
            nodes.append(Node(embedding=embedding,
                         text=chunk.page_content, parent=None))
    return nodes


def cluster_nodes(nodes: List[Node], config) -> pd.DataFrame:
    """
    Clusters nodes based on their embeddings and returns a DataFrame with cluster labels and other details.

    Parameters:
        nodes (List[Node]): The list of nodes to cluster.
        config (Config)uration instance containing clustering parameters.

    Returns:
        pd.DataFrame: DataFrame containing the text of each node, their embeddings, and assigned cluster labels.
    """
    embeddings = np.array([node.embedding for node in nodes])
    reduced_embeddings = reduce_cluster_embeddings(
        embeddings, config.embedding_dim, config)
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
    processed_count = 0  # Initialize a counter for processed files
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
                processed_count += 1  # Increment the counter after successfully writing a file

    print(f"{processed_count} files were processed and added to the '{destination_folder}' folder.")


def reduce_cluster_embeddings(embeddings: np.ndarray, dim: int, config) -> np.ndarray:
    """
    Applies dimensionality reduction on the embeddings using UMAP.

    Parameters:
        embeddings (np.ndarray): Array of embedding vectors to be reduced.
        dim (int): Target dimensionality for the embeddings.
        config (Config)uration object containing algorithm parameters.

    Returns:
        np.ndarray: Array of reduced embeddings.
    """
    return umap.UMAP(
        n_neighbors=int((len(embeddings) - 1) ** 0.5),
        n_components=dim,
        metric="cosine"
    ).fit_transform(embeddings)


def get_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 10, random_state: int = 1234):
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
        config (Config)uration object containing clustering settings.

    Returns:
        Tuple[List[List[int]], int]: A tuple containing a list of cluster labels for each point and the number of clusters used.
    """
    n_clusters = get_optimal_clusters(
        embeddings, config.max_clusters, config.random_state)
    gm = GaussianMixture(n_components=n_clusters,
                         random_state=config.random_state).fit(embeddings)
    probs = gm.predict_proba(embeddings)
    return [np.where(prob > config.cluster_threshold)[0] for prob in probs], n_clusters


def format_cluster_texts(df) -> Dict[int, str]:
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


def summarize_clusters(df: pd.DataFrame, model, prompt_template: str) -> Dict[int, str]:
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
