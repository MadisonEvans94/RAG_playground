from typing import Dict, List, Optional, Tuple
import os
import re
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.mixture import GaussianMixture


class Config:
    """
    Configuration class to hold settings and parameters for the application.

    Attributes:
        text_splitter (RecursiveCharacterTextSplitter): Text splitter utility for processing documents.
        source_folder (str): Directory where source markdown files are stored.
        destination_folder (str): Directory where processed text files are stored.
        phrases_to_remove (List[str]): List of phrases to filter out from the text.
        chunk_size (int): The size of each text chunk to process.
        chunk_overlap (int): The overlap between consecutive text chunks.
        model_temperature (float): Temperature setting for the AI model, affecting randomness.
        model_name (str): Name of the AI model used for processing text.
        embedding_dim (int): Dimensionality for embedding vectors.
        max_clusters (int): Maximum number of clusters for the clustering algorithm.
        random_state (int): Seed for the random number generator used in clustering.
        cluster_threshold (float): Threshold for cluster assignment probabilities.
        summary_template (str): Template used for generating summaries of text clusters.
    """

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=75, length_function=len, is_separator_regex=False)
        self.source_folder = 'core-notes'
        self.destination_folder = 'text-files'
        self.phrases_to_remove = [
            "video links", "Video Links", "Second Brain", "brain dump", "Brain Dump", "upstream"
        ]
        self.chunk_size = 500
        self.chunk_overlap = 75
        self.model_temperature = 0
        self.model_name = "gpt-3.5-turbo"
        self.embedding_dim = 2
        self.max_clusters = 10
        self.random_state = 1234
        self.cluster_threshold = 0.5
        self.summary_template = """You are an assistant to create a summary of the text input provided. It should be human-readable. It should contain a minimum of 1 words and a maximum of 4 words
        Text:
        {text}
        """


class Cluster:
    def __init__(self, name: str, embedding: List[float], children: List['Node'], parent: Optional['Cluster'] | None) -> None:
        self.name = name
        self.embedding = embedding
        self.children = children
        self.parent = parent


class Node:
    def __init__(self, embedding: float, text: str, parent: Optional[Cluster]):
        self.embedding = embedding
        self.text = text
        self.parent = parent


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


def create_nodes_from_documents(destination_folder: str, embedding_model: OpenAIEmbeddings, config: Config) -> List[Node]:
    """
    Processes documents from the specified folder and creates Node objects with embeddings.

    Parameters:
        destination_folder (str): The folder where text files are located.
        embedding_model (OpenAIEmbeddings): The model used to generate embeddings for text chunks.
        config (Config): Configuration instance containing settings like text_splitter.

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


def cluster_nodes(nodes: List[Node], config: Config) -> pd.DataFrame:
    """
    Clusters nodes based on their embeddings and returns a DataFrame with cluster labels and other details.

    Parameters:
        nodes (List[Node]): The list of nodes to cluster.
        config (Config): Configuration instance containing clustering parameters.

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


def markdown_to_text(source_folder: str, destination_folder: str, phrases_to_remove: List[str]) -> None:
    """
    Converts markdown files from a source directory to cleaned text files in a destination directory.

    Parameters:
        source_folder (str): Directory containing the markdown files to process.
        destination_folder (str): Target directory where the cleaned text files will be saved.
        phrases_to_remove (List[str]): List of phrases that should be removed from the text.

    Returns:
        None: This function does not return a value; it writes to files directly.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for filename in os.listdir(source_folder):
        if filename.endswith('.md'):
            with open(os.path.join(source_folder, filename), 'r', encoding='utf-8') as file:
                content = file.read()
            content = text_cleanup(content, phrases_to_remove)
            with open(os.path.join(destination_folder, filename.replace('.md', '.txt')), 'w', encoding='utf-8') as file:
                file.write(content)


def reduce_cluster_embeddings(embeddings: np.ndarray, dim: int, config: Config) -> np.ndarray:
    """
    Applies dimensionality reduction on the embeddings using UMAP.

    Parameters:
        embeddings (np.ndarray): Array of embedding vectors to be reduced.
        dim (int): Target dimensionality for the embeddings.
        config (Config): Configuration object containing algorithm parameters.

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


def gmm_clustering(embeddings: np.ndarray, config: Config) -> Tuple[List[List[int]], int]:
    """
    Applies Gaussian Mixture Modeling to cluster the given embeddings and returns cluster labels.

    Parameters:
        embeddings (np.ndarray): Array of embeddings to cluster.
        config (Config): Configuration object containing clustering settings.

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


phrases_to_remove = [
    "video links", "Video Links", "Second Brain", "brain dump", "Brain Dump", "upstream"
]


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75,
    length_function=len,
    is_separator_regex=False,
)


# Main execution function
def main():
    config = Config()
    markdown_to_text(config.source_folder,
                     config.destination_folder, config.phrases_to_remove)
    embedding_model = OpenAIEmbeddings()
    model = ChatOpenAI(temperature=config.model_temperature,
                       model=config.model_name)
    nodes = create_nodes_from_documents(
        config.destination_folder, embedding_model, config)
    df = cluster_nodes(nodes, config)
    visualize_clusters(df)
    summaries = summarize_clusters(df, model, config.summary_template)
    print(summaries)


if __name__ == "__main__":
    main()
