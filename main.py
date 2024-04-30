from typing import List, Optional
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
import tiktoken


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


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75,
    length_function=len,
    is_separator_regex=False,
)


def markdown_to_text(source_folder, destination_folder, phrases_to_remove) -> None:
    print("converting markdown to text...")
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for filename in os.listdir(source_folder):
        if filename.endswith('.md'):
            markdown_file_path = os.path.join(source_folder, filename)
            text_file_path = os.path.join(
                destination_folder, filename.replace('.md', '.txt'))
            try:
                with open(markdown_file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                lines = content.splitlines()
                filtered_content = []
                for line in lines:
                    if any(phrase in line for phrase in phrases_to_remove):
                        continue
                    if re.search(r'> \*\w+ on \d{4}-\d{2}-\d{2}', line) or line.strip().startswith('#'):
                        continue
                    line = line.replace('**links**: ', '').replace('---', '')
                    line = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', line)
                    line = re.sub(r'https?://\S+', '', line)
                    line = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
                    line = re.sub(r'\*(.*?)\*', r'\1', line)
                    line = re.sub(r'>\s?(.*?)', r'\1', line)
                    line = re.sub(r'\[\[.*?\]\]', '', line)
                    line = re.sub(r'^#+\s*', '', line)
                    filtered_content.append(line + '\n')
                with open(text_file_path, 'w', encoding='utf-8') as file:
                    file.writelines(filtered_content)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
    print(f"text documents written to {destination_folder}")


def reduce_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def get_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 10, random_state: int = 1234):
    max_clusters = min(max_clusters, len(embeddings))
    bics = [GaussianMixture(n_components=n, random_state=random_state).fit(embeddings).bic(embeddings)
            for n in range(1, max_clusters)]
    return np.argmin(bics) + 1


def gmm_clustering(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters,
                         random_state=random_state).fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def format_cluster_texts(df):
    clustered_texts = {}
    for cluster in df['Cluster'].unique():
        cluster_texts = df[df['Cluster'] == cluster]['Text'].tolist()
        clustered_texts[cluster] = " --- ".join(cluster_texts)
    return clustered_texts


phrases_to_remove = [
    "video links", "Video Links", "Second Brain", "brain dump", "Brain Dump", "upstream"
]

if __name__ == "__main__":
    source_folder = 'core-notes'
    destination_folder = 'text-files'
    markdown_to_text(source_folder, destination_folder, phrases_to_remove)
    embedding_model = OpenAIEmbeddings()
    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    loader = DirectoryLoader('text-files', glob="**/*.txt", show_progress=True)
    docs = loader.load()

    nodes = []
    for doc in docs:
        chunks = text_splitter.split_documents([doc])
        for chunk in chunks:
            embedding = embedding_model.embed_query(chunk.page_content)
            node = Node(embedding=embedding,
                        text=chunk.page_content, parent=None)
            nodes.append(node)
    global_embeddings = [node.embedding for node in nodes]
    global_embeddings_reduced = reduce_cluster_embeddings(
        global_embeddings, 2)

    labels, _ = gmm_clustering(global_embeddings_reduced, threshold=0.5)

    plot_labels = np.array(
        [label[0] if len(label) > 0 else -1 for label in labels])
    plt.figure(figsize=(10, 8))

    unique_labels = np.unique(plot_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = plot_labels == label
        plt.scatter(global_embeddings_reduced[mask, 0], global_embeddings_reduced[mask,
                    1], color=color, label=f'Cluster {label}', alpha=0.5)
    simple_labels = [label[0] if len(label) > 0 else -1 for label in labels]
    texts = [node.text for node in nodes]
    df = pd.DataFrame({
        'Text': texts,
        'Embedding': list(global_embeddings_reduced),
        'Cluster': simple_labels
    })
    clustered_texts = format_cluster_texts(df)
    template = """You are an assistant to create a summary of the text input prodived. It should be human readable. It should contain a minimum of 3 words and maximum of 10 words
    Text:
    {text}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()

    labels = {}

    for cluster, text in clustered_texts.items():
        summary = chain.invoke({"text": text})
        labels[cluster] = summary

    plt.title("Cluster Visualization of Global Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.show()
