
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from utils import create_nodes_from_documents, summarize_clusters, visualize_clusters, cluster_nodes, markdown_to_text


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
        self.source_folder = 'knowledge-vault/nodes'
        self.destination_folder = 'text-files'
        self.phrases_to_remove = [
            "video links", "Video Links", "Second Brain", "brain dump", "Brain Dump", "upstream"
        ]
        self.chunk_size = 500
        self.chunk_overlap = 75
        self.model_temperature = 0
        self.model_name = "gpt-3.5-turbo"
        self.embedding_dim = 2
        self.max_clusters = 50
        self.random_state = 1234
        self.cluster_threshold = 0.5
        self.summary_template = """You are an assistant to create a summary of the text input provided. It should be human-readable. It should contain a minimum of 1 words and a maximum of 4 words
        Text:
        {text}
        """


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
