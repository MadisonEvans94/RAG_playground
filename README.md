# Project: Advanced RAG Techniques on Zettelkasten Knowledge Base

## Overview

This project explores advanced Retrieval-Augmented Generation (RAG) techniques using a subset of markdown notes from an Obsidian vault as the knowledge base. The goal is to test and refine methods for efficiently querying and summarizing large, interconnected datasets such as a Zettelkasten knowledge base.

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) enhances retrieval-augmented language models by constructing hierarchical tree structures through recursive embedding, clustering, and summarizing of text chunks. This approach allows for retrieval at various abstraction levels, improving holistic document understanding and outperforming traditional retrieval methods in complex reasoning tasks. For more insights on RAG techniques, you can refer to the [research paper](https://arxiv.org/html/2401.18059v1).

## Project Structure

The project consists of:

-   **Main Script (`raptor.py`)**: Processes and clusters text data, creates recursive hierarchical structures, and generates contextually relevant answers using an OpenAI model.
-   **Utility Functions (`utils.py`)**: Contains functions for data processing, clustering, and summarization.
-   **Configuration Class**: Manages various parameters for text processing, clustering, and model interaction.

### Files and Directories

-   `raptor.py`: The main script for processing data and generating responses.
-   `utils.py`: Contains utility functions for text processing and clustering.
-   `requirements.txt`: Lists all Python dependencies required for the project.
-   `knowledge-vault/nodes`: Directory where markdown notes are stored.

## Setup Instructions

To set up this project on your local machine, follow these steps:

### Prerequisites

Ensure you have Python 3.7 or higher installed on your machine.

### Steps

1. **Clone the Repository**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Install Required Packages**
   Use the `requirements.txt` file to install all necessary packages:

    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare Your Data**

    - Place your markdown files in the `knowledge-vault/nodes` directory.

4. **Run the Script**
   Execute the main script to process the data and test the RAG techniques:
    ```bash
    python raptor.py
    ```

## Project Goals

This project aims to:

-   Enhance the ability to query large, interconnected datasets.
-   Improve summarization techniques for better knowledge extraction.
-   Test the effectiveness of RAG methods in real-world applications like personal knowledge management systems (e.g., Zettelkasten).

## Next Steps

The next phase of the project involves:

1. **Modularity**: Making the project more modular by allowing different Large Language Models (LLMs) to be easily swapped in place of the current OpenAI API.
2. **Performance**: Improving inference speed by leveraging GPUs or TPUs for faster processing.

Any contributions or suggestions to help achieve these goals are welcome.

## Motivation

The motivation behind this project is to explore advanced methods for managing and querying personal knowledge bases. By leveraging state-of-the-art RAG techniques, we aim to create a powerful tool that can enhance personal productivity and knowledge management.

## Contribution

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
