# Project: Advanced RAG Techniques on Zettelkasten Knowledge Base

## Overview

This project explores advanced Retrieval-Augmented Generation (RAG) techniques using a subset of markdown notes from an Obsidian vault as the knowledge base. The goal is to test and refine methods for efficiently querying and summarizing large, interconnected datasets such as a Zettelkasten knowledge base.

## Project Structure

The project consists of:

-   A Python script (`raptor.py`) that processes and clusters text data, creates recursive hierarchical structures, and generates contextually relevant answers using an OpenAI model.
-   Utility functions in `utils.py` for data processing, clustering, and summarization.
-   A configuration class to manage various parameters for text processing, clustering, and model interaction.

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

### Files and Directories

-   `raptor.py`: The main script for processing data and generating responses.
-   `utils.py`: Contains utility functions for text processing and clustering.
-   `requirements.txt`: Lists all Python dependencies required for the project.
-   `knowledge-vault/nodes`: Directory where markdown notes are stored.

## Project Goals

This project aims to:

-   Enhance the ability to query large, interconnected datasets.
-   Improve summarization techniques for better knowledge extraction.
-   Test the effectiveness of RAG methods in real-world applications like personal knowledge management systems (e.g., Zettelkasten).
