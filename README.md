# Business Information Retrieval Augmented Generation (RAG) System
This repository contains the work for a project focused on developing a Retrieval-Augmented Generation (RAG) system designed to improve information collation and summary within a business context. This is a sub-project of a larger LLM fine-tuning initiative.

The primary goal is to leverage Large Language Models (LLMs) to unlock significant business value by making internal information more accessible and ensuring the retention of corporate knowledge. A key focus of this project is the use of locally hosted, open-source models to guarantee data security and protect intellectual property.

# Current Status
- Preliminary Naive RAG Pipeline Implemented

- Synthetic Dataset Generation Underway

# The Problem
In many organizations, efficiently accessing specific information spread across numerous documents is a persistent challenge. This project targets two primary pain points:

- Organizational Procedural Advice: Employees often struggle to find specific details within a large and multi-layered body of procedural and policy documents. This can make it difficult to ensure compliance or even verify that a process exists.

- Lessons Learned Retention: Key insights and niche lessons from major activities (e.g., field trials, project reviews) can be lost over time, buried within a high volume of post-activity reports.

This system aims to empower the workforce by reducing the time spent searching for information and ensuring that valuable corporate knowledge is preserved and remains accessible.

# Objectives
The core objectives for this sub-project are:

- Develop a synthetic dataset for a representative, fictional company to serve as a consistent testbed.

- Implement and assess a baseline "naive" RAG pipeline using this dataset to establish performance benchmarks.

- Investigate and compare alternative RAG implementations (e.g., advanced chunking strategies, graph-based RAG).

- Analyze the relative strengths and weaknesses of each approach to provide clear insights for future development.

# Project Scope
## Preliminary Scope
To ensure a focused and achievable outcome, the initial scope is strictly defined:

- Dataset: Development of a synthetic dataset for a fictional company.

- Baseline Implementation: A "naive" RAG pipeline to act as a benchmark.

- Comparative Analysis: Investigation into alternative RAG implementations (e.g., different chunking strategies, graph-based RAG).

- Outcomes: A summary of the relative strengths and weaknesses of each tested RAG approach.

## Expansion Options
Potential future work for this project may include, in order of priority:

- Implementing an agentic system to support the RAG LLM.

- Conducting comparative benchmarks of entirely different architectures.

- Expanding the datasets to include more complex and challenging retrieval cases.

# Project Roadmap
The project is structured into the following high-level phases:

**Phase 1 (Week 2-4):** Project Refinement & Personal Development

**Phase 2 (Week 3-5):** Sub-Project Planning, Dataset Sourcing & Test Case Development

**Phase 3 (Week 5-8):**

- **3A:** Initial RAG Implementation & Iteration

- **3B:** Alternate Implementation Investigations

**Phase 4 (Week 9):** Final Benchmarking & Outcome Report Development

# Installation
To set up the environment, please follow these steps:

## Clone the Repository
```
git clone [URL]
cd [repository-name]
```

## Install Python Dependencies
```
pip install -r requirements.txt
```

## Install Ollama
This project relies on a locally running LLM managed by Ollama. Please follow the official instructions to install it on your system: https://ollama.com/download

## Download Required Models
Once Ollama is installed and running, you need to download the models for both embedding and generation.
```
# Download the embedding model
ollama pull nomic-embed-text

# Download the LLM for generation
ollama pull llama3:8b
```

# Usage
1. Prepare Your Documents
Before you can chat with your documents, you need to create a vector database from them.

Create a folder named ```documents``` in the root of the project directory.

Place all your source files (.pdf, .txt) into this documents folder.

2. Create the Vector Database
Run the following script to process your documents and create the local vector store. This only needs to be done once, or whenever you add, remove, or change the source documents.
```
python create_embeddings_ollama.py
```
This will create a faiss_index_ollama folder containing the vector database.

3. Run the Application
Prerequisite: Ensure the Ollama application is running in the background.

You can interact with the system in two ways:

## Web Interface
To use the web-based UI, start the server:
```
python ragbot_server.py
```
Then, open your web browser and navigate to http://127.0.0.1:8000.