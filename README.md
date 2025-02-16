# LLAMA Model Document Q&A with LoRA/QLoRA on US Census Dataset

## Overview

This project demonstrates the use of **Gemma Model**, enhanced with **LoRA/QLoRA**, for Document Question Answering (Q&A) on the **US Census Dataset**. The solution leverages advanced machine learning techniques such as **RAG (Retrieval-Augmented Generation)** and **vector embeddings** to provide accurate answers based on the content of a large dataset stored as PDFs. The model is fine-tuned using **LoRA/QLoRA** for computational efficiency, while **Pinecone** and **FAISS** serve as the vector stores for fast and scalable document retrieval.

## Key Features
- **Document Embedding** using Google Generative AI Embeddings.
- **Vector Stores** using **Pinecone** and **FAISS**.
- **LoRA/QLoRA** for efficient model fine-tuning on large datasets.
- **Retrieval-Augmented Generation (RAG)** for improved question-answering accuracy.
- Streamlit-based **interactive web interface** for easy query handling.

## Table of Contents
- [Objective](#objective)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [License](#license)

## Objective

The main objective of this project is to provide an efficient and scalable solution for **question answering** based on the **US Census Dataset**. The solution incorporates the **Gemma Model**, which is enhanced using **LoRA/QLoRA** techniques for fine-tuning large models. The system allows users to interact with the dataset, input queries, and receive answers derived from relevant documents.

Key features:
- **Vectorization** of documents into embeddings.
- **Retrieval of relevant documents** based on user queries.
- **RAG Scores**: Relevance scores of retrieved documents to show how closely they relate to the user's question.

## Technologies Used

- **LangChain**: Framework for LLM-based document retrieval.
- **LoRA/QLoRA**: Low-rank adaptation of transformer models for efficient fine-tuning.
- **Pinecone**: Managed vector database for scalable document retrieval.
- **FAISS**: A similarity search library for vector search.
- **Streamlit**: Framework for building interactive web applications.
- **Google Generative AI**: API for generating document embeddings.
- **PyTorch**: Deep learning framework for model fine-tuning and inference.
- **Transformers**: Library by Hugging Face for loading pre-trained language models.
- **peft**: Low-rank adaptation technique to reduce model size during fine-tuning.

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/gemma-document-qa.git
cd gemma-document-qa
```
### 2. Install Dependencies
* Make sure to install the necessary Python libraries. You can do this by running:
```bash
pip install -r requirements.txt
```

Dependencies include:
faiss-cpu
groq
langchain-groq
PyPDF2
langchain_google_genai
langchain
streamlit
langchain_community
python-dotenv
pypdf
google-cloud-aiplatform>=1.38
pinecone
peft
transformers
torch

### 3. Set Up Environment Variables
* Create a .env file in the project root and add your API keys for the following services:
```

GROQ API Key
Google API Key
Pinecone API Key
Pinecone Environment

```

Example .env file:
```makefile
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_env
```

* Usage
Run the Streamlit App: Start the Streamlit app by running the following command:

```bash
streamlit run app.py
```

* Interact with the App:

    * Input your query in the text box.
    * Select whether to use Pinecone or FAISS as the vector store.
    * Click Documents Embedding to initialize the vector store and embed documents from the US Census dataset.
    * The system will process your query and return answers based on relevant documents. The RAG Score of each document will also be displayed to show the relevance.


### 4. Results
* The application provides answers to queries based on the US Census Dataset. 
* For each query, the system retrieves relevant documents and computes a RAG Score to indicate the relevance of each document. 
* The response time and the documents' context are shown in the user interface.

Example query:

``` css
Enter Your Question From Documents: "What is the population of California in 2020?"
Example output:
```
```css
Response time: 3.5 seconds
Answer: "The population of California in 2020 was approximately 39.51 million."
Document Similarity Search: For each retrieved document, the system displays its RAG Score and a snippet of the document content:
```
```vbnet
Document 1 - RAG Score: 0.92
Content: "According to the 2020 US Census, the population of the United States increased by 7.4% since 2010..."

Document 2 - RAG Score: 0.88
Content: "The census data shows that the state of California has the largest population among all states..."
Future Enhancements
Improved Model Fine-tuning: Further fine-tuning the LoRA/QLoRA models with more specific datasets could improve accuracy.
Real-time Updates: Enabling real-time updates for the vector store when new data is available.
Hybrid Retrieval Models: Combine traditional NLP techniques with modern embedding-based retrieval for enhanced performance.
```
### 5. Conclusion
* The integration of LoRA/QLoRA with Gemma Model for document-based question answering provides an efficient solution for large-scale datasets like the US Census.
* The use of Pinecone or FAISS as vector stores enables high-speed retrieval of documents relevant to user queries. The RAG score adds an extra layer of transparency, allowing users to evaluate the relevance of retrieved documents.

This framework can be easily adapted to other large-scale datasets, offering scalable and efficient solutions for document-based question answering and retrieval tasks.

### 6. Future Enhancements
* Improved Model Fine-tuning: Further fine-tuning the LoRA/QLoRA model with specific datasets could improve accuracy.
* Advanced Document Retrieval: Integration of additional retrieval strategies (e.g., hybrid models using both traditional and modern NLP techniques).
* Real-time Updates: Allowing users to update the vector store with new data dynamically
