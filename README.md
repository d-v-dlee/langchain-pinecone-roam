### Project Overview
This project provides a convenient way to generate well-structured essays using OpenAI's GPT-3.5-turbo. By leveraging your own notes and highlights, the project combines different techniques, including LangChain, Pinecone, Sentence Transformers, and Roam database notes, to automatically generate an essay.

### Main Components
- Pinecone: The project uses Pinecone to handle the retrieval of relevant data for the essay generation process. It allows embedding queries, querying the Pinecone index, and performing research and summarization tasks.
- BasePromptTemplate: A base class for creating custom prompt templates. It initializes a language model and defines the process function to execute the prompt.
- SummaryTemplate: A template for getting a TL;DR of a context.
- TangentialIdeasTemplate: A template for extracting tangential ideas or themes from a given context.
- CompressTemplate: A template for compressing text in a lossless manner.
- EssayTemplate: A template for generating a well-structured essay based on the provided topics and context.
- PineconeHandler: A class that integrates with Pinecone to manage the retrieval and processing of data. It provides methods for embedding queries, querying the Pinecone index, and performing research and summarization tasks.

### Flow
1. User enter research topic
2. Query Pinecone for top 5 results
3. Summarize top 5 results
4. Using summaries, come up with 5 new ideas to research
5. Compare cosine similarity between current topic and 5 new topics; choose 3rd closest
6. repeat steps 2-5 for 5 rounds
7. compress the 25 summaries 
8. feed topics + summaries for final essay creation