import os
import time
import logging

import numpy as np
from dotenv import load_dotenv
import pinecone
from sentence_transformers import SentenceTransformer
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from chainer import (run_parallel_chains, flatten_results,
                SummaryTemplate, TangenitalIdeasTemplate, CompressTemplate, EssayTemplate
                )

logger = logging.getLogger('pinecone_handler')
logger.setLevel(logging.INFO)

load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

def sort_strings_by_similarity(ref_string, strings_list):
    """
    embeds and compares a reference string to a list of strings and returns the list of strings sorted by cosine similarity
    """
    
    # Encode the reference string and list of strings
    ref_string_embedding = model.encode(ref_string)
    strings_list_embeddings = model.encode(strings_list)

    # Compute cosine similarity between the reference string and list of strings
    similarities = cosine_similarity(
        ref_string_embedding.reshape(1, -1),
        strings_list_embeddings
    )

    # Create an array of indices sorted by cosine similarity, descending order
    sorted_indices = np.argsort(similarities[0])[::-1]

    # Sort the list of strings based on the sorted indices
    sorted_strings_list = [strings_list[i] for i in sorted_indices]

    return sorted_strings_list

class PineconeHandler:
    def __init__(self):
        pinecone.init(api_key=PINECONE_API_KEY, environment='eu-west1-gcp')

        self.index = pinecone.Index('roam-research')
        self.model = model
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo")


    def get_embedding(self, query):
        """returns a list of floats representing the embedding of the query"""
        return self.model.encode(query).tolist()
    
    def query_index(self, query_embedding, filter=None):
        """
        queries the index for the top 5 results.
        """
        if filter:
            results = self.index.query(query_embedding, top_k=5, includeMetadata=True, filter=filter)
        else:
            results = self.index.query(query_embedding, top_k=5, includeMetadata=True)
        print(f'Found {len(results["matches"])} results!')
        return results
    
    def embed_and_query(self, query, filter=None):
        """
        embeds the query and queries the index for the top 5 results.
        """
        logger.info(f'Processing query: {query}')
        query_embedding = self.get_embedding(query)
        results = self.query_index(query_embedding, filter=filter)
        return results
    
    def research_and_summarize(self, query, filter=None, num_rounds=5):
        """
        embeds the query and queries the index for the top 5 results.
        then summarizes the results + comes up with a list of tangenital topics to research next. 
        takes the 3rd most similar topic to the last query and repeats the process for a total of 5 rounds.
        to get around compression limits, compress the summarization results.
        finally pass the topics and compressed summarizations for a final essay.
        """
        llm_35 = ChatOpenAI(model_name="gpt-3.5-turbo")
        llm_4 = ChatOpenAI(model_name="gpt-4")

        # define chains
        summary_template = SummaryTemplate(llm_35)
        tangenital_template = TangenitalIdeasTemplate(llm_35)
        compress_template = CompressTemplate(llm_4)
        essay_template = EssayTemplate(llm_4)
        
        intermediate_results = {}
        # loop with tqdm
        for i in tqdm(range(num_rounds), desc='Rounds'):
            results = {'retreived_context': [], 'summarized_results': [], 'queries': []}
            # embed and query
            retrieved_context = self.embed_and_query(query, filter=filter)
            # get text
            texts = [x['metadata']['text'] for x in retrieved_context['matches']]
            # get summaries
            summaries = run_parallel_chains(texts, summary_template)
            
            # add to results
            results['retreived_context'].append(retrieved_context)
            results['summarized_results'].append(summaries)
            results['queries'].append(query)
            intermediate_results[i] = results

            # get potential next topics and choose 3rd most similar to last query
            potential_next_topics = run_parallel_chains(summaries, tangenital_template)
            potential_next_topics = [x.lower().split('primary topic:')[1].strip() for x in potential_next_topics]

            sorted_potential_topics = sort_strings_by_similarity(query, potential_next_topics)
            query = sorted_potential_topics[2]
            logger.info(f'New query: {query}')
            if i < num_rounds-1:
                print(f'New query: {query}')
                print(f'Moving to round {i+2}')
        
        logger.info(f'moving to compression...')
        topic_context_dict = flatten_results(intermediate_results, compress_template)
        

        logger.info(f'moving to essay...')
        logger.info('note: will often get openai error, langchain should automatically retry...')
        final_essay = essay_template.process(topic_context_dict)

        result = {
            'intermediate_results': intermediate_results,
            'compressed_results': topic_context_dict,
            'final_essay': final_essay
        }

        return result
        