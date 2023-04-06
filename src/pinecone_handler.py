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

from chainer import (run_parallel_chains, run_parallel_comprehension, flatten_results,
                SummaryTemplate, TangenitalIdeasTemplate, EssayTemplate
                )


logger = logging.getLogger('pinecone_handler')

load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

def sort_strings_by_similarity(ref_string, strings_list):
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    
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
        return self.model.encode(query).tolist()
    
    def query_index(self, query_embedding, filter=None):
        if filter:
            results = self.index.query(query_embedding, top_k=5, includeMetadata=True, filter=filter)
        else:
            results = self.index.query(query_embedding, top_k=5, includeMetadata=True)
        print(f'Found {len(results["matches"])} results!')
        return results
    
    def embed_and_query(self, query, filter=None):
        logger.info(f'Processing query: {query}')
        query_embedding = self.get_embedding(query)
        results = self.query_index(query_embedding, filter=filter)
        return results
    
    def research_and_summarize(self, query, filter=None, num_rounds=5):
        llm_35 = ChatOpenAI(model_name="gpt-3.5-turbo")
        llm_4 = ChatOpenAI(model_name="gpt-4")

        # define chains
        summary_template = SummaryTemplate(llm_35)
        tangenital_template = TangenitalIdeasTemplate(llm_35)
        essay_template = EssayTemplate(llm_4)
        
        results_all = {}
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
            results_all[i] = results

            # get potential next topics and choose 3rd most similar to last query
            potential_next_topics = run_parallel_comprehension(summaries, tangenital_template)
            potential_next_topics = [x.lower().split('primary topic:')[1].strip() for x in potential_next_topics]

            sorted_potential_topics = sort_strings_by_similarity(query, potential_next_topics)
            query = sorted_potential_topics[2]
            logger.info(f'New query: {query}')
            if i < num_rounds-1:
                print(f'New query: {query}')
                print(f'Moving to round {i+2}')
        
        topic_context_dict = flatten_results(results_all)
        
        # try 3 times, if fail wait 10 seconds and try again
        print('Working on creating final essay...')
        tries = 3
        for i in range(tries):
            try:
                final_essay = essay_template.process(topic_context_dict)
                return final_essay
            except:
                time.sleep(10)
                print('Failed to create final essay, trying again...')
        
        print('All 3 tries failed, returning intermediate results.')
        return results_all

    
    # def process(self, query, filter=None, num_rounds=5):
    #     logger.info(f'Processing query: {query}')
    #     query_embedding = self.get_embedding(query)
    #     results = self.query_index(query_embedding, filter=filter)
    #     texts = extract_text_from_matches(results['matches'])

    #     intermediate_results = [results]
    #     summarized_results = []
    #     queries = [query]
    #     for round_num in range(num_rounds):
    #         logger.info(f'Round {round_num + 1}/{num_rounds}: Processing retrieved context...')
    #         with concurrent.futures.ThreadPoolExecutor() as executor:
    #             futures = [executor.submit(process_match, query, context, self.llm) for context in texts]
    #             distilled_insights = [future.result() for future in concurrent.futures.as_completed(futures)]
    #             summarized_results.extend(distilled_insights)
        
    #         # Generate new phrase based on the context and repeat the loop
    #         logger.info(f'Round {round_num + 1}/{num_rounds}: Generating new phrases...')
    #         query = generate_new_phrase(query, distilled_insights, self.llm)
    #         queries.append(query)
    #         logger.info(f'New query: {query}')
    #         print(f'New query: {query}')
            
    #         query_embedding = self.get_embedding(query)
            
    #         results = self.query_index(query_embedding, filter=filter)
    #         intermediate_results.append(results)
            
    #         texts = extract_text_from_matches(results['matches'])

    #     results_dict = {
    #         'intermediate_results': intermediate_results,
    #         'summarized_results': summarized_results,
    #         'queries': queries
    #     }

    #     return results_dict

        # # Process the matches using parallel execution
        # logger.info('Processing retrieved context...')
        # with concurrent.futures.ThreadPoolExecutor() as executor:

        #     with concurrent.futures.ThreadPoolExecutor() as executor:
        #         futures = [executor.submit(process_match, query, context, self.llm) for context in texts]
        #         distilled_insights = [future.result() for future in concurrent.futures.as_completed(futures)]

        # # Combine distillations
        # combined_result = combine_distillations(query, distilled_insights, self.llm)
        # return combined_result, results

def process_match(user_query, context, llm):
    multiple_input_prompt = PromptTemplate(
        input_variables=["user_query", "context"], 
        template="""Distill the key ideas, insights, or principles from the following context related \
        to the user's query in 3-5 sentences. \
        Include another 3 sentence paragraph on any interesting tangential ideas or topics provided in the context.
        User query: {user_query}

        Context: {context}
        """
    )

    chain = LLMChain(llm=llm, prompt=multiple_input_prompt)
    return chain.run({'user_query': user_query, 'context': context})

def generate_new_phrase(previous_user_query, context, llm):
    new_phrase_prompt = PromptTemplate(
        input_variables=["previous_user_query", "context"], 
        template="""Imagine youre conducting research for a blog post. \
            Based on the context, suggest a new but related topic, phrase, or question that \
            can be explored further. Make sure it doesn't repeat the same idea or phrase from the previous \
            user query.  Be creative yet descriptive with the associated query.\
            user query: {previous_user_query}
            Context: {context}
            """
    )

    chain = LLMChain(llm=llm, prompt=new_phrase_prompt)
    return chain.run({'previous_user_query': previous_user_query, 'context': context})


def extract_text_from_matches(matches):
    return [match['metadata']['text'] for match in matches]

def combine_distillations(user_query, distillations, llm):
    context = ' '.join(x for x in distillations)

    summary_prompt = PromptTemplate(
        input_variables=["user_query", "context"], 
        template="""Distill and summarize the key ideas, insights, or principles from the following \
        context related to the user's query.
        User query: {user_query}

        Context: {context}
        """
    )

    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    return summary_chain.run({'user_query': user_query, 'context': context})
