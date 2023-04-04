import os
import concurrent.futures
import logging

import pinecone
from sentence_transformers import SentenceTransformer
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

logger = logging.getLogger('pinecone_handler')

PINECONE_API_KEY = ''
os.environ["OPENAI_API_KEY"] = ''

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

class PineconeHandler:
    def __init__(self):
        pinecone.init(api_key=PINECONE_API_KEY, environment='eu-west1-gcp')
        self.index = pinecone.Index('roam-research')
        self.model = model
        self.llm = ChatOpenAI(model_name="gpt-4")


    def get_embedding(self, query):
        return self.model.encode(query).tolist()
    
    def query_index(self, query_embedding, filter=None):
        if filter:
            results = self.index.query(query_embedding, top_k=5, includeMetadata=True, filter=filter)
        else:
            results = self.index.query(query_embedding, top_k=5, includeMetadata=True)
        print(f'Found {len(results["matches"])} results!')
        return results
    
    def process(self, query, filter=None):
        logger.info(f'Processing query: {query}')
        query_embedding = self.get_embedding(query)
        results = self.query_index(query_embedding, filter=filter)
        texts = extract_text_from_matches(results['matches'])

        # Process the matches using parallel execution
        logger.info('Processing retrieved context...')
        with concurrent.futures.ThreadPoolExecutor() as executor:

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_match, query, context, self.llm) for context in texts]
                distilled_insights = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Combine distillations
        combined_result = combine_distillations(query, distilled_insights, self.llm)
        return combined_result, results

def process_match(user_query, context, llm):
    multiple_input_prompt = PromptTemplate(
        input_variables=["user_query", "context"], 
        template="""Distill the key ideas, insights, or principles from the following context related \
        to the user's query in 3-5 sentences.
        User query: {user_query}

        Context: {context}
        """
    )

    chain = LLMChain(llm=llm, prompt=multiple_input_prompt)
    return chain.run({'user_query': user_query, 'context': context})


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
