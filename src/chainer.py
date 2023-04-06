import os
import concurrent.futures

from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


llm = ChatOpenAI(model_name="gpt-3.5-turbo")

def run_parallel_chains(input_data, XTemplate):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(XTemplate.process, data) for data in input_data]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    return results

def run_parallel_comprehension(texts, XTemplate):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(XTemplate.process, text) for text in texts]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    return results

def flatten_results(results):
    """helper function used to flatten results from multiple rounds of research and summarize"""
    topics = []
    context = []

    for k, v in results.items():
        topics.append(v['queries'][0])
        context.append(' '.join(v['summarized_results'][0]))

    # Join topics and context with numbered points
    flattened_results = {
        'topics': '\n'.join(f"{i+1}. {topic}" for i, topic in enumerate(topics)),
        'context': '\n'.join(f"{i+1}. {summary}" for i, summary in enumerate(context))
    }
    return flattened_results

class BasePromptTemplate:
    def __init__(self, llm, input_variables, template):
        self.llm = llm

        self.input_variables = input_variables
        self.template = template
        self.prompt = PromptTemplate(input_variables=self.input_variables, template=self.template)
    
    def process(self, input_data):
        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        return chain.run(input_data)

class SummaryTemplate(BasePromptTemplate):
    """
    Template for getting a TL;DR of a context.
    """
    def __init__(self, llm):
        self.input_variables = ['context']

        self.template_string = """In 1-3 sentences, provide short, concise TL;DR of the following context:

        Context: {context}
        """
        super().__init__(llm, self.input_variables, self.template_string)

class TangenitalIdeasTemplate(BasePromptTemplate):
    """
    Template for getting the tangential ideas of a context.
    """
    def __init__(self, llm):
        self.input_variables = ['context']

        # self.template_string = """Imagine you're writing a blog.
        # If you had to tag the following text with 7 topics that could then be connected to other passages or ideas, 
        # what would they be? Think step by step. Produce a mix of logical and surprising topics. 
        # The wording should be optimized for semantic search. Once you have shared the 7 topics, share one final topic that you think you should research next:

        # Context: {context}
        # """

        self.template_string = """Examine the context below and extract 7 themes or subjects that can establish connections with other relevant ideas or topics. Strive for a blend of expected and creative connections, with a focus on semantic search compatibility. Once you've identified the 7 themes, suggest the primary topic for your next research endeavor.
        Return the final topic like: 'Primary topic: <your answer>
        
        Context: {context}"""
        super().__init__(llm, self.input_variables, self.template_string)

class EssayTemplate(BasePromptTemplate):
    """
    Template for getting an essay on a context.
    """
    def __init__(self, llm):
        self.input_variables = ['topics', 'context']

        self.template_string = """Using the provided topics and context, create an engaging and well-structured essay. 
        The essay should cover key insights, notable examples, and any potential implications of the provided topics and context. 
        Use smooth transitions between paragraphs. Make each paragraph 3-5 sentences long each.


        Topics: {topics}

        Context: {context}
        """
        super().__init__(llm, self.input_variables, self.template_string)