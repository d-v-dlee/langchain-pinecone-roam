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
    """helper function used to run multiple chains in parallel"""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(XTemplate.process, data) for data in input_data]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    return results

# def run_parallel_comprehension(texts, XTemplate):
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = [executor.submit(XTemplate.process, text) for text in texts]
#         results = [future.result() for future in concurrent.futures.as_completed(futures)]
#     return results

def compress_text(text, compress_template):
    context = {'text': text}
    compressed_text = compress_template.process(context)
    return compressed_text

def flatten_results(results, compress_template):
    """helper function used to flatten results from multiple rounds of research and summarize"""
    topics = []
    context = []

    for k, v in results.items():
        topics.append(v['queries'][0])
        context.append(' '.join(v['summarized_results'][0]))

    # Compress summaries individually
    print('Compressing context...')
    compressed_summaries = [compress_text(summary, compress_template) for summary in context]

    # Join topics and context with numbered points
    flattened_topics = '\n'.join(f"{i+1}. {topic}" for i, topic in enumerate(topics))
    flattened_context = '\n'.join(f"{i+1}. {summary}" for i, summary in enumerate(compressed_summaries))

    compressed_flattened_results = {
        'topics': flattened_topics,
        'context': flattened_context
    }

    return compressed_flattened_results

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

        self.template_string = """Examine the context below and extract 5 themes or subjects that can establish connections with other relevant ideas or topics. Strive for a blend of expected and creative connections, with a focus on semantic search compatibility. Once you've identified the 7 themes, suggest the primary topic for your next research endeavor.
        Return the final topic like: 'Primary topic: <your answer>
        
        Context: {context}"""
        super().__init__(llm, self.input_variables, self.template_string)

class CompressTemplate(BasePromptTemplate):
    """
    Template for getting a compressed version of a context.
    """
    def __init__(self, llm):
        self.input_variables = ['text']

        self.template_string = """
        compress the following text in a way that is lossless but results in the minimum number of tokens which could be fed into an LLM like yourself as-is and produce the same output. feel free to use multiple languages, symbols, other up-front priming to lay down rules. this is entirely for yourself to recover and proceed from with the same conceptual priming, not for humans to decompress:
        
        Text: {text}
        """
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