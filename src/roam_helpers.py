# This file containers helpers that can: 1) find the latest Roam export 2) clean it 3) compare it with the previous export 4) return the differences
# Python Built-Ins:
import os
import logging
from glob import glob
import re
import shutil

# External Dependencies:
import markdown
from bs4 import BeautifulSoup
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Internal Dependencies:

# Instantiate logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)



def find_latest_export(files):
    """
    this function finds the lateset export.
    the files are in the format: roam_export_MM-DD-YY.zip or MM-DD-YYYY.zip
    """
    # assumes the files are in the format: roam_export_MM-DD-YY.zip or MM-DD-YYYY.zip
    dates = [x.split('/')[-1].split('_')[-1].split('.')[0] for x in files]

    # Sort the dates in descending order
    sorted_dates = sorted(dates, reverse=True, key=lambda x: (int(x.split('-')[2]), int(x.split('-')[0]), int(x.split('-')[1])))

    # Take the first two elements of the sorted list
    latest_date = sorted_dates[0]
    logger.info(f'The latest date is {latest_date}')

    # Find the index of latest date
    index = dates.index(latest_date)
    # indices = [dates.index(date) for date in top_dates]

    latest_zip_file = files[index]
    return latest_zip_file, latest_date

class RoamUnpacker():
    """
    this class finds the latest Roam export zip and unzips it. it returns a temporary path to the unzipped files.
    """
    def __init__(self, path='../data/roam_exports/zip_files'):
        self._path = path
    
    def find_latest_export(self):
        """
        find the latest export
        """
        # load files
        if self._path[-1] != '/':
            self._path += '/'
        files = sorted(glob(self._path + "*.zip"))
        logger.info(f'Number of zip files: {len(files)}')

        # find latest export
        latest_zip, latest_date = find_latest_export(files)
        logger.info(f'Latest zip file: {latest_zip} from {latest_date}')
        
        # create temp directory to unzip files
        temp_dir = self._path + f'temp_{latest_date}'
        os.system(f'mkdir -p {temp_dir}')
        os.system(f'unzip {latest_zip} -d {temp_dir}')
        logger.info(f'Unzipped files to {temp_dir}')
    
        return temp_dir

def find_highlight_files(directory):
    """
    Find files with "highlights" in their name and that end with ".md" in the specified directory.
    Args:
        directory (str): The directory to search for highlight files.
    Returns:
        list: A list of paths to the highlight files found in the specified directory.
    """
    # find files with "highlights" in their name and that end with ".md"
    files = glob(directory + '/*highlights*.md') + glob(directory + '/*Literature Notes*.md')
    print(len(files))
    return files

def read_markdown_file(file_path):
    """
    Read the contents of a Markdown file and return the text.
    Args:
        file_path (str): The path to the Markdown file.
    Returns:
        str: The text in the Markdown file.
    """
    # read the contents of the Markdown file
    with open(file_path, 'r') as f:
        markdown_text = f.read()

    # convert the Markdown text to plain text
    text = markdown.markdown(markdown_text)

    # return the plain text
    return text

def convert_markdown_to_text(markdown_text):
    """
    Convert Markdown text to plain text.
    Args:
        markdown_text (str): The Markdown text to convert.
    Returns:
        str: The plain text version of the Markdown text.
    """
    # convert the Markdown text to HTML
    html = markdown.markdown(markdown_text)

    # parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    # extract the text from the HTML, with indentation
    text = soup.get_text(separator='\n', strip=True)

    # return the plain text with indentation
    return text

def extract_metadata(text):
    """
    Extract the metadata (author, title, and category) from a string of text.
    Args:
        text (str): The text to extract the metadata from.
    Returns:
        tuple: A tuple containing the author, title, and category extracted from the text.
    """
    # define the regular expressions for the author, title, and category
    regex_author = re.compile(r'Author:: \[\[(.+?)\]\]')
    regex_title = re.compile(r'Full Title:: (.+?)\n')
    regex_category = re.compile(r'Category:: #(.+?)\n')

    # extract the author, title, and category from the text using the regular expressions
    author = regex_author.search(text).group(1)
    title = regex_title.search(text).group(1)
    category = regex_category.search(text).group(1)

    # return the extracted metadata
    return author, title, category

import unicodedata

def decode_string(s):
    ascii_string = s.encode('ascii', 'ignore')
    unicode_string = ascii_string.decode('ascii')
    return unicode_string

def extract_text_lines(text):
    """
    Extract the lines of text from a string of text.
    Args:
        text (str): The text to extract the lines from.
    Returns:
        list: A list of lines extracted from the text.
    """
    split_text = text.split('\n')

    start = False
    extracted_text = []
    for indx, line in enumerate(split_text):
        line = decode_string(line)
        # skip first few lines
        if 'Highlights first synced' in line:
            start = True
            continue
            
        if start:
            if len(line) > 13 and line[:20] != 'New highlights added' and line[0] != ':':
                line = line.replace('(', '').rstrip().lstrip()
                extracted_text.append(line)
    return extracted_text    


def process_md_file(file_path):
    """
    Process a Markdown file and extract its metadata and text.
    Args:
        file_path (str): The path to the Markdown file to process.
    Returns:
        dict: A dictionary containing the metadata and text extracted from the Markdown file.
    """
    # read the Markdown file
    text = read_markdown_file(file_path)
    # convert the Markdown text to plain text
    text = convert_markdown_to_text(text)

    # extract the metadata
    author, title, category = extract_metadata(text)
    extracted_text = extract_text_lines(text)

    # return the plain text
    return {'author': author, 'title': title, 'category': category, 'extracted_text': extracted_text}

def split_text_langchain(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, 
                                                   chunk_overlap=50)
    texts_recursive = text_splitter.split_text(text)
    print(len(texts_recursive)) 

    return texts_recursive

def split_text(strings, n=1000):
    """
    Split a list of strings into a list of strings with n words or fewer.
    """
    # Initialize a list to store the concatenated strings
    concatenated_strings = []

    # Initialize a string to store the current concatenated string
    current_string = ""

    # Loop over the strings
    for string in strings:
        # Split the string into words
        words = string.split(" ")

        # If the current string has fewer than n words, add the new string to it
        if len(current_string.split(" ")) + len(words) <= n:
            current_string += " " + string
        else:
            # If the current string has n or more words, add it to the list and reset the current string
            concatenated_strings.append(current_string.strip())
            current_string = string

    # Add the final concatenated string to the list
    concatenated_strings.append(current_string.strip())

    return concatenated_strings

def create_dataframe(files_cleaned):
    """
    Create a Pandas DataFrame from a list of dictionaries containing the metadata and text of the files.
    """

    doc_ids = []
    titles = []
    authors = []
    categories = []
    texts = []
    for indx, meta_dict in enumerate(files_cleaned):
        author = meta_dict['author']
        title = meta_dict['title']
        category = meta_dict['category']
        
        text_chunks = split_text_langchain(meta_dict['extracted_text'])
        doc_id = 1 # this is to keep track of which split it is within the document
        for text in text_chunks:
            
            doc_ids.append(doc_id)
            titles.append(title)
            authors.append(author)
            categories.append(category)
            texts.append(text)

            doc_id += 1
    
    df =  pd.DataFrame({
        'Doc Split': doc_ids, 
        'Title': titles,
        'Author': authors,
        'Category': categories,
        'Text': texts
        })
    
    df['Word Count'] = df['Text'].apply(lambda x: len(x.split(' ')))
    
    return df

class RoamCleaner():
    def __init__(self, temp_dir, save_path='../data/roam_exports/extracted'):
        self._temp_dir = temp_dir
        if save_path[-1] == '/':
            save_path = save_path[:-1]
        self._save_path = save_path
        self._date = self._temp_dir.split('temp_')[-1]
    
    def process_and_clean(self):
        # find highlight files
        highlight_files = find_highlight_files(self._temp_dir)
        logger.info(f'Number of highlight files: {len(highlight_files)}')

        # filter out tweets
        filtered_files = []
        for filepath in highlight_files:
            try:
                # include tweets
                meta = process_md_file(filepath)
                filtered_files.append(meta)

                # if 'tweets' not in meta['category']:
                #     filtered_files.append(meta)
            except:
                logger.error(f'error with {filepath}')

        logger.info(f'Number of highlight files: {len(filtered_files)}')
        # logger.info(f'Number of highlight files after filtering out tweets: {len(filtered_files)}')

        # split text into 1,000 word chunks and then convert into dataframe
        df = create_dataframe(filtered_files)

        # save df with timestamp
        save_path = f'{self._save_path}/roam_highlights_{self._date}.csv'
        df.to_csv(save_path, index=False)
        logger.info(f'Saved to {save_path}!')

        # delete temp directory
        shutil.rmtree(self._temp_dir)

        return df

# next class - check directory for saved dfs, find the latest
# compare for rows that are different, using something like df[['Doc Split', 'Title', 'Word Count']] 


        