import box
import glob
import os
import pandas as pd
import spacy
import yaml
from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SpacyTextSplitter
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.chains import SequentialChain, ConversationalRetrievalChain
from langchain.output_parsers.enum import EnumOutputParser
from langchain.chains import RetrievalQA
from enum import Enum
from datetime import datetime as dt
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
import logging
import pickle

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from dotenv import find_dotenv, load_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('../config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def read_data(file_path):
    df = pd.read_fwf(file_path, delimiter='||', header=None)
    df['datetime'] = df[0].str.split(' - ').str[0]
    df['temp'] = [a.replace(b + ' - ', '').strip() for a, b in zip(df[0], df['datetime'])]
    df['person'] = df.temp.str.split(':').str[0]
    df['content'] = [a.replace(b + ': ', '').strip() for a, b in zip(df['temp'], df['person'])]

    ## remove lame rows
    df = df.loc[df.content != '<Media omitted>']
    df = df.loc[~df.content.str.contains('http')]

    df = df.loc[df['person'] != df['content']]
    df['filter'] = df.apply(lambda x: x.person in x.datetime, axis=1)
    df = df.loc[~df['filter']]

    ## Format rows
    df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%dT%H:%M:%S')

    ## remove lame columns
    df.drop([0, 'temp', 'filter'], axis=1, inplace=True)

    return df


def process_text(folder_location, chunk_size=500):
    text_loader_kwargs = {'encoding': 'utf8'}

    dir_loader = DirectoryLoader(folder_location,glob='**/*.txt', show_progress=True, loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    # loader = TextLoader(file_location, encoding='utf8')
    documents = dir_loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, separator='\n')
    texts = text_splitter.split_documents(documents)
    return texts


def initiate_llm(model_name):
    llm = ChatOpenAI(
        model_name=model_name,  # 'gpt-3.5-turbo' or 'gpt-4'
        temperature=0,
        openai_api_key=os.environ['OPENAI_API_KEY'],
        max_tokens=cfg.TOKEN_LIMIT)

    return llm


def initiate_embeddings():
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    return embeddings


def initiate_docsearch(texts, embeddings, recreate=False):
    if texts is None:
        docsearch = Chroma(persist_directory=cfg.chroma_folder, embedding_function=embeddings)
    else:
        docsearch = Chroma.from_documents(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))],
                                          persist_directory=cfg.chroma_folder)
    if recreate:
        docsearch.persist()
    return docsearch.as_retriever()


def initiate_prompt(PROMPT):
    PROMPT_SUMMARY = PromptTemplate(template=cfg[PROMPT], input_variables=['context', 'question'])
    return PROMPT_SUMMARY


def initiate_memory():
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return memory


def initiate_all(PROMPT):
    logging.info('initiating llm')
    llm = initiate_llm('gpt-3.5-turbo')
    llm2 = initiate_llm('gpt-3.5-turbo')

    logging.info('initiating embeddings')
    embeddings = initiate_embeddings()

    print(os.path.exists('./' + cfg.chroma_folder))
    if os.path.exists('./' + cfg.chroma_folder):
        logging.info('initiating docsearch')
        docsearch = initiate_docsearch(None, embeddings)
    else:
        logging.info('docsearch does not exist, recreating one')

        logging.info('processing text')
        texts = process_text(cfg.folder_location)

        logging.info('initiating docsearch')
        docsearch = initiate_docsearch(texts, embeddings, recreate=True)

    logging.info('initiating prompt')
    PROMPT_SUMMARY = initiate_prompt(PROMPT)

    logging.info('initiating memory')
    memory = initiate_memory()

    logging.info('creating retrieval qa')
    # qa = RetrievalQA.from_chain_type(llm=llm,
    #                                  chain_type="stuff",
    #                                  retriever=docsearch,
    #                                  return_source_documents=True,
    #                                  chain_type_kwargs={
    #                                         "verbose": False,
    #                                         "prompt": PROMPT_SUMMARY
    #                                     })

    # llm_chain = LLMChain(llm=llm, prompt=PROMPT_SUMMARY)
    qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                               retriever=docsearch,
                                               return_source_documents=False,
                                               combine_docs_chain_kwargs={
                                                   "prompt": PROMPT_SUMMARY
                                               },
                                               condense_question_llm=llm2,
                                               memory=memory)
    return qa
