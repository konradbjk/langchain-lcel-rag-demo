import os
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from bs4 import BeautifulSoup
from pprint import pprint
from dotenv import load_dotenv

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain import hub

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

## Load keys from .env file
load_dotenv()

## One URL or a list of URLs
url = 'https://www.svpg.com/product-model-at-amazon/'

azure_embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("https://langchain-books.openai.azure.com/"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)
    
llm = AzureChatOpenAI(
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0,
    streaming=True
)



def extract_article(article):
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        # article, tags_to_extract=["p", "li", "ul", "div", "a"]
        article, tags_to_extract=["article"]
    )
  
    return docs_transformed

def fetch_article(url):
    print(url)
    try:
        loader = AsyncHtmlLoader([url])
        html = loader.load() # returns a list
        ## Compare regular BS4 output with Document Transformer
        # soup = BeautifulSoup(html[0].page_content, 'html.parser')
        # article = soup.find('article')
        # text1 = article.get_text()
        text2 = extract_article(html)
    except IndexError:
        print(f"Problem fetching page content")
        
    # return text1, text2
    return text2
    
    
# contetn1, content2 = fetch_article(url)
content2 = fetch_article(url)

## Compare Content1 and content2
# pprint(f"content1: \n{content1} \n\n\n\n content2: \n{content2[0].page_content}")

## play around with values on 
## https://langchain-text-splitter.streamlit.app/
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

splits = splitter.split_documents(content2)

vectorstore = Chroma.from_documents(documents=splits, embedding=azure_embeddings)
retriever = vectorstore.as_retriever()

## URL to this prompt: https://smith.langchain.com/hub/rlm/rag-prompt
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

## Ask our Chain a question
response = rag_chain.invoke("Is amazon faster than most startups?")
print(response)


## If we want to stream output instead of waiting
# for chunk in rag_chain.stream("Is amazon faster than most startups?"):
#     print(chunk, flush=True)