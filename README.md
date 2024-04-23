## Prerequisite

Python: **3.9** or **3.10**
Suggested virtual environment manager: **conda**
Possible venv: pottery

### Packages
```
pip install --upgrade --quiet dotenv langchain langchain-core langchain-community langchainhub langchain-openai langchain-chroma bs4
```

### env file
Please follow the `.env.template` to supply parameters. I used Azure OpenAI, can bea easily ported regular OpenAI (make sure to have credits on their platform - ChatGPT Plus subscription is not enough). Can be also switched to different models - just modify the `llm` and optionally `azure_embeddings`.

## Instruction
During this demo, we want to build a smiple RAG, using Langchain. 
1. Fetching articles from Marty Cagan's blog. Can be list of articles or just one
2. Process the HTML to contain only text
3. Split the content into smaller chunks
4. Embedd chunks and save into Chroma vector store
5. Turn vector store into a retriever
6. Pull prompt template from LangSmith Hub
7. Build a simple RAG chain using LCEL
8. Ask our chain some questions about the articles (you can either use invoke, or stream)