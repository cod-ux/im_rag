from langchain.document_loaders import DirectoryLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings


## Load documents

source_path = "/Users/suryaganesan/vscode/ml/learning/Gen AI/RAG docs/"
md_loader = DirectoryLoader(source_path+'md')

md_documents = md_loader.load()


print("No. of docs: ", len(md_documents))

####### Split documents

mark_down_splitter = MarkdownHeaderTextSplitter(headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
])

content = [doc.page_content for doc in md_documents]
seperator = " \n\n"
content = seperator.join(content)

text_chunks = mark_down_splitter.split_text(content)

print("No. of text, chunks: ", len(text_chunks))

#### Embeddings and db

embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

db = FAISS.from_documents(text_chunks, embeddings)
db.save_local('faiss_index')

print(f'Saved {len(text_chunks)} chunks')
