from langchain.document_loaders import DirectoryLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer

import pandas as pd
import os

# Remove file paths
index_path = "/Users/suryaganesan/Documents/GitHub/im_rag/im_rag/im_rag/embeddings/faiss_index"
corpus_path = "/Users/suryaganesan/Documents/GitHub/im_rag/im_rag/eval_utils/corpus.xlsx"
path_checks = [index_path, corpus_path]

queries_path = "/Users/suryaganesan/Documents/GitHub/im_rag/im_rag/eval_utils/queries.xlsx"

for file_path in path_checks:
    if os.path.exists(file_path):
        os.remove(file_path)

## Load documents

source_path = "/Users/suryaganesan/Documents/GitHub/im_rag/im_rag/RAG docs/"
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

## Embed and export text_chunks to faiss_index - For im_rag retrieval

print("Embedding chunks and exporting to faiss...")

embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
path = "/Users/suryaganesan/Documents/GitHub/im_rag/im_rag/embeddings/"

db = FAISS.from_documents(text_chunks, embeddings)
db.save_local(path+'faiss_index')

## Embed and export text_chunks to excel - For inference
print("Exporting corpus to excel...")

corpus_df = pd.DataFrame()
corpus_df["text"] = [p.page_content for p in text_chunks]


with pd.ExcelWriter(corpus_path) as writer:
    corpus_df.to_excel(writer, sheet_name="vector", index=False)

print("...Program terminated")