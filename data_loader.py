from langchain_community.document_loaders import PyPDFLoader
import re
import numpy as np
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
import shutil
import argparse

from get_embedding_function import get_embedding_function

DATA_PATH = "data/GDPR Art 1-21.pdf"
FAISS_PATH = "FAISS_db"

article_summaries = {
    1 : "**Article 1 - Subject-matter and objectives**: This article outlines the GDPR's purpose, which is to protect individuals' rights regarding their personal data and to regulate the processing of such data.",
    2 : "**Article 2 - Material scope**: Specifies the data processing activities that fall under the GDPR, including processing in the context of EU member states' activities, regardless of whether the processing occurs in the EU or not.",
    3 : "**Article 3 - Territorial scope**: Defines the geographical scope of the GDPR, applying to organizations based in the EU and those outside the EU that offer goods or services to, or monitor the behavior of, EU data subjects.",
    4 : "**Article 4 - Definitions**: Provides definitions for key terms used in the regulation, such as 'personal data', 'processing', 'controller', and 'processor'.",
    5 : "**Article 5 - Principles relating to processing of personal data**: Lists the core principles for processing personal data, including lawfulness, fairness, transparency, purpose limitation, data minimization, accuracy, storage limitation, integrity, and confidentiality.",
    6 : "**Article 6 - Lawfulness of processing**: Specifies the lawful bases for processing personal data, such as consent, contract necessity, legal obligation, vital interests, public task, and legitimate interests.",
    7 : "**Article 7 - Conditions for consent**: Details the conditions for obtaining valid consent from data subjects, emphasizing that consent must be freely given, specific, informed, and unambiguous.",
    8 : "**Article 8 - Conditions applicable to child's consent in relation to information society services**: Sets the age of consent for data processing related to information society services at 16, with the possibility for member states to lower it to no less than 13 years.",
    9 : "**Article 9 - Processing of special categories of personal data**: Prohibits processing of sensitive data (e.g., racial or ethnic origin, political opinions, religious beliefs) unless specific conditions are met, such as explicit consent or necessity for certain legal purposes.",
    10 : "**Article 10 - Processing of personal data relating to criminal convictions and offences**: States that processing personal data related to criminal convictions and offences requires a legal basis under EU or member state law.",
    11 : "**Article 11 - Processing which does not require identification**: Covers processing of data that doesn't require the identification of a data subject, setting limitations and obligations for controllers in such cases.",
    12 : "**Article 12 - Transparent information, communication and modalities for the exercise of the rights of the data subject**: Obligates controllers to provide information about data processing in a concise, transparent, and easily accessible form.",
    13 : "**Article 13 - Information to be provided where personal data are collected from the data subject**: Details the information that must be provided to data subjects when their data is collected directly, including the purpose of processing and the data retention period.",
    14 : "**Article 14 - Information to be provided where personal data have not been obtained from the data subject**: Specifies the information to be provided when data is not obtained directly from the data subject, including the source of the data.",
    15 : "**Article 15 - Right of access by the data subject**: Grants data subjects the right to access their personal data and obtain copies of it, along with other details about how and why their data is processed.",
    16 : "**Article 16 - Right to rectification**: Gives data subjects the right to have inaccurate personal data corrected and incomplete data completed.",
    17 : "**Article 17 - Right to erasure ('right to be forgotten')**: Allows data subjects to have their personal data erased under certain conditions, such as when the data is no longer necessary for its original purpose.",
    18 : "**Article 18 - Right to restriction of processing**: Provides data subjects the right to restrict processing of their data under certain circumstances, such as when the accuracy of the data is contested.",
    19 : "**Article 19 - Notification obligation regarding rectification or erasure of personal data or restriction of processing**: Requires controllers to notify all recipients of the data about any rectification, erasure, or restriction of processing, unless this proves impossible or involves disproportionate effort.",
    20 : "**Article 20 - Right to data portability**: Grants data subjects the right to receive their personal data in a structured, commonly used, and machine-readable format, and to transfer that data to another controller.",
    21 : "**Article 21 - Right to object**: Gives data subjects the right to object to the processing of their personal data based on certain grounds, including processing for direct marketing, research, or based on a public or legitimate interest."
}

def clear_database():
    if os.path.exists(FAISS_PATH):
        shutil.rmtree(FAISS_PATH)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true", help="Reproduces FAISS database of pdf info")
    args = parser.parse_args()

    if args.rerun:
        if os.path.exists(FAISS_PATH):
            print("Clearing FAISS Database...")
            clear_database()
            produce_and_save_db()
        else:
            print("No database present, producing FAISS database from provided pdf...")
            produce_and_save_db()
    else:
        if os.path.exists(FAISS_PATH):
            print("FAISS database already exists. To rerun analysis, provide argument --rerun")
        else:
            produce_and_save_db()

def produce_and_save_db():

    print("Loading pdf pages...")
    pages = load_pdf_pages()

    print("Extracting article numbers...")
    pages_with_article_number = extract_article_number(pages)

    print("Removing header and footer text from pages...")
    pages_cleaned = remove_header_footer_text(pages_with_article_number)

    print("Preparing article documents...")
    articles = produce_article_docs(pages_cleaned)

    print("Splitting article documents into chunks...")
    chunks = split_documents(articles)

    print("Retrieving embedding function...")
    embeddings = get_embedding_function()

    print("Preparing FAISS database from chunks, using embedding function...")
    db = FAISS.from_documents(chunks, embeddings)

    print("Saving database to local folder titled FAISS_db")
    db.save_local(folder_path="FAISS_db")

def load_pdf_pages():
    loader = PyPDFLoader(DATA_PATH)
    return loader.load()

def extract_article_number(pages: list[Document]):
    article_number = 0
    for page in pages:
        if page.page_content[:10] == "EN\nArticle":
            article_number += 1
        page.metadata["article_number"] = article_number
    return pages

def remove_header_footer_text(pages: list[Document]):
    header_text = 'www.gdpr-text.com/en'

    footer_text1 = 'www.data-privacy-\noffice.eu\nwww.gdpr-text.cominfo@data-privacy-\noffice.eu'
    footer_text2 = '\nGDPR training, consulting and DPO outsourcing'
    footer_text3 = r'page \d+ / \d+'

    for page in pages:
        full_text = page.page_content

        removed_header = full_text.replace(header_text, "").strip()

        removed_footer1 = removed_header.replace(footer_text1, "").strip()
        removed_footer2 = removed_footer1.replace(footer_text2, "").strip()
        removed_footer3 = re.sub(footer_text3, '', removed_footer2)

        page.page_content = removed_footer3

    return pages

def produce_article_docs(pages: list[Document], article_summaries = article_summaries):
    article_documents = []
    for article_number in np.arange(1, 22):
        full_text = "\n".join([page.page_content for page in pages if page.metadata['article_number'] == article_number])
        article_summary = article_summaries[article_number]

        doc = Document(
            page_content = full_text,
            metadata = {
                "article_number" : article_number,
                "article_summary" : article_summary
            }
        )

        article_documents.append(doc)

    return article_documents

def split_documents(article_documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(article_documents)
    chunks = add_chunk_ids(chunks)
    return chunks

def add_chunk_ids(chunks: list[Document]):
    last_chunk_article_number = None
    for chunk in chunks:
        current_chunk_article_number = chunk.metadata["article_number"]

        if current_chunk_article_number == last_chunk_article_number:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        current_chunk_id = f"{current_chunk_article_number}:{current_chunk_index}"
        last_chunk_article_number = current_chunk_article_number
        
        chunk.metadata["id"] = current_chunk_id
    return chunks

if __name__ == "__main__":
    main()
