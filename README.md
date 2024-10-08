# Applying RAG to query GDPR PDF

This python project harnesses the capabilities of RAG through open-source embeddings from HuggingFace and FAISS vector store, and the text generative abilities of open source models from Ollama, to allow user's to query the pdf containing information on GDPR Articles 1 - 21.

## Setting up your environment

To create and activate a new virtual environment for this project, run the following commands in your command prompt

```console
python -m venv .venv
.venv\Scripts\activate
```

When your new environment is active, download the relevant python libraries using the ```.requirements.txt ``` and [PIP](https://pip.pypa.io/en/stable/installation/), by running the following command

```console
pip install -r requirements.txt
```

## Downloading Ollama

This project utilises Ollama LLMs running locally. 

To download Ollama, follow this [link](https://ollama.com/download/windows) and the follow the download instructions.

When Ollama is downloaded, you will need to pull the LLM that we are using in this project. Specifically, ```mistel```. To do this, run the following command

```console
ollama pull mistral
```

## Usage

The GDPR pdf is stored inside ```./data```.

Running ```python data_loader.py``` from the root of this project folder will load the pdf into a list of documents, with the corresponding article number and article summary within the document's metadata. The script will then embed these documents and store the results as a FAISS vector field in a folder titled ```FAISS_db```.

Following the creation of ```FAISS_db```, running ```python query_data.py "user's query"``` will construct a prompt, containing the user's query and the top 5 documents related to the user's query, and pass this prompt on to the mistral LLM from Ollama. Time taken for this script to run will vary depending on hardware available.

## Example Usage

#### Consider the following query:

```What is meant by a person's right to deletion?```

#### To answer this query using this project, follow the below steps...

If you have not yet created the ```FAISS_db``` vector store folder, run the following command:

```console
python data_loader.py
```

Note: If you have already created the ```FAISS_db``` folder, but wish to re-create it, run the same command with ```--rerun``` argument, i.e.

```console
python data_loader.py --rerun
```

Then, run the ```query_data.py``` script along with the query, i.e.

```console
python query_data.py "What is meant by a person's right to deletion?"
```

This will first print the prompt that will be fed into the mistral LLM from Ollama (which will include the contents from the top 5 most relevent documents to the query), before printing the response from the LLM.

For example, here is the prompt that will be fed into the LLM after retrieving the top 5 most relevant documents...

```console
 ---
 Prompt:

Human:
Answer the question based only on the following context:

they can neither be processed in any other manner nor modified.
The right is exercised in cases where there is no clear indication of whether
personal data will be deleted on a precise legal ground or when. It is
especially useful when the right to erasure (article 17) cannot be invoked
immediately, if the organization has a legal obligation to retain the data, for
example.
A person has a limited right to restrict the processing of her/his data under
four scenarios:
if the accuracy of the concerned data is challenged;
if the processing of the data was unlawful but the person opposed to
their deleting;
if the data are needed to establish, exercise or fight a legal claim; or
if a person exercised her/his right to object (article 21) but the

---

EN
Article 17.
Right to erasure (‘right to be forgotten’)
Article 17.
1.The data subject shall have the right to obtain from the controller the erasure of personal data concerning him or
her without undue delay and the controller shall have the obligation to erase personal data without undue delay 
where one of the following grounds applies:
(a)the personal data are no longer necessary in relation to the purposes for which they were collected or otherwise
processed;
(b)the data subject withdraws consent on which the processing is based according to point (a) of Article 6(1), or
point (a) of Article 9(2), and where there is no other legal ground for the processing;
(c)the data subject objects to the processing pursuant to Article 21(1) and there are no overriding legitimate  

---

or historical research purposes or statistical purposes, or for the establishment, exercise or defence of legal 
claims.
(66)To strengthen the right to be forgotten in the online environment, the right to erasure should also be extended in
such a way that a controller who has made the personal data public should be obliged to inform the controllers which
are processing such personal data to erase any links to, or copies or replications of those personal data. In doing so,
that controller should take reasonable steps, taking into account available technology and the means available to the
controller, including technical measures, to inform the controllers which are processing the personal data of the data
subject's request.
Guidelines & Case Law
Documents

---

well as Article 9(3);
(d)for archiving purposes in the public interest, scientific or historical research purposes or statistical purposes in
accordance with Article 89(1) in so far as the right referred to in paragraph 1 is likely to render impossible or
seriously impair the achievement of the objectives of that processing; or
(e) for the establishment, exercise or defence of legal claims.
General Data Protection Regulation (EU GDPR)
The latest consolidated version of the Regulation with corrections by Corrigendum, OJ L 127, 23.5.2018, p. 2    
((EU) 2016/679). Source: EUR-lex.
Related information Article 17. Right to erasure (‘right to be
forgotten’)
Recitals
(65)A data subject should have the right to have personal data concerning him or her rectified and a ‘right to be

---

(a)the period for which the personal data will be stored, or if that is not possible, the criteria used to determine
that period;
(b)where the processing is based on point (f) of Article 6(1), the legitimate interests pursued by the controller or
by a third party;
(c)the existence of the right to request from the controller access to and rectification or erasure of personal 
data or
restriction of processing concerning the data subject and to object to processing as well as the right to data  
portability;
(d)where processing is based on point (a) of Article 6(1) or point (a) of Article 9(2), the existence of the right to
withdraw consent at any time, without affecting the lawfulness of processing based on consent before its        
withdrawal;

---

Answer the question based on the above context: What is meant by a persons right to deletion?
```

And here is the response back from the LLM...

```console
Response:  In the given context, a person's "right to deletion" refers to their right to have their personal data erased from the controller's database under certain conditions. This right is often referred to as the "right 
to be forgotten." However, this right can only be exercised when there is no clear legal ground or time for the 
deletion of the personal data, such as in cases where the organization has a legal obligation to retain the data. The right to deletion may be limited by factors like archiving purposes, scientific or historical research, or establishing, exercising, or defending legal claims.
Sources: ['18:3', '17:0', '17:6', '17:3', '14:2']
```

Where the sources are the unique IDs of the 5 most relevent documents - where the first number represents the article number, and the second is the chunk number within that article (e.g. '18:3' refers to the 3rd document 'chunk' from article 18)
