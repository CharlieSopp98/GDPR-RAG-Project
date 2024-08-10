from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function():

    modelPath = "sentence-transformers/all-MiniLM-l6-v2"
    model_kwargs = {'tokenizer_kwargs' : {'clean_up_tokenization_spaces' : True}}

    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,
        model_kwargs = model_kwargs
    )

    return embeddings