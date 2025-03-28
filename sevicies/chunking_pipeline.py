import torch
from utilities.configure_script import ConfigurationLoader
from utilities.chunk_utils import Chunk_processing
from utilities.utils import read_json
from sentence_transformers import SentenceTransformer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

def main():

    print(torch.cuda.is_available())

    configs = ConfigurationLoader.get_config()

    recepies = read_json(configs['paths']['recepies_path']['recepies_json'])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = MistralTokenizer.v3()
    model_name = "open-mistral-7b"
    tokenizer = MistralTokenizer.from_model(model_name)

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

    chunker = Chunk_processing(recepies,
                            configs['paths']['recepies_path']['recepies_chunked_json'],
                            configs['paths']['recepies_path']['recepies_doc_ids'],
                            configs['paths']['recepies_path']['recepies_embeddings'],
                            tokenizer,
                            model,
                            device,
                            context_chunking = True)

    chunker.create_chunking_pipeline()

if __name__ == "__main__":

    main()