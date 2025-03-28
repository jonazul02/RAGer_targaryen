import json
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


def get_text_embedding(input: str, model) -> List:
    # sentences = [str(input)]
    # print('cuantas oraciones hay', len(sentences))
    embeddings_batch_response = model.encode(str(input))
    embeddings_batch_response = embeddings_batch_response.reshape(1, 384)
    return np.array(embeddings_batch_response)


def get_token_count(text: str, tokenizer) -> Tuple[List[int], int]:
    """
    Cuenta el número de tokens en el texto dado.

    Parameters:
    text (str): Texto a contar los tokens.

    Returns:
    int: Número de tokens en el texto.
    """

    # Crea una solicitud de tokenización para el texto
    request = ChatCompletionRequest(messages=[UserMessage(content=str(text))])
    tokenized = tokenizer.encode_chat_completion(request)
    tokens = tokenized.tokens
    return tokens, len(tokenized.tokens)


class Chunk_processing:
    def __init__(
        self,
        json_file,
        output_save_file,
        output_save_ids_file,
        output_save_emb_file,
        tokenizer,
        embedding_model,
        device,
        context_chunking=True,
    ):
        self.data = json_file
        self.save_file_path = output_save_file
        self.save_ids_file_path = output_save_ids_file
        self.save_embeddings_file_path = output_save_emb_file
        self.tokenizer = tokenizer
        self.model = embedding_model
        self.device = device
        self.flag = context_chunking
        pass

    def token_count(self, text: str) -> Tuple[List[int], int]:
        """
        Cuenta el número de tokens en el texto dado.

        Parameters:
        text (str): Texto a contar los tokens.

        Returns:
        int: Número de tokens en el texto.
        """

        # Crea una solicitud de tokenización para el texto
        request = ChatCompletionRequest(messages=[UserMessage(content=str(text))])
        tokenized = self.tokenizer.encode_chat_completion(request)
        tokens = tokenized.tokens
        return tokens, len(tokenized.tokens)

    def chunk_by_token_count(self, text: str, max_tokens=4000) -> List[str]:
        """
        Divide el texto en chunks basados en un límite máximo de tokens.

        Parameters:
        text (str): Texto a dividir en chunks.
        max_tokens (int): Límite de tokens por chunk.

        Returns:
        list: Lista de chunks de texto.
        """
        token_list, token_count = self.token_count(text)

        if token_count > max_tokens:
            print("Spliting in chunks...")

            chunks = []
            current_chunk = []

            for token in token_list:
                current_chunk.append(token)
                if len(current_chunk) == max_tokens:
                    chunk_text = (
                        self.tokenizer.decode(current_chunk)
                        .replace("[INST] ", "")
                        .replace(" [/INST]", "")
                    )
                    chunks.append(chunk_text)
                    current_chunk = []

            if current_chunk:
                chunks.append(
                    self.tokenizer.decode(current_chunk)
                    .replace("[INST] ", "")
                    .replace(" [/INST]", "")
                )

            # print('chunks in this document: ', len(chunks))

        else:
            # print('Is no necesary splitting into chunks')
            chunks = [text]

        return chunks

    def contextual_chunking(self, text: str, max_tokens: int) -> List[str]:
        """
        Realiza un chunkeo del texto basado en contexto, considerando un máximo de tokens.

        Parameters:
        text (str): Texto a dividir en chunks contextuales.
        max_tokens (int): Límite de tokens por chunk.

        Returns:
        list: Lista de chunks contextuales de texto.
        """

        labels = [
            "Dish name: ",
            "\n\n List of ingredients: ",
            "\n\nPreparation instructions: ",
            "\n\nImage description: ",
        ]
        chunks = []
        current_position = 0

        for label in labels:
            start = text.find(label, current_position)
            if start != -1:
                if current_position != start:
                    chunk = text[current_position:start].strip()
                    if self.token_count(chunk)[1] > max_tokens:
                        chunks.extend(self.chunk_by_token_count(chunk, max_tokens))
                    else:
                        chunks.append(chunk)
                current_position = start + len(label)

        if current_position < len(text):
            chunk = text[current_position:].strip().replace("-", " ")
            if self.token_count(chunk)[1] > max_tokens:
                chunks.extend(self.chunk_by_token_count(chunk, max_tokens))
            else:
                chunks.append(chunk)

        return chunks

    def create_embedding(self, chunks: str) -> List[float]:
        """
        Template para la creación de embeddings, implementado en otro script.

        Parameters:
        chunk (str): Chunk de texto para generar un embedding.

        Returns:
        List[float]: Embedding del chunk de texto o un template si no está implementado.
        """
        embeddings = self.model.encode(chunks, device=self.device)
        embeddings_array = np.array(embeddings)
        print("Saving embeddings into a npy file..", "\n\n")
        np.save(self.save_embeddings_file_path, embeddings_array)
        return print("Calculation of embeddings finished!!")

    def write_json_3(self, list_file: List):
        """
        This function saves a list into a json file.
        """
        with open(self.save_file_path, "w", encoding="utf-8") as f:
            json.dump(list_file, f, ensure_ascii=False, indent=4)
        print(f"File Created at {self.save_file_path}")

    def write_txt(self, list_file: List):
        """
        This functions saves a list into a txt file
        """
        with open(self.save_ids_file_path, "w") as file:
            for item in list_file:
                file.write(f"{item}\n")

    def create_chunking_pipeline(self):
        """ """
        new_list = []
        chunk_list = []
        doc_ids = []
        print("1- Creating chunks per recepie...", "\n\n")
        for doc_id, dic in tqdm(enumerate(self.data), desc="Chunking docs"):
            if self.flag:
                chunks = [
                    item
                    for sublist in self.contextual_chunking(
                        text=dic["Recepie"], max_tokens=4000
                    )
                    for item in (sublist if isinstance(sublist, list) else [sublist])
                ]
                dic["Chunks"] = chunks
            else:
                chunks = [
                    item
                    for sublist in self.chunk_by_token_count(
                        text=dic["Recepie"], max_tokens=4000
                    )
                    for item in (sublist if isinstance(sublist, list) else [sublist])
                ]
                dic["Chunks"] = chunks
            new_list.append(dic)
            for chunk in dic["Chunks"]:
                chunk_list.append(chunk)
                doc_ids.append(doc_id)
        print("Chunks created!", "\n")
        print("2- saving it into a new JSON file")
        self.write_json_3(new_list)
        self.write_txt(doc_ids)
        print("3- Calculating embeddings...", "\n\n")
        self.create_embedding(chunks=chunk_list)
        print("Pipeline finished!!!")
