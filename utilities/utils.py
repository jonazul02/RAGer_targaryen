import os
import json
import ast
import base64
from typing import List, Dict
import pdfplumber
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
from tqdm import tqdm
from utilities.images2text import generate_image_description


def read_json(file_path)->List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file, strict=False)
    return data

def read_txt(file_path):
    with open(file_path, 'r') as file:
        my_list = [int(line.strip()) for line in file]
    return my_list

class recepies_preprocessing:
    '''
    The objective of this class is to format and transform csv and images in a folder into
    json files with the text and create a complete dataset.
    '''
    def __init__(self, input_dataframe_path, input_image_folder, output_save_file, processor, model, device):
        '''
        input_dataframe_path: path to csv file
        input_image_folder: path to directory of images
        output_save_file: path where the json file will be saved
        processor: in charge to decode the description by the model  
        model: model to transforms images to text descriptions
        device: main device cuda or cpu
        '''
        self.dataframe_path = input_dataframe_path
        self.img_folder_path = input_image_folder
        self.save_file_path = output_save_file
        self.processor_decoder = processor
        self.img2text_model = model
        self.device = device

    def df_processing_0(self) -> pd.DataFrame:
        '''
        This function helps to preprocess the dataframe and save only necessary rows.
        '''
        dataframe = pd.read_csv(self.dataframe_path, usecols=['Title', 'Ingredients', 'Instructions', 'Image_Name'])
        # Convertir los ingredientes de string a lista
        dataframe['Ingredients'] = dataframe['Ingredients'].apply(lambda x: ast.literal_eval(x))
        images = os.listdir(self.img_folder_path)
        images_processed = [img.replace('.jpg', '') for img in images]
        # Filtrar filas que tienen imágenes correspondientes
        final_df = dataframe[dataframe['Image_Name'].isin(images_processed)].copy()
        final_df['Image_paths'] = final_df['Image_Name'].apply(lambda name: os.path.join(self.img_folder_path, name +'.jpg'))
        list_descriptions = []
        for index, row in tqdm(final_df.iterrows(), desc='Creating images descriptions'):
            dish = row['Image_Name']
            image = Image.open(row['Image_paths']).convert("RGB")
            description = generate_image_description(image, self.processor_decoder, self.img2text_model, self.device)
            list_descriptions.append(dish.replace('-',' ') + ' ' + description)
        final_df['Image_descriptions'] = list_descriptions
        return final_df
    
    def create_text_1(self, title_string: str, ingredients_list: List[str], instructions_string: str, img_description_string: str) -> str:
        '''
        This function converts the three columns into a single string text to process into a json,
        which helps to create the text database.
        '''
        title_string = str(title_string) if title_string is not None else "No Title"
        ingredients_list = ingredients_list if isinstance(ingredients_list, list) else ["No ingredients available"]
        instructions_string = str(instructions_string) if instructions_string is not None else "No instructions available"

        dish_name = 'Dish name: ' + title_string + ' \n\n '
        ingredients_string = dish_name + 'List of ingredients: \n• ' + " \n• ".join(ingredients_list)  # Add bullets
        ingredients_string = ingredients_string + " \n\n"
        semifinal_text = ingredients_string + 'Preparation instructions: ' + instructions_string
        final_text = semifinal_text + '\n\n' + 'Image description: ' + img_description_string

        return final_text

    def encode_image_2(self, image_path: str) -> str:
        '''
        This function converts the image into a base64 representation.
        '''
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def write_json_3(self, list_file: List):
        '''
        This function saves a list into a json file.
        '''
        with open(self.save_file_path, 'w', encoding='utf-8') as f:
            json.dump(list_file, f, ensure_ascii=False, indent=4)
        print(f'File Created at {self.save_file_path}')

    def convert_df2json(self):
        '''
        This function processes a dataframe and converts it into a json file readable for LLMs.
        '''
        recetario = []
        print('Step 0 starting...')
        dataframe = self.df_processing_0()
        print('Step 0 finished')
        for index, row in tqdm(dataframe.iterrows(), desc='Trasforming data into JSON'):
            dic = {}
            dic['Recepie'] = self.create_text_1(row['Title'], row['Ingredients'], row['Instructions'], row['Image_descriptions'])
            dic['Image'] = row['Image_Name']
            dic['Image_base64'] = self.encode_image_2(os.path.join(self.img_folder_path, row['Image_Name'] + '.jpg'))
            recetario.append(dic)
        print('finished!!!!')
        self.write_json_3(recetario)

class pdf_processing:
    '''
    The objective of this class is to format and transform pdfs in a folder into
    json files with the text and create a complete dataset.
    '''

    def __init__(self, input_folder_path):
        self.pdf_folder = input_folder_path
        print(f"Processing PDFs in: {self.pdf_folder}")
        self.output_dir = os.path.join(self.pdf_folder, 'preprocessing')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.files = [f for f in os.listdir(self.pdf_folder) if f.endswith('.pdf')]

    @staticmethod
    def extract_text_from_pdf(self, pdf_path):
        '''
        Function to extract text from a single PDF
        '''
        text_data = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text_data.append({
                    "page": i + 1,
                    "text": page.extract_text()
                })
        return text_data

    @staticmethod
    def extract_images_from_pdf(self, pdf_path, output_dir):
        '''
        Function to extract images from a single PDF
        '''
        doc = fitz.open(pdf_path)
        image_data = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = f"image_page{i+1}_{img_index}.{image_ext}"
                image_path = os.path.join(output_dir, image_filename)
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                image_data.append({
                    "page": i + 1,
                    "image_index": img_index,
                    "image_path": image_path
                })
        return image_data

    def extract_pdf_content_to_json(self):
        '''
        Main function to extract both text and images for all PDFs in the folder and combine into JSON
        '''
        for pdf_file in self.files:
            pdf_path = os.path.join(self.pdf_folder, pdf_file)
            print(f"Processing {pdf_file}")

            # Create a subdirectory for each PDF file's output
            pdf_output_dir = os.path.join(self.output_dir, pdf_file.replace('.pdf', ''))
            os.makedirs(pdf_output_dir, exist_ok=True)

            # Extract text and images
            text_data = self.extract_text_from_pdf(pdf_path)
            image_data = self.extract_images_from_pdf(pdf_path, pdf_output_dir)

            # Combine into JSON
            pdf_content = {
                "text": text_data,
                "images": image_data
            }
            json_output_path = os.path.join(pdf_output_dir, "pdf_content.json")
            with open(json_output_path, "w") as json_file:
                json.dump(pdf_content, json_file, indent=4)

            print(f"PDF content saved to {json_output_path}")


