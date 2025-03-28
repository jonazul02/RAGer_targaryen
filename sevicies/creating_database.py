import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from utilities.configure_script import ConfigurationLoader
from utilities.utils import recepies_preprocessing

def main():

    configs = ConfigurationLoader.get_config()

    dataframe_path = configs['paths']['recepies_path']['csv_ingredients']
    img_folder_path = configs['paths']['recepies_path']['images_folder']
    save_file_path = configs['paths']['recepies_path']['recepies_json']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor_blip = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model_blip.to(device)


    processor = recepies_preprocessing(input_dataframe_path=dataframe_path,
                                       input_image_folder=img_folder_path,
                                       output_save_file=save_file_path,
                                       processor=processor_blip,
                                       model=model_blip,
                                       device=device
                                       )

    processor.convert_df2json()

if __name__ == "__main__":

    main()