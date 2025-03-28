import os

def generate_image_description( image_path: str, processor, model, device,  max_length=100, num_beams=7, temperature=1.0, top_p=0.9) -> str:
    '''
    This function converts images into a text description
    '''
    inputs = processor(image_path, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_length=max_length,           # Incrementa la longitud máxima
        num_beams=num_beams,             # Beam search para exploración más detallada
        temperature=temperature,         # Controla la creatividad
        top_p=top_p,                     # Nucleus sampling para generar texto más variado
        repetition_penalty=5.0,            # Penaliza la repetición de palabras
        do_sample=True
    )
    
    description = processor.decode(outputs[0], skip_special_tokens=True)
    return description