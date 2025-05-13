from inference.inference_utils import chameleon_generate, load_chameleon
from PIL import Image

def run_inference():
    """Contains the main logic for loading the model and generating text."""
    
    # Update these paths
    model_path = "/path/to/your/model"
    image_path = "test_green_img.png"
    
    print("Loading Chameleon model...")
    model = load_chameleon(model_path) 

    image = Image.new('RGB', (300, 300), color="green")
    image.save(image_path)
    
    content = [image_path, "<image> What color is this?"]
    modality = ["image", "text"]

    print("Generating text...")
    generated_sequences = chameleon_generate(model,
                                             content=content,
                                             modality=modality,
                                             task="text-gen", # Set to "image-gen" for image generation only
                                             sft=False, # Set to True for sft mode
                                             temp=None, # Adjust sampling temperature
                                             greedy=True, # Greedy decoding
                                             max_gen_len=100) # Maximum generation length
                                            
    generated_text = generated_sequences[0]
    print("Generated text:", generated_text)

if __name__ == '__main__':

    run_inference()
