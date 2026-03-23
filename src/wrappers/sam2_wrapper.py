import torch
import logging
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


class Sam2Wrapper:
    def __init__(self, model_name: str, generator_params: dict, device: str):
        self.device = device
        logging.info(f"Initializing SAM 2 Wrapper by loading '{model_name}' from Hugging Face Hub...")
        
        try:
            image_predictor = SAM2ImagePredictor.from_pretrained(model_name)
            sam_model = image_predictor.model
            self.mask_generator = SAM2AutomaticMaskGenerator(model=sam_model, **generator_params)
            
            logging.info("SAM 2 Wrapper Initialized successfully from Hugging Face.")

        except Exception as e:
            logging.error(f"Failed to initialize SAM 2 from Hugging Face: {e}", exc_info=True)
            raise

    def generate_masks(self, image: 'np.ndarray') -> list:
        logging.info("SAM 2 Wrapper: Generating masks...")
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            raw_masks = self.mask_generator.generate(image)
        return raw_masks
    