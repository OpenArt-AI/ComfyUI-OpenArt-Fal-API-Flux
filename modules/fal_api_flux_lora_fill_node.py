from .base_fal_api_flux_node import BaseFalAPIFluxNode
import logging

logger = logging.getLogger(__name__)

class FalAPIFluxLoraFillNode(BaseFalAPIFluxNode):
    def __init__(self):
        super().__init__()
        self.set_api_endpoint("fal-ai/flux-lora-fill")


    def set_api_endpoint(self, endpoint):
        super().set_api_endpoint(endpoint)
        
    @classmethod
    def INPUT_TYPES(cls):
        input_types = super().INPUT_TYPES()
        input_types["required"].update({
            "image": ("IMAGE",), 
            "mask_image": ("IMAGE",),
        })
        input_types["optional"].update({
            "lora_1": ("LORA_CONFIG",),
            "lora_2": ("LORA_CONFIG",),
            "lora_3": ("LORA_CONFIG",),
            "lora_4": ("LORA_CONFIG",),
            "lora_5": ("LORA_CONFIG",),
        })

        return input_types

    def prepare_arguments(self, image, mask_image, lora_1=None, lora_2=None, lora_3=None, lora_4=None, lora_5=None, **kwargs):
        arguments = super().prepare_arguments(**kwargs)

        image_url = self.upload_image(image)
        mask_image_url = self.upload_image(mask_image)
        logger.info(f"Uploaded target image. URL: {image_url}")
        logger.info(f"Uploaded mask image. URL: {mask_image_url}")

        arguments.update({
            "image_url": image_url,
            "mask_url": mask_image_url
        })

        # Collect all provided LoRA configurations
        loras = []
        for lora in [lora_1, lora_2, lora_3, lora_4, lora_5]:
            if lora is not None:
                loras.append(lora)
        
        if loras:
            arguments["loras"] = loras

        return arguments

NODE_CLASS_MAPPINGS = {
    "FalAPIFluxLoraFillNode": FalAPIFluxLoraFillNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalAPIFluxLoraFillNode": "Fal API Flux Lora Fill"
}