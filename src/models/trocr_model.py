import torch
import torch.nn as nn
from transformers import AutoTokenizer, GenerationConfig, VisionEncoderDecoderModel

class SinhalaTrOCR(nn.Module):
    def __init__(self, encoder_pretrained_path=None, decoder_model_name='keshan/SinhalaBERTo'):
        super().__init__()
        
        # Initialize the VisionEncoderDecoderModel
        # This automatically sets up the cross-attention layers in the SinBERT model 
        # so it can act as a decoder.
        print("Initializing VisionEncoderDecoderModel...")
        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            "google/vit-base-patch16-384", 
            decoder_model_name
        )
        
        # If we have weights from Stage 1, load them into the encoder
        if encoder_pretrained_path is not None:
            print(f"Loading Stage 1 encoder weights from {encoder_pretrained_path}...")
            self.model.encoder.load_state_dict(
                torch.load(encoder_pretrained_path, map_location="cpu", weights_only=True)
            )
            
        # Configure decoder tokens using the SinBERT tokenizer
        tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        
        self.model.config.decoder_start_token_id = tokenizer.cls_token_id
        self.model.config.pad_token_id = tokenizer.pad_token_id
        self.model.config.eos_token_id = tokenizer.sep_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

        # Generation settings must use generation_config (transformers 5.x)
        self.model.generation_config = GenerationConfig(
            decoder_start_token_id=tokenizer.cls_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.sep_token_id,
            max_length=128,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
            num_beams=4,
        )

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)
    
    def generate(self, pixel_values, decoder_input_ids=None, **kwargs):
        if decoder_input_ids is not None:
            return self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                **kwargs,
            )
        return self.model.generate(pixel_values, **kwargs)
