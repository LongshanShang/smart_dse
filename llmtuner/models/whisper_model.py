from transformers import WhisperForConditionalGeneration
from llmtuner.models.base_model import BaseModel


class WhisperModel(BaseModel):
    def __init__(self, model_name_or_path, language='ta', task="transcribe"):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.language = language
        self.task = task
        self.model = None

    def load(self):
        if not self.model:
            # # Set the quantized model to use forced decoding
            # self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name_or_path)
    
            # # Load a pre-trained WhisperProcessor model
            # processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    
            # # Get the decoder prompt IDs for the English language
            # decoder_prompt_ids = processor.get_decoder_prompt_ids(language="ta", task="transcribe")
    
            # Set the forced decoder IDs
            self.model.config.forced_decoder_ids = None
    
            # Set other configuration parameters
            self.model.config.suppress_tokens = []
            self.model.is_peft_applied = False
    
            self.is_peft_applied = self.model.is_peft_applied
        return self.model

    
    def save(self, save_path, *args, **kwargs):
        self.model.save_pretrained(save_path, from_pt=True)
