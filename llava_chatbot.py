import torch
from transformers import AutoTokenizer
from llava.model import LlavaLlamaForCausalLM
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
from io import BytesIO

class LLaVAChatBot:
    def __init__(self, model_path: str = 'liuhaotian/llava-v1.5-7b', device_map: str = 'cuda') -> None:
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.conv = None
        self.conv_img = None
        self.img_tensor = None
        self.roles = None
        self.stop_key = None
        self.load_models(model_path, device_map)

    def load_models(self, model_path: str, device_map: str) -> None:
        try:
            self.model = LlavaLlamaForCausalLM.from_pretrained(model_path, device_map=device_map)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            vision_tower = self.model.get_vision_tower()
            vision_tower.load_model()
            vision_tower.to(device=device_map)
            self.image_processor = vision_tower.image_processor
            disable_torch_init()
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def setup_image(self, img_path: str) -> None:
        try:
            if img_path.startswith('http') or img_path.startswith('https'):
                response = requests.get(img_path)
                response.raise_for_status()  # Raise an error for bad responses
                self.conv_img = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                self.conv_img = Image.open(img_path).convert('RGB')

            self.img_tensor = self.image_processor.preprocess(self.conv_img, return_tensors='pt')['pixel_values'].half()
            self.img_tensor = self.img_tensor.cuda() if torch.cuda.is_available() else self.img_tensor.cpu()
        except Exception as e:
            print(f"Error setting up image: {e}")
            raise

    def generate_answer(self, **kwargs) -> str:
        try:
            raw_prompt = self.conv.get_prompt()
            input_ids = tokenizer_image_token(raw_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
            input_ids = input_ids.cuda() if torch.cuda.is_available() else input_ids.cpu()
            stopping = KeywordsStoppingCriteria([self.stop_key], self.tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = self.model.generate(input_ids, images=self.img_tensor, stopping_criteria=[stopping], **kwargs)

            outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            self.conv.messages[-1][-1] = outputs

            return outputs.rsplit('</s>', 1)[0]
        except Exception as e:
            print(f"Error generating answer: {e}")
            raise

    def start_new_chat(self, img_path: str, prompt: str, do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True, **kwargs) -> str:
        try:
            conv_mode = "v1"
            self.setup_image(img_path)
            self.conv = conv_templates[conv_mode].copy()
            self.roles = self.conv.roles
            first_input = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt)
            self.conv.append_message(self.roles[0], first_input)
            self.conv.append_message(self.roles[1], None)

            self.stop_key = self.conv.sep2 if self.conv.sep_style == SeparatorStyle.TWO else self.conv.sep
            answer = self.generate_answer(do_sample=do_sample, temperature=temperature, max_new_tokens=max_new_tokens, use_cache=use_cache, **kwargs)
            return answer
        except Exception as e:
            print(f"Error starting new chat: {e}")
            raise

    def continue_chat(self, prompt: str, do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True, **kwargs) -> str:
        if self.conv is None:
            raise RuntimeError("No existing conversation found. Start a new conversation using the `start_new_chat` method.")
        
        try:
            self.conv.append_message(self.roles[0], prompt)
            self.conv.append_message(self.roles[1], None)
            answer = self.generate_answer(do_sample=do_sample, temperature=temperature, max_new_tokens=max_new_tokens, use_cache=use_cache, **kwargs)
            return answer
        except Exception as e:
            print(f"Error continuing chat: {e}")
            raise
