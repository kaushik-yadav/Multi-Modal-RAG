import base64
import os

from groq import Groq

from constants import GROQ_API_KEY


class GroqProcessor:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=GROQ_API_KEY)
    
    def caption_image(self, image_path: str) -> str:
        """
        Generate detailed caption for image using Groq LLaVA model
        """
        if not os.path.exists(image_path):
            return "Image not found"
        
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            # Convert image to base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            image_data_url = f"data:image/png;base64,{image_b64}"

            # Generate caption with detailed prompt
            completion = self.client.chat.completions.create(
                model='meta-llama/llama-4-scout-17b-16e-instruct',
                messages=[
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'text',
                                'text': 'Describe the image in detail along with its components under 150 words. Also mention the connection of components like which component is connected to which. Provide the control flow'
                            },
                            {
                                'type': 'image_url',
                                'image_url': {'url': image_data_url}
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_completion_tokens=1000,
                top_p=1,
                stream=True
            )
            
            caption = ''.join(chunk.choices[0].delta.content or '' for chunk in completion).strip()
            return caption if caption else "Unable to generate caption"
            
        except Exception as e:
            print(f"Error captioning image {image_path}: {e}")
            return f"Error generating caption: {str(e)}"

# Global instance
groq_processor = None

def init_groq(api_key: str):
    global groq_processor
    groq_processor = GroqProcessor(api_key)

def get_image_caption(image_path: str) -> str:
    if groq_processor is None:
        raise ValueError("Groq processor not initialized. Call init_groq() first.")
    return groq_processor.caption_image(image_path)