import gradio as gr
from diffusers import DiffusionPipeline
import torch
from PIL import Image, ImageDraw

class ChatbotIconGenerator:
    def __init__(self):
        # Predefined size options
        self.SIZE_OPTIONS = {
            "Small (128x128)": 128,
            "Medium (256x256)": 256,
            "Large (512x512)": 512,
            "Extra Large (1024x1024)": 1024
        }
        
        # Predefined prompt templates
        self.PROMPT_TEMPLATES = [
            "Cute cartoon chatbot mascot, big eyes, friendly smile, pastel colors, kawaii style",
            "Professional AI chatbot avatar, minimalist design, sleek geometric shapes, corporate blue and white",
            "Futuristic AI chatbot icon, glowing circuit patterns, metallic blue and silver",
            "Abstract geometric chatbot avatar, clean lines, single color gradient background",
            "Watercolor style chatbot icon, soft brush strokes, dreamy color blend",
            "Adorable robot character avatar, round shape, soft colors, playful expression",
            "Ultra-minimalist chatbot icon, simple geometric face, monochrome color scheme",
            "Cyberpunk chatbot avatar, neon accents, digital glitch effects",
            "Elegant corporate chatbot icon, modern flat design, clean lines",
            "Sketch-style chatbot avatar, hand-drawn look, pencil texture"
        ]
        
        # Rounding options
        self.CORNER_OPTIONS = {
            "No Rounding": 0,
            "Slight Rounding": 20,
            "Medium Rounding": 50,
            "Full Rounded": 100
        }
        
        # Load the model
        self.model = self.load_image_generator()
    
    def load_image_generator(self):
        try:
            # Use a more lightweight model
            model = DiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", # kopyl/ui-icons-256
                torch_dtype=torch.float16,
                safety_checker=None,  # Disable safety checker to reduce load
                requires_safety_checker=False
            )
            return model.to("cpu")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def round_image_corners(self, image, corner_radius):
        # Create a rounded corner mask
        if corner_radius == 0:
            return image
        
        # Create a new image with an alpha channel
        rounded_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
        
        # Create a mask for rounded corners
        mask = Image.new('L', image.size, 255)
        draw = ImageDraw.Draw(mask)
        
        # Draw rounded rectangle
        draw.rounded_rectangle(
            [0, 0, image.width-1, image.height-1], 
            radius=corner_radius, 
            fill=255
        )
        
        # Paste the original image with the mask
        rounded_image.paste(image, mask=mask)
        return rounded_image
    
    def generate_chatbot_icon(
        self, 
        prompt, 
        size, 
        corner_rounding,
        negative_prompt="low quality, bad composition, blurry, ugly, deformed",
        num_inference_steps=20,
        guidance_scale=7.5
    ):
        if self.model is None:
            raise ValueError("Model failed to load. Please check your dependencies.")
        
        try:
            # Generate the image
            generated_image = self.model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=size,
                width=size
            ).images[0]
            
            # Resize and round corners
            generated_image = generated_image.resize((size, size))
            rounded_image = self.round_image_corners(generated_image, 
                self.CORNER_OPTIONS[corner_rounding])
            
            return rounded_image
        
        except Exception as e:
            print(f"Error generating image: {e}")
            raise
    
    def create_gradio_interface(self):
        with gr.Blocks(title="🤖 Chatbot Icon Generator") as demo:
            gr.Markdown("# 🤖 Chatbot Icon Generator")
            
            with gr.Row():
                with gr.Column():
                    # Prompt selection
                    prompt_dropdown = gr.Dropdown(
                        label="Quick Templates", 
                        choices=self.PROMPT_TEMPLATES,
                        allow_custom_value=True
                    )
                    
                    # Custom prompt input
                    custom_prompt = gr.Textbox(
                        label="Custom Prompt (Optional)", 
                        placeholder="Enter your own detailed description..."
                    )
                    
                    # Size selection
                    size_dropdown = gr.Dropdown(
                        label="Icon Size", 
                        choices=list(self.SIZE_OPTIONS.keys()),
                        value="Medium (256x256)"
                    )
                    
                    # Corner rounding
                    corner_dropdown = gr.Dropdown(
                        label="Corner Rounding", 
                        choices=list(self.CORNER_OPTIONS.keys()),
                        value="Slight Rounding"
                    )
                    
                    # Generate button
                    generate_btn = gr.Button("Generate Icon", variant="primary")
                
                with gr.Column():
                    # Output image
                    output_image = gr.Image(label="Generated Chatbot Icon")
            
            # Logic for prompt selection
            def update_prompt(template):
                return template
            
            prompt_dropdown.change(
                fn=update_prompt, 
                inputs=[prompt_dropdown], 
                outputs=[custom_prompt]
            )
            
            # Generate button logic
            generate_btn.click(
                fn=lambda prompt, size, corners: self.generate_chatbot_icon(
                    prompt or "Cute minimalist chatbot avatar, clean design, friendly expression",
                    self.SIZE_OPTIONS[size],
                    corners
                ),
                inputs=[custom_prompt, size_dropdown, corner_dropdown], 
                outputs=[output_image]
            )
        
        return demo

# Launch the app
def main():
    generator = ChatbotIconGenerator()
    demo = generator.create_gradio_interface()
    demo.launch()

if __name__ == "__main__":
    main()