import openai
from typing import Optional
import os
from dotenv import load_dotenv
import base64
from pydantic import BaseModel

load_dotenv()

class OpenAIChatSession:
    """A simplified class to manage an OpenAI chat session using the new Responses API."""
    
    def __init__(
        self, 
        system_prompt: str, 
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ):
        """Initialize the OpenAI chat session."""
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set up API key
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        elif os.getenv("OPENAI_API_KEY"):
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError("OpenAI API key must be provided either as parameter or OPENAI_API_KEY environment variable")

    def _encode_image(self, image_path: str) -> str:
        """Encode an image file to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def set_temperature(self, temperature: float) -> None:
        """Update the global temperature for all future queries."""
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        self.temperature = temperature
    
    def get_system_prompt(self) -> str:
        """Get the current system prompt."""
        return self.system_prompt
    
    def update_system_prompt(self, new_prompt: str) -> None:
        """Update the system prompt."""
        self.system_prompt = new_prompt
    
    def attach_return_schema(self, return_schema: str) -> None:
        """Attach a return schema to the system prompt."""
        self.system_prompt = f"{self.system_prompt}\n\nReturn schema: {return_schema}"

    def query(self, message: str) -> str:
        """Send a query using the Responses API and return the response."""
        # Prepare input for API call
        input_content = [
            {"type": "message", "role": "system", "content": f"{self.system_prompt}"},
            {"type": "message", "role": "user", "content": f"{message}"}
        ]
        
        # Prepare API call parameters
        params = {
            "model": self.model,
            "input": input_content,
            "temperature": self.temperature
        }
        
        if self.max_tokens:
            params["max_output_tokens"] = self.max_tokens
        
        try:
            # Make API call using new Responses API
            response = self.client.responses.create(**params)
            
            # Extract assistant's response
            assistant_response = response.output_text
            
            return assistant_response
            
        except Exception as e:
            raise Exception(f"OpenAI Responses API call failed: {str(e)}")
    
    def query_json(self, message: str, evaluation: BaseModel) -> dict:
        """Send a query using the Responses API and return structured JSON response."""
        import json
        
        # Prepare input for API call
        input_content = [
            {"type": "message", "role": "system", "content": f"{self.system_prompt}"},
            {"type": "message", "role": "user", "content": f"{message}"}
        ]
        if self.max_tokens:
            params["max_output_tokens"] = self.max_tokens
        # Prepare API call parameters
        params = {
            "model": self.model,
            "input": input_content,
            "temperature": self.temperature,
        }
        
        try:
            # Make API call using new Responses API
            response = self.client.responses.parse(**params, text_format=evaluation)
            return response.output_parsed

            
        except Exception as e:
            print(f"Full OpenAI error in query_json: {repr(e)}")
            raise Exception(f"OpenAI Responses API call failed: {str(e)}")

    def query_with_image(self, message: str, image_path: str) -> str:
        """Send a query with an image using the Responses API and return the response."""
        # Check if image file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Encode the image
        base64_image = self._encode_image(image_path)
        
        # Prepare input for API call with image
        input_content = [
        {"type":"message","role":"system","content":self.system_prompt},
        {"type":"message","role":"user","content":message},
        {
            "type":"message",
            "role":"user",
            "content":[
            {"type":"input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"}
            ]
        }
        ]
        
        # Prepare API call parameters
        params = {
            "model": self.model,
            "input": input_content,
            "temperature": self.temperature
        }
        
        if self.max_tokens:
            params["max_output_tokens"] = self.max_tokens
        
        try:
            # Make API call using new Responses API
            response = self.client.responses.create(**params)
            
            return response.output_text
            
        except Exception as e:
            raise Exception(f"OpenAI Responses API call failed: {str(e)}")

    def query_with_images_provided(self, message: str, images: dict[str, str]) -> str:
        """Send a query with an image using the Responses API and return the response."""
        # Check if image file exists

        
        input_content = [
        {"type":"message","role":"system","content":self.system_prompt},
        {"type":"message","role":"user","content":message}
        ]
        
        for image in images.keys():
            input_content.append({
                "type":"message",
                "role":"user",
                "content":[
                {"type":"input_image", "image_url": f"data:image/jpeg;base64,{images[image]}"}
                ]
            })
        
        # Prepare API call parameters
        params = {
            "model": self.model,
            "input": input_content,
            "temperature": self.temperature
        }
        
        if self.max_tokens:
            params["max_output_tokens"] = self.max_tokens
        
        try:
            # Make API call using new Responses API
            response = self.client.responses.create(**params)
            
            return response.output_text
            
        except Exception as e:
            raise Exception(f"OpenAI Responses API call failed: {str(e)}")

    def query_json_with_image_provided(self, message: str, image: str, evaluation: BaseModel, json_schema: Optional[dict] = None) -> dict:
        """Send a query with an image using the Responses API and return structured JSON response."""
        
        # Validate and format the base64 image string
        try:
            # Remove any existing data URL prefix if present
            if image.startswith('data:image/'):
                # Extract just the base64 part
                image = image.split(',', 1)[1] if ',' in image else image
            
            # Validate base64 string
            import base64
            try:
                base64.b64decode(image)
            except Exception as e:
                raise ValueError(f"Invalid base64 string: {str(e)}")
            
            # Format as proper data URL
            formatted_image = f"data:image/png;base64,{image}"
            
        except Exception as e:
            raise Exception(f"Error formatting image data: {str(e)}")
        
        # Prepare input for API call with image
        input_content = [
        {"type":"message","role":"system","content":self.system_prompt},
        {"type":"message","role":"user","content":message},
        {
            "type":"message",
            "role":"user",
            "content":[
            {"type":"input_image", "image_url": formatted_image}
            ]
        }
        ]
        
        # Prepare API call parameters
        params = {
            "model": self.model,
            "input": input_content,
            "temperature": self.temperature
        }
        
        if self.max_tokens:
            params["max_output_tokens"] = self.max_tokens
        
        try:
            # Make API call using new Responses API
            response = self.client.responses.parse(**params, text_format=evaluation)

            return response.output_parsed
            
        except Exception as e:
            raise Exception(f"OpenAI Responses API call failed: {str(e)}")

    def query_json_with_images_provided(self, message: str, images: dict[str, str], evaluation: BaseModel, json_schema: Optional[dict] = None) -> dict:
        """Send a query with an image using the Responses API and return structured JSON response."""
        
        # Prepare input for API call with image
        input_content = [
        {"type":"message","role":"system","content":self.system_prompt},
        {"type":"message","role":"user","content":message}
        ]
        
        # Validate and format each base64 image string
        for image_key, image_data in images.items():
            try:
                # Remove any existing data URL prefix if present
                if image_data.startswith('data:image/'):
                    # Extract just the base64 part
                    image_data = image_data.split(',', 1)[1] if ',' in image_data else image_data
                
                # Validate base64 string
                import base64
                try:
                    base64.b64decode(image_data)
                except Exception as e:
                    raise ValueError(f"Invalid base64 string for {image_key}: {str(e)}")
                
                # Format as proper data URL
                formatted_image = f"data:image/png;base64,{image_data}"
                
                input_content.append({
                    "type":"message",
                    "role":"user",
                    "content":[
                    {"type":"input_image", "image_url": formatted_image}
                    ]
                })
                
            except Exception as e:
                raise Exception(f"Error formatting image data for {image_key}: {str(e)}")
        
        # Prepare API call parameters
        params = {
            "model": self.model,
            "input": input_content,
            "temperature": self.temperature
        }
        
        if self.max_tokens:
            params["max_output_tokens"] = self.max_tokens
        
        try:
            # Make API call using new Responses API
            response = self.client.responses.parse(**params, text_format=evaluation)

            return response.output_parsed
            
        except Exception as e:
            raise Exception(f"OpenAI Responses API call failed: {str(e)}")