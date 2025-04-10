# api_clients.py
import streamlit as st
from openai import OpenAI
from abc import ABC, abstractmethod
# NEW: Import requests and json
import requests
import json
import os
import base64
import io
from PIL import Image
import traceback # Keep for error logging

# --- Configuration ---
XAI_API_BASE_URL = "https://api.x.ai/v1"
# NEW: Google AI REST API Endpoint Configuration
GOOGLE_API_BASE_URL = "https://generativelanguage.googleapis.com" # Base URL

# --- Base Client Class ---
class BaseChatClient(ABC):
    @abstractmethod
    def chat_completion(self, model: str, messages: list, stream: bool = False, **kwargs):
        pass

# --- xAI Client Implementation ---
class XAIClient(BaseChatClient):
    # ... (Keep XAIClient exactly as it was) ...
    def __init__(self, api_key: str = None, base_url: str = XAI_API_BASE_URL):
        if api_key is None:
            api_key = st.secrets.get("XAI_API_KEY")
            if not api_key:
                st.error("xAI API key (XAI_API_KEY) not found in Streamlit secrets.")
                st.stop()
        if not base_url:
             st.error("xAI API Base URL is required.")
             st.stop()
        try:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except Exception as e:
            st.error(f"Failed to initialize xAI client: {e}")
            st.stop()

    def chat_completion(self, model: str, messages: list, stream: bool = False, **kwargs):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream,
                **kwargs
            )
            return response
        except Exception as e:
            st.error(f"xAI API Error: {e}")
            return None


# --- Google Client Implementation (using requests) ---
class GoogleClient(BaseChatClient):
    def __init__(self, api_key: str = None, base_url: str = GOOGLE_API_BASE_URL):
        if api_key is None:
            api_key = st.secrets.get("GOOGLE_API_KEY")
            if not api_key:
                st.error("Google API key (GOOGLE_API_KEY) not found in Streamlit secrets.")
                st.stop()
        self.api_key = api_key
        self.base_url = base_url
        # We don't need to configure a library client anymore

    def _prepare_google_rest_payload(self, messages: list):
        """Converts OpenAI message history to Google REST API contents format."""
        google_contents = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            # Google REST API uses 'user' and 'model' roles
            google_role = "user" if role == "user" else "model"

            parts = []
            if isinstance(content, str):
                # Simple text message
                parts.append({"text": content})
            elif isinstance(content, list):
                # Multimodal message (list of parts)
                for item in content:
                    item_type = item.get("type")
                    if item_type == "text":
                        parts.append({"text": item["text"]})
                    elif item_type == "image_url":
                        image_data_uri = item["image_url"]["url"]
                        try:
                            # Extract mime type and base64 data from data URI
                            header, encoded = image_data_uri.split(",", 1)
                            mime_type = header.split(":")[1].split(";")[0]
                            # Add image part in Google REST format
                            parts.append({
                                "inlineData": {
                                    "mimeType": mime_type,
                                    "data": encoded # Base64 data
                                }
                            })
                        except Exception as e:
                            st.warning(f"Could not process image data URI for Google REST API: {e}")
                            parts.append({"text": "[Image processing error]"}) # Placeholder
            else:
                # Handle unexpected content format if necessary
                 st.warning(f"Unsupported content type in message for Google REST: {type(content)}")
                 parts.append({"text": "[Unsupported content format]"})


            # Append the message with its parts to the contents list
            # Skip empty messages potentially caused by filtering/errors
            if parts:
                google_contents.append({"role": google_role, "parts": parts})

        return {"contents": google_contents}

    def chat_completion(self, model: str, messages: list, stream: bool = False, **kwargs):
        """Calls the Google Generative Language REST API."""
        if not stream:
            # Non-streaming is more complex to parse correctly with requests for Gemini
            # Let's focus on the streaming implementation first as it's more common for chat
            st.warning("Non-streaming not fully implemented for Google REST client yet. Using stream.")
            stream = True # Force stream for now

        # Construct the API endpoint URL
        # Using v1beta as it often gets newer models first
        api_endpoint = f"{self.base_url}/v1beta/models/{model}:streamGenerateContent"
        request_url = f"{api_endpoint}?key={self.api_key}"

        # Prepare the payload in Google REST format
        payload = self._prepare_google_rest_payload(messages)

        # Add generationConfig if needed (e.g., temperature from kwargs)
        # Example: payload["generationConfig"] = {"temperature": kwargs.get("temperature", 0.7)}

        headers = {'Content-Type': 'application/json'}

        try:
            # Make the POST request with streaming enabled
            response = requests.post(request_url, headers=headers, json=payload, stream=True)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            # Define a generator to process the stream
            def stream_processor(response_stream):
                buffer = ""
                try:
                    for chunk in response_stream.iter_lines():
                        if chunk:
                            decoded_chunk = chunk.decode('utf-8')
                            # Google streams often prefix with 'data: ' - remove if present
                            if decoded_chunk.startswith('data: '):
                                decoded_chunk = decoded_chunk[len('data: '):]
                            # Sometimes the stream sends chunks that aren't complete JSON objects
                            # accumulate until we can parse. Assume JSON objects don't span iter_lines chunks.
                            buffer = decoded_chunk # Replace buffer with the new line
                            try:
                                # Attempt to parse the buffer as JSON
                                data = json.loads(buffer)
                                # Extract text based on expected Gemini REST API structure
                                # This structure can vary slightly, adjust as needed based on actual API response
                                text_content = ""
                                if 'candidates' in data and data['candidates']:
                                    candidate = data['candidates'][0]
                                    if 'content' in candidate and 'parts' in candidate['content']:
                                        for part in candidate['content']['parts']:
                                            if 'text' in part:
                                                text_content += part['text']
                                # Yield extracted text if any
                                if text_content:
                                     yield text_content
                                # Check for finish reason or errors if needed
                                # if candidate.get('finishReason'): break
                                # Handle potential safety blocks within the stream
                                if candidate.get('finishReason') == 'SAFETY':
                                    yield "[Blocked by API due to safety settings]"
                                    break
                                # Reset buffer after successful parse
                                buffer = ""
                            except json.JSONDecodeError:
                                # If buffer is not valid JSON yet, wait for more chunks (might happen with partial chunks)
                                # Or if it's just stream metadata/empty lines, ignore
                                # print(f"Skipping non-JSON line: {buffer}") # Debugging
                                pass # Continue to next line
                except requests.exceptions.RequestException as http_err:
                     yield f"[Network Error during streaming: {http_err}]"
                except Exception as stream_err:
                     yield f"[Error processing stream: {stream_err}]"
                     traceback.print_exc() # Log full error

            return stream_processor(response) # Return the generator

        except requests.exceptions.HTTPError as errh:
            st.error(f"HTTP Error calling Google API: {errh}")
            # Try to get more details from the response body if possible
            try:
                error_details = errh.response.json()
                st.error(f"API Response: {error_details}")
            except:
                st.error(f"API Response Content: {errh.response.text}")
            return None # Indicate failure
        except requests.exceptions.ConnectionError as errc:
            st.error(f"Connection Error calling Google API: {errc}")
            return None
        except requests.exceptions.Timeout as errt:
            st.error(f"Timeout Error calling Google API: {errt}")
            return None
        except requests.exceptions.RequestException as err:
            st.error(f"Error calling Google API: {err}")
            return None
        except Exception as e:
             st.error(f"An unexpected error occurred in GoogleClient: {e}")
             traceback.print_exc()
             return None

# --- Add other clients here later ---