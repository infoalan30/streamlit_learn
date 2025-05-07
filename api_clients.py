# api_clients.py
import streamlit as st
from openai import OpenAI # Pastikan OpenAI diimpor jika XAIClient menggunakannya
from abc import ABC, abstractmethod
import requests
import json
import os
import base64
import io
from PIL import Image
import traceback

# --- Configuration ---
XAI_API_BASE_URL = "https://api.x.ai/v1"
GOOGLE_API_BASE_URL = "https://generativelanguage.googleapis.com"
OPENROUTER_API_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_INFO = {
    "deepseek_R1": "deepseek/deepseek-r1:free",
    "deepseek_V3": "deepseek/deepseek-chat-v3-0324:free",
    "gemini-2.0-flash-exp": "google/gemini-2.0-flash-exp:free",
    "gemini-2.5-pro-exp": "google/gemini-2.5-pro-exp-03-25:free"
}

# --- Base Client Class ---
class BaseChatClient(ABC):
    @abstractmethod
    def chat_completion(self, model: str, messages: list, stream: bool = False, **kwargs):
        pass

# --- xAI Client Implementation ---
class XAIClient(BaseChatClient):
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
                stream=stream
            )
            if stream and not hasattr(response, '__iter__'):
                st.error(f"xAI API returned non-iterable response for streaming: {type(response)}")
                return None # Explicitly return None on error
            return response
        except Exception as e:
            st.error(f"xAI API Error: {str(e)}")
            traceback.print_exc()
            return None # Explicitly return None on error


# --- Helper Class for Google Stream ---
class GoogleStreamWrapper:
    def __init__(self, generator, status_container, queries_container): # Tambahkan queries_container
        self.generator = generator
        self.web_search_status_container = status_container
        self.web_search_queries_container = queries_container # Simpan container query

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.generator)

    def close(self):
        if hasattr(self.generator, 'close'):
            self.generator.close()


# --- Google Client Implementation ---
class GoogleClient(BaseChatClient):
    def __init__(self, api_key: str = None, base_url: str = GOOGLE_API_BASE_URL):
        if api_key is None:
            api_key = st.secrets.get("GOOGLE_API_KEY")
            if not api_key:
                st.error("Google API key (GOOGLE_API_KEY) not found in Streamlit secrets.")
                st.stop()
        self.api_key = api_key
        self.base_url = base_url

    def _prepare_google_rest_payload(self, messages: list, google_search: bool = False):
        google_contents = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            google_role = "user" if role == "user" else "model"
            parts = []
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for item in content:
                    item_type = item.get("type")
                    if item_type == "text":
                        parts.append({"text": item["text"]})
                    elif item_type == "image_url":
                        image_data_uri = item["image_url"]["url"]
                        try:
                            header, encoded = image_data_uri.split(",", 1)
                            mime_type = header.split(":")[1].split(";")[0]
                            parts.append({
                                "inlineData": {
                                    "mimeType": mime_type,
                                    "data": encoded
                                }
                            })
                        except Exception as e:
                            st.warning(f"Could not process image data URI for Google REST API: {e}")
                            parts.append({"text": "[Image processing error]"})
            if parts:
                google_contents.append({"role": google_role, "parts": parts})
        payload = {"contents": google_contents}
        if google_search:
            payload["tools"] = [{"google_search": {}}]
        return payload

    def chat_completion(self, model: str, messages: list, stream: bool = False, google_search: bool = False, **kwargs):
        if not stream:
            st.warning("Non-streaming not fully implemented for Google REST client yet. Using stream.")
            stream = True
        
        api_endpoint = f"{self.base_url}/v1beta/models/{model}:streamGenerateContent"
        request_url = f"{api_endpoint}?alt=sse&key={self.api_key}"
        payload = self._prepare_google_rest_payload(messages, google_search=google_search)
        headers = {'Content-Type': 'application/json'}

        print("--- Google API Request ---")
        print(f"URL: {request_url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")

        try:
            response = requests.post(request_url, headers=headers, json=payload, stream=True)
            
            print(f"Google API Response Status Code: {response.status_code}")
            if response.status_code != 200:
                print(f"Google API Response Text: {response.text}")
            
            response.raise_for_status()

            _web_search_status_container = [False]
            _web_search_queries_list = [[]]

            def _streaming_generator_inner(response_stream, status_container, queries_list_container): # Ubah nama agar tidak bentrok
                buffer = ""
                try:
                    for chunk in response_stream.iter_lines():
                        if chunk:
                            decoded_chunk = chunk.decode('utf-8')
                            if decoded_chunk.startswith('data: '):
                                decoded_chunk = decoded_chunk[len('data: '):]
                            buffer += decoded_chunk
                            if buffer.strip().endswith('}'):
                                try:
                                    data = json.loads(buffer)
                                    buffer = "" 
                                    text_content = ""
                                    if 'candidates' in data and data['candidates']:
                                        candidate = data['candidates'][0]
                                        if 'content' in candidate and 'parts' in candidate['content']:
                                            for part in candidate['content']['parts']:
                                                if 'text' in part:
                                                    text_content += part['text']
                                        try:
                                            grounding_metadata = candidate.get('groundingMetadata', {})
                                            web_search_queries = grounding_metadata.get('webSearchQueries', [])
                                            if web_search_queries:
                                                status_container[0] = True
                                                for q in current_chunk_queries:
                                                    if q not in queries_list_container[0]:
                                                        queries_list_container[0].append(q)
                                        except Exception as e_meta:
                                            pass 
                                    if text_content:
                                        yield text_content
                                    if candidate.get('finishReason') == 'SAFETY':
                                        yield "[Blocked by API due to safety settings]"
                                        break
                                except json.JSONDecodeError:
                                    pass 
                except requests.exceptions.RequestException as http_err:
                    yield f"[Network Error during streaming: {http_err}]"
                except Exception as stream_err:
                    print(f"--- Error inside _streaming_generator_inner ---") # DEBUG
                    traceback.print_exc()                                     # DEBUG
                    yield f"[Error processing stream: {stream_err}]"
                    # traceback.print_exc() # Sudah ada di atas

            # Buat instance generator yang sebenarnya
            actual_generator = _streaming_generator_inner(response, _web_search_status_container, _web_search_queries_list)
            
            # Bungkus generator dengan GoogleStreamWrapper
            wrapped_stream = GoogleStreamWrapper(actual_generator, _web_search_status_container, _web_search_queries_list)
            
            return wrapped_stream # Kembalikan objek wrapper

        except requests.exceptions.HTTPError as errh:
            st.error(f"HTTP Error calling Google API: {errh}")
            try:
                error_details = errh.response.json()
                st.error(f"API Response (JSON): {error_details}")
            except json.JSONDecodeError:
                st.error(f"API Response (text): {errh.response.text}")
            print("--- Google API HTTPError Traceback ---")
            traceback.print_exc()
            return None # Pastikan return None secara eksplisit
        except requests.exceptions.RequestException as err:
            st.error(f"Request Exception calling Google API: {err}")
            print("--- Google API RequestException Traceback ---")
            traceback.print_exc()
            return None # Pastikan return None secara eksplisit
        except Exception as e:
            st.error(f"An unexpected error occurred in GoogleClient: {e}")
            print("--- Google API Unexpected Exception Traceback (in chat_completion) ---")
            traceback.print_exc()
            return None # Pastikan return None secara eksplisit


# --- OpenRouter Client Implementation ---
class OpenRouterClient(BaseChatClient):
    def __init__(self, api_key: str = None, base_url: str = OPENROUTER_API_BASE_URL):
        if api_key is None:
            api_key = st.secrets.get("OPENROUTER_API_KEY")
            if not api_key:
                st.error("OpenRouter API key (OPENROUTER_API_KEY) not found in Streamlit secrets.")
                st.stop()
        self.api_key = api_key
        self.base_url = base_url
        self.models_info = OPENROUTER_MODELS_INFO

    def chat_completion(self, model: str, messages: list, stream: bool = False, **kwargs):
        model_id = self.models_info.get(model, model)
        api_endpoint = f"{self.base_url}/chat/completions"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        payload = {
            "model": model_id,
            "messages": messages,
            "stream": stream
        }
        try:
            if stream:
                response = requests.post(api_endpoint, headers=headers, json=payload, stream=True)
                response.raise_for_status()
                def stream_processor(response_stream): # Ini adalah generator
                    for chunk in response_stream.iter_lines():
                        if chunk:
                            decoded_chunk = chunk.decode('utf-8')
                            if decoded_chunk.startswith('data: '):
                                decoded_chunk = decoded_chunk[len('data: '):]
                            if decoded_chunk == '[DONE]':
                                break
                            try:
                                data = json.loads(decoded_chunk)
                                delta = data.get('choices', [{}])[0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue # Abaikan jika JSON tidak valid, mungkin chunk tidak lengkap
                return stream_processor(response) # Mengembalikan generator
            else:
                response = requests.post(api_endpoint, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                return content # Mengembalikan string
        except requests.exceptions.HTTPError as errh:
            st.error(f"HTTP Error calling OpenRouter API: {errh}")
            try:
                error_details = errh.response.json()
                st.error(f"API Response: {error_details}")
            except:
                st.error(f"API Response Content: {errh.response.text}")
            return None # Eksplisit return None
        except requests.exceptions.RequestException as err:
            st.error(f"Error calling OpenRouter API: {err}")
            return None # Eksplisit return None
        except Exception as e:
            st.error(f"An unexpected error occurred in OpenRouterClient: {e}")
            traceback.print_exc()
            return None # Eksplisit return None
