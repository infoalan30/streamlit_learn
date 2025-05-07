import streamlit as st
from openai import OpenAI
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
                return None
            return response
        except Exception as e:
            st.error(f"xAI API Error: {str(e)}")
            traceback.print_exc()
            return None

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
        """Converts OpenAI message history to Google REST API contents format."""
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
        try:
            response = requests.post(request_url, headers=headers, json=payload, stream=True)
            response.raise_for_status()
            web_search_used = False
            def stream_processor(response_stream):
                nonlocal web_search_used
                buffer = ""
                web_search_used = False
                try:
                    for chunk in response_stream.iter_lines():
                        if chunk:
                            decoded_chunk = chunk.decode('utf-8')
                            if decoded_chunk.startswith('data: '):
                                decoded_chunk = decoded_chunk[len('data: '):]
                            buffer = decoded_chunk
                            try:
                                data = json.loads(buffer)
                                text_content = ""
                                if 'candidates' in data and data['candidates']:
                                    candidate = data['candidates'][0]
                                    if 'content' in candidate and 'parts' in candidate['content']:
                                        for part in candidate['content']['parts']:
                                            if 'text' in part:
                                                text_content += part['text']
                                    # Check for web search metadata
                                    try:
                                        grounding_metadata = candidate.get('groundingMetadata', {})
                                        web_search_queries = grounding_metadata.get('webSearchQueries', [])
                                        grounding_supports = grounding_metadata.get('groundingSupports', [])
                                        if web_search_queries and grounding_supports:
                                            web_search_used = True
                                    except:
                                        pass
                                if text_content:
                                    yield text_content
                                if candidate.get('finishReason') == 'SAFETY':
                                    yield "[Blocked by API due to safety settings]"
                                    break
                                buffer = ""
                            except json.JSONDecodeError:
                                pass
                except requests.exceptions.RequestException as http_err:
                    yield f"[Network Error during streaming: {http_err}]"
                except Exception as stream_err:
                    yield f"[Error processing stream: {stream_err}]"
                    traceback.print_exc()

            stream = stream_processor(response)
            # Process stream and collect content
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                response_content = message_placeholder.markdown_stream(stream)
                # Append web search status
                if google_search_enabled and web_search_used:
                    response_content += "\n**Web Search: YES**"
                else:
                    response_content += "\n**Web Search: NO**"
                message_placeholder.markdown(response_content)
            return stream
        except requests.exceptions.HTTPError as errh:
            st.error(f"HTTP Error calling Google API: {errh}")
            try:
                error_details = errh.response.json()
                st.error(f"API Response: {error_details}")
            except:
                st.error(f"API Response Content: {errh.response.text}")
            return None
        except requests.exceptions.RequestException as err:
            st.error(f"Error calling Google API: {err}")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred in GoogleClient: {e}")
            traceback.print_exc()
            return None

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
                def stream_processor(response_stream):
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
                                continue
                return stream_processor(response)
            else:
                response = requests.post(api_endpoint, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                return content
        except requests.exceptions.HTTPError as errh:
            st.error(f"HTTP Error calling OpenRouter API: {errh}")
            try:
                error_details = errh.response.json()
                st.error(f"API Response: {error_details}")
            except:
                st.error(f"API Response Content: {errh.response.text}")
            return None
        except requests.exceptions.RequestException as err:
            st.error(f"Error calling OpenRouter API: {err}")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred in OpenRouterClient: {e}")
            traceback.print_exc()
            return None
