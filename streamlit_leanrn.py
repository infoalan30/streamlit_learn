import streamlit as st
import base64
import io
from PIL import Image
import time
# Import clients
from api_clients import XAIClient, GoogleClient, OpenRouterClient # Assuming api_clients.py is correct
import traceback
import json

# --- Configuration ---
MODEL_PROVIDERS = {
    "xAI": ["grok-3-beta", "grok-3-mini-fast-beta", "grok-2-vision-1212"],
    "Gemini": ["gemini-2.0-flash", "gemini-1.5-pro"],
    "openrouter": ["gemini-2.0-flash-exp", "gemini-2.5-pro-exp", "deepseek_R1", "deepseek_V3"]
}
TEXT_MODELS = ["grok-3-beta", "grok-3-mini-fast-beta", "gemini-1.5-pro", "deepseek_R1", "deepseek_V3"]
VISION_MODELS = ["grok-2-vision-1212", "gemini-2.0-flash", "gemini-2.0-flash-exp", "gemini-2.5-pro-exp"]
DEFAULT_PROVIDER = "xAI"
DEFAULT_MODEL = "grok-3-mini-fast-beta"
CONTEXT_MESSAGES_COUNT = 3

# --- Helper Functions ---
def image_to_base64(image_bytes, format="JPEG"):
    # ... (keep as is) ...
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img_format = img.format or format
        if img_format not in ["JPEG", "PNG", "GIF", "WEBP"]:
             st.warning(f"Unsupported format {img_format}. Using JPEG.")
             img_format = "JPEG"
             if img.mode == 'RGBA' and img_format == 'JPEG': img = img.convert('RGB')
        buffered = io.BytesIO()
        save_args = {'quality': 95} if img_format == "JPEG" else {}
        img.save(buffered, format=img_format, **save_args)
        return base64.b64encode(buffered.getvalue()).decode("utf-8"), f"image/{img_format.lower()}"
    except Exception as e: st.error(f"Image processing error: {e}"); return None, None

# --- Session State Initialization ---
def init_session_state():
    """Initializes session state variables."""
    # Clients
    if "xai_client" not in st.session_state:
        try: st.session_state.xai_client = XAIClient()
        except Exception as e: st.error(f"XAI Init Err: {e}"); st.session_state.xai_client = None
    if "google_client" not in st.session_state:
        try: st.session_state.google_client = GoogleClient()
        except Exception as e: st.error(f"Google Init Err: {e}"); st.session_state.google_client = None
    if "openrouter_client" not in st.session_state:
        try: st.session_state.openrouter_client = OpenRouterClient()
        except Exception as e: st.error(f"OpenRouter Init Err: {e}"); st.session_state.openrouter_client = None
    # Chat state
    if "chat_sessions" not in st.session_state: st.session_state.chat_sessions = {}
    if "current_chat_id" not in st.session_state:
        first_chat_id = f"chat_{time.time()}"; st.session_state.chat_sessions[first_chat_id] = []; st.session_state.current_chat_id = first_chat_id
    if "context_cleared_for_next_turn" not in st.session_state: st.session_state.context_cleared_for_next_turn = False
    # Image/Upload state
    if "staged_images_bytes" not in st.session_state: st.session_state.staged_images_bytes = []
    if "uploader_key_counter" not in st.session_state: st.session_state.uploader_key_counter = 0
    # Model selection state
    if "selected_provider" not in st.session_state: st.session_state.selected_provider = DEFAULT_PROVIDER
    if "selected_child_model" not in st.session_state: st.session_state.selected_child_model = DEFAULT_MODEL

# --- Context Clearing Callback ---
def handle_clear_context():
    # ... (keep as is) ...
    st.session_state.context_cleared_for_next_turn = True
    if st.session_state.current_chat_id in st.session_state.chat_sessions:
        if st.session_state.chat_sessions[st.session_state.current_chat_id]:
             st.session_state.chat_sessions[st.session_state.current_chat_id].append({"role": "system", "content": "🧹 Context cleared."})
        else: st.toast("Context cleared.")
    else: st.warning("Chat session not found.")
    st.rerun()

# --- Main App Logic ---
st.set_page_config(layout="wide", page_title="Multimodal Chat")
init_session_state()

# --- Sidebar ---
with st.sidebar:
    # ... (Sidebar code remains the same) ...
    st.title("Chat Sessions")
    if st.button("➕ New Chat", use_container_width=True):
        new_chat_id = f"chat_{time.time()}"
        st.session_state.chat_sessions[new_chat_id] = []
        st.session_state.current_chat_id = new_chat_id
        st.session_state.staged_images_bytes = []
        st.session_state.context_cleared_for_next_turn = False
        st.session_state.uploader_key_counter = 0
        st.session_state.selected_model = DEFAULT_MODEL
        st.rerun()
    st.write("---")
    st.write("**History**")
    sorted_session_ids = sorted(st.session_state.chat_sessions.keys(), key=lambda x: float(x.split('_')[1]), reverse=True)
    if not st.session_state.chat_sessions: st.caption("No chats yet.")
    for session_id in sorted_session_ids:
        messages = st.session_state.chat_sessions.get(session_id, [])
        label = "Empty Chat"
        if messages:
            first_user_message_content = None
            for m in messages:
                if m['role'] == 'user': first_user_message_content = m['content']; break
            if isinstance(first_user_message_content, list):
                text_content = next((item['text'] for item in first_user_message_content if item['type'] == 'text'), "")
                has_image = any(item.get('type') == 'image_url' for item in first_user_message_content)
                label = f"{'🖼️ ' if has_image else ''}{text_content[:25]}..." if text_content else f"{'🖼️ ' if has_image else ''}Chat"
            elif messages[-1]['role'] == 'system':
                 if len(messages) > 1:
                     prev_content = messages[-2]['content']
                     if isinstance(prev_content, list): text_content = next((i['text'] for i in prev_content if i['type'] == 'text'), "")
                     elif isinstance(prev_content, str): text_content = prev_content
                     else: text_content = ""
                     label = f"🧹 {text_content[:25]}..." if text_content else "🧹 System Action"
                 else: label = "System Message"
            elif messages[-1]['role'] == 'assistant' and isinstance(messages[-1]['content'], str):
                 label = f"{messages[-1]['content'][:30]}..."
            else: label = "Chat Entry"
        button_type = "primary" if session_id == st.session_state.current_chat_id else "secondary"
        if st.button(label, key=f"select_{session_id}", use_container_width=True, type=button_type):
            if st.session_state.current_chat_id != session_id:
                st.session_state.current_chat_id = session_id
                st.session_state.staged_images_bytes = []
                st.session_state.context_cleared_for_next_turn = False
                st.session_state.uploader_key_counter = 0
                st.session_state.selected_model = DEFAULT_MODEL
                st.rerun()

# --- Main Chat Area ---
st.title(f"Multimodal Chat")
st.info("ℹ️ Chat history session-based. Select provider and model carefully based on content.")

# --- Simplified MODEL SELECTION DROPDOWN ---
# --- Model Selection Dropdowns ---
col_provider, col_model = st.columns([1, 2])
with col_provider:
    selected_provider = st.selectbox(
        "Select Provider:",
        options=list(MODEL_PROVIDERS.keys()),
        key="selected_provider",
        help="Choose the API provider."
    )
with col_model:
    # Get available models for the selected provider
    available_models = MODEL_PROVIDERS.get(st.session_state.selected_provider, [])
    # Ensure the selected child model is valid for the provider
    if st.session_state.selected_child_model not in available_models:
        st.session_state.selected_child_model = available_models[0] if available_models else DEFAULT_MODEL
    selected_child_model = st.selectbox(
        "Select Model:",
        options=available_models,
        key="selected_child_model",
        help="Choose the model for the selected provider."
    )

# --- Input Controls ---
control_cols = st.columns([1, 4], gap="small")
with control_cols[0]:
    st.button("🧹", key="clear_context_btn", help="Clear context",
        on_click=handle_clear_context, use_container_width=True)
with control_cols[1]:
    uploader_key = f"file_uploader_{st.session_state.current_chat_id}_{st.session_state.uploader_key_counter}"
    uploaded_files = st.file_uploader(
        "Attach Image(s)", type=["png", "jpg", "jpeg", "gif", "webp"],
        accept_multiple_files=True, label_visibility="collapsed",
        key=uploader_key, help="Attach image(s) if using a vision model."
        # No on_change needed here anymore
    )
    # Preview logic
    if uploaded_files:
        st.session_state.staged_images_bytes = [file.getvalue() for file in uploaded_files]
        st.write("Attached Images Preview:")
        cols_needed = len(uploaded_files); preview_cols = st.columns(cols_needed if cols_needed > 0 else 1)
        for i, file in enumerate(uploaded_files):
             with preview_cols[i % cols_needed]: st.image(file, width=80, caption=f"{file.name[:10]}...")
    # staged_images_bytes is cleared after sending now

# --- Display Chat Messages ---
st.markdown("---")
message_container = st.container()
with message_container:
    # (Display logic remains the same)
    current_messages = st.session_state.chat_sessions.get(st.session_state.current_chat_id, [])
    for msg in current_messages:
        role = msg["role"]; content = msg["content"]
        if role == "system": st.markdown(f'<div style="text-align: center; color: grey; font-style: italic; margin-top: 5px; margin-bottom: 5px;">--- {content} ---</div>', unsafe_allow_html=True)
        else:
            with st.chat_message(role):
                if role == "user":
                    if isinstance(content, list):
                         text_parts = [item['text'] for item in content if item.get('type') == 'text']
                         image_count = sum(1 for item in content if item.get('type') == 'image_url')
                         if text_parts: st.markdown(" ".join(text_parts))
                         if image_count > 0: st.markdown(f"_{image_count} Image(s) sent._")
                    else: st.error(f"ERR: User msg format {type(content)}"); st.markdown(str(content))
                elif role == "assistant":
                    # Display content, potentially including formatted error messages
                    if isinstance(content, str):
                        st.markdown(content, unsafe_allow_html=True) # Allow markdown like italics in error
                    else:
                        st.error(f"ERR: Assistant msg format {type(content)}"); st.markdown(str(content))


# --- Chat Input ---
prompt = st.chat_input("Enter message...", key=f"chat_input_{st.session_state.current_chat_id}")

if prompt:
    # --- Prepare user message (always OpenAI list format for storage) ---
    user_message_content_for_history = [{"type": "text", "text": prompt}]
    image_attachments_for_history = []
    if st.session_state.staged_images_bytes:
        st.toast(f"Attaching {len(st.session_state.staged_images_bytes)} image(s).")
        processed_count = 0
        for img_bytes in st.session_state.staged_images_bytes:
            base64_image, mime_type = image_to_base64(img_bytes)
            if base64_image: image_attachments_for_history.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}); processed_count += 1
            else: st.warning("Image processing failed.")
        if image_attachments_for_history:
            user_message_content_for_history.extend(image_attachments_for_history)
            if processed_count < len(st.session_state.staged_images_bytes): st.warning(f"Processed {processed_count}/{len(st.session_state.staged_images_bytes)} images.")
        else: st.error("All images failed processing.")

    # --- Add user message to history ---
    st.session_state.chat_sessions.setdefault(st.session_state.current_chat_id, []).append(
        {"role": "user", "content": user_message_content_for_history}
    )

    # --- Clear staged data and reset UI elements ---
    st.session_state.staged_images_bytes = [] # Clear the actual data AFTER using it
    st.session_state.uploader_key_counter += 1 # Increment key to clear widget visually

    st.rerun()


# --- API Call and Streaming Response ---
current_messages = st.session_state.chat_sessions.get(st.session_state.current_chat_id, [])
should_call_api = bool(current_messages and current_messages[-1]["role"] == "user")

if should_call_api:
    last_user_message_openai_format = current_messages[-1]
    target_api_family = None
    model_for_api_call = ""

    try:
        # --- Direct Model and Client Determination ---
        model_for_api_call = st.session_state.selected_child_model
        selected_provider = st.session_state.selected_provider
        print(f"DEBUG: API Call - Provider: '{selected_provider}', Model: '{model_for_api_call}'")

        client = None
        if selected_provider == "xAI":
            client = st.session_state.xai_client
            target_api_family = "openai"
        elif selected_provider == "Gemini":
            client = st.session_state.google_client
            target_api_family = "google"
        elif selected_provider == "openrouter":
            client = st.session_state.openrouter_client
            target_api_family = "openrouter"
        else:
            st.exception(f"Invalid provider '{selected_provider}' in state. Cannot determine API client.")
            st.stop()

        if client is None:
            st.exception(f"API Client for {target_api_family or model_for_api_call} is None (Initialization failed?). Check API keys.")
            st.stop()

        # --- CONTEXT PREPARATION ---
        api_context_openai_format = []
        if not st.session_state.context_cleared_for_next_turn:
            context_limit = CONTEXT_MESSAGES_COUNT * 2
            history_for_context = [m for m in current_messages[:-1] if m["role"] != "system"]
            context_candidates_openai = history_for_context[-context_limit:]
            is_target_text_model = model_for_api_call in TEXT_MODELS
            processed_context_openai = []
            for msg in context_candidates_openai:
                role = msg["role"]
                original_content = msg.get("content")
                processed_content = None
                if role == "assistant":
                    if isinstance(original_content, str):
                        processed_content = original_content
                elif role == "user":
                    if is_target_text_model:
                        if isinstance(original_content, list):
                            text_parts = [i['text'] for i in original_content if i.get('type') == 'text']
                            processed_content = " ".join(text_parts) if text_parts else "[ImgCtx]"
                    else:
                        if isinstance(original_content, list):
                            processed_content = original_content
                if processed_content is not None:
                    processed_context_openai.append({"role": role, "content": processed_content})
            api_context_openai_format = processed_context_openai
        else:
            st.toast("🧹 Context cleared.")
        st.session_state.context_cleared_for_next_turn = False

        # --- Prepare final messages payload ---
        messages_for_api = api_context_openai_format + [last_user_message_openai_format]

        # --- WARNING for Content/Model Mismatch ---
        payload_has_images = any(item.get('type') == 'image_url' for item in last_user_message_openai_format['content'])
        if payload_has_images and model_for_api_call in TEXT_MODELS:
            st.warning(f"⚠️ Sending image data to text-only model '{model_for_api_call}'. API will likely error.", icon="⚠️")
        elif not payload_has_images and model_for_api_call in VISION_MODELS:
            st.info(f"ℹ️ Sending text-only request to vision model '{model_for_api_call}'.", icon="ℹ️")

        # --- Special Handling for Gemini with Google Search ---
        google_search_enabled = False
        if selected_provider == "Gemini" and model_for_api_call == "gemini-2.0-flash":
            google_search_enabled = True

        # --- Call the selected API Client ---
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            print(f"INFO: Calling {target_api_family.upper()} API with model: {model_for_api_call}")

            # Make the API call and get the stream object
            stream = client.chat_completion(model=model_for_api_call, messages=messages_for_api, stream=True, google_search=google_search_enabled)

            # Check if the stream object is valid
            if stream:
                response_content = []
                if target_api_family == "openai":
                    # Handle OpenAI streaming format (XAIClient)
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            response_content.append(content)
                            message_placeholder.markdown("".join(response_content))
                else:
                    # Handle GoogleClient and OpenRouterClient (custom generators)
                    for chunk in stream:
                        response_content.append(chunk)
                        message_placeholder.markdown("".join(response_content))
                response_content = "".join(response_content)
                # Check for Google Search usage if applicable
                if google_search_enabled and hasattr(stream, 'web_search_used') and stream.web_search_used:
                    response_content += "\n**Web Search: YES**"
                else:
                    response_content += "\n**Web Search: NO**"
                message_placeholder.markdown(response_content)
            else:
                raise ValueError(f"{target_api_family.upper()} client returned None stream, possibly due to connection or setup error.")

        # Store the result
        assistant_content_to_store = response_content
        st.session_state.chat_sessions.setdefault(st.session_state.current_chat_id, []).append(
            {"role": "assistant", "content": assistant_content_to_store}
        )
        st.rerun()

    except Exception as e:
        api_origin = target_api_family.upper() if target_api_family else "UNKNOWN"
        if not api_origin and model_for_api_call:
            if model_for_api_call.startswith("grok"):
                api_origin = "OPENAI"
            elif model_for_api_call.startswith("gemini"):
                api_origin = "GOOGLE"
            elif model_for_api_call in MODEL_PROVIDERS["openrouter"]:
                api_origin = "OPENROUTER"
        error_content_for_history = f"*API Error ({api_origin}): {str(e)}*"
        st.error(error_content_for_history)
        print(f"\n--- API Call Exception ({api_origin}) ---")
        traceback.print_exc()
        print(f"Provider: {st.session_state.selected_provider}, Model Attempted: {model_for_api_call}")
        print(f"Messages Sent: {repr(messages_for_api)}")
        print("---\n")
        try:
            st.session_state.chat_sessions.setdefault(st.session_state.current_chat_id, []).append(
                {"role": "assistant", "content": error_content_for_history}
            )
        except Exception as e_inner:
            st.error(f"History Error: Failed to add error message to chat history: {e_inner}")
        st.rerun()
