import streamlit as st
import base64
import io
from PIL import Image
import time
# Import clients
from api_clients import XAIClient, GoogleClient # Assuming api_clients.py is correct
import traceback
import json

# --- Configuration ---
TEXT_MODELS = ["grok-2-latest"]
VISION_MODELS = ["grok-2-vision-latest", "gemini-2.0-flash"]
ALL_AVAILABLE_MODELS = TEXT_MODELS + VISION_MODELS
DEFAULT_MODEL = "grok-2-latest"
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
    # Chat state
    if "chat_sessions" not in st.session_state: st.session_state.chat_sessions = {}
    if "current_chat_id" not in st.session_state:
        first_chat_id = f"chat_{time.time()}"; st.session_state.chat_sessions[first_chat_id] = []; st.session_state.current_chat_id = first_chat_id
    if "context_cleared_for_next_turn" not in st.session_state: st.session_state.context_cleared_for_next_turn = False
    # Image/Upload state
    if "staged_images_bytes" not in st.session_state: st.session_state.staged_images_bytes = []
    if "uploader_key_counter" not in st.session_state: st.session_state.uploader_key_counter = 0
    # Model selection state
    if "selected_model" not in st.session_state: st.session_state.selected_model = DEFAULT_MODEL

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
st.info("ℹ️ Chat history session-based. Select model carefully based on content.")

# --- Simplified MODEL SELECTION DROPDOWN ---
st.selectbox(
    "Select Model:",
    options=ALL_AVAILABLE_MODELS, # Always show all
    key="selected_model",         # State updated by widget interaction
    help="Choose the desired model. Ensure it matches content (text/vision)."
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
    target_api_family = None # Initialize for use in except block
    model_for_api_call = ""  # Initialize for use in except block

    try:
        # --- Direct Model and Client Determination ---
        model_for_api_call = st.session_state.selected_model
        print(f"DEBUG: API Call - Using selected model from state: '{model_for_api_call}'")

        client = None
        if model_for_api_call.startswith("grok"):
            client = st.session_state.xai_client
            target_api_family = "openai"
        elif model_for_api_call.startswith("gemini"):
            client = st.session_state.google_client
            target_api_family = "google"
        else:
            # Use st.exception to show error and stop nicely
            st.exception(f"Invalid model name '{model_for_api_call}' in state. Cannot determine API client.")
            st.stop() # Stop execution

        if client is None:
             st.exception(f"API Client for {target_api_family or model_for_api_call} is None (Initialization failed?). Check API keys.")
             st.stop()
        # --- End of Direct Model and Client Determination ---

        # --- CONTEXT PREPARATION ---
        api_context_openai_format = []
        if not st.session_state.context_cleared_for_next_turn:
            context_limit = CONTEXT_MESSAGES_COUNT * 2
            history_for_context = [m for m in current_messages[:-1] if m["role"] != "system"]
            context_candidates_openai = history_for_context[-context_limit:]
            is_target_text_model = model_for_api_call in TEXT_MODELS
            processed_context_openai = []
            for msg in context_candidates_openai:
                role = msg["role"]; original_content = msg.get("content"); processed_content = None
                if role == "assistant":
                    if isinstance(original_content, str): processed_content = original_content
                elif role == "user":
                    if is_target_text_model:
                        if isinstance(original_content, list):
                            text_parts = [i['text'] for i in original_content if i.get('type') == 'text']
                            processed_content = " ".join(text_parts) if text_parts else "[ImgCtx]"
                    else: # Vision model
                        if isinstance(original_content, list): processed_content = original_content
                if processed_content is not None: processed_context_openai.append({"role": role, "content": processed_content})
            api_context_openai_format = processed_context_openai
        else: st.toast("🧹 Context cleared.")
        st.session_state.context_cleared_for_next_turn = False
        # --- END OF CONTEXT PREPARATION ---

        # --- Prepare final messages payload ---
        messages_for_api = api_context_openai_format + [last_user_message_openai_format]

        # --- WARNING for Content/Model Mismatch ---
        payload_has_images = any(item.get('type') == 'image_url' for item in last_user_message_openai_format['content'])
        if payload_has_images and model_for_api_call in TEXT_MODELS:
            st.warning(f"⚠️ Sending image data to text-only model '{model_for_api_call}'. API will likely error.", icon="⚠️")
        elif not payload_has_images and model_for_api_call in VISION_MODELS:
            st.info(f"ℹ️ Sending text-only request to vision model '{model_for_api_call}'.", icon="ℹ️")

        # --- Call the selected API Client ---
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            print(f"INFO: Calling {target_api_family.upper()} API with model: {model_for_api_call}")

            # Make the API call and get the stream object (or None on immediate failure)
            stream = client.chat_completion(model=model_for_api_call, messages=messages_for_api, stream=True)

            # Check if the stream object is valid before trying to write it
            if stream:
                 response_content = message_placeholder.write_stream(stream)
            else:
                 # Handle cases where the client explicitly returned None (e.g., connection error handled inside client)
                 # We raise an exception here to be caught by the main handler below, ensuring consistent error formatting
                 raise ValueError(f"{target_api_family.upper()} client returned None stream, possibly due to connection or setup error.")

        # If stream processing succeeded, store the result
        assistant_content_to_store = str(response_content)
        st.session_state.chat_sessions.setdefault(st.session_state.current_chat_id, []).append(
            {"role": "assistant", "content": assistant_content_to_store}
        )
        st.rerun() # Rerun after successful response

    # --- Centralized Exception Handling ---
    except Exception as e:
        # Determine API family again safely, defaulting if needed
        api_origin = target_api_family.upper() if target_api_family else "UNKNOWN"
        if not api_origin and model_for_api_call: # Try to guess from model name if family wasn't set
             if model_for_api_call.startswith("grok"): api_origin = "OPENAI"
             elif model_for_api_call.startswith("gemini"): api_origin = "GOOGLE"

        # Construct the detailed error message for chat history
        error_content_for_history = f"*API Error ({api_origin}): {e}*"

        # Display temporary error in UI
        st.error(error_content_for_history)

        # Print details to console
        print(f"\n--- API Call Exception ({api_origin}) ---"); traceback.print_exc()
        print(f"Model Attempted: {model_for_api_call}"); print(f"Messages Sent: {repr(messages_for_api)}"); print("---\n")

        # Add detailed error to chat history
        try:
            st.session_state.chat_sessions.setdefault(st.session_state.current_chat_id, []).append(
                 {"role": "assistant", "content": error_content_for_history} # Use the detailed message
            )
        except Exception as e_inner:
             st.error(f"History Error: Failed to add error message to chat history: {e_inner}")

        st.rerun() # Rerun to display the error message in the chat history
