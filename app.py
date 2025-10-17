import streamlit as st
import io
import numpy as np
import soundfile as sf
# Note: 'kokoro' is assumed to be an available library for TTS generation.
from kokoro import KPipeline 

# --- Configuration ---
# Assuming a fixed list of voices for demonstration
# Replace with actual available voices if known
AVAILABLE_VOICES = ['af_heart', 'mock_voice_1', 'mock_voice_2']
SAMPLE_RATE = 24000
LANG_CODE = 'a'
# Explicitly define the repository ID to suppress the warning during loading
KOKORO_REPO_ID = 'hexgrad/Kokoro-82M'

# --- Model Initialization ---
# Use st.cache_resource to load the heavy model only once across user sessions
@st.cache_resource
def load_tts_pipeline():
    """Initializes and caches the Kokoro TTS pipeline."""
    try:
        # st.info(f"Initializing KPipeline for language code '{LANG_CODE}' with repo_id='{KOKORO_REPO_ID}'...")
        # KPipeline is assumed to handle model loading
        # FIX: Added explicit repo_id to suppress the warning
        pipeline = KPipeline(lang_code=LANG_CODE, repo_id=KOKORO_REPO_ID) 
        st.success("TTS Pipeline initialized successfully!")
        return pipeline
    except Exception as e:
        # This handles cases where model files might be missing or dependencies fail
        st.error(f"ERROR: Could not initialize KPipeline for TTS: {e}")
        st.warning("The Text-to-Speech function will be disabled.")
        return None

tts_pipeline = load_tts_pipeline()

# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="Streamlit Text-to-Speech (TTS)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🗣️ Streamlit Text-to-Speech App")
st.markdown("Generate and track speech history using the `kokoro` TTS pipeline.")

# Initialize chat history in session state
if 'history' not in st.session_state:
    st.session_state.history = []

# --- TTS Generation Form ---
with st.form(key='tts_form'):
    input_text = st.text_area(
        "Text to convert to Speech (max 500 characters)",
        placeholder="Enter the text you want to convert to audio...",
        max_chars=500
    )
    
    selected_voice = st.selectbox(
        "Select Voice",
        options=AVAILABLE_VOICES,
        # Set the default selection
        index=AVAILABLE_VOICES.index('af_heart') if 'af_heart' in AVAILABLE_VOICES else 0
    )
    
    submit_button = st.form_submit_button(label='Generate Speech')

if submit_button:
    if tts_pipeline is None:
        st.error("Cannot generate speech because the TTS model failed to initialize.")
    elif not input_text.strip():
        st.warning("Please enter some text to generate speech.")
    else:
        # Show spinner while generating
        with st.spinner("Generating audio..."):
            try:
                # Generate audio using the Kokoro pipeline
                # The generator is expected to yield (pitch, loudness, audio_chunk)
                generator = tts_pipeline(input_text, voice=selected_voice)
                
                # Collect all audio chunks from the generator (third element is audio_chunk)
                all_audio_chunks = [audio_chunk for _, _, audio_chunk in generator]

                if not all_audio_chunks:
                    st.error("No audio could be generated for the provided text. Please try different text or voice.")
                else:
                    # Concatenate all audio segments into a single numpy array
                    audio_data = np.concatenate(all_audio_chunks)
                    
                    # Create an in-memory byte buffer for the WAV file
                    buffer = io.BytesIO()
                    
                    # Write the audio data to the buffer in WAV format
                    sf.write(buffer, audio_data, SAMPLE_RATE, format='WAV')
                    buffer.seek(0) # Rewind the buffer

                    # Play the audio in Streamlit
                    st.success("Audio generated successfully! Playing now...")
                    # WARNING FIX: Removed sample_rate argument as audio data is encoded (bytes)
                    st.audio(buffer.getvalue(), format='audio/wav')
                    
                    # --- Update History ---
                    # Store the raw bytes in the session state for replay in history
                    st.session_state.history.append({
                        'text': input_text,
                        'voice': selected_voice,
                        'audio_data': buffer.getvalue(), 
                    })
                    
            except Exception as e:
                st.error(f"An unexpected error occurred during TTS generation: {e}")

# --- Display History (in Sidebar) ---
st.sidebar.header("Generation History")

if st.session_state.history:
    # Iterate through history in reverse chronological order
    for i, entry in enumerate(reversed(st.session_state.history)):
        # Calculate the actual index in the non-reversed list for unique key generation
        original_index = len(st.session_state.history) - 1 - i
        
        # Display each entry in an expandable box
        with st.sidebar.expander(f"Message {original_index + 1} ({entry['voice']})", expanded=False):
            st.caption(f"**Voice:** {entry['voice']}")
            st.markdown(f"*{entry['text']}*")
            
            # Recreate buffer from stored bytes and use st.audio for replay
            audio_buffer = io.BytesIO(entry['audio_data'])
            
            # WARNING FIX: Removed sample_rate argument as audio data is encoded (bytes)
            st.audio(audio_buffer.getvalue(), format='audio/wav')

    # Button to clear the history
    if st.sidebar.button("Clear History"):
        st.session_state.history = []
        st.rerun() # Rerun to update the UI immediately
else:
    st.sidebar.info("No speech generated yet. History will appear here after your first submission.")
