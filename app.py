"""Streamlit chat interface for GRAM Legal LLM with configurable modules."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
import torch

# Import local modules
from config import config
from model import GRAMTransformer, ModelConfig, create_model
from tokenizer import LegalBPETokenizer, get_tokenizer


# Page configuration
st.set_page_config(
    page_title=config.app_title,
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS for chat styling
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .config-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .config-full { background-color: #4caf50; color: white; }
    .config-us { background-color: #2196f3; color: white; }
    .config-eu { background-color: #ff9800; color: white; }
    .stChatInput { position: fixed; bottom: 0; left: 0; right: 0; background: white; padding: 1rem; z-index: 100; }
    .main-content { padding-bottom: 120px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model and tokenizer...")
def load_model_and_tokenizer(
    checkpoint_path: str = None,
    enable_us_module: bool = True,
    enable_eu_module: bool = True,
) -> tuple:
    """Load model and tokenizer with caching."""
    # Find checkpoint
    if checkpoint_path is None:
        checkpoint_dir = config.checkpoints_dir
        candidates = [
            checkpoint_dir / "best_model.pt",
            checkpoint_dir / "final_model.pt",
        ]
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))
        if checkpoints:
            candidates.append(checkpoints[-1])

        for c in candidates:
            if c.exists():
                checkpoint_path = str(c)
                break

    if checkpoint_path is None or not Path(checkpoint_path).exists():
        st.error(f"No model checkpoint found. Checked: {[str(c) for c in candidates]}")
        return None, None

    # Load tokenizer
    tokenizer_path = config.tokenizer_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        st.error(f"Tokenizer not found at {tokenizer_path}")
        return None, None

    tokenizer = LegalBPETokenizer.load(str(tokenizer_path))

    # Load model
    device = torch.device(config.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_cfg = ModelConfig(
        vocab_size=len(tokenizer),
        enable_us_module=enable_us_module,
        enable_eu_module=enable_eu_module,
    )
    model = create_model(model_cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    # Update config flags
    model.set_config(enable_us_module=enable_us_module, enable_eu_module=enable_eu_module)

    st.success(f"Loaded model from {checkpoint_path} (step {checkpoint.get('step', 'unknown')})")
    return model, tokenizer


def get_config_display_name(config: Dict[str, bool]) -> str:
    """Get display name for configuration."""
    if config["enable_us_module"] and config["enable_eu_module"]:
        return "Full (US + EU)"
    elif config["enable_us_module"]:
        return "US-only"
    elif config["enable_eu_module"]:
        return "EU-only"
    return "Core only"


def get_config_class(config: Dict[str, bool]) -> str:
    """Get CSS class for config badge."""
    if config["enable_us_module"] and config["enable_eu_module"]:
        return "config-full"
    elif config["enable_us_module"]:
        return "config-us"
    elif config["enable_eu_module"]:
        return "config-eu"
    return ""


def generate_response(
    model: GRAMTransformer,
    tokenizer: LegalBPETokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
) -> str:
    """Generate response from model."""
    device = next(model.parameters()).device

    # Encode prompt
    encoded = tokenizer.encode(
        prompt,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].unsqueeze(0).to(device)

    # Generate
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode (skip prompt)
    response_ids = generated[0][input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response.strip()


def render_sidebar():
    """Render sidebar with configuration options."""
    with st.sidebar:
        st.title("⚖️ GRAM Legal Assistant")

        st.markdown("---")
        st.subheader("Configuration")

        # Mode selection
        mode = st.selectbox(
            "Model Configuration",
            config.app_mode_options,
            index=0,
            help="Select which modules are active",
        )

        selected_config = config.app_mode_to_config[mode]

        # Display active config
        config_class = get_config_class(config)
        st.markdown(
            f'<div class="config-badge {config_class}">Active: {get_config_display_name(config)}</div>',
            unsafe_allow_html=True,
        )

        # Module toggles
        st.markdown("**Module Status:**")
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox("US Module", value=selected_config["enable_us_module"], disabled=True)
        with col2:
            st.checkbox("EU Module", value=selected_config["enable_eu_module"], disabled=True)

        st.markdown("---")

        # Generation parameters
        st.subheader("Generation Settings")
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=config.gen_temperature,
            step=0.1,
            help="Higher = more creative, lower = more focused",
        )
        top_k = st.slider(
            "Top-K",
            min_value=1,
            max_value=100,
            value=config.gen_top_k,
            step=1,
        )
        top_p = st.slider(
            "Top-P (Nucleus)",
            min_value=0.1,
            max_value=1.0,
            value=config.gen_top_p,
            step=0.05,
        )
        max_tokens = st.slider(
            "Max New Tokens",
            min_value=50,
            max_value=512,
            value=config.gen_max_new_tokens,
            step=50,
        )
        repetition_penalty = st.slider(
            "Repetition Penalty",
            min_value=1.0,
            max_value=2.0,
            value=config.gen_repetition_penalty,
            step=0.05,
        )

        st.markdown("---")

        # Model info
        st.subheader("Model Info")
        st.info("""
        **GRAM Legal LLM** - Jurisdiction-aware modular language model.

        **Configurations:**
        - **Full**: Core + US + EU modules
        - **US-only**: Core + US (EU disabled)
        - **EU-only**: Core + EU (US disabled)

        Uses Gradient-Routed Auxiliary Modules (GRAM) for jurisdiction-specific training.
        """)

        # Clear chat button
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        return {
            "mode": mode,
            "config": selected_config,
            "generation_params": {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "max_new_tokens": max_tokens,
                "repetition_penalty": repetition_penalty,
            },
        }


def render_chat_interface(model, tokenizer, generation_params):
    """Render the main chat interface."""
    # Initialize messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                config_class = get_config_class(message.get("config", {}))
                config_name = get_config_display_name(message.get("config", {}))
                st.markdown(
                    f'<div class="config-badge {config_class}">Mode: {config_name}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a legal question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            config = st.session_state.get("current_config", {})
            config_class = get_config_class(config)
            config_name = get_config_display_name(config)
            st.markdown(
                f'<div class="config-badge {config_class}">Mode: {config_name}</div>',
                unsafe_allow_html=True,
            )

            with st.spinner("Generating response..."):
                try:
                    response = generate_response(
                        model,
                        tokenizer,
                        prompt,
                        **generation_params,
                    )
                    st.markdown(response)

                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "config": config,
                    })
                except Exception as e:
                    st.error(f"Error generating response: {e}")


def render_example_prompts():
    """Render example prompts in sidebar."""
    with st.sidebar:
        st.markdown("---")
        st.subheader("Example Prompts")

        examples = {
            "US Law": [
                "What are the elements of negligence in US tort law?",
                "Explain the doctrine of stare decisis in US courts.",
                "What is the standard for summary judgment under Rule 56?",
                "Describe the Erie doctrine in federal courts.",
            ],
            "EU Law": [
                "What is the principle of supremacy in EU law?",
                "Explain the preliminary ruling procedure under Article 267 TFEU.",
                "What are the four freedoms of the EU single market?",
                "Describe the principle of proportionality in EU law.",
            ],
            "Comparative": [
                "Compare US and EU approaches to data privacy.",
                "How do US and EU competition law differ?",
                "Contrast judicial review in US and EU systems.",
            ],
        }

        for category, prompts in examples.items():
            with st.expander(category):
                for p in prompts:
                    if st.button(p, key=f"ex_{p[:20]}", use_container_width=True):
                        st.session_state.example_prompt = p
                        st.rerun()


def main():
    """Main Streamlit app."""
    st.title("⚖️ GRAM Legal Assistant")
    st.caption("Jurisdiction-aware modular legal language model")

    # Check if model exists
    model_exists = (
        (config.checkpoints_dir / "best_model.pt").exists() or
        (config.checkpoints_dir / "final_model.pt").exists() or
        len(list(config.checkpoints_dir.glob("checkpoint_step_*.pt"))) > 0
    )
    tokenizer_exists = (config.tokenizer_dir / "tokenizer.json").exists()

    if not model_exists or not tokenizer_exists:
        st.error("""
        **Model or tokenizer not found!**

        Please run the following first:
        1. Train tokenizer: `python tokenizer.py`
        2. Train model: `python train.py`
        3. Then run this app: `streamlit run app.py`
        """)
        st.stop()

    # Render sidebar and get config
    sidebar_state = render_sidebar()
    ui_config = sidebar_state["config"]
    generation_params = sidebar_state["generation_params"]

    # Store current config in session
    st.session_state.current_config = ui_config

    # Handle example prompt
    if "example_prompt" in st.session_state:
        prompt = st.session_state.pop("example_prompt")
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    # Load model with current config
    model, tokenizer = load_model_and_tokenizer(
        enable_us_module=ui_config["enable_us_module"],
        enable_eu_module=ui_config["enable_eu_module"],
    )

    if model is None or tokenizer is None:
        st.stop()

    # Render example prompts
    render_example_prompts()

    # Render chat interface
    render_chat_interface(model, tokenizer, generation_params)


if __name__ == "__main__":
    main()