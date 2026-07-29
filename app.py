"""
GRAM Legal LLM - Streamlit Chat Interface
==========================================

Interactive chat interface for the GRAM Legal Language Model.
Switches between Full (US+EU), US-only, and EU-only configurations.

Run: streamlit run app.py
"""

import streamlit as st
import torch
from pathlib import Path

from config import config
from model import GRAMModel, ModelConfig
from tokenizer import load_tokenizer, encode_with_jurisdiction, decode_tokens, SPECIAL_TOKENS


st.set_page_config(
    page_title="GRAM Legal LLM",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model_and_tokenizer(checkpoint_path: str = None):
    """Load model and tokenizer (cached)."""
    tokenizer = load_tokenizer(str(config.tokenizer_dir))
    
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=config.max_seq_len,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        layer_norm_eps=config.layer_norm_eps,
        adapter_rank=config.adapter_rank,
        adapter_alpha=config.adapter_alpha,
        adapter_dropout=config.adapter_dropout,
        jurisdictions=config.jurisdictions,
    )
    
    model = GRAMModel(model_config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        st.success(f"Loaded checkpoint from {checkpoint_path}")
    else:
        st.warning("No checkpoint loaded. Using randomly initialized model.")
    
    return model, tokenizer, device


def generate_response(
    model,
    tokenizer,
    device,
    prompt: str,
    enable_us_module: bool,
    enable_eu_module: bool,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
) -> str:
    """Generate response from the model with specified module configuration."""
    
    jurisdiction = "US" if enable_us_module else ("EU" if enable_eu_module else "general")
    
    encoded = encode_with_jurisdiction(tokenizer, prompt, jurisdiction, max_length=config.max_seq_len - max_new_tokens)
    
    input_ids = torch.tensor([encoded["input_ids"]], device=device)
    attention_mask = torch.tensor([encoded["attention_mask"]], device=device)
    
    model.set_jurisdiction(jurisdiction)
    model.enable_us_module = enable_us_module
    model.enable_eu_module = enable_eu_module
    
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            jurisdiction=jurisdiction,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    new_tokens = generated[0, input_ids.shape[1]:].tolist()
    response = decode_tokens(tokenizer, new_tokens)
    
    return response


def format_prompt(messages: list, jurisdiction: str) -> str:
    """Format chat messages into a prompt."""
    jurisdiction_markers = {
        "US": "[US Law]",
        "EU": "[EU Law]",
        "general": "[General Legal]",
    }
    marker = jurisdiction_markers.get(jurisdiction, "[General Legal]")
    
    formatted = f"{marker}\n\n"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            formatted += f"User: {content}\n\n"
        elif role == "assistant":
            formatted += f"Assistant: {content}\n\n"
    
    formatted += "Assistant:"
    return formatted


def main():
    st.title("⚖️ GRAM Legal LLM")
    st.caption("Generic + Region-Adaptive Module for Legal Language Modeling")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        checkpoint_dir = config.checkpoint_dir
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        checkpoint_names = ["None (random init)"] + [cp.name for cp in checkpoints]
        
        selected_checkpoint = st.selectbox(
            "Model Checkpoint",
            checkpoint_names,
            index=0,
        )
        
        checkpoint_path = None
        if selected_checkpoint != "None (random init)":
            checkpoint_path = str(checkpoint_dir / selected_checkpoint)
        
        st.subheader("Model Configuration")
        config_mode = st.selectbox(
            "Configuration",
            ["Full (US + EU)", "US-only", "EU-only"],
            index=0,
            help=(
                "Full: core + US module + EU module\n"
                "US-only: core + US module, EU disabled\n"
                "EU-only: core + EU module, US disabled"
            ),
        )
        
        enable_us_module = config_mode in ["Full (US + EU)", "US-only"]
        enable_eu_module = config_mode in ["Full (US + EU)", "EU-only"]
        
        st.divider()
        
        st.subheader("Generation Parameters")
        max_new_tokens = st.slider("Max New Tokens", 64, 1024, 256, 32)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
        top_k = st.slider("Top-K", 1, 100, 50, 1)
        top_p = st.slider("Top-P", 0.1, 1.0, 0.9, 0.05)
        
        st.divider()
        
        if st.button("🔄 Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.caption("Model Info")
        st.caption(f"Vocab: {config.tokenizer_vocab_size:,}")
        st.caption(f"Layers: {config.n_layers}")
        st.caption(f"Dim: {config.d_model}")
        st.caption(f"Heads: {config.n_heads}")
        st.caption(f"Adapters: {len(config.jurisdictions)} × rank {config.adapter_rank}")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    
    if not st.session_state.model_loaded or st.session_state.get("last_checkpoint") != checkpoint_path:
        with st.spinner("Loading model..."):
            model, tokenizer, device = load_model_and_tokenizer(checkpoint_path)
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.device = device
            st.session_state.model_loaded = True
            st.session_state.last_checkpoint = checkpoint_path
    
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    device = st.session_state.device
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a legal question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                jurisdiction = "US" if enable_us_module else ("EU" if enable_eu_module else "general")
                formatted_prompt = format_prompt(st.session_state.messages, jurisdiction)
                
                response = generate_response(
                    model,
                    tokenizer,
                    device,
                    formatted_prompt,
                    enable_us_module=enable_us_module,
                    enable_eu_module=enable_eu_module,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                
                st.markdown(response)
                st.caption(f"Mode: {config_mode}")
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    with st.expander("📋 Model Architecture Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Core Model**")
            st.markdown(f"- Layers: {config.n_layers}")
            st.markdown(f"- Hidden Dim: {config.d_model}")
            st.markdown(f"- Attention Heads: {config.n_heads}")
            st.markdown(f"- FFN Dim: {config.d_ff}")
            st.markdown(f"- Max Seq Len: {config.max_seq_len}")
            st.markdown(f"- Dropout: {config.dropout}")
        
        with col2:
            st.markdown("**Jurisdiction Adapters (LoRA)**")
            st.markdown(f"- Rank: {config.adapter_rank}")
            st.markdown(f"- Alpha: {config.adapter_alpha}")
            st.markdown(f"- Dropout: {config.adapter_dropout}")
            st.markdown(f"- Jurisdictions: {', '.join(config.jurisdictions)}")
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            st.markdown(f"- Total Params: {total_params/1e6:.1f}M")
            st.markdown(f"- Trainable: {trainable_params/1e6:.1f}M")
    
    with st.expander("🔧 Special Tokens"):
        st.json(SPECIAL_TOKENS)


if __name__ == "__main__":
    main()