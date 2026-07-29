You are an expert PyTorch engineer, ML researcher, and Streamlit app developer.

Goal:
Implement an end-to-end prototype of a GRAM-style modular pretraining pipeline
for a jurisdiction-aware legal language model, PLUS a Streamlit interface
to chat with the model and switch between three configurations:
- Full (core + US + EU modules)
- US-only (core + US module, EU module disabled)
- EU-only (core + EU module, US module disabled)

Use GRAM (Gradient-Routed Auxiliary Modules) as introduced in the paper
“Modular Pretraining Enables Access Control”: apply data-dependent
gradient routing masks so that US- and EU-labeled batches update
only their respective auxiliary modules, while core parameters are
frozen on jurisdiction-specific batches.

Part 1: Model, training, and evaluation (same as before)

1. Framework and environment
   - Use Python 3 and PyTorch.
   - Assume a single or few GPUs (no massive distributed setup).
   - Use standard training utilities (DataLoader, etc.).
   - Code should be runnable as-is with minimal modifications.

2. Datasets:
   - US law dataset: Hugging Face "free-law/Caselaw_Access_Project".
     (public domain US cases). [US jurisdiction] [web:57]
   - EU law dataset: CEPS EurLex dataset from Kaggle (EU legal acts). [EU jurisdiction] [web:50]
   - Optionally a small "general" corpus (e.g., Wikipedia or generic English)
     to give the core model broader language skills.

   Implement:
   - A data module that:
     * Downloads/reads these datasets.
     * Cleans and normalizes text.
     * Splits documents into chunks of N tokens (e.g., 512–1024).
     * Assigns a label `jurisdiction` ∈ {"US", "EU", "general"} to each chunk.
     * Tokenizes text with a shared BPE/WordPiece vocabulary learned on combined data.

3. Model architecture (GRAM-style):

   Implement a small decoder-only Transformer (GPT-like), e.g. 100M–300M parameters:
   - Embedding layer.
   - Several Transformer blocks (self-attention + feed-forward).
   - Output projection to vocabulary size.

   Extend each Transformer block with GRAM-style modules:
   - Shared "core" sub-block (standard attention + MLP).
   - US module MLP: extra neurons dedicated to US law.
   - EU module MLP: extra neurons dedicated to EU law.

   Design the forward pass such that:
   - Core output is always computed.
   - Module outputs are added or concatenated to the core representation,
     controlled by flags indicating whether US/EU modules are enabled.

   Include a simple configuration mechanism:
   - `enable_us_module: bool`
   - `enable_eu_module: bool`
   that can be toggled at runtime to simulate ablating modules
   (e.g., turning US or EU capabilities off).

4. Gradient routing / training logic (GRAM):

   Implement GRAM-style gradient routing during training:
   - For batches where `jurisdiction == "US"`:
     * Freeze core parameters and freeze EU module parameters.
     * Enable forward and backward pass only for the US module.
   - For batches where `jurisdiction == "EU"`:
     * Freeze core parameters and freeze US module parameters.
     * Enable forward and backward pass only for the EU module.
   - For batches where `jurisdiction == "general"`:
     * Allow updates to core and both modules (or at least core),
       to maintain general language ability.

   Use:
   - Parameter groups for core, US module, EU module.
   - Logic to toggle `requires_grad` per batch.
   - Clean abstraction so the training loop remains readable.

5. Training script:

   Create a `train.py` that:
   - Sets up the tokenizer and loads the datasets.
   - Builds train/validation DataLoaders for US, EU, and general data.
   - Constructs the model with core + US + EU modules.
   - Uses cross-entropy loss for next-token prediction.
   - Implements a training schedule, for example:
     * Phase 1: general warm-up.
     * Phase 2: GRAM phase (alternate US and EU batches with gradient routing).
   - Logs training/validation loss per jurisdiction.

6. Evaluation / ablation experiments:

   Implement `evaluate.py` that:

   - Loads the trained model.
   - Defines small test sets for US and EU.
   - Evaluates three configs:
     1) Full: core + US + EU.
     2) US-ablated: core + EU only.
     3) EU-ablated: core + US only.

   For each config:
   - Compute perplexity or loss on US and EU test data.
   - Print a table showing performance drops or preservation, similar to GRAM. [web:6][web:16]

7. Code organization:

   Organize into:
   - `tokenizer.py` — build/load tokenizer.
   - `datasets.py` — dataset download, preprocessing, DataLoader creation.
   - `model.py` — Transformer + GRAM modules.
   - `train.py` — training loop + gradient routing.
   - `evaluate.py` — ablation experiments.
   - `config.py` — hyperparameters and paths.

Part 2: Streamlit interface to switch configurations and chat

8. Streamlit app (new requirement):

   Create `app.py` using Streamlit that:

   - Loads the trained model once at startup (use caching where appropriate).
   - Provides a sidebar or top-level control to choose the active configuration
     using a selectbox, e.g.:

       - "Full (US + EU)"
       - "US-only"
       - "EU-only"

     This can be done with `st.selectbox`. [web:62][web:69]

   - Based on the selected option, sets the model flags:
       - Full: `enable_us_module = True`, `enable_eu_module = True`.
       - US-only: `enable_us_module = True`, `enable_eu_module = False`.
       - EU-only: `enable_us_module = False`, `enable_eu_module = True`.

   - Implements a simple chat interface (like Streamlit's LLM chat examples): [web:63][web:73]
       * A text input box for the user’s legal question.
       * A "Send" button.
       * A chat history display (use `st.chat_message` or a simple text area).

   - When the user submits a question:
       * Run the model in the currently selected configuration.
       * Generate a response using basic sampling (top-k/top-p) from the model.
       * Display the response along with the current configuration name
         (so the user sees whether they’re in Full / US-only / EU-only mode).

   - Optionally, show a small info panel explaining the three modes.

   Make sure:
   - The Streamlit app does NOT retrain the model; it only loads and uses
     the trained weights.
   - Model loading is done once (e.g., with `st.cache_resource` or similar)
     to avoid reloading on every interaction.

9. File organization for the app:

   - `app.py` — main Streamlit file with:
     * Model loading.
     * Mode selectbox.
     * Chat UI and response generation.
   - Reuse `model.py` and tokenizer utilities from the training code.

10. Practical details for Streamlit:

   - Use `st.selectbox` for mode selection. [web:62][web:69]
   - Use basic layout (sidebar for mode, main area for chat).
   - Assume the app is run with `streamlit run app.py`.

11. General requirements:

   - Keep code readable and well-commented; this is a research prototype.
   - Include TODOs where large-scale optimization or deployment details
     would be needed.
   - Provide instructions at the top-level README comments:
     * How to install dependencies.
     * How to run training.
     * How to run evaluation.
     * How to run Streamlit (`streamlit run app.py`).

Please generate all core files with placeholder hyperparameters and
simple training loops, then implement the Streamlit app that switches
between Full, US-only, and EU-only configurations and provides a basic
chat interface to query the model.