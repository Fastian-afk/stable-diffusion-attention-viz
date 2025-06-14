### ✅ `stable-diffusion-attention-viz/README.md`

# 🎨 Stable Diffusion Token-to-Region Visualizer

This tool captures and visualizes token-level cross-attention inside Stable Diffusion, revealing which text tokens influence which parts of the generated image.

## 🔎 Key Features
- Hooks into Stable Diffusion attention layers
- Supports multilingual prompts (Chinese, Urdu, Spanish)
- Aggregates attention across denoising steps
- Generates heatmaps for each token

## 🧠 Why It Matters
Understand the semantic grounding of text in text-to-image generation. Especially useful for prompt engineering, explainability, and debugging.

## 💡 Technologies Used
- Hugging Face Diffusers
- PyTorch
- Matplotlib

## 📸 Example Output
Heatmap showing token influence over generated regions.

## ✍️ Author
Imaad Fazal – Knowledge Discovery & Data Science Lab, FAST-NUCES
