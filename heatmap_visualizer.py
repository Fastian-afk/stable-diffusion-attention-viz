import matplotlib.pyplot as plt

def show_token_attention(attention_tensor, token_index, step, decoded_tokens):
    token_attention = attention_tensor[step][token_index].reshape(64, 64)
    plt.figure(figsize=(6, 6))
    plt.imshow(token_attention, cmap='inferno')
    plt.title(f"Token: {decoded_tokens[token_index]} | Step {step}")
    plt.axis('off')
    plt.colorbar()
    plt.show()