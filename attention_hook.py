from diffusers.models.attention import BasicTransformerBlock
import torch

attention_store = []

def register_attention_hooks(pipe):
    for name, module in pipe.unet.named_modules():
        if isinstance(module, BasicTransformerBlock):
            original_forward = module.attn2.forward

            def custom_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
                query = self.to_q(hidden_states)
                key = self.to_k(encoder_hidden_states)
                value = self.to_v(encoder_hidden_states)

                query = self.head_to_batch_dim(query)
                key = self.head_to_batch_dim(key)
                value = self.head_to_batch_dim(value)

                attention_scores = torch.baddbmm(
                    torch.empty(query.size(0), query.size(1), key.size(1), device=query.device, dtype=query.dtype),
                    query,
                    key.transpose(-1, -2),
                    beta=0,
                    alpha=self.scale,
                )
                attention_probs = attention_scores.softmax(dim=-1)
                attention_store.append(attention_probs.detach().cpu())

                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = self.batch_to_head_dim(hidden_states)
                hidden_states = self.to_out[0](hidden_states)
                hidden_states = self.to_out[1](hidden_states)
                return hidden_states

            module.attn2.forward = custom_forward.__get__(module.attn2, type(module.attn2))

    return attention_store