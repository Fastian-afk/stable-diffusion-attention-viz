from transformers import pipeline


def translate_prompt(prompt, target_lang="ur"):
    model_map = {
        "ur": "Helsinki-NLP/opus-mt-en-ur",
        "es": "Helsinki-NLP/opus-mt-en-es",
        "zh": "Helsinki-NLP/opus-mt-en-zh"
    }
    if target_lang not in model_map:
        raise ValueError("Unsupported language.")

    translator = pipeline("translation", model=model_map[target_lang])
    return translator(prompt, max_length=128)[0]["translation_text"]