import json
import sentencepiece as spm
import os


def extract_text_from_manifest(manifest_path):
    """
    Extract grapheme text from G2P manifest for tokenizer training
    """
    texts = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            texts.append(entry['text_graphemes'])

    # Write to temporary text file
    output_file = 'armenian_tokenizer_input.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(texts))

    return output_file


def train_armenian_tokenizer(input_file, model_prefix='armenian_g2p'):
    """
    Train a SentencePiece tokenizer for Armenian G2P model

    Args:
    input_file (str): Path to input text file for training tokenizer
    model_prefix (str): Prefix for saved tokenizer model files
    """
    # Ensure output directory exists
    os.makedirs('tokenizer', exist_ok=True)

    # Full path for tokenizer model
    model_path = os.path.join('tokenizer', model_prefix)

    # SentencePiece training parameters
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_path,
        model_type='bpe',  # Byte-Pair Encoding
        vocab_size=1000,  # Adjust based on your dataset size
        character_coverage=1.0,  # Full coverage for Armenian script
        normalization_rule_name='identity',  # Preserve original script
        max_sentencepiece_length=16,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )

    # Correct model path
    full_model_path = f"{model_path}.model"
    print(f"Tokenizer model saved at {full_model_path}")
    return full_model_path


# Main execution
if __name__ == '__main__':
    # Path to your training manifest
    manifest_path = "/home/general/PhD/ArmG2PNemo/armenian_g2p_train_manifest.json"

    # Extract text for tokenizer training
    input_text_file = extract_text_from_manifest(manifest_path)

    # Train tokenizer
    tokenizer_model_path = train_armenian_tokenizer(input_text_file)

    # Print the exact path for use in NEMO configuration
    print(f"\nUse this path in your NEMO configuration:")
    print(f"model.tokenizer.dir={tokenizer_model_path}")

    # Optional: Test the tokenizer
    sp = spm.SentencePieceProcessor(model_file=tokenizer_model_path)

    # Print some sample tokenizations
    test_texts = ["աբա", "Աբադան", "հայաստան"]
    for text in test_texts:
        print(f"\nOriginal: {text}")
        print(f"Tokenized: {sp.encode(text, out_type=str)}")
        print(f"Piece IDs: {sp.encode(text)}")