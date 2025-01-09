import argparse
import nemo
from nemo.collections.tts.g2p.models.ctc import CTCG2PModel
import sys


def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Armenian Grapheme-to-Phoneme Converter')
    parser.add_argument('--model', type=str, required=False,
                        default="/home/general/PhD/ArmG2PNemo/NeMo/examples/tts/g2p/nemo_experiments/G2P-Conformer-CTC/2025-01-10_01-55-38/checkpoints/G2P-Conformer-CTC.nemo",
                        help='Path to the .nemo model file')
    parser.add_argument('--words', type=str, nargs='*',
                        help='Words to convert to phonemes. If not provided, runs in interactive mode')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for prediction')
    return parser.parse_args()


def load_g2p_model(model_path):
    """
    Load a pretrained G2P model

    Args:
        model_path (str): Path to the .nemo model file

    Returns:
        Loaded G2P model
    """
    try:
        print(f"Loading model from {model_path}...")
        g2p_model = CTCG2PModel.restore_from(model_path)
        print("Model loaded successfully!")
        return g2p_model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(f"Full error details: {repr(e)}")
        return None


def predict_phonemes(model, graphemes, batch_size=32):
    """
    Predict phonemes for given graphemes

    Args:
        model: Loaded G2P model
        graphemes (str or list): Input word(s) to convert
        batch_size (int): Batch size for prediction

    Returns:
        List of predicted phoneme sequences
    """
    try:
        # Create a temporary manifest for prediction
        import json
        import tempfile
        import os

        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_in, \
                tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_out:

            # Write input words to temporary manifest
            if isinstance(graphemes, str):
                graphemes = [graphemes]

            for word in graphemes:
                f_in.write(json.dumps({"text_graphemes": word}, ensure_ascii=False) + "\n")

            input_manifest = f_in.name
            output_manifest = f_out.name

        # Convert graphemes to phonemes
        predictions = model.convert_graphemes_to_phonemes(
            manifest_filepath=input_manifest,
            output_manifest_filepath=output_manifest,
            batch_size=batch_size
        )

        # Clean up temporary files
        os.unlink(input_manifest)
        os.unlink(output_manifest)

        return predictions

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        print(f"Full error details: {repr(e)}")
        return None


def interactive_mode(model):
    """
    Interactive mode for phoneme prediction
    """
    print("\n=== Armenian Grapheme-to-Phoneme Converter ===")
    print("Enter words to convert to phonemes. Type 'exit' to quit.")
    print("Type 'batch' to enter batch mode for multiple words.")

    while True:
        try:
            user_input = input("\nEnter a word (or 'exit'/'batch'): ").strip()

            if user_input.lower() == 'exit':
                break

            if user_input.lower() == 'batch':
                print("Enter words (one per line). Type 'done' when finished:")
                words = []
                while True:
                    word = input().strip()
                    if word.lower() == 'done':
                        break
                    if word:
                        words.append(word)

                if words:
                    phonemes = predict_phonemes(model, words)
                    if phonemes:
                        for word, phoneme in zip(words, phonemes):
                            print(f"\nGrapheme: {word}")
                            print(f"Phonemes: {phoneme}")
                continue

            if not user_input:
                continue

            phonemes = predict_phonemes(model, user_input)
            if phonemes:
                print(f"\nGrapheme: {user_input}")
                print(f"Phonemes: {phonemes[0]}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


def main():
    """
    Main function
    """
    # Parse command line arguments
    args = parse_arguments()

    # Load the model
    g2p_model = load_g2p_model(args.model)
    if not g2p_model:
        print("Failed to load the model. Exiting.")
        sys.exit(1)

    # If words are provided as arguments
    if args.words:
        predictions = predict_phonemes(g2p_model, args.words, args.batch_size)
        if predictions:
            for word, phoneme in zip(args.words, predictions):
                print(f"\nGrapheme: {word}")
                print(f"Phonemes: {phoneme}")
    # Interactive mode if no words provided
    else:
        interactive_mode(g2p_model)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)