import argparse
from src.preprocessing.image_loader import load_image
from src.preprocessing.segmentation import segment_letters
from src.inference.classifier import LetterClassifier
from src.inference.postprocess import assemble_text
from src.postprocessing.spellcorrect import correct_sentence

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--model', default=None)
    parser.add_argument('--scaler', default=None)
    args = parser.parse_args()
    img = load_image(args.image)
    letters, centers = segment_letters(img)
    clf = LetterClassifier(model_path=args.model, scaler_path=args.scaler) if args.model and args.scaler else LetterClassifier()
    raw = assemble_text([clf.predict(l) for l in letters], centers)
    print(correct_sentence(raw))

if __name__ == '__main__':
    main()