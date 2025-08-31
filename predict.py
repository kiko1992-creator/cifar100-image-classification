import argparse, numpy as np
from PIL import Image
import tensorflow as tf

def load_labels(p='labels.txt'):
    with open(p) as f: return [l.strip() for l in f]

def preprocess(img, size_hw):
    H, W = size_hw
    img = img.convert('RGB').resize((W, H), Image.BILINEAR)
    x = np.asarray(img).astype('float32') / 255.0
    return np.expand_dims(x, 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument('--weights', default='final_efficientnetv2b0.keras')
    ap.add_argument('--labels', default='labels.txt')
    ap.add_argument('--topk', type=int, default=5)
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.weights, compile=False)
    H, W = model.input_shape[1], model.input_shape[2]
    x = preprocess(Image.open(args.image), (H, W))
    probs = model.predict(x, verbose=0)[0]
    labels = load_labels(args.labels)
    for i in probs.argsort()[-args.topk:][::-1]:
        print(f'{labels[i]:<20s} {probs[i]:.4f}')

if __name__ == '__main__':
    main()
