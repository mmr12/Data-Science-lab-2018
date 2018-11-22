import argparse

from embedding.embedding import embedding
from similarity.similarity import similarity
from classifier.classifier import classifier


def _setup_argparser():
    parser = argparse.ArgumentParser(description="Select actions to run the program")
    parser.add_argument("-e", "--embedding", action="store",
                        choices=["word2vec", "doc2vec"],
                        default="doc2vec",
                        type=str,
                        help="select which embedding you want the model to use")
    #parser.add_argument("-t", "--train",
    #                    help="train the given model",
    #                    action="store_true")
    args, unknown = parser.parse_known_args()
    return args

if __name__== "__main__" :
    args = _setup_argparser()
    mod = args.embedding

    embedding(mod)
    similarity()
    classifier()