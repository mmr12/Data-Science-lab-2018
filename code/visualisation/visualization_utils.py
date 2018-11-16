from sklearn.manifold import TSNE
from matplotlib import pyplot
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


#function to keep track from where the word is coming from
def belongs_to(word, faq_ques_docs, faq_ans_docs, ticket_content_docs):
    faq_ques = False
    faq_ans = False
    tic = False
    faq_ques = any(word in x  for x in faq_ques_docs)
    faq_ans = any(word in x  for x in faq_ans_docs)
    tic = any(word in x  for x in ticket_content_docs)
    tot = (faq_ques or faq_ans) and tic
    if tot:
        out = 'g'
    else:
        if tic:
            out = 'y'
        else: out = 'b'
    return(out)

#function to compute and plot PCA
def pca_plot(tokens, col, fname, n_comp):
    pca = PCA(n_components=n_comp)
    pca_tokens = pca.fit_transform(tokens)

    if n_comp == 2:
        pyplot.figure(figsize=(16, 16))
        pyplot.scatter(pca_tokens[:, 0], pca_tokens[:, 1], c=col, alpha=.2)
        name = 'plots/' + str(fname) + '.pca.png'
        pyplot.savefig(name)
        # plt.show()
    else:
        fig = pyplot.figure(figsize=(16, 16))
        ax = Axes3D(fig)
        ax.scatter(pca_tokens[:, 0], pca_tokens[:, 1], pca_tokens[:, 2], c=col, alpha=.2)
        name = 'plots/' + str(fname) + '.pca.png'
        pyplot.savefig(name)
        # pyplot.show()

#function to compute and plot TSNE
def tsne_plot(tokens, col, fname, perp, n_iter, n_comp):
    pca = PCA(n_components=50)
    pca_tokens = pca.fit_transform(tokens)
    tsne_model = TSNE(perplexity=perp, n_components=n_comp, init='pca', n_iter=n_iter)
    new_values = tsne_model.fit_transform(pca_tokens)

    if n_comp == 2:
        pyplot.figure(figsize=(16, 16))
        pyplot.scatter(new_values[:, 0], new_values[:, 1], c=col, alpha=.2)
        name = 'plots/' + str(fname) + '.perp' + str(perp) + '.iter' + str(n_iter) + '.png'
        pyplot.savefig(name)
        # plt.show()
    else:
        fig = pyplot.figure(figsize=(16, 16))
        ax = Axes3D(fig)
        ax.scatter(new_values[:, 0], new_values[:, 1], new_values[:, 2], c=col, alpha=.2)
        name = 'plots/' + str(fname) + '.perp' + str(perp) + '.iter' + str(n_iter) + '.png'
        pyplot.savefig(name)
        # pyplot.show()