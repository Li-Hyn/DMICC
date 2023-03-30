import numpy as np
from sklearn.cluster import KMeans
from tools.metrics import metrics


def check_clustering_metrics(npc, train_loader):
    trainFeatures = npc.memory
    z = trainFeatures.cpu().numpy()
    y = np.array(train_loader.dataset.labels)
    n_clusters = len(np.unique(y))
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z)
    return metrics.acc(y, y_pred), metrics.nmi(y, y_pred), metrics.ari(y, y_pred)
