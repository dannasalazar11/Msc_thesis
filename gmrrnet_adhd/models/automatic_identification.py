import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC


def automatic_identification(X,y):
    # Run ANOVA
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    
    # Get p-values and scores
    p_values = selector.pvalues_
    scores = selector.scores_
    
    # Create mask for p-values <= 0.5
    mask = p_values <= 0.5
    
    # Apply the mask to X
    X_selected = X[:, mask]
    
    print("Selected features:", np.sum(mask))
    print("X shape after ANOVA filter:", X_selected.shape)

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Apply PCA
    pca = PCA(n_components=0.90)  # Retain 90% of the variance
    X_pca = pca.fit_transform(X_scaled)
    
    # Print results
    print("Original ANOVA-selected shape:", X_selected.shape)
    print("Reduced shape after PCA:", X_pca.shape)
    print("Number of components selected:", X_pca.shape[1])

    return X_pca, SVC(kernel='rbf', gamma='scale', C=1.0)    
