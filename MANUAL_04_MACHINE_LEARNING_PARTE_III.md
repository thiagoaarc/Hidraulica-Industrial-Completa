# Manual de Machine Learning - Sistema Hidráulico Industrial - Parte III

## 📋 Algoritmos de Machine Learning

---

## 🌲 Isolation Forest

### Fundamentos do Algoritmo

O **Isolation Forest** é o algoritmo principal para **detecção de anomalias** no sistema. Baseia-se no princípio de que anomalias são mais fáceis de isolar que pontos normais.

#### 🧮 Matemática do Isolation Forest

##### **Princípio de Isolamento**

```python
def isolation_principle_explanation():
    """
    Princípio matemático do Isolation Forest:
    
    1. Anomalias requerem menos divisões para serem isoladas
    2. Path Length médio em BST: E(h(x)) = 2H(n-1) - (2(n-1)/n)
    3. Score de anomalia: s(x,n) = 2^(-E(h(x))/c(n))
    
    Onde:
    - h(x) = path length de x
    - c(n) = comprimento médio de path em BST de n pontos
    - H(i) = número harmônico
    """
    
    # Número harmônico
    def harmonic_number(n):
        return sum(1/i for i in range(1, n+1))
    
    # Path length médio esperado
    def average_path_length(n):
        if n > 2:
            return 2 * harmonic_number(n-1) - (2*(n-1)/n)
        elif n == 2:
            return 1.0
        else:
            return 0.0
    
    # Score de anomalia
    def anomaly_score(path_length, n_samples):
        c_n = average_path_length(n_samples)
        if c_n > 0:
            return 2**(-path_length / c_n)
        else:
            return 0.5
    
    return {
        'harmonic_number': harmonic_number,
        'average_path_length': average_path_length, 
        'anomaly_score': anomaly_score
    }
```

#### 🏗️ Implementação Adaptativa

```python
def train_isolation_forest(self, X_scaled, y_labels=None):
    """
    Treina Isolation Forest com parâmetros adaptativos
    
    Parâmetros otimizados:
    - contamination: taxa adaptativa baseada em histórico
    - n_estimators: balanceado entre precisão e velocidade
    - max_samples: otimizado para tamanho do dataset
    """
    
    n_samples = len(X_scaled)
    
    # Taxa de contaminação adaptativa
    if y_labels is not None and len(y_labels) > 0:
        # Se temos labels, usa taxa empírica
        contamination_rate = min(0.1, max(0.01, np.mean(y_labels)))
    else:
        # Taxa padrão conservadora
        contamination_rate = 0.05
    
    # Número de estimadores baseado no tamanho dos dados
    if n_samples < 1000:
        n_estimators = 100
    elif n_samples < 5000:
        n_estimators = 150
    else:
        n_estimators = 200
    
    # Tamanho máximo de amostra para cada árvore
    max_samples = min(256, max(64, n_samples // 4))
    
    self.logger.info(f"Configurando Isolation Forest: contamination={contamination_rate:.3f}, "
                    f"n_estimators={n_estimators}, max_samples={max_samples}")
    
    # Inicializa modelo
    self.leak_detector = IsolationForest(
        contamination=contamination_rate,
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=42,
        n_jobs=-1,  # Paralelização
        bootstrap=True,
        verbose=0
    )
    
    # Treinamento
    self.leak_detector.fit(X_scaled)
    
    # Armazena threshold de decisão
    scores = self.leak_detector.decision_function(X_scaled)
    self.detection_threshold = np.percentile(scores, 100 * contamination_rate)
    
    self.logger.info(f"Isolation Forest treinado. Threshold: {self.detection_threshold:.3f}")
    
    return {
        'contamination_rate': contamination_rate,
        'n_estimators': n_estimators,
        'max_samples': max_samples,
        'threshold': self.detection_threshold
    }
```

#### 📊 Interpretação dos Scores

```python
def interpret_isolation_score(self, anomaly_score):
    """
    Interpreta score do Isolation Forest
    
    Score ranges:
    - [0.6, 1.0]: Anomalia muito provável
    - [0.55, 0.6]: Anomalia provável  
    - [0.45, 0.55]: Região incerta
    - [0.0, 0.45]: Comportamento normal
    """
    
    if anomaly_score >= 0.6:
        return {
            'classification': 'anomaly_high',
            'probability': 'muito_alta',
            'color': '#ff4444',
            'action': 'investigacao_imediata',
            'confidence': min(1.0, (anomaly_score - 0.6) / 0.4)
        }
    elif anomaly_score >= 0.55:
        return {
            'classification': 'anomaly_medium',
            'probability': 'alta', 
            'color': '#ff8800',
            'action': 'monitoramento_intensificado',
            'confidence': (anomaly_score - 0.55) / 0.05
        }
    elif anomaly_score >= 0.45:
        return {
            'classification': 'uncertain',
            'probability': 'incerta',
            'color': '#ffcc00',
            'action': 'observacao_continuada',
            'confidence': 0.5
        }
    else:
        return {
            'classification': 'normal',
            'probability': 'baixa',
            'color': '#44ff44', 
            'action': 'monitoramento_rotineiro',
            'confidence': min(1.0, (0.45 - anomaly_score) / 0.45)
        }
```

---

## 🌳 Random Forest

### Classificação de Tipos de Anomalias

O **Random Forest** atua como **classificador** quando há dados rotulados suficientes, categorizando tipos específicos de problemas detectados.

#### 🎯 Implementação do Random Forest

```python
def train_random_forest_classifier(self, X_scaled, y_labels, leak_types=None):
    """
    Treina Random Forest para classificação de tipos de vazamento
    
    Classes possíveis:
    - 'normal': Operação normal
    - 'small_leak': Vazamento pequeno  
    - 'large_leak': Vazamento grande
    - 'blockage': Bloqueio parcial
    - 'sensor_drift': Deriva de sensor
    """
    
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Verifica se temos exemplos positivos suficientes
    positive_samples = np.sum(y_labels)
    if positive_samples < 5:
        self.logger.warning(f"Poucos exemplos positivos ({positive_samples}). Pulando Random Forest.")
        return None
    
    # Prepara labels multiclasse se disponível
    if leak_types is not None:
        y_multiclass = np.array(leak_types)
    else:
        # Converte para binário
        y_multiclass = np.where(y_labels, 'leak', 'normal')
    
    # Parâmetros para otimização
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Inicializa modelo base
    base_rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True
    )
    
    # Grid search com validação cruzada estratificada
    cv = StratifiedKFold(n_splits=min(5, len(np.unique(y_multiclass))), shuffle=True, random_state=42)
    
    self.leak_classifier = GridSearchCV(
        base_rf,
        param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    # Treinamento
    self.leak_classifier.fit(X_scaled, y_multiclass)
    
    # Extrai melhor modelo
    best_model = self.leak_classifier.best_estimator_
    
    # Avaliação do modelo
    y_pred = best_model.predict(X_scaled)
    classification_rep = classification_report(y_multiclass, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_multiclass, y_pred)
    
    # Importância das features
    feature_importance = best_model.feature_importances_
    important_features = []
    
    for i, importance in enumerate(feature_importance):
        if importance > 0.01:  # Features com pelo menos 1% de importância
            important_features.append({
                'feature_name': self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}',
                'importance': float(importance)
            })
    
    # Ordena por importância
    important_features.sort(key=lambda x: x['importance'], reverse=True)
    
    self.logger.info(f"Random Forest treinado. Melhores parâmetros: {self.leak_classifier.best_params_}")
    self.logger.info(f"Acurácia OOB: {best_model.oob_score_:.3f}")
    
    return {
        'best_params': self.leak_classifier.best_params_,
        'oob_score': float(best_model.oob_score_),
        'classification_report': classification_rep,
        'confusion_matrix': conf_matrix.tolist(),
        'feature_importance': important_features[:20],  # Top 20
        'n_classes': len(np.unique(y_multiclass))
    }
```

#### 🧠 Interpretação da Importância das Features

```python
def analyze_feature_importance(self, feature_importances):
    """
    Analisa e interpreta importância das features no Random Forest
    
    Categoriza features por tipo e identifica padrões
    """
    
    # Categorização das features
    feature_categories = {
        'statistical': ['mean', 'std', 'min', 'max', 'q25', 'q75', 'median', 'var', 'trend'],
        'gradient': ['grad_mean', 'grad_std', 'grad_max'],
        'correlation': ['corr_max_norm', 'delay_samples', 'corr_peaks', 'corr_std', 'corr_pos_mean'],
        'spectral': ['low_energy', 'mid_energy', 'high_energy', 'dom_freq'],
        'relational': ['corr_', 'xcorr_max_'],
        'temporal': ['stability', 'cv_max']
    }
    
    # Agrupa importâncias por categoria
    category_importance = {}
    
    for category, keywords in feature_categories.items():
        category_importance[category] = 0.0
        
        for feature in feature_importances:
            feature_name = feature['feature_name']
            
            # Verifica se feature pertence à categoria
            for keyword in keywords:
                if keyword in feature_name:
                    category_importance[category] += feature['importance']
                    break
    
    # Normaliza importâncias por categoria
    total_importance = sum(category_importance.values())
    if total_importance > 0:
        category_importance = {k: v/total_importance for k, v in category_importance.items()}
    
    # Identifica categoria dominante
    dominant_category = max(category_importance.keys(), key=lambda k: category_importance[k])
    
    # Interpretação
    interpretation = {
        'statistical': 'Padrões baseados em estatísticas básicas dos sinais',
        'gradient': 'Mudanças temporais e transientes são críticos',
        'correlation': 'Análise sônica é fundamental para detecção',
        'spectral': 'Conteúdo de frequências revela anomalias',
        'relational': 'Interações entre variáveis são indicativas',
        'temporal': 'Estabilidade temporal é fator chave'
    }
    
    return {
        'category_importance': category_importance,
        'dominant_category': dominant_category,
        'interpretation': interpretation.get(dominant_category, 'Padrão não identificado'),
        'feature_diversity': len([cat for cat, imp in category_importance.items() if imp > 0.1])
    }
```

---

## 🎯 Support Vector Machines (SVM)

### Classificação com Margem Máxima

O **SVM** é usado como modelo complementar para **classificação binária** (normal vs. anômalo) com alta precisão.

#### 🧮 Implementação com Kernel RBF

```python
def train_svm_classifier(self, X_scaled, y_labels):
    """
    Treina SVM com kernel RBF para classificação binária
    
    Otimização de hiperparâmetros:
    - C: parâmetro de regularização
    - gamma: parâmetro do kernel RBF
    """
    
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import roc_auc_score, roc_curve
    
    # Verifica balanceamento das classes
    class_counts = np.bincount(y_labels.astype(int))
    class_weight = 'balanced' if min(class_counts) / max(class_counts) < 0.3 else None
    
    # Grid de parâmetros
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    }
    
    # Modelo base
    base_svm = SVC(
        kernel='rbf',
        probability=True,  # Para scores de probabilidade
        class_weight=class_weight,
        random_state=42
    )
    
    # Grid search
    svm_grid = GridSearchCV(
        base_svm,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    
    # Treinamento
    svm_grid.fit(X_scaled, y_labels)
    self.svm_classifier = svm_grid.best_estimator_
    
    # Avaliação
    y_pred_proba = self.svm_classifier.predict_proba(X_scaled)[:, 1]
    auc_score = roc_auc_score(y_labels, y_pred_proba)
    
    # Curva ROC
    fpr, tpr, thresholds = roc_curve(y_labels, y_pred_proba)
    
    # Encontra threshold ótimo (maximiza F1-score)
    f1_scores = []
    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_labels, y_pred_thresh)
        f1_scores.append(f1)
    
    optimal_threshold_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_threshold_idx]
    
    self.logger.info(f"SVM treinado. AUC: {auc_score:.3f}, Threshold ótimo: {optimal_threshold:.3f}")
    
    return {
        'best_params': svm_grid.best_params_,
        'auc_score': float(auc_score),
        'optimal_threshold': float(optimal_threshold),
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
    }
```

#### ⚖️ Interpretação da Margem de Separação

```python
def analyze_svm_decision_boundary(self, X_scaled):
    """
    Analisa a margem de separação do SVM
    
    Identifica:
    - Support vectors
    - Distância da margem
    - Confiança da classificação
    """
    
    if not hasattr(self, 'svm_classifier') or self.svm_classifier is None:
        return {'error': 'SVM não treinado'}
    
    # Distância da margem para cada ponto
    decision_scores = self.svm_classifier.decision_function(X_scaled)
    
    # Support vectors
    if hasattr(self.svm_classifier, 'support_'):
        support_vector_indices = self.svm_classifier.support_
        n_support_vectors = len(support_vector_indices)
    else:
        support_vector_indices = []
        n_support_vectors = 0
    
    # Probabilidades de classe
    probabilities = self.svm_classifier.predict_proba(X_scaled)
    
    # Análise da confiança
    confidence_scores = np.max(probabilities, axis=1)
    high_confidence_mask = confidence_scores > 0.8
    low_confidence_mask = confidence_scores < 0.6
    
    return {
        'n_support_vectors': n_support_vectors,
        'support_vector_ratio': n_support_vectors / len(X_scaled),
        'decision_scores': decision_scores.tolist(),
        'confidence_scores': confidence_scores.tolist(),
        'high_confidence_count': int(np.sum(high_confidence_mask)),
        'low_confidence_count': int(np.sum(low_confidence_mask)),
        'margin_quality': 'good' if n_support_vectors / len(X_scaled) < 0.3 else 'poor'
    }
```

---

## 🔍 DBSCAN Clustering

### Detecção de Densidade

O **DBSCAN** identifica **clusters de densidade** nos dados, útil para detectar regimes operacionais e outliers.

#### 🎯 Implementação Adaptativa

```python
def perform_dbscan_analysis(self, X_scaled):
    """
    Executa DBSCAN para análise de clusters
    
    Parâmetros adaptativos:
    - eps: distância baseada na distribuição dos dados
    - min_samples: baseado na dimensionalidade
    """
    
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    
    # Estimativa automática de eps usando k-distance
    k = max(4, int(X_scaled.shape[1] * 0.1))  # k baseado na dimensionalidade
    
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)
    
    # k-distance (distância ao k-ésimo vizinho)
    k_distances = distances[:, k-1]
    k_distances_sorted = np.sort(k_distances)
    
    # Método do "joelho" para encontrar eps
    # Procura ponto de máxima curvatura
    diffs = np.diff(k_distances_sorted)
    diff2 = np.diff(diffs)
    
    if len(diff2) > 0:
        knee_idx = np.argmax(diff2) + 2  # +2 devido aos diffs
        eps = k_distances_sorted[min(knee_idx, len(k_distances_sorted)-1)]
    else:
        eps = np.percentile(k_distances, 90)  # Fallback
    
    # min_samples baseado na dimensionalidade
    min_samples = max(2, min(10, X_scaled.shape[1] // 4))
    
    # Executa DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = dbscan.fit_predict(X_scaled)
    
    # Análise dos resultados
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    # Características dos clusters
    cluster_info = {}
    
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:
            continue  # Pula outliers
        
        cluster_mask = cluster_labels == cluster_id
        cluster_data = X_scaled[cluster_mask]
        
        cluster_info[cluster_id] = {
            'size': int(np.sum(cluster_mask)),
            'centroid': np.mean(cluster_data, axis=0).tolist(),
            'std': np.std(cluster_data, axis=0).tolist(),
            'density': float(np.sum(cluster_mask) / len(X_scaled))
        }
    
    # Análise de outliers
    outlier_indices = np.where(cluster_labels == -1)[0].tolist()
    outlier_ratio = n_noise / len(X_scaled)
    
    self.logger.info(f"DBSCAN: {n_clusters} clusters, {n_noise} outliers ({outlier_ratio:.1%})")
    
    return {
        'n_clusters': n_clusters,
        'n_outliers': n_noise,
        'outlier_ratio': float(outlier_ratio),
        'cluster_labels': cluster_labels.tolist(),
        'cluster_info': cluster_info,
        'outlier_indices': outlier_indices,
        'parameters': {
            'eps': float(eps),
            'min_samples': min_samples,
            'k_for_eps': k
        }
    }
```

#### 📊 Interpretação dos Clusters

```python
def interpret_dbscan_clusters(self, cluster_analysis, feature_names):
    """
    Interpreta os clusters encontrados pelo DBSCAN
    
    Classifica clusters como:
    - Normal operation
    - Transient state  
    - Anomalous behavior
    """
    
    interpretations = {}
    
    for cluster_id, cluster_data in cluster_analysis['cluster_info'].items():
        size = cluster_data['size']
        density = cluster_data['density']
        centroid = np.array(cluster_data['centroid'])
        
        # Classificação baseada no tamanho e densidade
        if density > 0.6:  # Cluster dominante
            cluster_type = 'normal_operation'
            interpretation = 'Operação normal - cluster principal'
            color = '#4CAF50'
            
        elif density > 0.2:  # Cluster significativo
            # Analisa características do centróide
            # Features de correlação (índices 45-49)
            if len(centroid) > 45:
                corr_features = centroid[45:50]
                avg_correlation = np.mean(corr_features)
                
                if avg_correlation < -0.5:  # Correlação baixa
                    cluster_type = 'anomalous_behavior'
                    interpretation = 'Possível comportamento anômalo - correlação baixa'
                    color = '#f44336'
                else:
                    cluster_type = 'transient_state'
                    interpretation = 'Estado transiente ou regime secundário'
                    color = '#FF9800'
            else:
                cluster_type = 'secondary_mode'
                interpretation = 'Modo operacional secundário'
                color = '#2196F3'
                
        else:  # Cluster pequeno
            cluster_type = 'outlier_group'
            interpretation = 'Grupo de outliers - possíveis eventos isolados'
            color = '#9C27B0'
        
        interpretations[cluster_id] = {
            'type': cluster_type,
            'interpretation': interpretation,
            'color': color,
            'confidence': min(1.0, density * 2),  # Confiança baseada na densidade
            'size': size,
            'density': density
        }
    
    # Interpretação geral
    n_clusters = len(interpretations)
    
    if n_clusters == 1:
        overall = 'Sistema operando em regime único - muito estável'
    elif n_clusters == 2:
        overall = 'Sistema com dois regimes - normal vs. transiente/anômalo'
    elif n_clusters >= 3:
        overall = 'Sistema com múltiplos regimes - complexidade operacional alta'
    else:
        overall = 'Todos os pontos são outliers - instabilidade severa'
    
    return {
        'cluster_interpretations': interpretations,
        'overall_assessment': overall,
        'stability_score': 1.0 / max(1, n_clusters - 1),  # Menos clusters = mais estável
        'outlier_concern': cluster_analysis['outlier_ratio'] > 0.1
    }
```

---

**CONTINUAÇÃO NA PARTE IV**

A Parte III cobriu detalhadamente:

- ✅ **Isolation Forest** - Matemática, implementação adaptativa, interpretação de scores
- ✅ **Random Forest** - Classificação multiclasse, importância de features
- ✅ **SVM** - Kernel RBF, margem de separação, curva ROC
- ✅ **DBSCAN** - Clustering por densidade, detecção automática de eps

**Próxima parte** - MANUAL_04_MACHINE_LEARNING_PARTE_IV.md:

- 🧠 **Análise PCA** - Componentes principais, redução de dimensionalidade
- 🎯 **Correlação Canônica** - CCA entre conjuntos de variáveis
- 🔮 **Predição e Fusão** - Combinação de modelos, ensemble
- 📊 **Métricas de Performance** - Avaliação, retreino adaptativo

Continuar com a Parte IV (final)?
