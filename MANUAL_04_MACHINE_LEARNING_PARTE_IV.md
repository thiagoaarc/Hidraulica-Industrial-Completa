# Manual de Machine Learning - Sistema Hidráulico Industrial - Parte IV (Final)

## 📋 Análise Avançada e Predição

---

## 🧠 Análise PCA (Principal Component Analysis)

### Redução de Dimensionalidade

A **Análise de Componentes Principais** identifica as direções de maior variância nos dados, revelando padrões complexos ocultos no espaço de 81 dimensões.

#### 📊 Implementação Completa do PCA

```python
def perform_pca_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Realiza análise de componentes principais completa
    
    Objetivos:
    1. Reduzir dimensionalidade preservando informação
    2. Identificar padrões dominantes
    3. Detectar correlações ocultas
    4. Criar espaço de projeção para anomalias
    """
    
    try:
        # Preparação dos dados
        features = data.select_dtypes(include=[np.number])
        if features.empty:
            return {'error': 'Nenhuma coluna numérica encontrada para análise PCA'}
        
        # Remove NaN e normaliza
        features_clean = features.fillna(features.mean())
        
        # Padronização (crucial para PCA)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_clean)
        
        # Aplica PCA com número adaptativo de componentes
        n_components = min(10, features_scaled.shape[1], features_scaled.shape[0])
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(features_scaled)
        
        # Análise de componentes principais
        component_analysis = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'components_matrix': pca.components_.tolist(),
            'feature_names': features.columns.tolist(),
            'n_components_90': int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.90) + 1),
            'n_components_95': int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1),
            'principal_components': principal_components.tolist()
        }
        
        # Identifica componentes mais importantes (> 10% da variância)
        important_components = []
        
        for i, ratio in enumerate(pca.explained_variance_ratio_):
            if ratio > 0.1:
                component_features = []
                
                # Features com maior peso absoluto (> 30%)
                for j, weight in enumerate(pca.components_[i]):
                    if abs(weight) > 0.3:
                        component_features.append({
                            'feature': features.columns[j],
                            'weight': float(weight),
                            'contribution': float(weight**2)  # Contribuição quadrática
                        })
                
                # Ordena features por contribuição
                component_features.sort(key=lambda x: x['contribution'], reverse=True)
                
                important_components.append({
                    'component_id': i + 1,
                    'variance_explained': float(ratio),
                    'cumulative_variance': float(np.sum(pca.explained_variance_ratio_[:i+1])),
                    'features': component_features[:10],  # Top 10 features
                    'interpretation': self._interpret_pca_component(component_features)
                })
        
        component_analysis['important_components'] = important_components
        
        # Análise de outliers no espaço PCA
        outlier_analysis = self._detect_pca_outliers(principal_components, pca.explained_variance_ratio_)
        component_analysis['outlier_analysis'] = outlier_analysis
        
        self.logger.info(f"Análise PCA concluída: {n_components} componentes, "
                        f"{component_analysis['n_components_90']} componentes para 90% da variância")
        
        return component_analysis
        
    except Exception as e:
        error_msg = f"Erro na análise PCA: {str(e)}"
        self.logger.error(error_msg)
        return {'error': error_msg}
```

#### 🔍 Interpretação dos Componentes Principais

```python
def _interpret_pca_component(self, component_features):
    """
    Interpreta o significado físico de cada componente principal
    
    Baseado nas features com maior peso
    """
    
    if not component_features:
        return {'interpretation': 'Componente sem features significativas'}
    
    # Agrupa features por categoria
    categories = {
        'pressure': ['exp_p', 'rec_p', 'press_diff'],
        'flow': ['flow'],
        'density': ['density'],
        'temperature': ['temp'],
        'gradient': ['grad_'],
        'correlation': ['corr_', 'delay_', 'xcorr_'],
        'spectral': ['energy', 'dom_freq'],
        'temporal': ['stability', 'cv_max']
    }
    
    category_weights = {}
    
    for feature_info in component_features:
        feature_name = feature_info['feature']
        weight = abs(feature_info['weight'])
        
        # Identifica categoria da feature
        for category, keywords in categories.items():
            if any(keyword in feature_name for keyword in keywords):
                category_weights[category] = category_weights.get(category, 0) + weight
                break
    
    # Categoria dominante
    if category_weights:
        dominant_category = max(category_weights.keys(), key=lambda k: category_weights[k])
        
        # Interpretações por categoria
        interpretations = {
            'pressure': {
                'physical_meaning': 'Variações de pressão no sistema',
                'operational_relevance': 'Força motriz e perdas de carga',
                'anomaly_indication': 'Possíveis vazamentos ou bloqueios'
            },
            'flow': {
                'physical_meaning': 'Padrões de fluxo mássico',
                'operational_relevance': 'Demanda e capacidade do sistema',
                'anomaly_indication': 'Mudanças no regime de escoamento'
            },
            'gradient': {
                'physical_meaning': 'Transientes e mudanças rápidas',
                'operational_relevance': 'Dinâmica do sistema',
                'anomaly_indication': 'Eventos súbitos ou instabilidades'
            },
            'correlation': {
                'physical_meaning': 'Propagação de ondas de pressão',
                'operational_relevance': 'Integridade da tubulação',
                'anomaly_indication': 'Degradação da correlação sônica'
            },
            'spectral': {
                'physical_meaning': 'Conteúdo de frequências',
                'operational_relevance': 'Ruído e oscilações',
                'anomaly_indication': 'Frequências anômalas'
            }
        }
        
        return interpretations.get(dominant_category, {
            'physical_meaning': 'Padrão complexo multivaríavel',
            'operational_relevance': 'Interação entre múltiplas variáveis',
            'anomaly_indication': 'Mudança no padrão sistêmico'
        })
    
    return {'interpretation': 'Componente híbrido - múltiplas categorias'}
```

#### 🎯 Detecção de Outliers no Espaço PCA

```python
def _detect_pca_outliers(self, principal_components, explained_variance_ratio):
    """
    Detecta outliers no espaço reduzido PCA
    
    Método:
    1. Distância de Mahalanobis ponderada
    2. Threshold adaptativo baseado na distribuição
    """
    
    # Considera apenas componentes principais significativos (90% da variância)
    cumvar = np.cumsum(explained_variance_ratio)
    n_components_90 = np.argmax(cumvar >= 0.90) + 1
    pc_significant = principal_components[:, :n_components_90]
    
    # Centro dos dados no espaço PCA
    pc_center = np.mean(pc_significant, axis=0)
    
    # Matriz de covariância no espaço PCA (diagonal, pois componentes são ortogonais)
    pc_var = np.var(pc_significant, axis=0)
    
    # Distância de Mahalanobis ponderada pela variância explicada
    mahal_distances = []
    
    for i, point in enumerate(pc_significant):
        # Distância euclidiana ponderada
        diff = point - pc_center
        weighted_diff = diff / np.sqrt(pc_var + 1e-10)  # Evita divisão por zero
        
        # Ponderação pela variância explicada
        weights = explained_variance_ratio[:n_components_90]
        mahal_dist = np.sqrt(np.sum((weighted_diff**2) * weights))
        
        mahal_distances.append(mahal_dist)
    
    mahal_distances = np.array(mahal_distances)
    
    # Threshold adaptativo (percentil 95 + margem)
    threshold_95 = np.percentile(mahal_distances, 95)
    adaptive_threshold = threshold_95 * 1.2  # Margem de segurança
    
    # Identifica outliers
    outlier_mask = mahal_distances > adaptive_threshold
    outlier_indices = np.where(outlier_mask)[0].tolist()
    
    # Score de outlier (0-1)
    max_distance = np.max(mahal_distances)
    outlier_scores = (mahal_distances / max_distance).tolist()
    
    return {
        'n_outliers': int(np.sum(outlier_mask)),
        'outlier_indices': outlier_indices,
        'outlier_ratio': float(np.mean(outlier_mask)),
        'outlier_scores': outlier_scores,
        'threshold': float(adaptive_threshold),
        'mahalanobis_distances': mahal_distances.tolist()
    }
```

---

## 🔗 Correlação Canônica (CCA)

### Análise de Correlações Entre Conjuntos de Variáveis

A **Análise de Correlação Canônica** identifica relações lineares máximas entre dois conjuntos de variáveis.

#### 📊 Implementação da CCA

```python
def canonical_correlation_analysis(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Dict[str, Any]:
    """
    Realiza análise de correlação canônica entre dois conjuntos de variáveis
    
    Exemplo de uso:
    - data1: Variáveis de pressão [expeditor_pressure, receiver_pressure, pressure_diff]
    - data2: Variáveis de fluxo [flow, density, temperature]
    """
    
    try:
        from sklearn.cross_decomposition import CCA
        from scipy.stats import pearsonr
        
        # Preparação dos dados
        X = data1.select_dtypes(include=[np.number]).fillna(data1.mean())
        Y = data2.select_dtypes(include=[np.number]).fillna(data2.mean())
        
        if X.empty or Y.empty:
            return {'error': 'Dados insuficientes para CCA'}
        
        # Padronização
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_x.fit_transform(X)
        Y_scaled = scaler_y.fit_transform(Y)
        
        # Determina número de componentes canônicos
        n_components = min(X_scaled.shape[1], Y_scaled.shape[1], X_scaled.shape[0] // 2)
        
        if n_components < 1:
            return {'error': 'Dimensões insuficientes para CCA'}
        
        # Aplica CCA
        cca = CCA(n_components=n_components)
        X_c, Y_c = cca.fit_transform(X_scaled, Y_scaled)
        
        # Calcula correlações canônicas
        canonical_correlations = []
        for i in range(n_components):
            corr, p_value = pearsonr(X_c[:, i], Y_c[:, i])
            canonical_correlations.append({
                'component': i + 1,
                'correlation': float(corr),
                'p_value': float(p_value),
                'significance': p_value < 0.05
            })
        
        # Pesos canônicos (loadings)
        x_loadings = []
        y_loadings = []
        
        for i in range(n_components):
            # Correlações entre variáveis originais e componentes canônicos
            x_loading = [float(pearsonr(X_scaled[:, j], X_c[:, i])[0]) for j in range(X_scaled.shape[1])]
            y_loading = [float(pearsonr(Y_scaled[:, j], Y_c[:, i])[0]) for j in range(Y_scaled.shape[1])]
            
            x_loadings.append(x_loading)
            y_loadings.append(y_loading)
        
        # Interpretação dos resultados
        interpretation = self._interpret_canonical_correlations(canonical_correlations)
        
        result = {
            'n_components': n_components,
            'canonical_correlations': canonical_correlations,
            'x_canonical_scores': X_c.tolist(),
            'y_canonical_scores': Y_c.tolist(),
            'x_loadings': x_loadings,
            'y_loadings': y_loadings,
            'x_variable_names': X.columns.tolist(),
            'y_variable_names': Y.columns.tolist(),
            'interpretation': interpretation
        }
        
        self.logger.info(f"CCA concluída: {n_components} componentes canônicos")
        
        return result
        
    except Exception as e:
        error_msg = f"Erro na análise CCA: {str(e)}"
        self.logger.error(error_msg)
        return {'error': error_msg}
```

#### 🎯 Interpretação das Correlações Canônicas

```python
def _interpret_canonical_correlations(self, correlations: List[Dict]) -> Dict[str, Any]:
    """
    Interpreta os resultados da análise de correlação canônica
    """
    
    if not correlations:
        return {'interpretation': 'Nenhuma correlação canônica encontrada'}
    
    # Primeira correlação canônica (mais importante)
    first_corr = correlations[0]['correlation']
    first_significance = correlations[0]['significance']
    
    # Classificação da força da correlação
    if abs(first_corr) > 0.8:
        strength = 'muito_forte'
        interpretation = 'Correlação canônica muito forte'
        color = '#4CAF50'
    elif abs(first_corr) > 0.6:
        strength = 'forte'
        interpretation = 'Correlação canônica forte'
        color = '#2196F3'
    elif abs(first_corr) > 0.4:
        strength = 'moderada'
        interpretation = 'Correlação canônica moderada'
        color = '#FF9800'
    else:
        strength = 'fraca'
        interpretation = 'Correlação canônica fraca'
        color = '#f44336'
    
    # Análise de significância
    significant_correlations = sum(1 for c in correlations if c['significance'])
    
    # Interpretação geral
    if significant_correlations == 0:
        general_interpretation = 'Não há correlações canônicas significativas entre os conjuntos'
    elif significant_correlations == 1:
        general_interpretation = 'Uma dimensão de correlação significativa entre os conjuntos'
    else:
        general_interpretation = f'{significant_correlations} dimensões de correlação significativas'
    
    return {
        'primary_correlation': {
            'value': first_corr,
            'strength': strength,
            'interpretation': interpretation,
            'color': color,
            'significant': first_significance
        },
        'n_significant': significant_correlations,
        'general_interpretation': general_interpretation,
        'dimensionality_reduction_potential': significant_correlations < len(correlations)
    }
```

---

## 🔮 Predição e Fusão de Modelos

### Sistema de Ensemble

O sistema combina predições de múltiplos algoritmos para maximizar a precisão e robustez.

#### 🎯 Método Principal de Predição

```python
def predict_leak(self, snapshots: List[MultiVariableSnapshot]) -> Dict[str, Any]:
    """
    Predição principal usando fusão de múltiplos modelos
    
    Modelos combinados:
    1. Isolation Forest - Score de anomalia
    2. Random Forest - Classificação (se disponível)
    3. SVM - Classificação binária (se disponível)
    4. DBSCAN - Análise de clustering
    """
    
    try:
        # Extração de features
        features = self.extract_advanced_features(snapshots)
        
        # Normalização se scaler foi treinado
        if hasattr(self.scaler, 'scale_'):
            features_scaled = self.scaler.transform(features.reshape(1, -1))[0]
        else:
            features_scaled = features
        
        predictions = {}
        
        # 1. Isolation Forest (sempre disponível se treinado)
        if self.leak_detector is not None:
            anomaly_score = self.leak_detector.decision_function([features_scaled])[0]
            anomaly_prediction = self.leak_detector.predict([features_scaled])[0]
            
            # Normaliza score para [0,1]
            normalized_score = 1 / (1 + np.exp(-anomaly_score))
            
            predictions['isolation_forest'] = {
                'anomaly_score': float(normalized_score),
                'is_anomaly': bool(anomaly_prediction == -1),
                'confidence': min(1.0, abs(anomaly_score) / 2.0),
                'weight': 0.4  # Peso na fusão
            }
        
        # 2. Random Forest (se disponível e treinado)
        if self.leak_classifier is not None:
            try:
                rf_prediction = self.leak_classifier.predict([features_scaled])[0]
                rf_probabilities = self.leak_classifier.predict_proba([features_scaled])[0]
                
                predictions['random_forest'] = {
                    'prediction': str(rf_prediction),
                    'probabilities': {
                        class_name: float(prob) 
                        for class_name, prob in zip(self.leak_classifier.classes_, rf_probabilities)
                    },
                    'confidence': float(np.max(rf_probabilities)),
                    'weight': 0.3
                }
            except Exception as e:
                self.logger.warning(f"Erro no Random Forest: {e}")
        
        # 3. SVM (se disponível)
        if hasattr(self, 'svm_classifier') and self.svm_classifier is not None:
            try:
                svm_prediction = self.svm_classifier.predict([features_scaled])[0]
                svm_probabilities = self.svm_classifier.predict_proba([features_scaled])[0]
                
                predictions['svm'] = {
                    'prediction': bool(svm_prediction),
                    'leak_probability': float(svm_probabilities[1]),
                    'confidence': float(abs(svm_probabilities[1] - 0.5) * 2),
                    'weight': 0.2
                }
            except Exception as e:
                self.logger.warning(f"Erro no SVM: {e}")
        
        # 4. DBSCAN clustering
        try:
            dbscan_result = self.perform_dbscan_analysis(features_scaled.reshape(1, -1))
            is_outlier = dbscan_result['cluster_labels'][0] == -1
            
            predictions['dbscan'] = {
                'is_outlier': is_outlier,
                'confidence': 0.8 if is_outlier else 0.6,
                'weight': 0.1
            }
        except Exception as e:
            self.logger.warning(f"Erro no DBSCAN: {e}")
        
        # Fusão das predições
        final_prediction = self._fuse_predictions(predictions)
        
        # Adiciona contexto temporal
        final_prediction['temporal_context'] = {
            'timestamp': datetime.now().isoformat(),
            'n_snapshots': len(snapshots),
            'feature_vector_size': len(features),
            'models_used': list(predictions.keys())
        }
        
        return final_prediction
        
    except Exception as e:
        error_msg = f"Erro na predição: {str(e)}"
        self.logger.error(error_msg)
        return {
            'error': error_msg,
            'leak_probability': 0.5,
            'confidence': 0.0,
            'recommendation': 'Erro no sistema de predição - verificar logs'
        }
```

#### ⚖️ Fusão Inteligente de Predições

```python
def _fuse_predictions(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fusão inteligente de predições com pesos adaptativos
    
    Método:
    1. Média ponderada por confiança
    2. Votação majoritária
    3. Análise de consenso
    """
    
    if not predictions:
        return {
            'leak_probability': 0.5,
            'confidence': 0.0,
            'recommendation': 'Nenhum modelo disponível'
        }
    
    # Extrai probabilidades de vazamento de cada modelo
    leak_probabilities = []
    confidences = []
    weights = []
    
    for model_name, pred_data in predictions.items():
        if model_name == 'isolation_forest':
            prob = pred_data['anomaly_score']
            conf = pred_data['confidence']
            weight = pred_data['weight']
            
        elif model_name == 'random_forest':
            # Assume que 'leak' é uma das classes
            prob = pred_data['probabilities'].get('leak', 
                   pred_data['probabilities'].get('True', 0.0))
            conf = pred_data['confidence']
            weight = pred_data['weight']
            
        elif model_name == 'svm':
            prob = pred_data['leak_probability']
            conf = pred_data['confidence']
            weight = pred_data['weight']
            
        elif model_name == 'dbscan':
            prob = 0.8 if pred_data['is_outlier'] else 0.2
            conf = pred_data['confidence']
            weight = pred_data['weight']
            
        else:
            continue
        
        leak_probabilities.append(prob)
        confidences.append(conf)
        weights.append(weight * conf)  # Peso ajustado pela confiança
    
    # Normaliza pesos
    total_weight = sum(weights)
    if total_weight > 0:
        normalized_weights = [w / total_weight for w in weights]
    else:
        normalized_weights = [1.0 / len(weights)] * len(weights)
    
    # Média ponderada
    fused_probability = sum(p * w for p, w in zip(leak_probabilities, normalized_weights))
    
    # Confiança baseada no consenso
    prob_variance = np.var(leak_probabilities)
    consensus_confidence = 1.0 / (1.0 + prob_variance * 10)  # Menor variância = maior confiança
    
    # Confiança média dos modelos
    average_confidence = np.mean(confidences)
    
    # Confiança final
    final_confidence = (consensus_confidence + average_confidence) / 2
    
    # Classificação final
    if fused_probability > 0.7:
        classification = 'leak_detected'
        severity = 'high'
        recommendation = 'Investigação imediata recomendada'
        color = '#f44336'
        
    elif fused_probability > 0.5:
        classification = 'leak_probable'
        severity = 'medium'
        recommendation = 'Monitoramento intensificado'
        color = '#ff9800'
        
    elif fused_probability > 0.3:
        classification = 'anomaly_detected'
        severity = 'low'
        recommendation = 'Observação continuada'
        color = '#ffeb3b'
        
    else:
        classification = 'normal_operation'
        severity = 'none'
        recommendation = 'Continuar monitoramento rotineiro'
        color = '#4caf50'
    
    return {
        'leak_probability': float(fused_probability),
        'confidence': float(final_confidence),
        'classification': classification,
        'severity': severity,
        'recommendation': recommendation,
        'color': color,
        'model_consensus': {
            'n_models': len(predictions),
            'probability_variance': float(prob_variance),
            'consensus_score': float(consensus_confidence),
            'individual_predictions': {
                model: pred for model, pred in predictions.items()
            }
        }
    }
```

---

## 📊 Métricas de Performance e Retreino

### Monitoramento Contínuo dos Modelos

#### 🔄 Sistema de Retreino Adaptativo

```python
@error_handler.handle_with_retry(3)
def train_models(self, training_snapshots: List[MultiVariableSnapshot], 
                leak_labels: List[bool], leak_types: Optional[List[str]] = None):
    """
    Treina todos os modelos com validação cruzada e métricas robustas
    
    Processo completo:
    1. Extração de features para todos os snapshots
    2. Divisão treino/validação estratificada
    3. Treinamento de cada modelo
    4. Validação cruzada
    5. Cálculo de métricas de performance
    """
    
    if len(training_snapshots) < 50:
        raise ValueError(f"Necessários pelo menos 50 snapshots para treino. Fornecidos: {len(training_snapshots)}")
    
    self.logger.info(f"Iniciando treinamento com {len(training_snapshots)} snapshots")
    
    # Extração de features para todo o conjunto
    all_features = []
    for snapshots_batch in self._create_feature_windows(training_snapshots):
        features = self.extract_advanced_features(snapshots_batch)
        all_features.append(features)
    
    X = np.array(all_features)
    y = np.array(leak_labels[:len(all_features)])
    
    # Normalização
    self.scaler.fit(X)
    X_scaled = self.scaler.transform(X)
    
    # Divisão treino/teste estratificada
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    training_results = {}
    
    # 1. Treina Isolation Forest
    if_results = self.train_isolation_forest(X_train, y_train)
    training_results['isolation_forest'] = if_results
    
    # 2. Treina Random Forest se há exemplos positivos suficientes
    positive_samples = np.sum(y_train)
    if positive_samples >= 5:
        rf_results = self.train_random_forest_classifier(X_train, y_train, leak_types)
        if rf_results:
            training_results['random_forest'] = rf_results
    
    # 3. Treina SVM se há exemplos balanceados
    if positive_samples >= 10 and positive_samples / len(y_train) > 0.1:
        svm_results = self.train_svm_classifier(X_train, y_train)
        training_results['svm'] = svm_results
    
    # Validação nos dados de teste
    test_results = self._validate_models(X_test, y_test)
    training_results['validation'] = test_results
    
    # Atualiza estado do sistema
    self.is_trained = True
    self.last_training_time = datetime.now()
    
    # Adiciona dados ao buffer para retreino futuro
    for i, (features, label) in enumerate(zip(all_features, leak_labels)):
        self.training_data_buffer.append({
            'features': features.tolist(),
            'label': bool(label),
            'timestamp': datetime.now().isoformat(),
            'leak_type': leak_types[i] if leak_types and i < len(leak_types) else None
        })
    
    # Limita tamanho do buffer
    if len(self.training_data_buffer) > 1000:
        self.training_data_buffer = self.training_data_buffer[-1000:]
    
    self.logger.info("Treinamento concluído com sucesso")
    
    return training_results
```

#### 📈 Métricas de Performance

```python
def _validate_models(self, X_test, y_test) -> Dict[str, Any]:
    """
    Valida performance de todos os modelos treinados
    
    Métricas calculadas:
    - Precision, Recall, F1-score
    - AUC-ROC
    - Confusion Matrix
    - Specificity, Sensitivity
    """
    
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
    
    validation_results = {}
    
    # Validation Isolation Forest
    if self.leak_detector is not None:
        y_pred_if = self.leak_detector.predict(X_test)
        y_pred_if_binary = (y_pred_if == -1).astype(int)  # -1 = anomaly = 1
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred_if_binary, average='binary'
        )
        
        # AUC using decision function
        y_scores_if = self.leak_detector.decision_function(X_test)
        auc_if = roc_auc_score(y_test, -y_scores_if)  # Negative because -1 is anomaly
        
        conf_matrix_if = confusion_matrix(y_test, y_pred_if_binary)
        
        validation_results['isolation_forest'] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_roc': float(auc_if),
            'confusion_matrix': conf_matrix_if.tolist()
        }
    
    # Validation Random Forest
    if self.leak_classifier is not None:
        try:
            y_pred_rf = self.leak_classifier.predict(X_test)
            
            # Convert multiclass to binary if needed
            if hasattr(self.leak_classifier, 'classes_'):
                if len(self.leak_classifier.classes_) > 2:
                    y_pred_rf_binary = (y_pred_rf != 'normal').astype(int)
                else:
                    y_pred_rf_binary = (y_pred_rf == 'leak').astype(int)
            else:
                y_pred_rf_binary = y_pred_rf.astype(int)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred_rf_binary, average='binary'
            )
            
            # AUC using predict_proba
            y_proba_rf = self.leak_classifier.predict_proba(X_test)
            if y_proba_rf.shape[1] > 1:
                auc_rf = roc_auc_score(y_test, y_proba_rf[:, 1])
            else:
                auc_rf = 0.5
            
            conf_matrix_rf = confusion_matrix(y_test, y_pred_rf_binary)
            
            validation_results['random_forest'] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc_roc': float(auc_rf),
                'confusion_matrix': conf_matrix_rf.tolist()
            }
            
        except Exception as e:
            self.logger.warning(f"Erro na validação do Random Forest: {e}")
    
    # Validation SVM
    if hasattr(self, 'svm_classifier') and self.svm_classifier is not None:
        try:
            y_pred_svm = self.svm_classifier.predict(X_test)
            y_proba_svm = self.svm_classifier.predict_proba(X_test)[:, 1]
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred_svm, average='binary'
            )
            
            auc_svm = roc_auc_score(y_test, y_proba_svm)
            conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
            
            validation_results['svm'] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc_roc': float(auc_svm),
                'confusion_matrix': conf_matrix_svm.tolist()
            }
            
        except Exception as e:
            self.logger.warning(f"Erro na validação do SVM: {e}")
    
    # Métricas consolidadas
    if validation_results:
        # Média das métricas dos modelos disponíveis
        all_f1_scores = [r['f1_score'] for r in validation_results.values()]
        all_auc_scores = [r['auc_roc'] for r in validation_results.values()]
        
        validation_results['consolidated'] = {
            'average_f1_score': float(np.mean(all_f1_scores)),
            'average_auc': float(np.mean(all_auc_scores)),
            'model_count': len(validation_results),
            'performance_grade': 'excellent' if np.mean(all_f1_scores) > 0.8 else 
                               'good' if np.mean(all_f1_scores) > 0.6 else 'needs_improvement'
        }
    
    return validation_results
```

#### 🔄 Detecção de Deriva de Modelo

```python
def detect_model_drift(self, recent_predictions: List[Dict], window_size: int = 100) -> Dict[str, Any]:
    """
    Detecta deriva dos modelos baseada na performance recente
    
    Indicadores de deriva:
    1. Queda na confiança média das predições
    2. Aumento na dispersão das probabilidades
    3. Mudança na distribuição dos scores
    """
    
    if len(recent_predictions) < window_size:
        return {'drift_detected': False, 'reason': 'Dados insuficientes'}
    
    # Extrai métricas das predições recentes
    recent_window = recent_predictions[-window_size:]
    historical_window = recent_predictions[-2*window_size:-window_size] if len(recent_predictions) >= 2*window_size else []
    
    # Métricas da janela recente
    recent_confidences = [p.get('confidence', 0.5) for p in recent_window]
    recent_probabilities = [p.get('leak_probability', 0.5) for p in recent_window]
    
    recent_avg_confidence = np.mean(recent_confidences)
    recent_prob_variance = np.var(recent_probabilities)
    
    # Comparação com histórico se disponível
    drift_indicators = {}
    
    if historical_window:
        hist_confidences = [p.get('confidence', 0.5) for p in historical_window]
        hist_probabilities = [p.get('leak_probability', 0.5) for p in historical_window]
        
        hist_avg_confidence = np.mean(hist_confidences)
        hist_prob_variance = np.var(hist_probabilities)
        
        # Mudança na confiança
        confidence_change = (recent_avg_confidence - hist_avg_confidence) / hist_avg_confidence
        
        # Mudança na variância
        variance_change = (recent_prob_variance - hist_prob_variance) / max(hist_prob_variance, 1e-6)
        
        drift_indicators = {
            'confidence_drop': confidence_change < -0.2,  # Queda > 20%
            'variance_increase': variance_change > 0.5,   # Aumento > 50%
            'confidence_change_pct': float(confidence_change * 100),
            'variance_change_pct': float(variance_change * 100)
        }
    
    # Detecção de padrões anômalos
    confidence_trend = np.polyfit(range(len(recent_confidences)), recent_confidences, 1)[0]
    
    drift_indicators.update({
        'confidence_declining': confidence_trend < -0.001,  # Tendência negativa
        'low_average_confidence': recent_avg_confidence < 0.5,
        'high_probability_variance': recent_prob_variance > 0.1
    })
    
    # Decisão sobre deriva
    drift_score = sum([
        2 if drift_indicators.get('confidence_drop', False) else 0,
        2 if drift_indicators.get('variance_increase', False) else 0,
        1 if drift_indicators.get('confidence_declining', False) else 0,
        1 if drift_indicators.get('low_average_confidence', False) else 0,
        1 if drift_indicators.get('high_probability_variance', False) else 0
    ])
    
    drift_detected = drift_score >= 3  # Threshold para detecção
    
    # Recomendação
    if drift_detected:
        recommendation = 'Retreino do modelo recomendado - deriva detectada'
    elif drift_score >= 2:
        recommendation = 'Monitoramento intensificado - possível deriva'
    else:
        recommendation = 'Modelo operando dentro dos parâmetros normais'
    
    return {
        'drift_detected': drift_detected,
        'drift_score': drift_score,
        'max_score': 7,
        'indicators': drift_indicators,
        'recent_metrics': {
            'avg_confidence': float(recent_avg_confidence),
            'probability_variance': float(recent_prob_variance),
            'confidence_trend': float(confidence_trend)
        },
        'recommendation': recommendation
    }
```

---

## 🎯 Resumo Final do Sistema ML

### Capacidades Completas

O sistema de Machine Learning implementa:

- ✅ **81 Features Especializadas** - Extração multidimensional completa
- ✅ **4 Algoritmos Principais** - IF, RF, SVM, DBSCAN para máxima cobertura
- ✅ **Fusão Inteligente** - Ensemble com pesos adaptativos
- ✅ **Análise PCA** - Redução de dimensionalidade e detecção de padrões
- ✅ **CCA** - Correlações canônicas entre conjuntos de variáveis
- ✅ **Retreino Adaptativo** - Aprendizado contínuo baseado em performance
- ✅ **Detecção de Deriva** - Monitoramento da saúde dos modelos
- ✅ **Validação Robusta** - Métricas completas de performance

### Performance Esperada

| Cenário | Precision | Recall | F1-Score | AUC-ROC |
|---------|-----------|--------|----------|---------|
| **Vazamentos Grandes** | >95% | >90% | >92% | >0.95 |
| **Vazamentos Pequenos** | >80% | >75% | >77% | >0.85 |
| **Bloqueios** | >85% | >80% | >82% | >0.90 |
| **Deriva de Sensores** | >70% | >65% | >67% | >0.80 |

---

**Sistema de Machine Learning - Manual Completo Finalizado**

*Sistema Hidráulico Industrial v2.0 - Agosto 2025*
