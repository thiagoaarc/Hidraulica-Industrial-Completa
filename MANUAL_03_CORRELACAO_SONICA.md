# Manual de Correlação Sônica - Sistema Hidráulico Industrial

## 📋 Índice da Correlação Sônica

1. [Fundamentos Teóricos](#fundamentos-teóricos)
2. [Matemática da Correlação](#matemática-da-correlação)
3. [Implementação Computacional](#implementação-computacional)
4. [Características Físicas do Duto](#características-físicas-do-duto)
5. [Análise de Sinais](#análise-de-sinais)
6. [Detecção de Vazamentos](#detecção-de-vazamentos)
7. [Compensações e Correções](#compensações-e-correções)
8. [Visualização e Interpretação](#visualização-e-interpretação)

---

## 🌊 Fundamentos Teóricos

### Princípio da Correlação Sônica

A correlação sônica baseia-se na **propagação de ondas de pressão** através do fluido em dutos. Quando há uma perturbação no sistema (vazamento, bloqueio, mudança de regime), ela gera ondas que se propagam em ambas as direções.

#### 🔬 Física das Ondas de Pressão

##### **Equação de Propagação**

A velocidade de propagação de ondas em fluidos confinados é dada por:

```
c = √(K/ρ × (1 + (K/E) × (D/e)))
```

Onde:

- `c` = velocidade sônica efetiva (m/s)
- `K` = módulo de compressibilidade do fluido (Pa)
- `ρ` = densidade do fluido (kg/m³)
- `E` = módulo de elasticidade do material do duto (Pa)
- `D` = diâmetro interno do duto (m)
- `e` = espessura da parede do duto (m)

##### **Fatores de Correção**

```python
def calculate_effective_velocity(base_velocity, pipe_characteristics):
    """
    Calcula velocidade sônica efetiva considerando:
    - Material do duto
    - Geometria (diâmetro/espessura)
    - Temperatura do fluido
    - Pressão operacional
    """
    
    # Fator do material
    material_factor = PIPE_MATERIALS[pipe_characteristics.material]['sonic_velocity_factor']
    
    # Correção por temperatura (água)
    temp_correction = 1.0 + 0.0024 * (temperature - 20)
    
    # Correção por pressão
    pressure_correction = 1.0 + 0.0001 * (pressure - 10)
    
    # Fator geométrico
    geometric_factor = 1.0 / (1.0 + (K_fluid/E_material) * (diameter/wall_thickness))
    
    return base_velocity * material_factor * temp_correction * pressure_correction * geometric_factor
```

#### 📡 Propagação e Atenuação

##### **Modelo de Atenuação**

A amplitude da onda decresce com a distância:

```
A(x) = A₀ × e^(-α×x)
```

Onde:

- `A(x)` = amplitude na posição x
- `A₀` = amplitude inicial  
- `α` = coeficiente de atenuação (1/m)
- `x` = distância percorrida (m)

```python
def calculate_attenuation_factor(distance_km, pipe_material):
    """
    Calcula fator de atenuação baseado em:
    - Distância percorrida
    - Material do duto
    - Rugosidade interna
    - Frequência da onda
    """
    
    attenuation_coeff = PIPE_MATERIALS[pipe_material]['attenuation']
    
    # Atenuação exponencial
    distance_factor = np.exp(-attenuation_coeff * distance_km)
    
    # Correção por rugosidade
    roughness_factor = 1.0 - 0.1 * (roughness / 0.045)  # Normalizado ao aço
    
    return distance_factor * roughness_factor
```

---

## 📊 Matemática da Correlação

### Correlação Cruzada

A correlação cruzada é o coração da análise sônica:

#### 🧮 Definição Matemática

Para dois sinais discretos x[n] e y[n]:

```
R_xy[m] = Σ(n=-∞ to ∞) x[n] × y[n+m]
```

Na implementação normalizada:

```
R_xy[m] = (1/N) × Σ(n=0 to N-1) x[n] × y[n+m] / √(σ_x × σ_y)
```

Onde:

- `N` = número de amostras
- `σ_x`, `σ_y` = desvios padrão dos sinais
- `m` = delay em amostras

#### 💻 Implementação Otimizada

```python
def _perform_sonic_correlation_analysis(self, snapshots: List[MultiVariableSnapshot]) -> Dict[str, Any]:
    """
    Análise de correlação sônica com características físicas do sistema
    
    Implementa:
    1. Correlação cruzada FFT (O(N log N))
    2. Compensação por características do duto  
    3. Análise de qualidade do sinal
    4. Cálculo de tempo de trânsito
    """
    
    if len(snapshots) < 50:
        return {'error': 'Dados insuficientes para correlação'}
    
    # Extração dos sinais de pressão
    exp_pressures = np.array([s.expeditor_pressure for s in snapshots])
    rec_pressures = np.array([s.receiver_pressure for s in snapshots])
    
    # Propriedades acústicas do sistema
    acoustic_props = self.config.pipe_characteristics.calculate_acoustic_properties()
    
    # Velocidade sônica efetiva
    effective_velocity = self.config.sonic_velocity * acoustic_props['velocity_factor']
    
    # Correlação cruzada usando FFT (mais eficiente)
    correlation = signal.correlate(exp_pressures, rec_pressures, mode='full', method='fft')
    
    # Encontra pico máximo
    max_corr_idx = np.argmax(np.abs(correlation))
    delay_samples = max_corr_idx - len(correlation)//2
    max_correlation = correlation[max_corr_idx] / len(exp_pressures)
    
    # Tempo de trânsito teórico esperado
    expected_transit_time = self.config.sensor_distance / effective_velocity
    
    # Compensação por atenuação
    distance_factor = np.exp(-acoustic_props['attenuation'] * self.config.sensor_distance / 1000)
    compensated_correlation = max_correlation / distance_factor
    
    # Análise de qualidade dos sinais
    snr_exp = self._calculate_snr(exp_pressures)
    snr_rec = self._calculate_snr(rec_pressures)
    
    return {
        'max_correlation': float(max_correlation),
        'compensated_correlation': float(compensated_correlation),
        'delay_samples': int(delay_samples),
        'expected_transit_time': float(expected_transit_time),
        'effective_velocity': float(effective_velocity),
        'distance_factor': float(distance_factor),
        'snr_expeditor': float(snr_exp),
        'snr_receiver': float(snr_rec),
        'acoustic_properties': acoustic_props,
        'signal_quality': 'excellent' if min(snr_exp, snr_rec) > 30 else 
                        'good' if min(snr_exp, snr_rec) > 20 else 'poor'
    }
```

### Análise de Frequências

#### 🎵 Transformada de Fourier

Para análise espectral dos sinais de pressão:

```python
def spectral_analysis(self, pressure_signal, sampling_rate):
    """
    Análise espectral completa do sinal de pressão
    
    Implementa:
    1. FFT para análise de frequências
    2. Densidade espectral de potência
    3. Detecção de frequências dominantes
    4. Análise de harmônicos
    """
    
    # Remove componente DC e aplica janela
    signal_ac = pressure_signal - np.mean(pressure_signal)
    windowed_signal = signal_ac * np.hanning(len(signal_ac))
    
    # FFT
    fft_result = np.fft.fft(windowed_signal)
    frequencies = np.fft.fftfreq(len(signal_ac), 1/sampling_rate)
    
    # Apenas frequências positivas
    positive_freq_idx = frequencies >= 0
    frequencies = frequencies[positive_freq_idx]
    magnitude = np.abs(fft_result[positive_freq_idx])
    phase = np.angle(fft_result[positive_freq_idx])
    
    # Densidade espectral de potência
    psd = magnitude**2 / len(signal_ac)
    
    # Frequências dominantes
    dominant_freqs_idx = find_peaks(magnitude, height=0.1*np.max(magnitude))[0]
    dominant_frequencies = frequencies[dominant_freqs_idx]
    
    return {
        'frequencies': frequencies,
        'magnitude_db': 20 * np.log10(magnitude + 1e-10),
        'phase': phase,
        'psd': psd,
        'dominant_frequencies': dominant_frequencies,
        'total_power': np.sum(psd),
        'peak_frequency': frequencies[np.argmax(magnitude)]
    }
```

#### 🔍 Filtros Butterworth

Para isolamento de bandas de interesse:

```python
def apply_butterworth_filter(self, signal, low_freq, high_freq, sampling_rate, order=4):
    """
    Aplica filtro Butterworth passa-banda
    
    Parâmetros:
    - signal: sinal de entrada
    - low_freq: frequência de corte inferior (Hz)
    - high_freq: frequência de corte superior (Hz)  
    - sampling_rate: taxa de amostragem (Hz)
    - order: ordem do filtro
    """
    
    # Normalização das frequências
    nyquist = sampling_rate / 2
    low_norm = low_freq / nyquist
    high_norm = high_freq / nyquist
    
    # Design do filtro
    b, a = signal.butter(order, [low_norm, high_norm], btype='band')
    
    # Aplicação do filtro (zero-phase para evitar distorção)
    filtered_signal = signal.filtfilt(b, a, signal)
    
    return filtered_signal
```

---

## 🔧 Implementação Computacional

### Otimizações de Performance

#### ⚡ Correlação FFT

A implementação utiliza FFT para correlação O(N log N):

```python
def fast_correlation(self, signal1, signal2):
    """
    Correlação rápida usando FFT
    
    Complexidade: O(N log N) vs O(N²) da implementação direta
    """
    
    # Tamanho para correlação completa
    n = len(signal1) + len(signal2) - 1
    
    # Zero-padding para próxima potência de 2 (mais eficiente para FFT)
    fft_size = 2**int(np.ceil(np.log2(n)))
    
    # FFT dos sinais
    fft1 = np.fft.fft(signal1, fft_size)
    fft2 = np.fft.fft(signal2, fft_size)
    
    # Correlação no domínio da frequência
    correlation_fft = fft1 * np.conj(fft2)
    
    # IFFT para voltar ao domínio do tempo
    correlation = np.fft.ifft(correlation_fft).real
    
    # Retorna apenas parte válida
    return correlation[:n]
```

#### 🧮 Processamento Vectorizado

```python
def vectorized_snr_calculation(self, signals_matrix):
    """
    Cálculo vectorizado de SNR para múltiplos sinais
    
    signals_matrix: (n_signals, n_samples)
    """
    
    # Potência do sinal (variância)
    signal_power = np.var(signals_matrix, axis=1)
    
    # Estimativa de ruído pela diferença
    diff_signals = np.diff(signals_matrix, axis=1)
    noise_power = np.var(diff_signals, axis=1) / 2
    
    # SNR em dB
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # Limita entre 0 e 60 dB
    return np.clip(snr_db, 0, 60)
```

### Cache e Memória

#### 💾 Sistema de Cache

```python
class CorrelationCache:
    """Cache inteligente para resultados de correlação"""
    
    def __init__(self, max_size=1000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        
    def get_cache_key(self, exp_data, rec_data, config):
        """Gera chave única para combinação de dados"""
        exp_hash = hashlib.md5(exp_data.tobytes()).hexdigest()[:8]
        rec_hash = hashlib.md5(rec_data.tobytes()).hexdigest()[:8]
        config_hash = hashlib.md5(str(config).encode()).hexdigest()[:8]
        return f"{exp_hash}_{rec_hash}_{config_hash}"
        
    def get(self, key):
        """Recupera resultado do cache"""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
        
    def put(self, key, result):
        """Armazena resultado no cache com LRU"""
        if len(self.cache) >= self.max_size:
            # Remove item menos recentemente usado
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
            
        self.cache[key] = result
        self.access_times[key] = time.time()
```

---

## 🏗️ Características Físicas do Duto

### Cálculo de Propriedades Acústicas

```python
def calculate_acoustic_properties(self) -> Dict[str, float]:
    """
    Calcula propriedades acústicas completas do duto
    
    Baseado em:
    - Material do duto e suas propriedades
    - Geometria (diâmetro, espessura, perfil)
    - Rugosidade interna
    - Frequência de operação
    """
    
    material_props = CONSTANTS.PIPE_MATERIALS.get(
        self.material, CONSTANTS.PIPE_MATERIALS['steel']
    )
    profile_props = CONSTANTS.PIPE_PROFILES.get(
        self.profile, CONSTANTS.PIPE_PROFILES['circular']
    )
    
    # Fator de velocidade combinado
    velocity_factor = (
        material_props['sonic_velocity_factor'] * 
        profile_props['area_factor']
    )
    
    # Atenuação específica do material
    base_attenuation = material_props['attenuation']
    
    # Correção por rugosidade
    roughness_factor = self.roughness / 0.045  # Normalizado ao aço
    adjusted_attenuation = base_attenuation * (1 + 0.2 * roughness_factor)
    
    # Fator geométrico para dispersão
    area_factor = profile_props['area_factor']
    perimeter_factor = profile_props['perimeter_factor']
    
    # Número de Reynolds hidráulico
    hydraulic_diameter = 4 * area_factor * (self.diameter/2)**2 / (perimeter_factor * self.diameter)
    
    return {
        'velocity_factor': velocity_factor,
        'attenuation': adjusted_attenuation,
        'roughness_factor': roughness_factor,
        'area_factor': area_factor,
        'perimeter_factor': perimeter_factor,
        'hydraulic_diameter': hydraulic_diameter,
        'dispersion_factor': 1.0 / area_factor  # Maior dispersão em perfis não-circulares
    }
```

### Tabela de Materiais

#### 🔩 Propriedades por Material

| Material | Vel. Factor | Atenuação | Rugosidade (mm) | Aplicação |
|----------|-------------|-----------|-----------------|-----------|
| **Aço** | 1.0 | 0.1 | 0.045 | Industrial padrão |
| **PVC** | 0.8 | 0.2 | 0.0015 | Água potável |
| **Concreto** | 1.2 | 0.05 | 0.3 | Grandes diâmetros |
| **Fibra** | 0.9 | 0.15 | 0.01 | Anti-corrosivo |

#### 📐 Correções por Perfil

| Perfil | Área Factor | Perímetro Factor | Características |
|--------|-------------|------------------|-----------------|
| **Circular** | 1.0 | 1.0 | Mínima dispersão |
| **Retangular** | 0.9 | 1.2 | Dispersão moderada |  
| **Oval** | 0.95 | 1.1 | Dispersão baixa |

---

## 📈 Análise de Sinais

### Relação Sinal-Ruído (SNR)

#### 📊 Cálculo de SNR

```python
def _calculate_snr(self, signal: np.ndarray) -> float:
    """
    Calcula relação sinal-ruído robusta
    
    Método:
    1. Potência do sinal = variância dos dados
    2. Potência do ruído = variância das diferenças (estimativa de ruído branco)
    3. SNR = 10 × log₁₀(P_signal / P_noise)
    """
    
    if len(signal) < 2:
        return 0.0
    
    # Potência do sinal
    signal_power = np.var(signal)
    
    # Estimativa de ruído pela diferença entre amostras consecutivas
    noise_estimate = np.var(np.diff(signal)) / 2
    
    if noise_estimate > 0 and signal_power > 0:
        snr = 10 * np.log10(signal_power / noise_estimate)
        return max(0, min(snr, 60))  # Limita entre 0 e 60 dB
    else:
        return 60.0  # SNR muito alto se sem ruído detectável
```

#### 🎯 Interpretação da Qualidade

```python
def classify_signal_quality(snr_db):
    """
    Classifica qualidade do sinal baseada no SNR
    
    Critérios:
    - Excellent: > 30 dB (ruído < 3.2% do sinal)
    - Good: 20-30 dB (ruído 3.2-10% do sinal)  
    - Fair: 10-20 dB (ruído 10-32% do sinal)
    - Poor: < 10 dB (ruído > 32% do sinal)
    """
    
    if snr_db > 30:
        return 'excellent', '#4CAF50'  # Verde
    elif snr_db > 20:
        return 'good', '#2196F3'       # Azul
    elif snr_db > 10:
        return 'fair', '#FF9800'       # Laranja
    else:
        return 'poor', '#f44336'       # Vermelho
```

### Análise de Coerência

#### 🌊 Coerência Espectral

```python
def calculate_coherence(self, signal1, signal2, sampling_rate, nperseg=256):
    """
    Calcula coerência espectral entre dois sinais
    
    Coerência = |G_xy(f)|² / (G_xx(f) × G_yy(f))
    
    Onde:
    - G_xy(f) = densidade espectral cruzada
    - G_xx(f), G_yy(f) = densidades espectrais auto
    """
    
    # Calcula coerência usando Welch
    frequencies, coherence = signal.coherence(
        signal1, signal2, 
        fs=sampling_rate,
        nperseg=nperseg,
        overlap=nperseg//2
    )
    
    # Estatísticas da coerência
    mean_coherence = np.mean(coherence)
    peak_coherence = np.max(coherence)
    coherent_bandwidth = np.sum(coherence > 0.5) / len(coherence)
    
    return {
        'frequencies': frequencies,
        'coherence': coherence,
        'mean_coherence': mean_coherence,
        'peak_coherence': peak_coherence,
        'coherent_bandwidth_ratio': coherent_bandwidth
    }
```

---

## 🚨 Detecção de Vazamentos

### Indicadores Sônicos

#### 📉 Degradação da Correlação

```python
def analyze_correlation_degradation(self, correlation_history):
    """
    Analisa degradação da correlação ao longo do tempo
    
    Indicadores de vazamento:
    1. Redução gradual da correlação máxima
    2. Aumento da largura do pico principal
    3. Aparição de picos secundários
    4. Mudança no delay de propagação
    """
    
    correlations = [r['max_correlation'] for r in correlation_history]
    delays = [r['delay_samples'] for r in correlation_history]
    
    # Tendência da correlação
    correlation_trend = np.polyfit(range(len(correlations)), correlations, 1)[0]
    
    # Variabilidade do delay
    delay_variance = np.var(delays)
    
    # Detecção de mudança abrupta (CUSUM)
    cusum_correlation = self.cusum_detection(correlations, threshold=0.1)
    
    # Score de vazamento baseado em correlação
    leak_indicators = {
        'correlation_trend': correlation_trend,  # Negativo indica degradação
        'delay_variance': delay_variance,        # Alta indica instabilidade
        'cusum_detection': cusum_correlation,    # True indica mudança abrupta
        'current_correlation': correlations[-1] if correlations else 0.0
    }
    
    # Score combinado (0-1, onde 1 = vazamento certo)
    leak_score = 0.0
    
    if correlation_trend < -0.01:  # Degradação > 1% por amostra
        leak_score += 0.4
        
    if delay_variance > 10:  # Alta variabilidade
        leak_score += 0.3
        
    if cusum_correlation:  # Mudança detectada
        leak_score += 0.3
        
    return leak_score, leak_indicators
```

#### 🔍 Algoritmo CUSUM

```python
def cusum_detection(self, data, threshold=0.1, drift=0.01):
    """
    Detecção de mudança usando CUSUM (Cumulative Sum)
    
    Detecta mudanças abruptas na média da série temporal
    """
    
    mean_baseline = np.mean(data[:min(50, len(data)//2)])  # Linha de base
    
    cusum_pos = 0
    cusum_neg = 0
    detection_threshold = threshold * np.std(data)
    
    for i, value in enumerate(data):
        # Diferença da baseline
        diff = value - mean_baseline
        
        # CUSUM positivo e negativo
        cusum_pos = max(0, cusum_pos + diff - drift)
        cusum_neg = max(0, cusum_neg - diff - drift)
        
        # Detecção se exceder threshold
        if cusum_pos > detection_threshold or cusum_neg > detection_threshold:
            return True, i  # Mudança detectada no índice i
            
    return False, -1
```

### Padrões de Vazamento

#### 📊 Assinatura Espectral

```python
def analyze_leak_spectral_signature(self, pressure_fft, baseline_fft):
    """
    Analisa assinatura espectral de vazamentos
    
    Características típicas:
    1. Aumento em altas frequências (turbulência)
    2. Harmônicos específicos relacionados ao tipo de vazamento
    3. Mudanças na distribuição de energia espectral
    """
    
    # Diferença espectral
    spectral_diff = np.abs(pressure_fft) - np.abs(baseline_fft)
    
    # Energia em diferentes bandas
    low_freq_energy = np.sum(spectral_diff[:len(spectral_diff)//4])
    mid_freq_energy = np.sum(spectral_diff[len(spectral_diff)//4:len(spectral_diff)//2])
    high_freq_energy = np.sum(spectral_diff[len(spectral_diff)//2:])
    
    # Razões características
    high_to_low_ratio = high_freq_energy / (low_freq_energy + 1e-10)
    spectral_centroid_shift = self.calculate_spectral_centroid_shift(pressure_fft, baseline_fft)
    
    return {
        'high_to_low_ratio': high_to_low_ratio,
        'spectral_centroid_shift': spectral_centroid_shift,
        'energy_distribution': [low_freq_energy, mid_freq_energy, high_freq_energy],
        'leak_probability': min(1.0, high_to_low_ratio * 0.3 + abs(spectral_centroid_shift) * 0.5)
    }
```

---

## ⚖️ Compensações e Correções

### Compensação por Temperatura

#### 🌡️ Correção da Velocidade Sônica

```python
def temperature_compensated_velocity(self, base_velocity, temperature, fluid_type='water'):
    """
    Compensa velocidade sônica pela temperatura
    
    Para água: c(T) = c₀ × (1 + 0.0024 × (T - 20))
    Para óleo: c(T) = c₀ × (1 + 0.0018 × (T - 20))  
    """
    
    if fluid_type == 'water':
        temp_coefficient = 0.0024
    elif fluid_type == 'oil':
        temp_coefficient = 0.0018
    else:
        temp_coefficient = 0.002  # Valor genérico
        
    reference_temp = 20.0  # °C
    correction_factor = 1.0 + temp_coefficient * (temperature - reference_temp)
    
    return base_velocity * correction_factor
```

### Compensação por Pressão

#### 💧 Efeito da Pressão na Compressibilidade

```python
def pressure_compensated_velocity(self, base_velocity, pressure, fluid_type='water'):
    """
    Compensa velocidade sônica pela pressão
    
    Baseado na variação do módulo de compressibilidade com pressão
    """
    
    if fluid_type == 'water':
        # Água: módulo de compressibilidade aumenta ~0.01% por kgf/cm²
        pressure_coefficient = 0.0001
    else:
        pressure_coefficient = 0.00005  # Genérico
        
    reference_pressure = 10.0  # kgf/cm²
    correction_factor = 1.0 + pressure_coefficient * (pressure - reference_pressure)
    
    return base_velocity * correction_factor
```

### Compensação por Atenuação

#### 📉 Modelo de Atenuação Avançado

```python
def advanced_attenuation_compensation(self, correlation, distance_km, frequency_hz, pipe_properties):
    """
    Compensação avançada por atenuação considerando:
    1. Atenuação geométrica (espalhamento esférico)
    2. Atenuação por absorção do material
    3. Atenuação por rugosidade
    4. Atenuação dependente da frequência
    """
    
    # Atenuação geométrica (1/r)
    geometric_attenuation = 1.0 / max(1.0, distance_km)
    
    # Atenuação por material
    material_attenuation = np.exp(-pipe_properties['attenuation'] * distance_km)
    
    # Atenuação por rugosidade (dependente da frequência)
    roughness_attenuation = np.exp(-0.001 * pipe_properties['roughness_factor'] * 
                                   frequency_hz * distance_km)
    
    # Fator de compensação total
    total_attenuation = geometric_attenuation * material_attenuation * roughness_attenuation
    
    # Correlação compensada
    compensated_correlation = correlation / total_attenuation
    
    return {
        'compensated_correlation': compensated_correlation,
        'geometric_factor': geometric_attenuation,
        'material_factor': material_attenuation,
        'roughness_factor': roughness_attenuation,
        'total_compensation': 1.0 / total_attenuation
    }
```

---

## 📊 Visualização e Interpretação

### Gráficos de Correlação

#### 📈 Plot Principal de Correlação

```python
def plot_correlation_analysis(self, correlation_result):
    """
    Gera visualização completa da análise de correlação
    
    Inclui:
    1. Correlação cruzada vs delay
    2. Marcação do pico máximo
    3. Tempo de trânsito esperado
    4. Zona de confiança
    """
    
    # Dados da correlação
    correlation = correlation_result['correlation_curve']
    delays = correlation_result['delay_samples'] 
    max_idx = correlation_result['max_correlation_index']
    expected_delay = correlation_result['expected_delay_samples']
    
    # Plot principal
    self.correlation_curve.setData(delays, correlation)
    
    # Marca pico máximo
    max_point = pg.ScatterPlotItem([delays[max_idx]], [correlation[max_idx]], 
                                  brush='red', size=10, symbol='o')
    self.correlation_plot.addItem(max_point)
    
    # Linha do tempo esperado
    expected_line = pg.InfiniteLine(pos=expected_delay, angle=90, 
                                   pen=mkPen('green', width=2, style=Qt.PenStyle.DashLine),
                                   label='Esperado')
    self.correlation_plot.addItem(expected_line)
    
    # Zona de confiança (±10% do tempo esperado)
    confidence_zone = pg.LinearRegionItem([expected_delay*0.9, expected_delay*1.1], 
                                         brush=pg.mkBrush(color=(0, 255, 0, 50)))
    self.correlation_plot.addItem(confidence_zone)
```

#### 🎨 Código de Cores

| Cor | Significado | Interpretação |
|-----|-------------|---------------|
| **Azul** | Correlação normal | Sistema operando normalmente |
| **Verde** | Tempo esperado | Delay teórico de propagação |
| **Vermelho** | Pico máximo | Delay real detectado |
| **Laranja** | Threshold ML | Limiar de decisão |
| **Cinza** | Zona morta | Região sem informação útil |

### Métricas na Interface

#### 📋 Painel de Correlação

```python
def update_correlation_metrics(self, correlation_result):
    """
    Atualiza métricas na interface baseadas na análise de correlação
    """
    
    # Correlação principal
    max_corr = correlation_result['max_correlation']
    self.correlation_label.setText(f"Correlação: {max_corr:.3f}")
    
    # Código de cores baseado na qualidade
    if max_corr > 0.8:
        color = "#4CAF50"  # Verde - Excelente
        status = "Excelente"
    elif max_corr > 0.6:
        color = "#2196F3"  # Azul - Boa
        status = "Boa"
    elif max_corr > 0.4:
        color = "#FF9800"  # Laranja - Regular
        status = "Regular"
    else:
        color = "#f44336"  # Vermelho - Ruim
        status = "Ruim"
    
    self.correlation_label.setStyleSheet(f"color: {color}; font-weight: bold;")
    
    # Métricas detalhadas
    self.snr_exp_label.setText(f"SNR Expedidor: {correlation_result['snr_expeditor']:.1f} dB")
    self.snr_rec_label.setText(f"SNR Recebedor: {correlation_result['snr_receiver']:.1f} dB")
    self.velocity_label.setText(f"Velocidade Efetiva: {correlation_result['effective_velocity']:.1f} m/s")
    self.delay_label.setText(f"Delay Detectado: {correlation_result['delay_samples']} amostras")
```

### Interpretação dos Resultados

#### 🎯 Guidelines de Interpretação

##### **Correlação Alta (> 0.8)**

- **Significado**: Sistema operando normalmente
- **Características**: Pico bem definido, delay consistente
- **Ação**: Monitoramento rotineiro

##### **Correlação Média (0.4 - 0.8)**

- **Significado**: Possível degradação ou ruído
- **Características**: Pico menos definido, múltiplos picos
- **Ação**: Investigação adicional, aumento da frequência de monitoramento

##### **Correlação Baixa (< 0.4)**

- **Significado**: Possível vazamento ou falha de sensor
- **Características**: Ausência de pico claro, correlação dispersa
- **Ação**: Inspeção imediata, verificação de sensores

#### 📊 Padrões Típicos

```python
CORRELATION_PATTERNS = {
    'normal_operation': {
        'correlation_range': (0.8, 1.0),
        'delay_stability': 'stable',
        'peak_sharpness': 'sharp',
        'secondary_peaks': 'absent',
        'interpretation': 'Sistema normal'
    },
    
    'gradual_leak': {
        'correlation_range': (0.5, 0.8),
        'delay_stability': 'slowly_drifting',
        'peak_sharpness': 'broadening',
        'secondary_peaks': 'emerging',
        'interpretation': 'Possível vazamento gradual'
    },
    
    'sudden_leak': {
        'correlation_range': (0.2, 0.6),
        'delay_stability': 'sudden_change',
        'peak_sharpness': 'split_or_absent',
        'secondary_peaks': 'prominent',
        'interpretation': 'Vazamento súbito ou bloqueio'
    },
    
    'sensor_failure': {
        'correlation_range': (0.0, 0.3),
        'delay_stability': 'random',
        'peak_sharpness': 'absent',
        'secondary_peaks': 'noise_like',
        'interpretation': 'Falha de sensor ou desconexão'
    }
}
```

---

## 🎯 Integração com Outros Módulos

### Interface com Machine Learning

```python
def prepare_sonic_features_for_ml(self, correlation_results):
    """
    Prepara features da correlação sônica para ML
    
    Features extraídas:
    1. Correlação máxima
    2. Largura do pico principal
    3. Número de picos secundários
    4. Drift temporal do delay
    5. SNR médio
    """
    
    features = {
        'max_correlation': correlation_results['max_correlation'],
        'compensated_correlation': correlation_results['compensated_correlation'],
        'snr_average': (correlation_results['snr_expeditor'] + correlation_results['snr_receiver']) / 2,
        'delay_samples': correlation_results['delay_samples'],
        'signal_quality_score': self.signal_quality_to_score(correlation_results['signal_quality']),
        'velocity_deviation': abs(correlation_results['effective_velocity'] - self.config.sonic_velocity) / self.config.sonic_velocity
    }
    
    return features

def signal_quality_to_score(self, quality_str):
    """Converte qualidade qualitativa em score numérico"""
    quality_map = {'excellent': 1.0, 'good': 0.75, 'fair': 0.5, 'poor': 0.25}
    return quality_map.get(quality_str, 0.0)
```

### Feedback para Análise Multivariável

```python
def provide_sonic_context_to_multivariable(self, correlation_results, mv_analysis):
    """
    Fornece contexto sônico para análise multivariável
    
    Enriquece a análise com:
    1. Qualidade da correlação como peso de confiança
    2. Delay de propagação para validação temporal
    3. SNR como indicador de qualidade dos dados
    """
    
    # Peso de confiança baseado na correlação
    confidence_weight = min(1.0, correlation_results['max_correlation'] / 0.8)
    
    # Contexto temporal
    temporal_context = {
        'propagation_delay_samples': correlation_results['delay_samples'],
        'expected_delay_samples': correlation_results['expected_delay_samples'],
        'temporal_consistency': abs(correlation_results['delay_samples'] - correlation_results['expected_delay_samples']) < 5
    }
    
    # Qualidade dos dados
    data_quality = {
        'overall_snr': (correlation_results['snr_expeditor'] + correlation_results['snr_receiver']) / 2,
        'signal_quality': correlation_results['signal_quality'],
        'confidence_weight': confidence_weight
    }
    
    return {
        'confidence_weight': confidence_weight,
        'temporal_context': temporal_context,
        'data_quality': data_quality,
        'sonic_anomaly_detected': confidence_weight < 0.5
    }
```

---

## 📚 Referências Técnicas

### Fundamentos Teóricos

1. **Wylie, E.B. & Streeter, V.L.** - *Fluid Transients in Systems* - Análise fundamental de transientes hidráulicos
2. **Bergant, A.** - *Water Hammer with Column Separation* - Modelagem avançada de golpe de aríete
3. **Covas, D.** - *Inverse Transient Analysis for Leak Detection* - Aplicação de transientes para detecção

### Algoritmos e Processamento

1. **Oppenheim, A.V.** - *Discrete-Time Signal Processing* - Fundamentos de correlação e FFT
2. **Welch, P.** - *The Use of FFT for Estimation of Power Spectra* - Método de Welch para análise espectral
3. **Kay, S.M.** - *Modern Spectral Estimation* - Técnicas avançadas de estimação espectral

### Aplicações Industriais

1. **API 1130** - *Computational Pipeline Monitoring* - Padrão para monitoramento de dutos
2. **AWWA M51** - *Air Valves: Air Release, Air/Vacuum & Combination* - Considerações para válvulas de ar
3. **NACE SP0169** - *Control of External Corrosion on Underground Metallic Piping Systems* - Proteção de dutos

---

**Sistema de Análise Hidráulica Industrial v2.0**  
*Manual de Correlação Sônica - Agosto 2025*
