# Manual do Sistema de Análise Hidráulica Industrial

## 📋 Índice Geral

1. [Visão Geral do Sistema](#visão-geral-do-sistema)
2. [Arquitetura Tecnológica](#arquitetura-tecnológica)
3. [Constantes e Configurações Industriais](#constantes-e-configurações-industriais)
4. [Sistema de Logging e Tratamento de Erros](#sistema-de-logging-e-tratamento-de-erros)
5. [Estruturas de Dados](#estruturas-de-dados)
6. [Database Industrial](#database-industrial)
7. [Fluxo de Execução Principal](#fluxo-de-execução-principal)

---

## 🎯 Visão Geral do Sistema

### Descrição Funcional

O **Sistema Industrial de Análise Hidráulica** é uma solução completa para monitoramento e análise de sistemas de dutos industriais, utilizando técnicas avançadas de:

- **Correlação Sônica Multivariável**: Análise de propagação de ondas acústicas em dutos
- **Machine Learning Adaptativo**: Detecção inteligente de anomalias e vazamentos
- **Processamento Assíncrono**: Interface responsiva com análise em tempo real
- **Validação Física**: Verificação de consistência baseada em leis físicas
- **Interface Industrial**: PyQt6 com visualização profissional de dados

### Funcionalidades Principais

#### ✅ Análises Implementadas

1. **Correlação Sônica**: Medição de tempo de trânsito de ondas de pressão
2. **Análise Multivariável**: Correlação entre pressão, vazão, densidade e temperatura
3. **Machine Learning**: Classificação e detecção de anomalias com múltiplos algoritmos
4. **Análise Espectral**: FFT, filtros Butterworth e transformada de Hilbert
5. **PCA e CCA**: Análise de componentes principais e correlação canônica
6. **Filtros Temporais**: Interpolação física e detecção de outliers
7. **Status Operacional**: Detecção automática de estados do sistema
8. **Análise de Ruídos**: Caracterização de interferências no sinal

#### 📊 Capacidades de Processamento

- **Dados por Exceção**: Processamento de timestamps irregulares
- **Múltiplos Sistemas**: Gestão de diferentes configurações de dutos
- **Processamento Paralelo**: Utilização de múltiplos núcleos de CPU
- **Cache Inteligente**: Otimização de memória para grandes volumes
- **Validação Cruzada**: Verificação de consistência física dos dados

---

## 🏗️ Arquitetura Tecnológica

### Stack Tecnológico

```
┌─────────────────────────────────────────────────────────┐
│                    Interface Gráfica                    │
│                     PyQt6 + PyQtGraph                  │
├─────────────────────────────────────────────────────────┤
│                Processamento Assíncrono                 │
│              QThread + ThreadPoolExecutor              │
├─────────────────────────────────────────────────────────┤
│                  Análise de Dados                       │
│    NumPy + SciPy + Pandas + Scikit-learn + Joblib     │
├─────────────────────────────────────────────────────────┤
│                  Armazenamento                          │
│                SQLite + JSON + Pickle                  │
├─────────────────────────────────────────────────────────┤
│                 Sistema Operacional                     │
│                    Windows/Linux                        │
└─────────────────────────────────────────────────────────┘
```

### Módulos do Sistema

#### 📦 Estrutura Modular

```python
hydraulic_system_complete.py
├── config/constants.py          # Constantes industriais
├── config/logging_config.py     # Sistema de logging
├── utils/error_handler.py       # Tratamento de erros
├── models/data_structures.py    # Estruturas de dados
├── models/database.py           # Base de dados
├── core/processor.py            # Processador principal
├── core/ml_system.py            # Sistema de ML
├── filters/temporal_filter.py   # Filtros temporais
├── gui/main_window.py           # Interface principal
├── gui/dialogs.py               # Diálogos auxiliares
└── async_components/            # Componentes assíncronos
```

---

## ⚙️ Constantes e Configurações Industriais

### Classe `IndustrialConstants`

A classe centraliza todas as configurações críticas do sistema industrial:

```python
@dataclass
class IndustrialConstants:
    # Variáveis físicas com unidades industriais
    VARIABLES = {
        'pressure': {
            'unit': 'kgf/cm²',
            'min': 0, 'max': 100,
            'decimals': 3,
            'typical_range': (1, 50)
        },
        'flow': {
            'unit': 'm³/h',
            'min': 0, 'max': 10000,
            'decimals': 2,
            'typical_range': (10, 5000)
        },
        'density': {
            'unit': 'g/cm³',
            'min': 0.1, 'max': 2.0,
            'decimals': 4,
            'typical_range': (0.6, 1.2)
        },
        'temperature': {
            'unit': '°C',
            'min': -50, 'max': 300,
            'decimals': 1,
            'typical_range': (10, 80)
        }
    }
```

#### 🔧 Parâmetros de Configuração

##### Processamento Sônico

```python
SONIC_VELOCITY_DEFAULT: float = 1500.0  # m/s
DISTANCE_SENSORS_DEFAULT: float = 100.0  # metros
SAMPLING_RATE_DEFAULT: float = 100.0     # Hz
```

##### Buffers e Performance

```python
MAX_BUFFER_SIZE: int = 10000      # Máximo de pontos em buffer
MAX_CACHE_SIZE: int = 2000        # Cache de resultados
CORRELATION_WINDOW: int = 500     # Janela para correlação
```

##### Machine Learning

```python
ML_TRAINING_WINDOW: int = 1000        # Amostras para treinamento
ML_PREDICTION_THRESHOLD: float = 0.7  # Limite de confiança
ML_RETRAIN_INTERVAL: int = 24         # Horas entre retreinos
```

##### Sensibilidades de Detecção

```python
LEAK_SENSITIVITY_LOW: float = 0.3     # Baixa sensibilidade
LEAK_SENSITIVITY_MEDIUM: float = 0.5  # Média sensibilidade  
LEAK_SENSITIVITY_HIGH: float = 0.8    # Alta sensibilidade
```

#### 🏗️ Características Físicas dos Dutos

##### Materiais Suportados

```python
PIPE_MATERIALS = {
    'steel': {
        'sonic_velocity_factor': 1.0,
        'attenuation': 0.1,
        'roughness': 0.045
    },
    'pvc': {
        'sonic_velocity_factor': 0.8,
        'attenuation': 0.2,
        'roughness': 0.0015
    },
    'concrete': {
        'sonic_velocity_factor': 1.2,
        'attenuation': 0.05,
        'roughness': 0.3
    },
    'fiberglass': {
        'sonic_velocity_factor': 0.9,
        'attenuation': 0.15,
        'roughness': 0.01
    }
}
```

##### Perfis Geométricos

```python
PIPE_PROFILES = {
    'circular': {
        'area_factor': 1.0,
        'perimeter_factor': 1.0
    },
    'rectangular': {
        'area_factor': 0.9,
        'perimeter_factor': 1.2
    },
    'oval': {
        'area_factor': 0.95,
        'perimeter_factor': 1.1
    }
}
```

#### 🔍 Validação Física

##### Correlações Esperadas

```python
PHYSICAL_VALIDATION = {
    'density_temp_correlation': 0.8,     # Densidade vs Temperatura
    'pressure_flow_correlation': 0.6,    # Pressão vs Vazão
    'max_pressure_drop_rate': 5.0,       # kgf/cm²/min
    'max_flow_change_rate': 100.0        # m³/h/min
}
```

---

## 📊 Sistema de Logging e Tratamento de Erros

### Classe `IndustrialLogger`

Sistema robusto de logging multi-nível para ambiente industrial:

#### 🎯 Características

- **Logs Rotativos**: Máximo 10MB por arquivo, 5 backups
- **Múltiplos Níveis**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Formatação Detalhada**: Timestamp, módulo, função, mensagem
- **Console e Arquivo**: Saída dupla para debugging e histórico

#### 📝 Formato de Log

```
2025-08-30 10:30:45 | hydraulic_system.processor | INFO | process_data | Processando 1500 amostras
2025-08-30 10:30:46 | hydraulic_system.ml_system | WARNING | predict | Confiança baixa: 0.65
2025-08-30 10:30:47 | hydraulic_system.error_handler | ERROR | validate_data | Pressão fora de faixa: 150 kgf/cm²
```

#### 💾 Estrutura de Arquivos

```
logs/
├── hydraulic_system.log      # Log atual
├── hydraulic_system.log.1    # Backup mais recente
├── hydraulic_system.log.2    # Backup anterior
├── hydraulic_system.log.3    # ...
├── hydraulic_system.log.4    # ...
└── hydraulic_system.log.5    # Backup mais antigo
```

### Classe `IndustrialErrorHandler`

Sistema de tratamento robusto de erros com retry automático:

#### 🛡️ Tipos de Erro Especializados

```python
class HydraulicError(Exception):
    """Exceção base para erros do sistema hidráulico"""

class DataValidationError(HydraulicError):
    """Erro de validação de dados"""

class CalibrationError(HydraulicError):
    """Erro de calibração"""

class MLModelError(HydraulicError):
    """Erro relacionado ao modelo de ML"""
```

#### 🔄 Sistema de Retry

```python
@error_handler.handle_with_retry(max_retries=3)
def critical_function():
    # Função crítica com retry automático
    # Backoff exponencial: 0.1s, 0.2s, 0.3s
    pass
```

#### ✅ Validações Implementadas

##### Validação de Tipos

```python
def validate_data_type(self, data: Any, expected_type: type, name: str = "data"):
    """Validação robusta de tipos com mensagens claras"""
```

##### Validação de Faixas Físicas

```python
def validate_physical_range(self, value: float, variable: str, system_id: str = "unknown"):
    """
    Valida se valores estão dentro de faixas físicas realistas:
    - Faixa absoluta: valores fisicamente impossíveis
    - Faixa típica: valores incomuns mas possíveis
    """
```

##### Validação Cruzada

```python
def validate_cross_variables(self, readings: Dict[str, float], system_id: str = "unknown"):
    """
    Validação cruzada entre variáveis:
    - Densidade vs Temperatura: correlação esperada
    - Pressão vs Vazão: consistência física
    - Taxas de mudança: limites de variação temporal
    """
```

---

## 📁 Estruturas de Dados

### Classe `MultiVariableSnapshot`

Estrutura fundamental que encapsula uma medição completa do sistema:

```python
@dataclass
class MultiVariableSnapshot:
    """Snapshot multivariável com timestamp"""
    timestamp: datetime
    system_id: str
    expeditor_pressure: float    # kgf/cm²
    receiver_pressure: float     # kgf/cm²
    flow_rate: float            # m³/h
    temperature: float          # °C
    density: float              # g/cm³
    viscosity: Optional[float] = None     # cP
    operational_status: str = 'unknown'   # normal/abnormal/maintenance
    quality_flags: Dict[str, bool] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validação automática pós-inicialização"""
        # Calcula pressão diferencial
        self.pressure_diff = self.expeditor_pressure - self.receiver_pressure
        
        # Flags de qualidade padrão
        if not self.quality_flags:
            self.quality_flags = {
                'pressure_valid': True,
                'flow_valid': True,
                'temperature_valid': True,
                'density_valid': True
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversão para dicionário para análise"""
        return {
            'timestamp': self.timestamp,
            'system_id': self.system_id,
            'expeditor_pressure': self.expeditor_pressure,
            'receiver_pressure': self.receiver_pressure,
            'pressure_diff': self.pressure_diff,
            'flow_rate': self.flow_rate,
            'temperature': self.temperature,
            'density': self.density,
            'viscosity': self.viscosity,
            'operational_status': self.operational_status,
            'quality_flags': self.quality_flags
        }
    
    def validate_physics(self) -> List[str]:
        """Validação física dos dados"""
        warnings = []
        
        # Pressão diferencial
        if abs(self.pressure_diff) > 50:
            warnings.append(f"Pressão diferencial alta: {self.pressure_diff:.2f} kgf/cm²")
        
        # Densidade vs Temperatura (água)
        expected_density = 1.0 - (self.temperature - 20) * 0.0002
        if abs(self.density - expected_density) > 0.1:
            warnings.append(f"Densidade inconsistente com temperatura")
        
        return warnings
```

### Classe `SystemConfiguration`

Configuração completa de um sistema hidráulico:

```python
@dataclass
class SystemConfiguration:
    """Configuração completa de um sistema hidráulico"""
    system_id: str
    name: str
    location: str
    
    # Unidades do sistema
    expeditor_unit: str = "Expedidora"
    expeditor_alias: str = "BAR" 
    receiver_unit: str = "Recebedora"
    receiver_alias: str = "PLN"
    
    # Características físicas
    pipe_characteristics: 'PipeCharacteristics' = None
    sensor_distance: float = 0.1  # km
    
    # Propriedades do fluido
    fluid_type: str = "água"
    nominal_density: float = 1.0     # g/cm³
    nominal_temperature: float = 20.0 # °C
    nominal_pressure: float = 10.0   # kgf/cm²
    nominal_flow: float = 1000.0     # m³/h
    sonic_velocity: float = 1500.0   # m/s
    
    def __post_init__(self):
        """Inicialização automática de características do duto"""
        if self.pipe_characteristics is None:
            self.pipe_characteristics = PipeCharacteristics()
    
    def get_sensor_info(self) -> Dict[str, str]:
        """Informações dos sensores baseadas nos aliases"""
        return {
            'pressure_expeditor': f'{self.expeditor_alias}_PT',
            'pressure_receiver': f'{self.receiver_alias}_PT', 
            'flow_expeditor': f'{self.expeditor_alias}_FT',
            'flow_receiver': f'{self.receiver_alias}_FT',
            'temperature_expeditor': f'{self.expeditor_alias}_TT',
            'temperature_receiver': f'{self.receiver_alias}_TT',
            'density_expeditor': f'{self.expeditor_alias}_DT',
            'density_receiver': f'{self.receiver_alias}_DT'
        }
```

### Classe `PipeCharacteristics`

Características físicas detalhadas do duto:

```python
@dataclass 
class PipeCharacteristics:
    """Características físicas do duto"""
    diameter: float = 0.5           # metros
    length: float = 1.0             # km
    wall_thickness: float = 10.0    # mm
    material: str = 'steel'         # steel/pvc/concrete/fiberglass
    profile: str = 'circular'       # circular/rectangular/oval
    roughness: float = 0.045        # mm
    
    def calculate_area(self) -> float:
        """Área da seção transversal"""
        if self.profile == 'circular':
            return np.pi * (self.diameter/2)**2
        elif self.profile == 'rectangular':
            # Aproximação para retangular (assume aspectRatio 2:1)
            width = self.diameter * 1.2
            height = self.diameter * 0.8  
            return width * height
        elif self.profile == 'oval':
            # Aproximação elíptica
            a = self.diameter / 2 * 1.1
            b = self.diameter / 2 * 0.9
            return np.pi * a * b
        return np.pi * (self.diameter/2)**2
    
    def calculate_acoustic_properties(self) -> Dict[str, float]:
        """Propriedades acústicas do duto"""
        material_props = CONSTANTS.PIPE_MATERIALS.get(self.material, CONSTANTS.PIPE_MATERIALS['steel'])
        profile_props = CONSTANTS.PIPE_PROFILES.get(self.profile, CONSTANTS.PIPE_PROFILES['circular'])
        
        return {
            'velocity_factor': material_props['sonic_velocity_factor'] * profile_props['area_factor'],
            'attenuation': material_props['attenuation'],
            'roughness_factor': self.roughness / 0.045,  # Normalizado ao aço
            'area_factor': profile_props['area_factor'],
            'perimeter_factor': profile_props['perimeter_factor']
        }
```

---

## 💾 Database Industrial

### Classe `IndustrialDatabase`

Sistema de persistência robusto com SQLite:

#### 🗄️ Estrutura das Tabelas

##### Tabela Systems

```sql
CREATE TABLE IF NOT EXISTS systems (
    system_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    location TEXT,
    config_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

##### Tabela Readings

```sql  
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    system_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    expeditor_pressure REAL,
    receiver_pressure REAL,
    flow_rate REAL,
    temperature REAL,
    density REAL,
    viscosity REAL,
    operational_status TEXT DEFAULT 'unknown',
    quality_flags TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (system_id) REFERENCES systems (system_id)
)
```

##### Tabela Analysis Results

```sql
CREATE TABLE IF NOT EXISTS analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    system_id TEXT NOT NULL,
    analysis_type TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    results_json TEXT NOT NULL,
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (system_id) REFERENCES systems (system_id)
)
```

#### 🔧 Operações Principais

##### Salvar Sistema

```python
def save_system(self, config: SystemConfiguration) -> bool:
    """
    Salva configuração do sistema com validação:
    - Converte para JSON
    - Atualiza timestamp
    - Transação atômica
    """
```

##### Carregar Sistema  

```python
def load_system(self, system_id: str) -> Optional[SystemConfiguration]:
    """
    Carrega configuração do sistema:
    - Busca por ID
    - Deserializa JSON  
    - Reconstrói objetos
    """
```

##### Salvar Leituras em Lote

```python
def save_readings_batch(self, readings: List[MultiVariableSnapshot]) -> bool:
    """
    Salva leituras em lote com otimização:
    - Transação única
    - Prepared statements
    - Validação prévia
    """
```

##### Recuperar Leituras por Período

```python
def get_readings_by_period(self, system_id: str, start_time: datetime, end_time: datetime) -> List[MultiVariableSnapshot]:
    """
    Busca otimizada por período:
    - Índice por timestamp
    - Paginação automática
    - Conversão para objetos
    """
```

---

## ⚡ Fluxo de Execução Principal

### Inicialização do Sistema

```python
def main():
    """Função principal com inicialização completa"""
    
    # 1. Configuração de logging
    industrial_logger = IndustrialLogger()
    logger = industrial_logger.get_logger()
    
    # 2. Inicialização da base de dados
    database = IndustrialDatabase()
    database.initialize()
    
    # 3. Configuração da aplicação PyQt6
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Estilo industrial
    
    # 4. Janela principal
    main_window = HydraulicAnalysisMainWindow()
    main_window.show()
    
    # 5. Loop principal
    sys.exit(app.exec())
```

### Ciclo de Análise

#### 1️⃣ Carregamento de Dados

```
Arquivos Excel/CSV → Validação → MultiVariableSnapshot → Database
```

#### 2️⃣ Processamento

```
Snapshots → IndustrialHydraulicProcessor → Múltiplas Análises → Resultados
```

#### 3️⃣ Visualização

```
Resultados → PyQtGraph → Interface Gráfica → Usuário
```

#### 4️⃣ Persistência  

```
Resultados → Database → Cache → Histórico
```

### Processamento Assíncrono

O sistema utiliza múltiplas threads para manter a interface responsiva:

#### 🧵 Threads Principais

- **Thread Principal**: Interface gráfica (PyQt6)
- **Thread de Carregamento**: `AsyncFileLoader` para arquivos
- **Thread de Processamento**: `ParallelDataProcessor` para análise  
- **ThreadPool**: Processamento paralelo de lotes

#### 📊 Comunicação por Sinais

```python
# Sinais PyQt6 para comunicação thread-safe
data_loaded = pyqtSignal(object)      # Dados carregados
progress_updated = pyqtSignal(int)    # Progresso atualizado
analysis_complete = pyqtSignal(dict)  # Análise completa
error_occurred = pyqtSignal(str)      # Erro ocorrido
```

---

## 📈 Métricas de Performance

### Capacidades Testadas

- ✅ **10.000+ amostras**: Processamento em tempo real
- ✅ **Multiple sistemas**: Gerenciamento simultâneo
- ✅ **Arquivos grandes**: Excel com 50MB+ suportados
- ✅ **Interface responsiva**: Sem travamentos durante processamento
- ✅ **Análise completa**: < 5 segundos para 5000 amostras

### Otimizações Implementadas

- **Vectorização NumPy**: Operações matemáticas otimizadas
- **Cache inteligente**: Resultados intermediários armazenados
- **Processamento paralelo**: Utilização de múltiplos cores
- **Lazy loading**: Carregamento sob demanda
- **Memory mapping**: Para arquivos muito grandes

---

## 🎯 Próximos Manuais

Este manual geral será complementado pelos seguintes manuais especializados:

1. **MANUAL_02_INTERFACE_GRAFICA.md** - Interface completa e componentes
2. **MANUAL_03_CORRELACAO_SONICA.md** - Análise sônica e matemática
3. **MANUAL_04_MACHINE_LEARNING.md** - Sistema de ML e algoritmos
4. **MANUAL_05_ANALISE_MULTIVARIAVEL.md** - Análise de múltiplas variáveis
5. **MANUAL_06_FILTROS_TEMPORAIS.md** - Processamento temporal
6. **MANUAL_07_ANALISE_ESPECTRAL.md** - FFT e análise de frequências
7. **MANUAL_08_VALIDACAO_FISICA.md** - Validação e consistência
8. **MANUAL_09_CONFIGURACAO_SISTEMA.md** - Configuração e calibração

---

## 📞 Suporte

Para dúvidas técnicas ou suporte:

- Consulte os manuais específicos de cada módulo
- Verifique os logs em `logs/hydraulic_system.log`
- Execute os testes em `test_*.py` para diagnósticos

---

**Sistema de Análise Hidráulica Industrial v2.0**  
*Desenvolvido para ambientes industriais críticos*  
*Documentação completa - Agosto 2025*
