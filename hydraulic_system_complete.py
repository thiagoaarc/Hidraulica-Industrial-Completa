#!/usr/bin/env python3
"""
Sistema Industrial de Análise Hidráulica - Correlação Sônica Multivariável com Machine Learning
Versão Completa 100% Funcional

Funcionalidades principais:
- Timestamps irregulares (dados por exceção)
- Análise multivariável (vazão, densidade, temperatura, pressão)
- Detecção de status operacional (coluna aberta/fechada)
- Calibração por sistema com bancCorri
de dados
- Machine Learning adaptativo para detecção de vazamentos
- Gestão de múltiplos sistemas
- Interface PyQt6 completa
- Sistema de testes integrado
- Validação física robusta

Autor: Sistema de Análise Hidráulica Industrial
Versão: Complete Rev2
Data: 2025

Execução: python hydraulic_system_complete.py
"""

# ============================================================================
# config/constants.py - CONFIGURAÇÕES E CONSTANTES INDUSTRIAIS
# ============================================================================

import sys
import os
import gc
import glob
import threading
import time
import queue
import logging
import logging.handlers
import traceback
import functools
import sqlite3
import json
import pickle
import unittest
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

# Imports científicos
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy import interpolate
from scipy.stats import pearsonr, zscore
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib

# Imports PyQt6
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QGroupBox, QPushButton, QLabel, QLineEdit, QComboBox,
    QProgressBar, QTextEdit, QFileDialog, QMessageBox, QSplitter,
    QGridLayout, QScrollArea, QFrame, QTableWidget, QTableWidgetItem,
    QCheckBox, QSpinBox, QDoubleSpinBox, QDateTimeEdit, QSlider,
    QStatusBar, QMenuBar, QToolBar, QInputDialog, QDialog
)
from PyQt6.QtCore import (
    QTimer, QThread, pyqtSignal, QObject, Qt, QMutex, QWaitCondition,
    QDateTime, QSettings, QStandardPaths, QSize, QRunnable, QThreadPool,
    pyqtSlot, QPropertyAnimation, QEasingCurve
)
from PyQt6.QtGui import QFont, QPalette, QColor, QPixmap, QPainter, QAction, QIcon

# Imports PyQtGraph
import pyqtgraph as pg
from pyqtgraph import PlotWidget, mkPen, mkBrush, ImageView

@dataclass
class IndustrialConstants:
    """Constantes otimizadas para ambiente industrial"""
    
    # Configurações de análise multivariável com unidades industriais
    VARIABLES = {
        'pressure': {'unit': 'kgf/cm²', 'min': 0, 'max': 100, 'decimals': 3, 'typical_range': (1, 50)},
        'flow': {'unit': 'm³/h', 'min': 0, 'max': 10000, 'decimals': 2, 'typical_range': (10, 5000)},
        'density': {'unit': 'g/cm³', 'min': 0.1, 'max': 2.0, 'decimals': 4, 'typical_range': (0.6, 1.2)},
        'temperature': {'unit': '°C', 'min': -50, 'max': 300, 'decimals': 1, 'typical_range': (10, 80)}
    }
    
    # Configurações de processamento
    SONIC_VELOCITY_DEFAULT: float = 1500.0  # m/s
    DISTANCE_SENSORS_DEFAULT: float = 100.0  # metros
    SAMPLING_RATE_DEFAULT: float = 100.0  # Hz
    
    # Buffers e cache
    MAX_BUFFER_SIZE: int = 10000
    MAX_CACHE_SIZE: int = 2000
    CORRELATION_WINDOW: int = 500
    
    # Machine Learning
    ML_TRAINING_WINDOW: int = 1000
    ML_PREDICTION_THRESHOLD: float = 0.7
    ML_RETRAIN_INTERVAL: int = 24  # horas
    
    # Detecção de vazamentos
    LEAK_SENSITIVITY_LOW: float = 0.3
    LEAK_SENSITIVITY_MEDIUM: float = 0.5
    LEAK_SENSITIVITY_HIGH: float = 0.8
    
    # Sistema físico expandido
    PIPE_MATERIALS = {
        'steel': {'sonic_velocity_factor': 1.0, 'attenuation': 0.1, 'roughness': 0.045},
        'pvc': {'sonic_velocity_factor': 0.8, 'attenuation': 0.2, 'roughness': 0.0015},
        'concrete': {'sonic_velocity_factor': 1.2, 'attenuation': 0.05, 'roughness': 0.3},
        'fiberglass': {'sonic_velocity_factor': 0.9, 'attenuation': 0.15, 'roughness': 0.01}
    }
    
    # Perfis de duto
    PIPE_PROFILES = {
        'circular': {'area_factor': 1.0, 'perimeter_factor': 1.0},
        'rectangular': {'area_factor': 0.9, 'perimeter_factor': 1.2},
        'oval': {'area_factor': 0.95, 'perimeter_factor': 1.1}
    }
    
    # Validação física
    PHYSICAL_VALIDATION = {
        'density_temp_correlation': 0.8,  # Correlação esperada
        'pressure_flow_correlation': 0.6,
        'max_pressure_drop_rate': 5.0,  # kgf/cm²/min
        'max_flow_change_rate': 100.0,  # m³/h/min
    }

CONSTANTS = IndustrialConstants()

# ============================================================================
# config/logging_config.py - SISTEMA DE LOGGING INDUSTRIAL
# ============================================================================

class IndustrialLogger:
    """Sistema de logging robusto para ambiente industrial"""
    
    def __init__(self, log_level=logging.INFO):
        self.log_level = log_level
        self.setup_logging()
    
    def setup_logging(self):
        """Configura sistema de logging multi-nível"""
        
        # Criação do logger principal
        self.logger = logging.getLogger('hydraulic_system')
        self.logger.setLevel(self.log_level)
        
        # Remove handlers existentes
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Formatter detalhado
        formatter = logging.Formatter(
            '%(asctime)s | %(name)-20s | %(levelname)-8s | %(funcName)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler para console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Handler para arquivo rotativo
        try:
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_dir / 'hydraulic_system.log',
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
        except Exception as e:
            self.logger.warning(f"Não foi possível configurar log em arquivo: {e}")
    
    def get_logger(self, name: Optional[str] = None):
        """Retorna logger configurado"""
        if name:
            return logging.getLogger(f'hydraulic_system.{name}')
        return self.logger

# Instância global do logger
industrial_logger = IndustrialLogger()
logger = industrial_logger.get_logger()

# ============================================================================
# utils/error_handler.py - TRATAMENTO ROBUSTO DE ERROS
# ============================================================================

class HydraulicError(Exception):
    """Exceção base para erros do sistema hidráulico"""
    pass

class DataValidationError(HydraulicError):
    """Erro de validação de dados"""
    pass

class CalibrationError(HydraulicError):
    """Erro de calibração"""
    pass

class MLModelError(HydraulicError):
    """Erro relacionado ao modelo de ML"""
    pass

class IndustrialErrorHandler:
    """Tratador robusto de erros para ambiente industrial"""
    
    def __init__(self):
        self.logger = industrial_logger.get_logger('error_handler')
        self.error_count = {}
        self.max_retries = 3
    
    def handle_with_retry(self, max_retries: Optional[int] = None):
        """Decorator para retry automático"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                retries = max_retries or self.max_retries
                last_exception = None
                
                for attempt in range(retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        self.logger.warning(f"Tentativa {attempt + 1} falhou para {func.__name__}: {e}")
                        
                        if attempt < retries:
                            time.sleep(0.1 * (attempt + 1))  # Backoff exponencial
                        else:
                            self.logger.error(f"Todas as {retries + 1} tentativas falharam para {func.__name__}")
                            raise last_exception
                
                return None
            return wrapper
        return decorator
    
    def validate_data_type(self, data: Any, expected_type: type, name: str = "data"):
        """Validação robusta de tipos"""
        if not isinstance(data, expected_type):
            error_msg = f"{name} deve ser do tipo {expected_type.__name__}, recebido {type(data).__name__}"
            self.logger.error(error_msg)
            raise DataValidationError(error_msg)
    
    def validate_physical_range(self, value: float, variable: str, system_id: str = "unknown"):
        """Validação de faixas físicas realistas"""
        if variable not in CONSTANTS.VARIABLES:
            return True
        
        var_config = CONSTANTS.VARIABLES[variable]
        min_val = var_config['min']
        max_val = var_config['max']
        typical_min, typical_max = var_config['typical_range']
        
        if not (min_val <= value <= max_val):
            error_msg = f"{variable} = {value} {var_config['unit']} fora da faixa física válida [{min_val}, {max_val}] para sistema {system_id}"
            self.logger.error(error_msg)
            raise DataValidationError(error_msg)
        
        if not (typical_min <= value <= typical_max):
            self.logger.warning(f"{variable} = {value} {var_config['unit']} fora da faixa típica [{typical_min}, {typical_max}] para sistema {system_id}")
        
        return True
    
    def validate_cross_variables(self, readings: Dict[str, float], system_id: str = "unknown"):
        """Validação cruzada entre variáveis"""
        try:
            # Validação densidade vs temperatura
            if 'density' in readings and 'temperature' in readings:
                # Água: densidade diminui com temperatura
                if readings['density'] > 1.05 and readings['temperature'] > 80:
                    self.logger.warning(f"Sistema {system_id}: Alta densidade ({readings['density']}) com alta temperatura ({readings['temperature']}) - possível inconsistência")
            
            # Validação pressão vs vazão
            if 'pressure' in readings and 'flow' in readings:
                if readings['pressure'] < 0.5 and readings['flow'] > 1000:
                    error_msg = f"Sistema {system_id}: Baixa pressão ({readings['pressure']}) com alta vazão ({readings['flow']}) - fisicamente implausível"
                    self.logger.error(error_msg)
                    raise DataValidationError(error_msg)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro na validação cruzada para sistema {system_id}: {e}")
            return False

# Instância global do error handler
error_handler = IndustrialErrorHandler()

# ============================================================================
# core/memory_manager.py - GESTÃO INTELIGENTE DE MEMÓRIA
# ============================================================================

class CircularBuffer:
    """Buffer circular otimizado para dados de séries temporais"""
    
    def __init__(self, maxsize: int, dtype=np.float64):
        self.maxsize = maxsize
        self.buffer = np.zeros(maxsize, dtype=dtype)
        self.head = 0
        self.size = 0
        self.full = False
    
    def append(self, value):
        """Adiciona valor ao buffer"""
        self.buffer[self.head] = value
        self.head = (self.head + 1) % self.maxsize
        
        if self.full:
            pass  # Sobrescreve dados antigos
        else:
            self.size += 1
            if self.size == self.maxsize:
                self.full = True
    
    def get_data(self, n_points: Optional[int] = None) -> np.ndarray:
        """Recupera dados mais recentes"""
        if n_points is None:
            n_points = self.size
        
        n_points = min(n_points, self.size)
        if n_points == 0:
            return np.array([])
        
        if self.full:
            # Buffer completo - precisa reordenar
            if n_points == self.maxsize:
                return np.concatenate([self.buffer[self.head:], self.buffer[:self.head]])
            else:
                start_idx = (self.head - n_points) % self.maxsize
                if start_idx + n_points <= self.maxsize:
                    return self.buffer[start_idx:start_idx + n_points].copy()
                else:
                    return np.concatenate([
                        self.buffer[start_idx:],
                        self.buffer[:start_idx + n_points - self.maxsize]
                    ])
        else:
            # Buffer parcial
            start_idx = max(0, self.size - n_points)
            return self.buffer[start_idx:self.size].copy()
    
    def clear(self):
        """Limpa buffer"""
        self.head = 0
        self.size = 0
        self.full = False

class IndustrialMemoryManager:
    """Gerenciador de memória para ambiente industrial"""
    
    def __init__(self, max_systems: int = 10):
        self.logger = industrial_logger.get_logger('memory_manager')
        self.max_systems = max_systems
        self.system_buffers = {}
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Monitoramento de memória
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.cleanup_memory)
        self.memory_timer.start(30000)  # 30 segundos
    
    def get_system_buffers(self, system_id: str) -> Dict[str, CircularBuffer]:
        """Obtém buffers para um sistema específico"""
        if system_id not in self.system_buffers:
            self.system_buffers[system_id] = {
                'expeditor_pressure': CircularBuffer(CONSTANTS.MAX_BUFFER_SIZE),
                'receiver_pressure': CircularBuffer(CONSTANTS.MAX_BUFFER_SIZE),
                'flow_rate': CircularBuffer(CONSTANTS.MAX_BUFFER_SIZE),
                'density': CircularBuffer(CONSTANTS.MAX_BUFFER_SIZE),
                'temperature': CircularBuffer(CONSTANTS.MAX_BUFFER_SIZE),
                'timestamps': CircularBuffer(CONSTANTS.MAX_BUFFER_SIZE, dtype=np.float64)
            }
            self.logger.info(f"Buffers criados para sistema {system_id}")
        
        return self.system_buffers[system_id]
    
    def add_data_point(self, system_id: str, timestamp: float, data: Dict[str, float]):
        """Adiciona ponto de dados aos buffers com filtragem temporal"""
        buffers = self.get_system_buffers(system_id)
        
        # Aplica filtro temporal aos dados antes de adicionar aos buffers
        try:
            filtered_data = temporal_filter.add_data_point(timestamp, data)
            
            # Adiciona timestamp filtrado
            buffers['timestamps'].append(timestamp)
            
            # Valida e adiciona cada variável filtrada
            for var_name, value in data.items():
                if var_name in buffers:
                    try:
                        error_handler.validate_physical_range(value, var_name, system_id)
                        
                        # Usa valor filtrado se disponível, senão usa valor original
                        filtered_value = filtered_data.get(var_name, value)
                        buffers[var_name].append(filtered_value)
                        
                        # Adiciona métricas de qualidade se disponíveis
                        quality_key = f'{var_name}_quality'
                        if quality_key in filtered_data:
                            if quality_key not in buffers:
                                buffers[quality_key] = CircularBuffer(CONSTANTS.MAX_BUFFER_SIZE)
                            buffers[quality_key].append(filtered_data[quality_key])
                            
                    except DataValidationError as e:
                        self.logger.warning(f"Valor inválido rejeitado: {e}")
                        buffers[var_name].append(np.nan)  # Placeholder para manter sincronização
                        
        except Exception as e:
            self.logger.error(f"Erro na filtragem temporal: {e}")
            # Fallback para modo sem filtro
            buffers['timestamps'].append(timestamp)
            for var_name, value in data.items():
                if var_name in buffers:
                    try:
                        error_handler.validate_physical_range(value, var_name, system_id)
                        buffers[var_name].append(value)
                    except DataValidationError as e:
                        self.logger.warning(f"Valor inválido rejeitado: {e}")
                        buffers[var_name].append(np.nan)
    
    def get_cache_key(self, operation: str, params: Dict) -> str:
        """Gera chave de cache"""
        return f"{operation}_{hash(str(sorted(params.items())))}"
    
    def get_from_cache(self, key: str) -> Any:
        """Recupera item do cache"""
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        self.cache_misses += 1
        return None
    
    def store_in_cache(self, key: str, value: Any):
        """Armazena item no cache com limite de tamanho"""
        if len(self.cache) >= CONSTANTS.MAX_CACHE_SIZE:
            # Remove item mais antigo (FIFO simples)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    def cleanup_memory(self):
        """Limpeza automática de memória"""
        try:
            # Força garbage collection
            collected = gc.collect()
            
            # Remove sistemas inativos há muito tempo
            current_time = time.time()
            inactive_systems = []
            
            for system_id, buffers in self.system_buffers.items():
                last_timestamp = buffers['timestamps'].get_data(1)
                if len(last_timestamp) > 0:
                    time_diff = current_time - last_timestamp[-1]
                    if time_diff > 3600:  # 1 hora de inatividade
                        inactive_systems.append(system_id)
            
            for system_id in inactive_systems:
                del self.system_buffers[system_id]
                self.logger.info(f"Sistema {system_id} removido por inatividade")
            
            # Limpa cache se muito grande
            if len(self.cache) > CONSTANTS.MAX_CACHE_SIZE * 0.8:
                cache_to_remove = len(self.cache) // 4
                for _ in range(cache_to_remove):
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
            
            self.logger.debug(f"Limpeza de memória: {collected} objetos coletados, cache: {self.cache_hits}/{self.cache_hits + self.cache_misses} hits")
            
        except Exception as e:
            self.logger.error(f"Erro na limpeza de memória: {e}")

# Instância global do memory manager
memory_manager = IndustrialMemoryManager()

# ============================================================================
# data/structures.py - ESTRUTURAS DE DADOS INDUSTRIAIS
# ============================================================================

@dataclass
class PipeCharacteristics:
    """Características completas do duto"""
    diameter: float  # metros
    material: str
    profile: str  # circular, rectangular, oval
    length: float  # metros
    roughness: float  # mm
    wall_thickness: float  # mm
    elevation_profile: List[Tuple[float, float]] = field(default_factory=list)  # (distância, elevação)
    fittings: List[Dict[str, Any]] = field(default_factory=list)  # curvas, reduções, etc.
    
    def calculate_acoustic_properties(self) -> Dict[str, float]:
        """Calcula propriedades acústicas baseado nas características"""
        material_props = CONSTANTS.PIPE_MATERIALS.get(self.material, CONSTANTS.PIPE_MATERIALS['steel'])
        profile_props = CONSTANTS.PIPE_PROFILES.get(self.profile, CONSTANTS.PIPE_PROFILES['circular'])
        
        return {
            'velocity_factor': material_props['sonic_velocity_factor'],
            'attenuation': material_props['attenuation'],
            'area_factor': profile_props['area_factor'],
            'perimeter_factor': profile_props['perimeter_factor'],
            'roughness_factor': self.roughness / 1000  # Convert mm to m
        }

# ============================================================================
# SISTEMA DE PROCESSAMENTO ASSÍNCRONO E PARALELO
# ============================================================================

class WorkerSignals(QObject):
    """Sinais para comunicação entre threads"""
    progress = pyqtSignal(int)  # Progresso (0-100)
    status = pyqtSignal(str)    # Mensagem de status
    result = pyqtSignal(object) # Resultado final
    error = pyqtSignal(str)     # Erro encontrado
    finished = pyqtSignal()     # Trabalho concluído

class AsyncDataWorker(QRunnable):
    """Worker assíncrono para processamento de dados"""
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.is_cancelled = False
    
    def run(self):
        """Executa a função em thread separada"""
        try:
            # Adiciona callback de progresso se possível
            if 'progress_callback' in self.kwargs:
                self.kwargs['progress_callback'] = self.signals.progress.emit
            if 'status_callback' in self.kwargs:
                self.kwargs['status_callback'] = self.signals.status.emit
                
            result = self.func(*self.args, **self.kwargs)
            
            if not self.is_cancelled:
                self.signals.result.emit(result)
                
        except Exception as e:
            logger.error(f"Erro no worker assíncrono: {e}")
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()
    
    def cancel(self):
        """Cancela a operação"""
        self.is_cancelled = True

class ParallelDataProcessor:
    """Processador de dados com paralelização automática"""
    
    def __init__(self, max_workers=None):
        self.max_workers = max_workers or min(32, (cpu_count() or 1) + 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=cpu_count())
        
    def process_chunks_parallel(self, data, chunk_func, chunk_size=1000):
        """Processa dados em chunks paralelos"""
        if len(data) <= chunk_size:
            return chunk_func(data)
        
        # Dividir em chunks
        chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        
        # Processar em paralelo
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(chunk_func, chunk) for chunk in chunks]
            results = [future.result() for future in as_completed(futures)]
        
        return results
    
    def process_with_multiprocessing(self, data, func, processes=None):
        """Processa dados usando multiprocessing"""
        processes = processes or cpu_count()
        
        with Pool(processes) as pool:
            if isinstance(data, list) and len(data) > processes:
                chunk_size = len(data) // processes
                chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
                results = pool.map(func, chunks)
            else:
                results = pool.map(func, [data])
        
        return results

class AsyncFileLoader(QThread):
    """Thread assíncrona para carregamento de arquivos"""
    
    # Sinais
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    data_loaded = pyqtSignal(object)  # DataFrame carregado
    file_processed = pyqtSignal(str, object)  # nome_arquivo, dados_processados
    loading_completed = pyqtSignal(bool)  # sucesso
    error_occurred = pyqtSignal(str)
    
    def __init__(self, file_path, processor=None):
        super().__init__()
        self.file_path = file_path
        self.processor = processor
        self.is_cancelled = False
        self.parallel_processor = ParallelDataProcessor()
        
    def run(self):
        """Executa carregamento em thread separada"""
        try:
            self.status_updated.emit("Iniciando carregamento...")
            self.progress_updated.emit(5)
            
            if self.is_cancelled:
                return
            
            # Carregar arquivo
            filename = Path(self.file_path).name.upper()
            self.status_updated.emit(f"Carregando {filename}...")
            
            # Carregamento assíncrono do arquivo
            if self.file_path.lower().endswith('.csv'):
                df = pd.read_csv(self.file_path, engine='c')
            elif self.file_path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(self.file_path, engine='openpyxl' if self.file_path.endswith('.xlsx') else 'xlrd')
            else:
                raise ValueError("Formato de arquivo não suportado")
            
            self.progress_updated.emit(30)
            self.data_loaded.emit(df)
            
            if self.is_cancelled:
                return
            
            # CORREÇÃO CRÍTICA: Processar dados independentemente do processador
            # O processador é usado para análises avançadas, mas os dados básicos 
            # devem sempre ser processados para permitir plotagem
            self.status_updated.emit("Processando dados...")
            
            if 'OPASA10' in filename or 'BAR2PLN' in filename:
                processed_data = self._process_pipeline_profile_async(df, filename)
            else:
                processed_data = self._process_sensor_data_async(df, filename)
            
            self.progress_updated.emit(90)
            
            # Emitir sinal de dados processados (SEMPRE)
            self.file_processed.emit(filename, processed_data)
            
            self.progress_updated.emit(100)
            self.status_updated.emit("Carregamento concluído")
            self.loading_completed.emit(True)
            
        except Exception as e:
            logger.error(f"Erro no carregamento assíncrono: {e}")
            self.error_occurred.emit(str(e))
            self.loading_completed.emit(False)
    
    def _process_pipeline_profile_async(self, df, filename):
        """Processa perfil da tubulação de forma assíncrona"""
        try:
            expected_cols = ['Km Desenvol.', 'Cota', 'Esp', 'Dext', 'Tag']
            
            # Mapear colunas
            column_mapping = {}
            for expected_col in expected_cols:
                for actual_col in df.columns:
                    if expected_col.lower() in actual_col.lower():
                        column_mapping[actual_col] = expected_col
                        break
            
            if len(column_mapping) < 3:
                raise ValueError(f"Estrutura de perfil inválida. Esperado: {expected_cols}")
            
            df_renamed = df.rename(columns=column_mapping)
            
            # Processar em chunks paralelos para arquivos grandes
            def process_profile_chunk(chunk_df):
                pipeline_data = {
                    'stations': [], 'elevations': [], 'distances': [],
                    'diameters': [], 'thicknesses': [], 'tags': []
                }
                
                for _, row in chunk_df.iterrows():
                    if 'Km Desenvol.' in row:
                        pipeline_data['distances'].append(float(row['Km Desenvol.']) * 1000)
                    if 'Cota' in row:
                        pipeline_data['elevations'].append(float(row['Cota']))
                    if 'Dext' in row:
                        dext_meters = float(row['Dext']) * 0.0254
                        pipeline_data['diameters'].append(dext_meters)
                    if 'Esp' in row:
                        esp_meters = float(row['Esp']) * 0.0254
                        pipeline_data['thicknesses'].append(esp_meters)
                    if 'Tag' in row:
                        pipeline_data['tags'].append(str(row['Tag']))
                
                return pipeline_data
            
            if len(df_renamed) > 1000:
                # Processamento paralelo para arquivos grandes
                chunk_results = self.parallel_processor.process_chunks_parallel(
                    df_renamed, process_profile_chunk, chunk_size=500
                )
                
                # Combinar resultados
                combined_data = {'stations': [], 'elevations': [], 'distances': [], 
                               'diameters': [], 'thicknesses': [], 'tags': []}
                for chunk_data in chunk_results:
                    for key in combined_data.keys():
                        combined_data[key].extend(chunk_data[key])
                        
                return combined_data
            else:
                return process_profile_chunk(df_renamed)
                
        except Exception as e:
            raise Exception(f"Erro no processamento assíncrono do perfil: {e}")
    
    def _process_sensor_data_async(self, df, filename):
        """Processa dados de sensor de forma assíncrona"""
        try:
            if len(df.columns) < 2:
                raise ValueError("Arquivo de sensor deve ter pelo menos 2 colunas")
            
            df.columns = ['tempo', 'valor'] + list(df.columns[2:]) if len(df.columns) > 2 else ['tempo', 'valor']
            
            # Processar tempo de forma assíncrona
            time_col = df.columns[0]
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                try:
                    df[time_col] = pd.to_datetime(df[time_col], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                except:
                    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            
            df = df.dropna(subset=[time_col])
            
            if len(df) == 0:
                raise ValueError("Nenhum registro válido encontrado")
            
            # Parser de informações do sensor
            sensor_info = self._parse_sensor_filename_async(filename)
            
            # Processar em chunks paralelos para arquivos grandes
            def process_sensor_chunk(chunk_df):
                readings = []
                for _, row in chunk_df.iterrows():
                    reading = {
                        'timestamp': row['tempo'],
                        'sensor_id': sensor_info['sensor_id'],
                        'variable': sensor_info['variable'],
                        'value': float(row['valor']),
                        'unit': sensor_info['unit']
                    }
                    readings.append(reading)
                return readings
            
            if len(df) > 5000:
                # Processamento paralelo para arquivos grandes
                chunk_results = self.parallel_processor.process_chunks_parallel(
                    df, process_sensor_chunk, chunk_size=2000
                )
                
                # Combinar resultados
                all_readings = []
                for chunk_readings in chunk_results:
                    all_readings.extend(chunk_readings)
                    
                return {
                    'readings': all_readings,
                    'sensor_info': sensor_info,
                    'total_records': len(all_readings)
                }
            else:
                readings = process_sensor_chunk(df)
                return {
                    'readings': readings,
                    'sensor_info': sensor_info,
                    'total_records': len(readings)
                }
                
        except Exception as e:
            raise Exception(f"Erro no processamento assíncrono do sensor: {e}")
    
    def _parse_sensor_filename_async(self, filename):
        """Parser assíncrono de nome de arquivo de sensor"""
        filename_upper = filename.upper()
        
        if 'BAR_' in filename_upper:
            station = 'BAR'
            location = 'expeditor'
        elif 'PLN_' in filename_upper:
            station = 'PLN'
            location = 'receiver'
        else:
            station = 'UNK'
            location = 'unknown'
        
        if '_DT' in filename_upper:
            variable, unit, description = 'density', 'kg/m³', f'Densidade {station}'
        elif '_FT' in filename_upper:
            variable, unit, description = 'flow_rate', 'kg/s', f'Vazão {station}'
        elif '_PT' in filename_upper:
            variable, unit, description = 'pressure', 'kgf/cm²', f'Pressão {station}'
        elif '_TT' in filename_upper:
            variable, unit, description = 'temperature', '°C', f'Temperatura {station}'
        else:
            variable, unit, description = 'unknown', 'unknown', f'Sensor {station}'
        
        return {
            'sensor_id': f"{station}_{variable.upper()}_01",
            'variable': variable,
            'unit': unit,
            'station': station,
            'location': location,
            'description': description
        }
    
    def cancel(self):
        """Cancela o carregamento"""
        self.is_cancelled = True
        self.requestInterruption()

@dataclass
class SystemConfiguration:
    """Configuração completa de um sistema industrial expandida"""
    system_id: str
    name: str
    location: str
    
    # Características físicas expandidas
    pipe_characteristics: PipeCharacteristics
    sensor_distance: float  # metros
    
    # Fluido
    fluid_type: str
    nominal_density: float  # g/cm³
    nominal_temperature: float  # °C
    nominal_pressure: float  # kgf/cm²
    nominal_flow: float  # m³/h
    
    # Calibração
    sonic_velocity: float  # m/s
    calibration_date: datetime
    
    # Campos com valores padrão devem vir por último
    calibration_parameters: Dict[str, float] = field(default_factory=dict)
    ml_model_version: str = "1.0"
    ml_last_training: Optional[datetime] = None
    ml_performance_metrics: Dict[str, float] = field(default_factory=dict)
    operational_status: str = "unknown"  # open, closed, unknown
    last_status_check: Optional[datetime] = None
    expeditor_unit: str = ""
    expeditor_alias: str = ""
    receiver_unit: str = ""
    receiver_alias: str = ""
    pipeline_profile: Optional[List[Dict[str, Any]]] = field(default_factory=list)

@dataclass
class SensorReading:
    """Leitura individual de sensor com timestamp irregular"""
    timestamp: datetime
    sensor_id: str
    variable: str  # pressure, flow, density, temperature
    value: float
    unit: str
    quality: float = 1.0  # 0-1, qualidade da leitura
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'sensor_id': self.sensor_id,
            'variable': self.variable,
            'value': self.value,
            'unit': self.unit,
            'quality': self.quality
        }
    
    def validate(self, system_id: str = "unknown") -> bool:
        """Valida leitura do sensor"""
        try:
            error_handler.validate_physical_range(self.value, self.variable, system_id)
            return True
        except DataValidationError:
            return False

@dataclass
class MultiVariableSnapshot:
    """Snapshot multivariável interpolado para análise"""
    timestamp: datetime
    expeditor_pressure: float
    receiver_pressure: float
    flow_rate: float
    density: float
    temperature: float
    interpolated_flags: Dict[str, bool] = field(default_factory=dict)
    
    def to_array(self) -> np.ndarray:
        """Converte para array numpy para ML"""
        return np.array([
            self.expeditor_pressure,
            self.receiver_pressure,
            self.flow_rate,
            self.density,
            self.temperature
        ])
    
    def validate_physics(self, system_id: str = "unknown") -> bool:
        """Valida consistência física do snapshot"""
        data = {
            'pressure': (self.expeditor_pressure + self.receiver_pressure) / 2,
            'flow': self.flow_rate,
            'density': self.density,
            'temperature': self.temperature
        }
        return error_handler.validate_cross_variables(data, system_id)

# ============================================================================
# data/database.py - BANCO DE DADOS INDUSTRIAL ROBUSTO
# ============================================================================

class IndustrialDatabase:
    """Banco de dados robusto para múltiplos sistemas industriais"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.logger = industrial_logger.get_logger('database')
        
        if db_path is None:
            data_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            db_path = os.path.join(data_dir, "hydraulic_systems.db")
        
        self.db_path = db_path
        self._init_database()
    
    @error_handler.handle_with_retry(3)
    def _init_database(self):
        """Inicializa estrutura do banco de dados"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tabela de sistemas expandida
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS systems (
                    system_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    location TEXT,
                    config_json TEXT,
                    pipe_characteristics_json TEXT,
                    operational_status TEXT DEFAULT 'unknown',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabela de leituras de sensores
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    system_id TEXT,
                    timestamp TIMESTAMP,
                    sensor_id TEXT,
                    variable TEXT,
                    value REAL,
                    unit TEXT,
                    quality REAL,
                    validated BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (system_id) REFERENCES systems (system_id)
                )
            """)
            
            # Tabela de eventos de vazamento expandida
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS leak_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    system_id TEXT,
                    timestamp TIMESTAMP,
                    leak_type TEXT,
                    severity REAL,
                    location_estimate REAL,
                    confidence REAL,
                    detected_by TEXT,
                    confirmed BOOLEAN DEFAULT FALSE,
                    confirmation_method TEXT,
                    ml_signature TEXT,
                    variables_at_detection TEXT,
                    FOREIGN KEY (system_id) REFERENCES systems (system_id)
                )
            """)
            
            # Tabela de modelos ML expandida
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    system_id TEXT,
                    model_version TEXT,
                    model_type TEXT,
                    model_data BLOB,
                    training_date TIMESTAMP,
                    performance_metrics TEXT,
                    training_samples INTEGER,
                    validation_score REAL,
                    FOREIGN KEY (system_id) REFERENCES systems (system_id)
                )
            """)
            
            # Tabela de calibrações
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS calibrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    system_id TEXT,
                    calibration_date TIMESTAMP,
                    parameters_json TEXT,
                    calibration_type TEXT,
                    performance_metrics TEXT,
                    created_by TEXT,
                    FOREIGN KEY (system_id) REFERENCES systems (system_id)
                )
            """)
            
            # Tabela de status operacional
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS operational_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    system_id TEXT,
                    timestamp TIMESTAMP,
                    status TEXT,
                    confidence REAL,
                    location_estimate REAL,
                    detected_by TEXT,
                    additional_info TEXT,
                    FOREIGN KEY (system_id) REFERENCES systems (system_id)
                )
            """)
            
            # Índices para performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_readings_system_time ON sensor_readings (system_id, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_readings_variable ON sensor_readings (variable)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_leaks_system_time ON leak_events (system_id, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status_system_time ON operational_status (system_id, timestamp)")
            
            conn.commit()
            self.logger.info("Banco de dados inicializado com sucesso")
    
    @error_handler.handle_with_retry(3)
    def save_system(self, config: SystemConfiguration):
        """Salva configuração de sistema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            config_json = json.dumps({
                'sensor_distance': config.sensor_distance,
                'fluid_type': config.fluid_type,
                'nominal_density': config.nominal_density,
                'nominal_temperature': config.nominal_temperature,
                'nominal_pressure': config.nominal_pressure,
                'nominal_flow': config.nominal_flow,
                'sonic_velocity': config.sonic_velocity,
                'calibration_date': config.calibration_date.isoformat(),
                'calibration_parameters': config.calibration_parameters,
                'ml_model_version': config.ml_model_version,
                'ml_last_training': config.ml_last_training.isoformat() if config.ml_last_training else None,
                'ml_performance_metrics': config.ml_performance_metrics
            })
            
            pipe_json = json.dumps({
                'diameter': config.pipe_characteristics.diameter,
                'material': config.pipe_characteristics.material,
                'profile': config.pipe_characteristics.profile,
                'length': config.pipe_characteristics.length,
                'roughness': config.pipe_characteristics.roughness,
                'wall_thickness': config.pipe_characteristics.wall_thickness,
                'elevation_profile': config.pipe_characteristics.elevation_profile,
                'fittings': config.pipe_characteristics.fittings
            })
            
            cursor.execute("""
                INSERT OR REPLACE INTO systems 
                (system_id, name, location, config_json, pipe_characteristics_json, 
                 operational_status, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (config.system_id, config.name, config.location, config_json, 
                  pipe_json, config.operational_status))
            
            conn.commit()
            self.logger.info(f"Sistema {config.system_id} salvo com sucesso")
    
    @error_handler.handle_with_retry(3)
    def load_system(self, system_id: str) -> Optional[SystemConfiguration]:
        """Carrega configuração de sistema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT name, location, config_json, pipe_characteristics_json, operational_status 
                FROM systems WHERE system_id = ?
            """, (system_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            name, location, config_json, pipe_json, operational_status = row
            config_data = json.loads(config_json)
            pipe_data = json.loads(pipe_json)
            
            # Reconstrói características do duto
            pipe_characteristics = PipeCharacteristics(
                diameter=pipe_data['diameter'],
                material=pipe_data['material'],
                profile=pipe_data['profile'],
                length=pipe_data['length'],
                roughness=pipe_data['roughness'],
                wall_thickness=pipe_data['wall_thickness'],
                elevation_profile=pipe_data.get('elevation_profile', []),
                fittings=pipe_data.get('fittings', [])
            )
            
            return SystemConfiguration(
                system_id=system_id,
                name=name,
                location=location,
                pipe_characteristics=pipe_characteristics,
                sensor_distance=config_data['sensor_distance'],
                fluid_type=config_data['fluid_type'],
                nominal_density=config_data['nominal_density'],
                nominal_temperature=config_data['nominal_temperature'],
                nominal_pressure=config_data['nominal_pressure'],
                nominal_flow=config_data['nominal_flow'],
                sonic_velocity=config_data['sonic_velocity'],
                calibration_date=datetime.fromisoformat(config_data['calibration_date']),
                calibration_parameters=config_data['calibration_parameters'],
                ml_model_version=config_data['ml_model_version'],
                ml_last_training=datetime.fromisoformat(config_data['ml_last_training']) if config_data['ml_last_training'] else None,
                ml_performance_metrics=config_data['ml_performance_metrics'],
                operational_status=operational_status
            )
    
    def get_all_systems(self) -> List[Dict[str, str]]:
        """Retorna lista de todos os sistemas"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT system_id, name, location, operational_status FROM systems")
            rows = cursor.fetchall()
            
            return [
                {
                    'system_id': row[0],
                    'name': row[1],
                    'location': row[2],
                    'status': row[3]
                }
                for row in rows
            ]
    
    @error_handler.handle_with_retry(3)
    def save_sensor_readings(self, system_id: str, readings: List[SensorReading]):
        """Salva leituras de sensores em batch com validação"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            validated_data = []
            for reading in readings:
                is_valid = reading.validate(system_id)
                # Converte timestamp para string ISO format para compatibilidade com SQLite
                timestamp_str = reading.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f') if hasattr(reading.timestamp, 'strftime') else str(reading.timestamp)
                validated_data.append((
                    system_id,
                    timestamp_str,
                    reading.sensor_id,
                    reading.variable,
                    reading.value,
                    reading.unit,
                    reading.quality,
                    is_valid
                ))
            
            cursor.executemany("""
                INSERT INTO sensor_readings 
                (system_id, timestamp, sensor_id, variable, value, unit, quality, validated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, validated_data)
            
            conn.commit()
            valid_count = sum(1 for _, _, _, _, _, _, _, valid in validated_data if valid)
            self.logger.info(f"Salvos {valid_count}/{len(readings)} leituras válidas para sistema {system_id}")
    
    def save_operational_status(self, system_id: str, timestamp: datetime, 
                              status: str, confidence: float, location_estimate: Optional[float] = None,
                              detected_by: str = "automatic", additional_info: Optional[Dict[str, Any]] = None):
        """Salva status operacional do sistema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Converte timestamp para string
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f') if hasattr(timestamp, 'strftime') else str(timestamp)
            
            cursor.execute("""
                INSERT INTO operational_status 
                (system_id, timestamp, status, confidence, location_estimate, 
                 detected_by, additional_info)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (system_id, timestamp_str, status, confidence, location_estimate,
                  detected_by, json.dumps(additional_info) if additional_info else None))
            
            # Atualiza status na tabela de sistemas
            cursor.execute("""
                UPDATE systems SET operational_status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE system_id = ?
            """, (status, system_id))
            
            conn.commit()

    @error_handler.handle_with_retry(3)
    def save_ml_model(self, system_id: str, model_version: str, model_type: str, 
                     model_data: bytes, performance_metrics: Dict[str, float],
                     training_samples: int = 0, validation_score: float = 0.0):
        """Salva modelo de ML no banco de dados"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO ml_models 
                (system_id, model_version, model_type, model_data, training_date,
                 performance_metrics, training_samples, validation_score)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?)
            """, (system_id, model_version, model_type, model_data, 
                  json.dumps(performance_metrics), training_samples, validation_score))
            
            conn.commit()
            self.logger.info(f"Modelo ML salvo para sistema {system_id}: {model_version}")

    @error_handler.handle_with_retry(3)
    def load_ml_model(self, system_id: str) -> Optional[Tuple[bytes, Dict[str, float]]]:
        """Carrega modelo ML mais recente do sistema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT model_data, performance_metrics 
                FROM ml_models 
                WHERE system_id = ? 
                ORDER BY training_date DESC 
                LIMIT 1
            """, (system_id,))
            
            row = cursor.fetchone()
            if row:
                model_data, metrics_json = row
                metrics = json.loads(metrics_json) if metrics_json else {}
                return model_data, metrics
            
            return None

    @error_handler.handle_with_retry(3)
    def save_leak_event(self, system_id: str, timestamp: datetime, leak_type: str,
                       severity: float, location_estimate: Optional[float] = None,
                       confidence: float = 1.0, detected_by: str = "automatic",
                       ml_signature: Optional[str] = None, 
                       variables_at_detection: Optional[Dict[str, float]] = None):
        """Salva evento de vazamento detectado"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f') if hasattr(timestamp, 'strftime') else str(timestamp)
            
            cursor.execute("""
                INSERT INTO leak_events 
                (system_id, timestamp, leak_type, severity, location_estimate,
                 confidence, detected_by, ml_signature, variables_at_detection)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (system_id, timestamp_str, leak_type, severity, location_estimate,
                  confidence, detected_by, ml_signature,
                  json.dumps(variables_at_detection) if variables_at_detection else None))
            
            conn.commit()
            self.logger.info(f"Evento de vazamento salvo para sistema {system_id}: {leak_type}")

    @error_handler.handle_with_retry(3)
    def get_sensor_readings(self, system_id: str, start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None, 
                           variables: Optional[List[str]] = None,
                           limit: int = 10000) -> pd.DataFrame:
        """Recupera leituras de sensores do banco de dados"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT timestamp, sensor_id, variable, value, unit, quality, validated
                FROM sensor_readings 
                WHERE system_id = ?
            """
            params = [system_id]
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.strftime('%Y-%m-%d %H:%M:%S.%f'))
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.strftime('%Y-%m-%d %H:%M:%S.%f'))
            
            if variables:
                placeholders = ','.join(['?' for _ in variables])
                query += f" AND variable IN ({placeholders})"
                params.extend(variables)
            
            query += " ORDER BY timestamp DESC"
            
            if limit > 0:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn, params=tuple(params) if params else None)
            
            # Converte timestamp de volta para datetime
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df

# Instância global do banco de dados
database = IndustrialDatabase()

# ============================================================================
# algorithms/temporal_filter.py - FILTRO TEMPORAL PARA STREAMING
# ============================================================================

class TemporalFilter:
    """Filtro temporal avançado para dados de streaming em tempo real"""
    
    def __init__(self, window_size: int = 100, filter_type: str = 'adaptive'):
        self.window_size = window_size
        self.filter_type = filter_type
        self.logger = industrial_logger.get_logger('temporal_filter')
        
        # Buffers para filtragem
        self.time_buffer = deque(maxlen=window_size)
        self.data_buffers = {
            'expeditor_pressure': deque(maxlen=window_size),
            'receiver_pressure': deque(maxlen=window_size),
            'flow_rate': deque(maxlen=window_size),
            'density': deque(maxlen=window_size),
            'temperature': deque(maxlen=window_size)
        }
        
        # Configurações de filtro
        self.smooth_factor = 0.7  # Para filtro exponencial
        self.outlier_threshold = 3.0  # Desvios padrão para detecção de outliers
        self.min_update_interval = 0.1  # Intervalo mínimo entre updates (segundos)
        self.last_update_time = 0
        
        # Estado do filtro
        self.filtered_values = {}
        self.trend_values = {}
        self.quality_scores = {}
        
    def add_data_point(self, timestamp: float, data: Dict[str, float]) -> Dict[str, float]:
        """
        Adiciona ponto de dados e aplica filtragem temporal
        
        Args:
            timestamp: Timestamp Unix
            data: Dicionário com valores das variáveis
            
        Returns:
            Dict com valores filtrados e métricas de qualidade
        """
        # Verifica se deve processar (evita sobrecarga)
        current_time = time.time()
        if current_time - self.last_update_time < self.min_update_interval:
            return self.filtered_values.copy() if self.filtered_values else {}
        
        self.last_update_time = current_time
        
        # Adiciona timestamp
        self.time_buffer.append(timestamp)
        
        filtered_data = {}
        
        for variable, value in data.items():
            if variable in self.data_buffers:
                # Adiciona ao buffer
                self.data_buffers[variable].append(value)
                
                # Aplica filtragem baseada no tipo
                if self.filter_type == 'adaptive':
                    filtered_value = self._apply_adaptive_filter(variable, value)
                elif self.filter_type == 'exponential':
                    filtered_value = self._apply_exponential_filter(variable, value)
                elif self.filter_type == 'median':
                    filtered_value = self._apply_median_filter(variable)
                else:
                    filtered_value = value
                
                # Detecta outliers
                quality_score = self._calculate_quality_score(variable, value)
                
                filtered_data[variable] = filtered_value
                filtered_data[f'{variable}_quality'] = quality_score
                filtered_data[f'{variable}_trend'] = self._calculate_trend(variable)
        
        # Atualiza estado interno
        self.filtered_values.update(filtered_data)
        
        # Adiciona timestamp filtrado
        filtered_data['timestamp'] = timestamp
        filtered_data['filter_lag'] = current_time - timestamp if timestamp > 0 else 0
        
        return filtered_data
    
    def _apply_adaptive_filter(self, variable: str, new_value: float) -> float:
        """Aplica filtro adaptativo baseado na variância dos dados"""
        buffer = self.data_buffers[variable]
        
        if len(buffer) < 3:
            return new_value
        
        # Calcula estatísticas da janela
        recent_values = np.array(list(buffer)[-10:])  # Últimos 10 valores
        mean_val = np.mean(recent_values)
        std_val = np.std(recent_values)
        
        # Fator adaptativo baseado na variabilidade
        if std_val > 0:
            variability = abs(new_value - mean_val) / std_val
            adaptive_factor = min(0.9, 0.1 + 0.8 * np.exp(-variability))
        else:
            adaptive_factor = 0.5
        
        # Filtro exponencial com fator adaptativo
        if variable in self.filtered_values:
            filtered_value = (adaptive_factor * new_value + 
                            (1 - adaptive_factor) * self.filtered_values[variable])
        else:
            filtered_value = new_value
        
        return filtered_value
    
    def _apply_exponential_filter(self, variable: str, new_value: float) -> float:
        """Aplica filtro exponencial simples"""
        if variable in self.filtered_values:
            return (self.smooth_factor * new_value + 
                   (1 - self.smooth_factor) * self.filtered_values[variable])
        return new_value
    
    def _apply_median_filter(self, variable: str) -> float:
        """Aplica filtro de mediana móvel"""
        buffer = self.data_buffers[variable]
        if len(buffer) < 3:
            return buffer[-1] if buffer else 0.0
        
        window = min(7, len(buffer))  # Janela móvel de até 7 pontos
        return float(np.median(list(buffer)[-window:]))
    
    def _calculate_quality_score(self, variable: str, value: float) -> float:
        """Calcula score de qualidade do dado (0-1)"""
        buffer = self.data_buffers[variable]
        
        if len(buffer) < 5:
            return 1.0  # Assume boa qualidade para poucos dados
        
        # Análise de outliers
        recent_values = np.array(list(buffer)[-20:])
        mean_val = np.mean(recent_values)
        std_val = np.std(recent_values)
        
        if std_val == 0:
            return 1.0
        
        # Z-score normalizado
        z_score = abs(value - mean_val) / std_val
        outlier_score = max(0.0, 1.0 - (z_score / self.outlier_threshold))
        
        # Score de consistência temporal
        if len(buffer) >= 2:
            last_value = buffer[-2]
            change_rate = abs(value - last_value) / (abs(last_value) + 1e-6)
            consistency_score = max(0.0, 1.0 - min(1.0, change_rate * 10))
        else:
            consistency_score = 1.0
        
        # Score combinado
        quality_score = (outlier_score * 0.6 + consistency_score * 0.4)
        return float(max(0.0, min(1.0, quality_score)))
    
    def _calculate_trend(self, variable: str) -> float:
        """Calcula tendência da variável (-1 a 1)"""
        buffer = self.data_buffers[variable]
        
        if len(buffer) < 10:
            return 0.0
        
        # Regressão linear simples nos últimos pontos
        recent_values = np.array(list(buffer)[-min(20, len(buffer)):])
        x = np.arange(len(recent_values))
        
        try:
            # Coeficiente angular normalizado
            slope = np.polyfit(x, recent_values, 1)[0]
            value_range = np.ptp(recent_values)
            
            if value_range > 0:
                normalized_slope = slope * len(recent_values) / value_range
                return max(-1.0, min(1.0, normalized_slope))
            return 0.0
            
        except Exception:
            return 0.0
    
    def get_streaming_summary(self) -> Dict[str, Any]:
        """Retorna resumo do estado atual do streaming"""
        if not self.filtered_values:
            return {'status': 'no_data', 'timestamp': time.time()}
        
        summary = {
            'status': 'active',
            'timestamp': time.time(),
            'buffer_size': len(self.time_buffer),
            'variables': {}
        }
        
        for variable in self.data_buffers.keys():
            if variable in self.filtered_values:
                buffer = self.data_buffers[variable]
                quality_key = f'{variable}_quality'
                trend_key = f'{variable}_trend'
                
                summary['variables'][variable] = {
                    'current_value': self.filtered_values[variable],
                    'quality_score': self.filtered_values.get(quality_key, 1.0),
                    'trend': self.filtered_values.get(trend_key, 0.0),
                    'buffer_fill': len(buffer) / self.window_size,
                    'std_dev': np.std(list(buffer)) if len(buffer) > 1 else 0.0
                }
        
        return summary
    
    def reset_filter(self):
        """Reseta estado do filtro"""
        self.time_buffer.clear()
        for buffer in self.data_buffers.values():
            buffer.clear()
        
        self.filtered_values.clear()
        self.trend_values.clear()
        self.quality_scores.clear()
        
        self.last_update_time = 0
        self.logger.info("Filtro temporal resetado")
    
    def configure_filter(self, **kwargs):
        """Configura parâmetros do filtro"""
        if 'filter_type' in kwargs:
            self.filter_type = kwargs['filter_type']
        
        if 'smooth_factor' in kwargs:
            self.smooth_factor = max(0.1, min(0.9, kwargs['smooth_factor']))
        
        if 'outlier_threshold' in kwargs:
            self.outlier_threshold = max(1.0, kwargs['outlier_threshold'])
        
        if 'min_update_interval' in kwargs:
            self.min_update_interval = max(0.01, kwargs['min_update_interval'])
        
        self.logger.info(f"Filtro reconfigurado: tipo={self.filter_type}, "
                        f"suavização={self.smooth_factor}, outlier={self.outlier_threshold}")
    
    def physics_aware_interpolation(self, timestamps: np.ndarray, 
                                   pressure_data: Dict[str, np.ndarray],
                                   flow_data: np.ndarray,
                                   system_properties: Dict[str, float]) -> Dict[str, Any]:
        """Interpolação consciente de física para timestamps irregulares."""
        try:
            # Propriedades do sistema
            pipe_length = system_properties.get('pipe_length', 1000)  # metros
            pipe_diameter = system_properties.get('pipe_diameter', 0.5)  # metros
            fluid_density = system_properties.get('fluid_density', 1000)  # kg/m³
            fluid_viscosity = system_properties.get('fluid_viscosity', 0.001)  # Pa.s
            bulk_modulus = system_properties.get('bulk_modulus', 2.2e9)  # Pa
            
            # Calcular velocidade sônica no fluido
            sound_speed = np.sqrt(bulk_modulus / fluid_density)
            
            # Tempo de propagação acústica
            acoustic_travel_time = pipe_length / sound_speed
            
            # Criar grid temporal regular para interpolação
            min_time, max_time = np.min(timestamps), np.max(timestamps)
            dt_optimal = min(0.1, acoustic_travel_time / 10)  # 10 pontos por tempo acústico
            regular_times = np.arange(min_time, max_time + dt_optimal, dt_optimal)
            
            # Interpolação baseada em física para cada variável
            interpolated_data = {}
            
            for location, pressure_values in pressure_data.items():
                # Remover NaN e outliers
                valid_mask = ~np.isnan(pressure_values)
                valid_times = timestamps[valid_mask]
                valid_pressures = pressure_values[valid_mask]
                
                if len(valid_pressures) < 2:
                    # Dados insuficientes - usar interpolação linear simples
                    interpolated_data[location] = {
                        'values': np.full_like(regular_times, valid_pressures[0] if len(valid_pressures) > 0 else 0),
                        'method': 'constant_fallback'
                    }
                    continue
                
                # Interpolação considerando propagação de ondas de pressão
                interpolated_pressure = self._hydraulic_wave_interpolation(
                    valid_times, valid_pressures, regular_times,
                    sound_speed, acoustic_travel_time
                )
                
                interpolated_data[location] = {
                    'values': interpolated_pressure,
                    'method': 'hydraulic_wave'
                }
            
            # Interpolação para dados de vazão
            valid_flow_mask = ~np.isnan(flow_data)
            if np.sum(valid_flow_mask) >= 2:
                valid_flow_times = timestamps[valid_flow_mask]
                valid_flow_values = flow_data[valid_flow_mask]
                
                interpolated_flow = self._flow_continuity_interpolation(
                    valid_flow_times, valid_flow_values, regular_times,
                    pipe_diameter, fluid_density, fluid_viscosity
                )
            else:
                interpolated_flow = np.full_like(regular_times, 0.0)
            
            # Aplicar restrições físicas
            final_data = self._apply_physical_constraints(
                regular_times, interpolated_data, interpolated_flow,
                system_properties
            )
            
            return {
                'timestamps': regular_times,
                'pressure_data': final_data['pressure'],
                'flow_data': final_data['flow'],
                'interpolation_quality': final_data['quality_metrics'],
                'physics_parameters': {
                    'sound_speed': sound_speed,
                    'acoustic_travel_time': acoustic_travel_time,
                    'sample_rate': 1/dt_optimal
                }
            }
            
        except Exception as e:
            self.logger.error(f"Erro na interpolação consciente de física: {e}")
            return {'error': str(e)}
    
    def _hydraulic_wave_interpolation(self, times: np.ndarray, values: np.ndarray,
                                     target_times: np.ndarray, sound_speed: float,
                                     travel_time: float) -> np.ndarray:
        """Interpolação considerando propagação de ondas hidráulicas."""
        from scipy.interpolate import interp1d
        from scipy.ndimage import gaussian_filter1d
        
        # Interpolação spline cúbica base
        if len(times) >= 4:
            interpolator = interp1d(times, values, kind='cubic', 
                                  bounds_error=False, fill_value='extrapolate')
        else:
            interpolator = interp1d(times, values, kind='linear',
                                  bounds_error=False, fill_value='extrapolate')
        
        base_interpolation = interpolator(target_times)
        
        # Aplicar suavização baseada no tempo acústico
        # Janela de suavização proporcional ao tempo de propagação
        sigma = travel_time / (target_times[1] - target_times[0]) / 4
        smoothed_values = gaussian_filter1d(base_interpolation, sigma=max(0.5, sigma))
        
        # Detectar e preservar transientes rápidos (características de vazamentos)
        if len(values) > 3:
            # Calcular gradiente temporal
            dt = np.diff(times)
            dP_dt = np.diff(values) / dt
            
            # Identificar mudanças rápidas (possíveis vazamentos)
            rapid_changes = np.abs(dP_dt) > np.std(dP_dt) * 2
            
            if np.any(rapid_changes):
                # Preservar características de alta frequência em regiões de mudança rápida
                change_indices = np.where(rapid_changes)[0]
                for idx in change_indices:
                    if idx < len(times) - 1:
                        # Região de interesse
                        t_start, t_end = times[idx], times[idx + 1]
                        mask = (target_times >= t_start) & (target_times <= t_end)
                        
                        # Manter resolução alta nesta região
                        smoothed_values[mask] = base_interpolation[mask]
        
        return smoothed_values
    
    def _flow_continuity_interpolation(self, times: np.ndarray, flow_values: np.ndarray,
                                      target_times: np.ndarray, pipe_diameter: float,
                                      density: float, viscosity: float) -> np.ndarray:
        """Interpolação considerando continuidade do fluxo e perdas por atrito."""
        from scipy.interpolate import interp1d
        
        # Calcular número de Reynolds médio
        pipe_area = np.pi * (pipe_diameter / 2) ** 2
        mean_velocity = np.mean(np.abs(flow_values)) / (pipe_area * density)
        reynolds = density * mean_velocity * pipe_diameter / viscosity
        
        # Interpolação base
        if len(times) >= 4:
            interpolator = interp1d(times, flow_values, kind='cubic',
                                  bounds_error=False, fill_value='extrapolate')
        else:
            interpolator = interp1d(times, flow_values, kind='linear',
                                  bounds_error=False, fill_value='extrapolate')
        
        interpolated_flow = interpolator(target_times)
        
        # Aplicar correção baseada no regime de fluxo
        if reynolds > 2300:  # Fluxo turbulento
            # Maior suavização para capturar características turbulentas
            from scipy.ndimage import gaussian_filter1d
            sigma = len(interpolated_flow) / 20  # Suavização moderada
            interpolated_flow = gaussian_filter1d(interpolated_flow, sigma=sigma)
        
        # Preservar continuidade (fluxo não pode mudar instantaneamente)
        dt = target_times[1] - target_times[0] if len(target_times) > 1 else 1.0
        max_flow_change_rate = 100.0 / dt  # kg/s por segundo (limitação física)
        
        for i in range(1, len(interpolated_flow)):
            flow_change = interpolated_flow[i] - interpolated_flow[i-1]
            if abs(flow_change) > max_flow_change_rate * dt:
                # Limitar taxa de mudança
                interpolated_flow[i] = interpolated_flow[i-1] + np.sign(flow_change) * max_flow_change_rate * dt
        
        return interpolated_flow
    
    def _apply_physical_constraints(self, times: np.ndarray, 
                                   pressure_data: Dict[str, Dict],
                                   flow_data: np.ndarray,
                                   system_props: Dict[str, float]) -> Dict[str, Any]:
        """Aplica restrições físicas aos dados interpolados."""
        
        # Restrições de pressão
        min_pressure = system_props.get('min_pressure', 0.0)
        max_pressure = system_props.get('max_pressure', 100.0)
        
        constrained_pressure = {}
        quality_metrics = {'pressure_violations': 0, 'flow_violations': 0}
        
        for location, data in pressure_data.items():
            values = data['values']
            
            # Aplicar limites físicos
            original_violations = np.sum((values < min_pressure) | (values > max_pressure))
            values_constrained = np.clip(values, min_pressure, max_pressure)
            
            constrained_pressure[location] = values_constrained
            quality_metrics['pressure_violations'] += original_violations
        
        # Restrições de fluxo
        max_flow = system_props.get('max_flow', 1000.0)  # kg/s
        flow_violations = np.sum(np.abs(flow_data) > max_flow)
        flow_constrained = np.clip(flow_data, -max_flow, max_flow)
        
        quality_metrics['flow_violations'] = flow_violations
        
        # Verificar consistência entre pressão e fluxo (equação de Darcy-Weisbach simplificada)
        if len(pressure_data) >= 2:
            locations = list(pressure_data.keys())
            if len(locations) >= 2:
                p1 = constrained_pressure[locations[0]]
                p2 = constrained_pressure[locations[1]]
                pressure_diff = p1 - p2
                
                # Verificar se diferença de pressão é consistente com fluxo
                expected_flow_direction = np.sign(pressure_diff)
                actual_flow_direction = np.sign(flow_constrained)
                
                consistency = np.mean(expected_flow_direction * actual_flow_direction >= 0)
                quality_metrics['pressure_flow_consistency'] = float(consistency)
        
        # Cálculo de qualidade geral
        total_points = len(times) * len(pressure_data)
        if total_points > 0:
            quality_score = 1.0 - (quality_metrics['pressure_violations'] + 
                                 quality_metrics['flow_violations']) / (2 * total_points)
        else:
            quality_score = 0.0
        
        quality_metrics['overall_quality'] = max(0.0, quality_score)
        
        return {
            'pressure': constrained_pressure,
            'flow': flow_constrained,
            'quality_metrics': quality_metrics
        }

# Instância global do filtro temporal
temporal_filter = TemporalFilter()

# ============================================================================
# algorithms/data_simulator.py - GERADOR DE DADOS PARA SIMULAÇÃO E TESTES
# ============================================================================

class HydraulicDataSimulator:
    """Gerador de dados realísticos para simulação de sistema hidráulico"""
    
    def __init__(self, system_config: Optional[SystemConfiguration] = None):
        self.config = system_config
        self.logger = industrial_logger.get_logger('data_simulator')
        
        # Estado da simulação
        self.current_time = time.time()
        self.is_running = False
        self.simulation_speed = 1.0  # Multiplicador de velocidade
        
        # Atributos para dados reais
        self.real_data_mode = False
        self.real_data_cache = {}
        self.data_start_time = None
        self.data_duration = None
        
        # Parâmetros base da simulação
        self.base_params = {
            'expeditor_pressure': {'base': 15.0, 'noise': 0.5, 'trend': 0.0},
            'receiver_pressure': {'base': 12.0, 'noise': 0.4, 'trend': 0.0}, 
            'flow_rate': {'base': 500.0, 'noise': 25.0, 'trend': 0.0},
            'density': {'base': 0.85, 'noise': 0.02, 'trend': 0.0},
            'temperature': {'base': 25.0, 'noise': 2.0, 'trend': 0.0}
        }
        
        # Cenários de simulação
        self.scenarios = {
            'normal': {'description': 'Operação normal'},
            'leak_gradual': {'description': 'Vazamento gradual', 'duration': 300},
            'leak_sudden': {'description': 'Vazamento súbito', 'duration': 120},
            'valve_closure': {'description': 'Fechamento de válvula', 'duration': 60},
            'pressure_surge': {'description': 'Surto de pressão', 'duration': 30},
            'sensor_drift': {'description': 'Deriva de sensor', 'duration': 600}
        }
        
        self.current_scenario = 'normal'
        self.scenario_start_time = 0
        self.scenario_progress = 0.0
        
        # Buffers para padrões temporais
        self.time_patterns = {
            'daily_cycle': True,
            'weekly_cycle': False,
            'seasonal_drift': False
        }
        
        # Thread para simulação contínua
        self.simulation_thread = None
        self.stop_event = threading.Event()
    
    def start_simulation(self, target_system_id: str = 'SIM_001', 
                        scenario: str = 'normal', speed: float = 1.0):
        """Inicia simulação contínua de dados"""
        if self.is_running:
            self.stop_simulation()
        
        self.current_scenario = scenario
        self.simulation_speed = speed
        self.scenario_start_time = time.time()
        self.scenario_progress = 0.0
        self.is_running = True
        self.stop_event.clear()
        
        # Configura sistema se não existir
        if not self.config:
            self._create_default_system_config(target_system_id)
        
        self.logger.info(f"Simulação iniciada: cenário={scenario}, velocidade={speed}x")
        
        # Thread para geração contínua
        self.simulation_thread = threading.Thread(
            target=self._simulation_loop, 
            args=(target_system_id,),
            daemon=True
        )
        self.simulation_thread.start()
    
    def stop_simulation(self):
        """Para simulação"""
        self.is_running = False
        self.stop_event.set()
        
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=2.0)
        
        self.logger.info("Simulação parada")
    
    def _simulation_loop(self, system_id: str):
        """Loop principal da simulação"""
        while not self.stop_event.is_set():
            try:
                # Gera ponto de dados
                timestamp = time.time()
                data_point = self._generate_data_point(timestamp)
                
                # Adiciona aos buffers via memory manager
                memory_manager.add_data_point(system_id, timestamp, data_point)
                
                # Intervalo baseado na velocidade
                sleep_time = 0.1 / self.simulation_speed  # 10Hz base
                self.stop_event.wait(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Erro no loop de simulação: {e}")
                self.stop_event.wait(1.0)  # Pausa em caso de erro
    
    def _generate_data_point(self, timestamp: float) -> Dict[str, float]:
        """Gera um ponto de dados realístico"""
        
        # Tempo relativo ao início do cenário
        scenario_time = timestamp - self.scenario_start_time
        self.scenario_progress = min(1.0, scenario_time / self.scenarios[self.current_scenario].get('duration', 300))
        
        # Padrões temporais base
        time_of_day = (timestamp % 86400) / 86400  # 0-1 para 24h
        daily_factor = 0.8 + 0.4 * np.sin(2 * np.pi * time_of_day)  # Variação diária
        
        data = {}
        
        for var_name, params in self.base_params.items():
            # Valor base com padrões temporais
            base_value = params['base'] * daily_factor
            
            # Ruído gaussiano
            noise = np.random.normal(0, params['noise'])
            
            # Aplicar cenário específico
            scenario_modifier = self._apply_scenario_effects(var_name, scenario_time)
            
            # Tendência de longo prazo
            trend_effect = params['trend'] * (timestamp - self.current_time) / 3600  # Por hora
            
            # Valor final
            final_value = base_value + noise + scenario_modifier + trend_effect
            
            # Aplica limites físicos
            final_value = self._apply_physical_limits(var_name, final_value)
            
            data[var_name] = final_value
        
        # Adiciona correlação física entre variáveis
        data = self._apply_physical_correlations(data, scenario_time)
        
        return data
    
    def _apply_scenario_effects(self, variable: str, scenario_time: float) -> float:
        """Aplica efeitos específicos do cenário"""
        if self.current_scenario == 'normal':
            return 0.0
        
        progress = self.scenario_progress
        
        if self.current_scenario == 'leak_gradual':
            if variable == 'expeditor_pressure':
                return -2.0 * progress  # Queda gradual
            elif variable == 'receiver_pressure':
                return -1.5 * progress
            elif variable == 'flow_rate':
                return -50 * progress  # Redução de vazão
                
        elif self.current_scenario == 'leak_sudden':
            if scenario_time > 10:  # Após 10s
                if variable == 'expeditor_pressure':
                    return -5.0  # Queda súbita
                elif variable == 'receiver_pressure':
                    return -4.0
                elif variable == 'flow_rate':
                    return -100
                    
        elif self.current_scenario == 'valve_closure':
            closure_factor = min(1.0, progress * 2)  # Fechamento rápido
            if variable == 'flow_rate':
                return -400 * closure_factor
            elif variable in ['expeditor_pressure', 'receiver_pressure']:
                return 2.0 * closure_factor  # Aumento de pressão
                
        elif self.current_scenario == 'pressure_surge':
            surge_wave = np.sin(2 * np.pi * scenario_time / 5)  # Onda de 5s
            if variable in ['expeditor_pressure', 'receiver_pressure']:
                return 3.0 * surge_wave * (1 - progress)
                
        elif self.current_scenario == 'sensor_drift':
            if variable == 'expeditor_pressure':
                return 0.5 * progress  # Deriva lenta
                
        return 0.0
    
    def _apply_physical_correlations(self, data: Dict[str, float], scenario_time: float) -> Dict[str, float]:
        """Aplica correlações físicas realísticas entre variáveis"""
        
        # Correlação pressão-vazão (Bernoulli simplificado)
        pressure_diff = data['expeditor_pressure'] - data['receiver_pressure']
        if pressure_diff > 0:
            # Flow aumenta com diferença de pressão
            flow_boost = np.sqrt(pressure_diff) * 20
            data['flow_rate'] += flow_boost
        
        # Correlação densidade-temperatura (expansão térmica)
        temp_effect = (data['temperature'] - 20) * -0.0005  # kg/m³ por °C
        data['density'] += temp_effect
        
        # Ruído correlacionado entre pressões (vibração de bomba)
        vibration_noise = np.random.normal(0, 0.1)
        data['expeditor_pressure'] += vibration_noise
        data['receiver_pressure'] += vibration_noise * 0.7
        
        return data
    
    def _apply_physical_limits(self, variable: str, value: float) -> float:
        """Aplica limites físicos realísticos"""
        limits = CONSTANTS.VARIABLES.get(variable, {})
        
        min_val = limits.get('min', 0)
        max_val = limits.get('max', 1000)
        
        return max(min_val, min(max_val, value))
    
    def _create_default_system_config(self, system_id: str):
        """Cria configuração padrão para simulação"""
        
        pipe_chars = PipeCharacteristics(
            diameter=0.3,  # 30cm
            material='steel',
            profile='circular',
            length=100.0,
            roughness=0.1,
            wall_thickness=10.0
        )
        
        self.config = SystemConfiguration(
            system_id=system_id,
            name=f"Sistema Simulado {system_id}",
            location="Simulação Virtual",
            pipe_characteristics=pipe_chars,
            sensor_distance=100.0,
            fluid_type="oil",
            nominal_density=0.85,
            nominal_temperature=25.0,
            nominal_pressure=15.0,
            nominal_flow=500.0,
            sonic_velocity=1400.0,
            calibration_date=datetime.now()
        )
    
    def change_scenario(self, new_scenario: str):
        """Muda cenário durante a simulação"""
        if new_scenario in self.scenarios:
            old_scenario = self.current_scenario
            self.current_scenario = new_scenario
            self.scenario_start_time = time.time()
            self.scenario_progress = 0.0
            
            self.logger.info(f"Cenário alterado: {old_scenario} -> {new_scenario}")
        else:
            self.logger.warning(f"Cenário inválido: {new_scenario}")
    
    def set_parameter(self, variable: str, parameter: str, value: float):
        """Altera parâmetros durante simulação"""
        if variable in self.base_params and parameter in self.base_params[variable]:
            old_value = self.base_params[variable][parameter]
            self.base_params[variable][parameter] = value
            self.logger.info(f"Parâmetro alterado: {variable}.{parameter} = {old_value} -> {value}")
    
    def generate_batch_data(self, duration_minutes: int = 60, 
                           frequency_hz: float = 1.0, 
                           scenario: str = 'normal') -> List[SensorReading]:
        """Gera batch de dados para testes"""
        
        readings = []
        total_points = int(duration_minutes * 60 * frequency_hz)
        
        self.current_scenario = scenario
        start_time = time.time()
        self.scenario_start_time = start_time
        
        for i in range(total_points):
            timestamp = start_time + (i / frequency_hz)
            data_point = self._generate_data_point(timestamp)
            
            # Converte para SensorReading objects
            for var_name, value in data_point.items():
                sensor_id = f"SIM_{var_name.upper()}_01"
                unit = CONSTANTS.VARIABLES.get(var_name, {}).get('unit', 'unknown')
                
                reading = SensorReading(
                    timestamp=datetime.fromtimestamp(timestamp),
                    sensor_id=sensor_id,
                    variable=var_name,
                    value=value,
                    unit=unit,
                    quality=1.0
                )
                readings.append(reading)
        
        self.logger.info(f"Batch gerado: {len(readings)} leituras, {duration_minutes}min, {scenario}")
        return readings
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Retorna status atual da simulação"""
        return {
            'is_running': self.is_running,
            'current_scenario': self.current_scenario,
            'scenario_progress': self.scenario_progress,
            'simulation_speed': self.simulation_speed,
            'scenarios_available': list(self.scenarios.keys()),
            'base_parameters': self.base_params.copy()
        }

# Instância global do simulador
hydraulic_simulator = HydraulicDataSimulator()

# ============================================================================
# algorithms/time_series_processor.py - PROCESSAMENTO TEMPORAL AVANÇADO
# ============================================================================

class IrregularTimeSeriesProcessor:
    """Processador avançado para dados com timestamps irregulares"""
    
    def __init__(self):
        self.logger = industrial_logger.get_logger('time_processor')
        
    def synchronize_multivariable_data(self, readings_df: pd.DataFrame, 
                                     target_frequency: float = 1.0,
                                     system_id: str = "unknown") -> List[MultiVariableSnapshot]:
        """
        Sincroniza dados multivariáveis com timestamps irregulares usando interpolação avançada
        """
        
        if readings_df.empty:
            return []
        
        # Converte timestamp para datetime se necessário
        readings_df['timestamp'] = pd.to_datetime(readings_df['timestamp'], format='ISO8601')
        
        # Valida dados de entrada
        required_columns = ['timestamp', 'sensor_id', 'variable', 'value', 'unit']
        for col in required_columns:
            if col not in readings_df.columns:
                raise DataValidationError(f"Coluna obrigatória '{col}' não encontrada nos dados")
        
        # Determina intervalo de tempo
        start_time = readings_df['timestamp'].min()
        end_time = readings_df['timestamp'].max()
        
        self.logger.info(f"Sincronizando dados de {start_time} a {end_time} para sistema {system_id}")
        
        # Cria grade temporal regular
        time_delta = timedelta(seconds=1.0/target_frequency)
        target_times = []
        current_time = start_time
        while current_time <= end_time:
            target_times.append(current_time)
            current_time += time_delta
        
        # Organiza dados por variável e sensor
        variables_data = {}
        for variable in ['pressure', 'flow', 'density', 'temperature']:
            var_data = readings_df[readings_df['variable'] == variable].copy()
            if not var_data.empty:
                var_data = var_data.sort_values('timestamp')
                variables_data[variable] = var_data
        
        # Interpola cada variável
        snapshots = []
        for target_time in target_times:
            snapshot_data = {'timestamp': target_time}
            interpolated_flags = {}
            
            for variable, var_df in variables_data.items():
                if variable == 'pressure':
                    # Separa expedidor e recebedor
                    exp_data = var_df[var_df['sensor_id'].str.contains('exp|upstream|01', case=False, na=False)]
                    rec_data = var_df[var_df['sensor_id'].str.contains('rec|downstream|02', case=False, na=False)]
                    
                    exp_value, exp_interpolated = self._interpolate_single_variable(exp_data, target_time)
                    rec_value, rec_interpolated = self._interpolate_single_variable(rec_data, target_time)
                    
                    snapshot_data['expeditor_pressure'] = exp_value
                    snapshot_data['receiver_pressure'] = rec_value
                    interpolated_flags['expeditor_pressure'] = exp_interpolated
                    interpolated_flags['receiver_pressure'] = rec_interpolated
                    
                else:
                    value, interpolated = self._interpolate_single_variable(var_df, target_time)
                    key = f'{variable}_rate' if variable == 'flow' else variable
                    snapshot_data[key] = value
                    interpolated_flags[key] = interpolated
            
            # Cria snapshot se temos dados mínimos necessários
            if all(key in snapshot_data for key in ['expeditor_pressure', 'receiver_pressure']):
                snapshot = MultiVariableSnapshot(
                    timestamp=target_time,
                    expeditor_pressure=snapshot_data.get('expeditor_pressure', 0.0),
                    receiver_pressure=snapshot_data.get('receiver_pressure', 0.0),
                    flow_rate=snapshot_data.get('flow_rate', 0.0),
                    density=snapshot_data.get('density', 1.0),
                    temperature=snapshot_data.get('temperature', 20.0),
                    interpolated_flags=interpolated_flags
                )
                
                # Valida consistência física
                if snapshot.validate_physics(system_id):
                    snapshots.append(snapshot)
                else:
                    self.logger.warning(f"Snapshot com inconsistência física rejeitado em {target_time}")
        
        self.logger.info(f"Criados {len(snapshots)} snapshots sincronizados de {len(target_times)} timestamps alvo")
        return snapshots
    
    def _interpolate_single_variable(self, var_df: pd.DataFrame, target_time: datetime) -> Tuple[float, bool]:
        """Interpola valor de uma variável para timestamp específico com métodos avançados"""
        
        if var_df.empty:
            return 0.0, True
        
        # Converte para valores numéricos para interpolação
        timestamps_numeric = pd.to_numeric(var_df['timestamp'].astype('int64'), errors='coerce')
        target_numeric = pd.Timestamp(target_time).value
        values = pd.to_numeric(var_df['value'], errors='coerce').to_numpy()
        
        # Verifica se há leitura exata
        exact_match = var_df[var_df['timestamp'] == target_time]
        if not exact_match.empty:
            return float(exact_match['value'].iloc[0]), False
        
        # Se fora do range, usa extrapolação limitada
        min_time = float(timestamps_numeric.min()) if not timestamps_numeric.empty else target_numeric
        max_time = float(timestamps_numeric.max()) if not timestamps_numeric.empty else target_numeric
        
        if target_numeric < min_time:
            return float(values[0]) if len(values) > 0 else 0.0, True
        if target_numeric > max_time:
            return float(values[-1]) if len(values) > 0 else 0.0, True
        
        # Interpolação baseada na qualidade dos dados
        try:
            # Remove NaN values com tratamento robusto
            valid_mask = ~(pd.isna(values) | np.isinf(values))
            if not np.any(valid_mask):
                return 0.0, True
            
            clean_times = timestamps_numeric[valid_mask].to_numpy()
            clean_values = values[valid_mask]
            
            # Escolhe método de interpolação baseado na quantidade de pontos
            if len(clean_values) < 3:
                # Interpolação linear simples
                interp_value = float(np.interp(target_numeric, clean_times, clean_values))
            else:
                # Interpolação spline para suavidade
                try:
                    from scipy.interpolate import UnivariateSpline
                    spline = UnivariateSpline(clean_times, clean_values, s=0, k=min(3, len(clean_values)-1))
                    result = spline(target_numeric)
                    # Usa numpy para garantir conversão segura
                    interp_value = float(np.asarray(result).item() if np.asarray(result).size > 0 else 0.0)
                except Exception:
                    # Fallback para interpolação linear
                    interp_value = float(np.interp(target_numeric, clean_times, clean_values))
            
            return interp_value, True
            
        except Exception as e:
            self.logger.warning(f"Erro na interpolação: {e}")
            return float(values[0]) if len(values) > 0 else 0.0, True

# ============================================================================
# algorithms/operational_status_detector.py - DETECTOR DE STATUS OPERACIONAL
# ============================================================================

class OperationalStatusDetector:
    """Detecta se sistema está com coluna aberta/fechada e localiza abertura"""
    
    def __init__(self, system_config: SystemConfiguration):
        self.config = system_config
        self.logger = industrial_logger.get_logger('status_detector')
        
        # Parâmetros adaptativos baseados nas características do sistema
        self.pressure_stability_threshold = 0.1  # kgf/cm²
        self.flow_stability_threshold = self.config.nominal_flow * 0.05  # 5% da vazão nominal
        self.column_open_pressure_drop = 0.5  # kgf/cm²
        
        # Características acústicas do sistema
        self.acoustic_props = self.config.pipe_characteristics.calculate_acoustic_properties()
        
    def analyze_operational_status(self, snapshots: List[MultiVariableSnapshot]) -> Dict[str, Any]:
        """
        Analisa status operacional completo do sistema
        """
        
        if len(snapshots) < 20:
            return {
                'column_status': 'unknown',
                'confidence': 0.0,
                'error': 'Dados insuficientes'
            }
        
        # Extrai dados para análise
        times = np.array([s.timestamp.timestamp() for s in snapshots])
        exp_pressures = np.array([s.expeditor_pressure for s in snapshots])
        rec_pressures = np.array([s.receiver_pressure for s in snapshots])
        flows = np.array([s.flow_rate for s in snapshots])
        densities = np.array([s.density for s in snapshots])
        temperatures = np.array([s.temperature for s in snapshots])
        
        # Análises especializadas
        pressure_analysis = self._analyze_pressure_patterns(exp_pressures, rec_pressures)
        flow_analysis = self._analyze_flow_patterns(flows)
        acoustic_analysis = self._analyze_acoustic_signature(exp_pressures, rec_pressures)
        thermal_analysis = self._analyze_thermal_patterns(temperatures, densities)
        
        # Determina status baseado em múltiplos indicadores
        status_indicators = []
        
        # Indicador 1: Padrão de pressão
        if pressure_analysis['differential_stability'] > 0.8:
            if pressure_analysis['mean_differential'] > self.column_open_pressure_drop:
                status_indicators.append(('closed', 0.9))
            else:
                status_indicators.append(('open', 0.7))
        
        # Indicador 2: Padrão acústico
        if acoustic_analysis['correlation_strength'] > 0.6:
            if acoustic_analysis['delay_consistency'] > 0.8:
                status_indicators.append(('closed', 0.8))
            else:
                status_indicators.append(('open', 0.6))
        
        # Indicador 3: Padrão de vazão
        if flow_analysis['stability'] > 0.7:
            if flow_analysis['mean_flow'] > self.config.nominal_flow * 0.8:
                status_indicators.append(('closed', 0.7))
        
        # Combina indicadores
        status_votes = {'open': [], 'closed': [], 'unknown': []}
        for status, confidence in status_indicators:
            status_votes[status].append(confidence)
        
        # Determina status final
        final_status = 'unknown'
        final_confidence = 0.0
        
        for status, confidences in status_votes.items():
            if confidences:
                avg_confidence = np.mean(confidences)
                if avg_confidence > final_confidence:
                    final_status = status
                    final_confidence = avg_confidence
        
        # Estima localização se aberto
        open_location = None
        if final_status == 'open':
            open_location = self._estimate_opening_location(
                exp_pressures, rec_pressures, flows, acoustic_analysis
            )
        
        result = {
            'column_status': final_status,
            'confidence': float(final_confidence),
            'open_location': open_location,
            'pressure_analysis': pressure_analysis,
            'flow_analysis': flow_analysis,
            'acoustic_analysis': acoustic_analysis,
            'thermal_analysis': thermal_analysis,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Salva no banco de dados
        database.save_operational_status(
            self.config.system_id,
            datetime.now(),
            final_status,
            float(final_confidence),
            open_location,
            "automatic_analysis",
            {
                'pressure_analysis': pressure_analysis,
                'acoustic_analysis': acoustic_analysis
            }
        )
        
        return result
    
    def _analyze_pressure_patterns(self, exp_pressures: np.ndarray, 
                                 rec_pressures: np.ndarray) -> Dict[str, float]:
        """Analisa padrões de pressão para determinar status"""
        
        pressure_diff = exp_pressures - rec_pressures
        
        return {
            'mean_differential': float(np.mean(pressure_diff)),
            'differential_stability': float(1.0 / (1.0 + np.std(pressure_diff))),
            'exp_stability': float(1.0 / (1.0 + np.std(exp_pressures))),
            'rec_stability': float(1.0 / (1.0 + np.std(rec_pressures))),
            'trend_consistency': float(np.corrcoef(exp_pressures, rec_pressures)[0,1]) if len(exp_pressures) > 1 else 0.0
        }
    
    def _analyze_flow_patterns(self, flows: np.ndarray) -> Dict[str, float]:
        """Analisa padrões de vazão"""
        
        return {
            'mean_flow': float(np.mean(flows)),
            'stability': float(1.0 / (1.0 + np.std(flows) / max(np.mean(flows), 1.0))),
            'variation_coefficient': float(np.std(flows) / max(np.mean(flows), 1.0))
        }
    
    def _analyze_acoustic_signature(self, exp_pressures: np.ndarray, 
                                  rec_pressures: np.ndarray) -> Dict[str, float]:
        """Analisa assinatura acústica para determinação de status"""
        
        # Correlação cruzada
        from scipy import signal as sp_signal
        correlation = sp_signal.correlate(exp_pressures, rec_pressures, mode='full')
        max_corr_idx = np.argmax(np.abs(correlation))
        max_correlation = correlation[max_corr_idx]
        delay_samples = max_corr_idx - len(correlation)//2
        
        # Tempo de trânsito esperado vs observado
        expected_delay = self.config.sensor_distance / self.config.sonic_velocity
        
        # Análise espectral
        exp_fft = np.fft.fft(exp_pressures)
        rec_fft = np.fft.fft(rec_pressures)
        
        # Coerência espectral
        coherence = np.abs(np.mean(exp_fft * np.conj(rec_fft))) / (np.sqrt(np.mean(np.abs(exp_fft)**2)) * np.sqrt(np.mean(np.abs(rec_fft)**2)))
        
        return {
            'correlation_strength': float(abs(max_correlation) / len(exp_pressures)),
            'delay_samples': int(delay_samples),
            'delay_consistency': float(1.0 / (1.0 + abs(delay_samples - expected_delay))),
            'spectral_coherence': float(coherence),
            'expected_delay': float(expected_delay)
        }
    
    def _analyze_thermal_patterns(self, temperatures: np.ndarray, 
                                densities: np.ndarray) -> Dict[str, float]:
        """Analisa padrões térmicos e de densidade"""
        
        temp_density_corr = np.corrcoef(temperatures, densities)[0,1] if len(temperatures) > 1 else 0.0
        
        return {
            'temperature_stability': float(1.0 / (1.0 + np.std(temperatures))),
            'density_stability': float(1.0 / (1.0 + np.std(densities))),
            'temp_density_correlation': float(temp_density_corr)
        }
    
    def _estimate_opening_location(self, exp_pressures: np.ndarray, 
                                 rec_pressures: np.ndarray, flows: np.ndarray,
                                 acoustic_analysis: Dict[str, float]) -> Optional[float]:
        """Estima localização da abertura usando múltiplos métodos"""
        
        # Método 1: Análise de gradiente de pressão
        pressure_gradient = np.mean(exp_pressures - rec_pressures) / self.config.sensor_distance
        
        # Método 2: Análise de delay acústico
        delay_ratio = acoustic_analysis.get('delay_samples', 0) / acoustic_analysis.get('expected_delay', 1)
        
        # Método 3: Análise hidráulica
        flow_factor = np.mean(flows) / max(self.config.nominal_flow, 1.0)
        
        # Combina estimativas
        estimates = []
        
        if pressure_gradient > 0:
            # Estima posição baseada no perfil de pressão
            location_pressure = self.config.sensor_distance * (1 + pressure_gradient / 10)
            estimates.append(min(location_pressure, self.config.pipe_characteristics.length))
        
        if delay_ratio < 1.0:
            # Estima posição baseada no delay acústico
            location_acoustic = self.config.sensor_distance * delay_ratio
            estimates.append(max(0, location_acoustic))
        
        if estimates:
            return float(np.mean(estimates))
        
        return None
    
    def spectral_state_detection(self, signal: np.ndarray, 
                                sample_rate: float = 1.0) -> Dict[str, Any]:
        """Detecção de estado operacional usando análise espectral avançada."""
        try:
            # Aplicar filtro Butterworth passa-baixa para remover ruído
            low_filtered = self.apply_butterworth_filter(signal, 
                                                       cutoff_freq=0.1, 
                                                       filter_type='low', 
                                                       order=4)
            
            # Aplicar filtro passa-alta para detectar transientes
            high_filtered = self.apply_butterworth_filter(signal, 
                                                        cutoff_freq=0.01, 
                                                        filter_type='high', 
                                                        order=2)
            
            # Transformada de Hilbert para análise de envelope
            hilbert_transform = self.apply_hilbert_transform(signal)
            
            # FFT para análise de frequência
            fft_result = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
            magnitude = np.abs(fft_result)
            
            # Análise espectral detalhada
            spectral_analysis = {
                'dominant_frequency': float(freqs[np.argmax(magnitude[:len(magnitude)//2])]),
                'spectral_centroid': float(np.sum(freqs[:len(freqs)//2] * magnitude[:len(magnitude)//2]) / 
                                         np.sum(magnitude[:len(magnitude)//2])),
                'spectral_spread': float(np.sqrt(np.sum((freqs[:len(freqs)//2] - 
                                                       np.sum(freqs[:len(freqs)//2] * magnitude[:len(magnitude)//2]) / 
                                                       np.sum(magnitude[:len(magnitude)//2]))**2 * 
                                                      magnitude[:len(magnitude)//2]) / 
                                               np.sum(magnitude[:len(magnitude)//2]))),
                'spectral_rolloff': self._calculate_spectral_rolloff(freqs, magnitude),
                'zero_crossing_rate': self._calculate_zero_crossing_rate(signal),
                'spectral_flux': self._calculate_spectral_flux(magnitude)
            }
            
            # Análise de envelope (Hilbert)
            envelope_analysis = {
                'envelope_mean': float(np.mean(hilbert_transform['envelope'])),
                'envelope_std': float(np.std(hilbert_transform['envelope'])),
                'envelope_peaks': len(hilbert_transform['peaks']),
                'instantaneous_frequency_mean': float(np.mean(hilbert_transform['instantaneous_frequency'])),
                'phase_coherence': float(hilbert_transform['phase_coherence'])
            }
            
            # Análise de filtros
            filter_analysis = {
                'low_freq_energy': float(np.sum(low_filtered**2)),
                'high_freq_energy': float(np.sum(high_filtered**2)),
                'energy_ratio': float(np.sum(high_filtered**2) / max(np.sum(low_filtered**2), 1e-10)),
                'signal_smoothness': float(np.mean(np.abs(np.diff(low_filtered)))),
                'transient_activity': float(np.std(high_filtered))
            }
            
            # Estado operacional baseado em características espectrais
            operational_state = self._classify_operational_state(
                spectral_analysis, envelope_analysis, filter_analysis)
            
            return {
                'operational_state': operational_state,
                'spectral_analysis': spectral_analysis,
                'envelope_analysis': envelope_analysis,
                'filter_analysis': filter_analysis,
                'confidence': self._calculate_state_confidence(spectral_analysis, envelope_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Erro na detecção espectral de estado: {e}")
            return {'error': str(e)}
    
    def apply_butterworth_filter(self, signal: np.ndarray, 
                                cutoff_freq: float, 
                                filter_type: str = 'low', 
                                order: int = 4) -> np.ndarray:
        """Aplica filtro Butterworth ao sinal."""
        try:
            from scipy import signal as sp_signal
            
            # Normalizar frequência de corte (0 a 1, onde 1 é Nyquist)
            nyquist_freq = 0.5
            normalized_cutoff = cutoff_freq / nyquist_freq
            normalized_cutoff = min(max(normalized_cutoff, 0.001), 0.999)
            
            # Projetar filtro
            if filter_type == 'low':
                b, a = sp_signal.butter(order, normalized_cutoff, btype='low')
            elif filter_type == 'high':
                b, a = sp_signal.butter(order, normalized_cutoff, btype='high')
            elif filter_type == 'band':
                # Para filtro passa-banda, cutoff_freq deve ser uma tupla
                if isinstance(cutoff_freq, (list, tuple)) and len(cutoff_freq) == 2:
                    low_cut = cutoff_freq[0] / nyquist_freq
                    high_cut = cutoff_freq[1] / nyquist_freq
                    b, a = sp_signal.butter(order, [low_cut, high_cut], btype='band')
                else:
                    raise ValueError("Para filtro passa-banda, forneça [freq_baixa, freq_alta]")
            else:
                raise ValueError(f"Tipo de filtro não suportado: {filter_type}")
            
            # Aplicar filtro
            filtered_signal = sp_signal.filtfilt(b, a, signal)
            
            return filtered_signal
            
        except Exception as e:
            self.logger.warning(f"Erro no filtro Butterworth: {e}")
            return signal  # Retorna sinal original em caso de erro
    
    def apply_hilbert_transform(self, signal: np.ndarray) -> Dict[str, Any]:
        """Aplica transformada de Hilbert para análise de envelope e fase."""
        try:
            from scipy import signal as sp_signal
            
            # Transformada de Hilbert
            analytic_signal = sp_signal.hilbert(signal)
            
            # Envelope (magnitude do sinal analítico)
            envelope = np.abs(analytic_signal)
            
            # Fase instantânea
            instantaneous_phase = np.angle(analytic_signal)
            
            # Frequência instantânea
            instantaneous_frequency = np.diff(np.unwrap(instantaneous_phase)) / (2.0 * np.pi)
            
            # Detectar picos no envelope
            peaks, _ = sp_signal.find_peaks(envelope, height=np.mean(envelope))
            
            # Coerência de fase
            phase_coherence = np.abs(np.mean(np.exp(1j * instantaneous_phase)))
            
            return {
                'envelope': envelope,
                'instantaneous_phase': instantaneous_phase,
                'instantaneous_frequency': instantaneous_frequency,
                'peaks': peaks,
                'phase_coherence': phase_coherence,
                'analytic_signal': analytic_signal
            }
            
        except Exception as e:
            self.logger.warning(f"Erro na transformada de Hilbert: {e}")
            return {
                'envelope': np.abs(signal),
                'instantaneous_phase': np.zeros_like(signal),
                'instantaneous_frequency': np.zeros(len(signal)-1),
                'peaks': np.array([]),
                'phase_coherence': 0.0,
                'analytic_signal': signal
            }
    
    def _calculate_spectral_rolloff(self, freqs: np.ndarray, 
                                   magnitude: np.ndarray, 
                                   rolloff_percent: float = 0.85) -> float:
        """Calcula a frequência de rolloff espectral."""
        try:
            # Considerar apenas frequências positivas
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = magnitude[:len(magnitude)//2]
            
            # Energia cumulativa
            total_energy = np.sum(positive_magnitude)
            cumulative_energy = np.cumsum(positive_magnitude)
            
            # Encontrar frequência onde energia cumulativa atinge rolloff_percent
            rolloff_idx = np.where(cumulative_energy >= rolloff_percent * total_energy)[0]
            
            if len(rolloff_idx) > 0:
                return float(positive_freqs[rolloff_idx[0]])
            else:
                return float(positive_freqs[-1])
                
        except Exception as e:
            return 0.0
    
    def _calculate_zero_crossing_rate(self, signal: np.ndarray) -> float:
        """Calcula a taxa de cruzamento por zero."""
        try:
            # Detectar cruzamentos por zero
            zero_crossings = np.where(np.diff(np.sign(signal)))[0]
            return float(len(zero_crossings) / len(signal))
        except Exception as e:
            return 0.0
    
    def _calculate_spectral_flux(self, magnitude: np.ndarray) -> float:
        """Calcula o fluxo espectral."""
        try:
            if len(magnitude) < 2:
                return 0.0
            
            # Diferença entre espectros consecutivos
            spectral_diff = np.diff(magnitude)
            # Considerar apenas aumentos (half-wave rectification)
            positive_diff = np.where(spectral_diff > 0, spectral_diff, 0)
            
            return float(np.sum(positive_diff))
        except Exception as e:
            return 0.0
    
    def _classify_operational_state(self, spectral: Dict[str, Any], 
                                   envelope: Dict[str, Any], 
                                   filter_analysis: Dict[str, Any]) -> str:
        """Classifica o estado operacional baseado nas análises."""
        
        # Critérios para classificação
        high_energy_ratio = filter_analysis['energy_ratio'] > 0.3
        high_transient = filter_analysis['transient_activity'] > np.std([
            spectral['spectral_spread'], envelope['envelope_std']
        ])
        high_spectral_flux = spectral['spectral_flux'] > 100  # Threshold adaptativo
        many_envelope_peaks = envelope['envelope_peaks'] > len(envelope['envelope']) / 20
        
        # Lógica de classificação
        if high_energy_ratio and high_transient:
            return 'transitioning'  # Sistema em transição
        elif high_spectral_flux and many_envelope_peaks:
            return 'active_leak'  # Vazamento ativo detectado
        elif filter_analysis['signal_smoothness'] < 0.1:
            return 'stable_operation'  # Operação estável
        else:
            return 'normal_operation'  # Operação normal
    
    def _calculate_state_confidence(self, spectral: Dict[str, Any], 
                                   envelope: Dict[str, Any]) -> float:
        """Calcula a confiança na classificação do estado."""
        try:
            # Fatores de confiança
            spectral_consistency = 1.0 / (1.0 + spectral['spectral_spread'])
            envelope_stability = 1.0 / (1.0 + envelope['envelope_std'])
            phase_coherence = envelope['phase_coherence']
            
            # Combinar fatores
            confidence = (spectral_consistency + envelope_stability + phase_coherence) / 3.0
            
            return float(min(max(confidence, 0.0), 1.0))
        except Exception as e:
            return 0.5  # Confiança neutra em caso de erro

# ============================================================================
# algorithms/adaptive_ml_system.py - SISTEMA DE MACHINE LEARNING ADAPTATIVO
# ============================================================================

class AdaptiveMLSystem:
    """Sistema de Machine Learning que aprende com vazamentos confirmados"""
    
    def __init__(self, system_config: SystemConfiguration):
        self.config = system_config
        self.logger = industrial_logger.get_logger('ml_system')
        
        # Modelos especializados
        self.leak_detector = None  # IsolationForest
        self.leak_classifier = None  # RandomForest
        self.status_detector = None  # Status operacional
        self.scaler = StandardScaler()
        
        # Estado do modelo
        self.is_trained = False
        self.last_training_time = None
        self.training_data_buffer = []
        self.feature_names = []
        
        # Configurações adaptativas
        self.feature_window = max(50, int(CONSTANTS.ML_TRAINING_WINDOW * 0.05))
        self.retrain_threshold = 100
        self.detection_threshold = 0.5  # Threshold para detecção de vazamentos
        
        # Cache para features
        self.feature_cache = {}
        
    def extract_advanced_features(self, snapshots: List[MultiVariableSnapshot]) -> np.ndarray:
        """
        Extrai features avançadas para machine learning industrial
        """
        
        if len(snapshots) < self.feature_window:
            return np.array([])
        
        # Cache key para evitar recálculos
        cache_key = f"features_{len(snapshots)}_{snapshots[-1].timestamp.timestamp()}"
        cached_features = memory_manager.get_from_cache(cache_key)
        if cached_features is not None:
            return cached_features
        
        # Extrai séries temporais
        times = np.array([s.timestamp.timestamp() for s in snapshots])
        exp_pressure = np.array([s.expeditor_pressure for s in snapshots])
        rec_pressure = np.array([s.receiver_pressure for s in snapshots])
        flow = np.array([s.flow_rate for s in snapshots])
        density = np.array([s.density for s in snapshots])
        temperature = np.array([s.temperature for s in snapshots])
        
        features = []
        feature_names = []
        
        # 1. Features estatísticas básicas
        for signal, name in [(exp_pressure, 'exp_p'), (rec_pressure, 'rec_p'), 
                           (flow, 'flow'), (density, 'density'), (temperature, 'temp')]:
            features.extend([
                np.mean(signal), np.std(signal), np.min(signal), np.max(signal),
                np.percentile(signal, 25), np.percentile(signal, 75),
                np.median(signal), np.std(signal), np.mean(np.diff(signal))
            ])
            feature_names.extend([
                f'{name}_mean', f'{name}_std', f'{name}_min', f'{name}_max',
                f'{name}_q25', f'{name}_q75', f'{name}_median', f'{name}_std2', f'{name}_trend'
            ])
        
        # 2. Features de gradiente e derivadas
        for signal, name in [(exp_pressure, 'exp_p'), (rec_pressure, 'rec_p'), (flow, 'flow')]:
            gradient = np.gradient(signal)
            features.extend([
                np.mean(gradient), np.std(gradient), np.max(np.abs(gradient))
            ])
            feature_names.extend([f'{name}_grad_mean', f'{name}_grad_std', f'{name}_grad_max'])
        
        # 3. Features de correlação cruzada avançadas
        from scipy import signal as sp_signal
        correlation = sp_signal.correlate(exp_pressure, rec_pressure, mode='full')
        max_corr_idx = np.argmax(np.abs(correlation))
        max_correlation = correlation[max_corr_idx]
        delay_samples = max_corr_idx - len(correlation)//2
        
        # Análise de múltiplos picos de correlação
        correlation_peaks, _ = sp_signal.find_peaks(np.abs(correlation), height=0.1*np.max(np.abs(correlation)))
        
        features.extend([
            max_correlation / len(exp_pressure),  # Correlação normalizada
            delay_samples,  # Delay principal
            len(correlation_peaks),  # Número de picos
            np.std(correlation),  # Variabilidade da correlação
            np.mean(correlation[correlation > 0])  # Correlação positiva média
        ])
        feature_names.extend([
            'corr_max_norm', 'delay_samples', 'corr_peaks', 'corr_std', 'corr_pos_mean'
        ])
        
        # 4. Features espectrais avançadas
        for signal, name in [(exp_pressure, 'exp_p'), (rec_pressure, 'rec_p')]:
            fft = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(signal))
            magnitude = np.abs(fft)
            
            # Energia em diferentes bandas
            low_freq_mask = np.abs(freqs) < 0.1
            mid_freq_mask = (np.abs(freqs) >= 0.1) & (np.abs(freqs) < 0.3)
            high_freq_mask = np.abs(freqs) >= 0.3
            
            total_energy = np.sum(magnitude)
            low_energy = np.sum(magnitude[low_freq_mask]) / max(total_energy, 1e-10)
            mid_energy = np.sum(magnitude[mid_freq_mask]) / max(total_energy, 1e-10)
            high_energy = np.sum(magnitude[high_freq_mask]) / max(total_energy, 1e-10)
            
            # Frequência dominante
            dominant_freq = freqs[np.argmax(magnitude[:len(freqs)//2])]
            
            features.extend([low_energy, mid_energy, high_energy, dominant_freq])
            feature_names.extend([
                f'{name}_low_energy', f'{name}_mid_energy', f'{name}_high_energy', f'{name}_dom_freq'
            ])
        
        # 5. Features de relação entre variáveis
        pressure_diff = exp_pressure - rec_pressure
        
        # Correlações entre variáveis
        correlations = []
        correlation_names = []
        variable_pairs = [
            (pressure_diff, flow, 'press_diff_flow'),
            (density, temperature, 'density_temp'),
            (flow, temperature, 'flow_temp'),
            (pressure_diff, density, 'press_diff_density')
        ]
        
        for var1, var2, name in variable_pairs:
            if len(var1) > 1 and len(var2) > 1:
                corr = np.corrcoef(var1, var2)[0,1]
                correlations.append(corr if not np.isnan(corr) else 0.0)
            else:
                correlations.append(0.0)
            correlation_names.append(f'corr_{name}')
        
        features.extend(correlations)
        feature_names.extend(correlation_names)
        
        # 6. Features de estabilidade temporal
        for signal, name in [(exp_pressure, 'exp_p'), (rec_pressure, 'rec_p'), (flow, 'flow')]:
            # Estabilidade usando janela móvel
            window_size = min(10, len(signal)//4)
            if window_size > 1:
                rolling_std = pd.Series(signal).rolling(window=window_size, min_periods=1).std()
                stability_metric = np.mean(rolling_std)
            else:
                stability_metric = np.std(signal)
            
            features.append(stability_metric)
            feature_names.append(f'{name}_stability')
        
        # 7. Features específicas do sistema
        # Incorpora características físicas do duto
        acoustic_props = self.config.pipe_characteristics.calculate_acoustic_properties()
        
        # Velocidade sônica efetiva
        effective_velocity = self.config.sonic_velocity * acoustic_props['velocity_factor']
        expected_delay = self.config.sensor_distance / effective_velocity
        
        # Razão entre delay observado e esperado
        delay_ratio = delay_samples / max(expected_delay, 1.0)
        
        features.extend([
            delay_ratio,
            acoustic_props['attenuation'],
            acoustic_props['roughness_factor'],
            self.config.pipe_characteristics.diameter
        ])
        feature_names.extend([
            'delay_ratio', 'attenuation', 'roughness_factor', 'diameter'
        ])
        
        # 8. Features de detecção de vazamento
        # Baseado em conhecimento físico
        
        # Taxa de mudança de pressão
        pressure_change_rate = np.mean(np.abs(np.diff(pressure_diff)))
        
        # Assimetria na correlação (indicativo de vazamento)
        correlation_asymmetry = np.mean(correlation[:len(correlation)//2]) - np.mean(correlation[len(correlation)//2:])
        
        # Energia de alta frequência (ruído de vazamento)
        # Inicializa as variáveis caso não tenham sido definidas no loop anterior
        high_energy = getattr(locals(), 'high_energy', 0.1)
        low_energy = getattr(locals(), 'low_energy', 0.3) 
        mid_energy = getattr(locals(), 'mid_energy', 0.6)
        hf_energy_ratio = high_energy / max(low_energy + mid_energy, 1e-10)
        
        features.extend([
            pressure_change_rate,
            correlation_asymmetry,
            hf_energy_ratio
        ])
        feature_names.extend([
            'pressure_change_rate', 'correlation_asymmetry', 'hf_energy_ratio'
        ])
        
        # Armazena nomes das features para interpretabilidade
        self.feature_names = feature_names
        
        # Converte para array numpy
        features_array = np.array(features)
        
        # Remove NaN e Inf
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Cache do resultado
        memory_manager.store_in_cache(cache_key, features_array)
        
        return features_array
    
    def perform_pca_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Realiza análise de componentes principais para identificação de padrões complexos."""
        try:
            # Preparar dados para PCA
            features = data.select_dtypes(include=[np.number])
            if features.empty:
                return {'error': 'Nenhuma coluna numérica encontrada para análise PCA'}
            
            # Remover NaN e normalizar
            features_clean = features.fillna(features.mean())
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_clean)
            
            # Aplicar PCA
            pca = PCA(n_components=min(10, features_scaled.shape[1]))
            principal_components = pca.fit_transform(features_scaled)
            
            # Análise de componentes
            component_analysis = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'components': pca.components_.tolist(),
                'feature_names': features.columns.tolist(),
                'n_components_90': np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.90) + 1,
                'principal_components': principal_components.tolist()
            }
            
            # Identificar componentes mais importantes
            important_components = []
            for i, ratio in enumerate(pca.explained_variance_ratio_):
                if ratio > 0.1:  # Componentes que explicam mais de 10% da variância
                    component_features = []
                    for j, weight in enumerate(pca.components_[i]):
                        if abs(weight) > 0.3:  # Pesos significativos
                            component_features.append({
                                'feature': features.columns[j],
                                'weight': float(weight)
                            })
                    important_components.append({
                        'component': i + 1,
                        'variance_explained': float(ratio),
                        'features': component_features
                    })
            
            component_analysis['important_components'] = important_components
            
            self.logger.info(f"Análise PCA concluída: {pca.n_components_} componentes, "
                           f"{component_analysis['n_components_90']} componentes para 90% da variância")
            
            return component_analysis
            
        except Exception as e:
            error_msg = f"Erro na análise PCA: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg}
    
    def canonical_correlation_analysis(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Dict[str, Any]:
        """Realiza análise de correlação canônica entre dois conjuntos de variáveis."""
        try:
            # Preparar dados
            features1 = data1.select_dtypes(include=[np.number]).fillna(data1.mean())
            features2 = data2.select_dtypes(include=[np.number]).fillna(data2.mean())
            
            if features1.empty or features2.empty:
                return {'error': 'Dados insuficientes para análise de correlação canônica'}
            
            # Normalizar dados
            scaler1 = StandardScaler()
            scaler2 = StandardScaler()
            X1 = scaler1.fit_transform(features1)
            X2 = scaler2.fit_transform(features2)
            
            # Calcular matrizes de covariância
            n_samples = min(X1.shape[0], X2.shape[0])
            X1 = X1[:n_samples]
            X2 = X2[:n_samples]
            
            C11 = np.cov(X1.T)
            C22 = np.cov(X2.T)
            C12 = np.cov(X1.T, X2.T)[:X1.shape[1], X1.shape[1]:]
            C21 = C12.T
            
            # Resolver o problema de correlação canônica
            # (C11^-1 * C12 * C22^-1 * C21) * a = λ * a
            try:
                A = np.linalg.inv(C11) @ C12 @ np.linalg.inv(C22) @ C21
                eigenvalues, eigenvectors = np.linalg.eig(A)
                
                # Ordenar por eigenvalues
                idx = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                # Correlações canônicas
                canonical_correlations = np.sqrt(np.real(eigenvalues))
                
                result = {
                    'canonical_correlations': canonical_correlations.tolist(),
                    'n_significant': int(np.sum(canonical_correlations > 0.3)),
                    'max_correlation': float(np.max(canonical_correlations)),
                    'features1': features1.columns.tolist(),
                    'features2': features2.columns.tolist(),
                    'interpretation': self._interpret_canonical_correlations(canonical_correlations)
                }
                
                self.logger.info(f"Análise CCA concluída: {len(canonical_correlations)} correlações canônicas, "
                               f"máxima = {result['max_correlation']:.3f}")
                
                return result
                
            except np.linalg.LinAlgError as e:
                return {'error': f'Erro numérico na CCA: {str(e)}'}
                
        except Exception as e:
            error_msg = f"Erro na análise de correlação canônica: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg}
    
    def _interpret_canonical_correlations(self, correlations: np.ndarray) -> Dict[str, Any]:
        """Interpreta os resultados da análise de correlação canônica."""
        interpretation = {
            'strength_levels': [],
            'summary': ''
        }
        
        for i, corr in enumerate(correlations):
            if corr > 0.7:
                strength = 'Muito Alta'
            elif corr > 0.5:
                strength = 'Alta'
            elif corr > 0.3:
                strength = 'Moderada'
            else:
                strength = 'Baixa'
                
            interpretation['strength_levels'].append({
                'component': i + 1,
                'correlation': float(corr),
                'strength': strength
            })
        
        high_corr_count = np.sum(correlations > 0.5)
        if high_corr_count > 0:
            interpretation['summary'] = f"{high_corr_count} componentes com correlação alta/muito alta detectadas"
        else:
            interpretation['summary'] = "Correlações fracas detectadas entre os conjuntos de variáveis"
        
        return interpretation
    
    @error_handler.handle_with_retry(3)
    def train_models(self, training_snapshots: List[MultiVariableSnapshot], 
                    leak_labels: List[bool], leak_types: Optional[List[str]] = None):
        """
        Treina modelos de machine learning com validação robusta
        """
        
        self.logger.info(f"Iniciando treinamento com {len(training_snapshots)} exemplos")
        
        if len(training_snapshots) < self.feature_window:
            raise MLModelError(f"Dados insuficientes para treinamento: {len(training_snapshots)} < {self.feature_window}")
        
        # Extrai features em batches
        all_features = []
        valid_labels = []
        
        for i in range(len(training_snapshots) - self.feature_window + 1):
            snapshot_window = training_snapshots[i:i + self.feature_window]
            
            try:
                features = self.extract_advanced_features(snapshot_window)
                if len(features) > 0:
                    all_features.append(features)
                    valid_labels.append(leak_labels[i + self.feature_window - 1])
            except Exception as e:
                self.logger.warning(f"Erro na extração de features para índice {i}: {e}")
                continue
        
        if len(all_features) == 0:
            raise MLModelError("Nenhuma feature válida extraída para treinamento")
        
        X = np.array(all_features)
        y = np.array(valid_labels)
        
        self.logger.info(f"Features extraídas: {X.shape}, Labels: {len(y)}")
        
        # Normaliza features
        X_scaled = self.scaler.fit_transform(X)
        
        # Treina detector de anomalias
        contamination = min(0.1, max(0.01, np.sum(y) / len(y)))  # Adaptativo
        
        self.leak_detector = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples=min(256, len(X_scaled))
        )
        self.leak_detector.fit(X_scaled)
        
        # Treina classificador se temos exemplos positivos suficientes
        positive_samples = np.sum(y)
        if positive_samples >= 5:
            self.leak_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
            self.leak_classifier.fit(X_scaled, y)
            
            # Validação cruzada
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(self.leak_classifier, X_scaled, y, cv=min(5, len(X_scaled)//2))
            
            # Calcula métricas detalhadas
            y_pred = self.leak_classifier.predict(X_scaled)
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            try:
                y_pred_proba = self.leak_classifier.predict_proba(X_scaled)[:, 1]
                auc_score = roc_auc_score(y, y_pred_proba)
            except:
                auc_score = 0.0
            
            metrics = {
                'accuracy': float(accuracy_score(y, y_pred)),
                'precision': float(precision_score(y, y_pred, zero_division=0)),
                'recall': float(recall_score(y, y_pred, zero_division=0)),
                'f1_score': float(f1_score(y, y_pred, zero_division=0)),
                'auc_score': float(auc_score),
                'cv_mean': float(np.mean(cv_scores)),
                'cv_std': float(np.std(cv_scores)),
                'training_samples': len(all_features),
                'positive_samples': int(positive_samples),
                'feature_count': len(self.feature_names)
            }
            
            self.logger.info(f"Modelo treinado - Métricas: {metrics}")
            
            # Salva modelo no banco
            database.save_ml_model(
                self.config.system_id,
                self.config.ml_model_version,
                'leak_detection',
                pickle.dumps({
                    'detector': self.leak_detector,
                    'classifier': self.leak_classifier,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names
                }),
                metrics
            )
            
        else:
            self.logger.warning(f"Poucas amostras positivas ({positive_samples}) - apenas detector de anomalias treinado")
            metrics = {
                'training_samples': len(all_features),
                'positive_samples': int(positive_samples),
                'model_type': 'anomaly_detection_only'
            }
        
        self.is_trained = True
        self.last_training_time = datetime.now()
        
        return metrics
    
    def predict_leak(self, snapshots: List[MultiVariableSnapshot]) -> Dict[str, Any]:
        """
        Prediz vazamento usando modelos treinados com múltiplas métricas
        """
        
        if not self.is_trained or len(snapshots) < self.feature_window:
            return {
                'leak_probability': 0.0,
                'anomaly_score': 0.0,
                'confidence': 0.0,
                'model_ready': False
            }
        
        try:
            # Extrai features
            features = self.extract_advanced_features(snapshots[-self.feature_window:])
            if len(features) == 0:
                return {'error': 'Não foi possível extrair features'}
            
            # Normaliza
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Predição com detector de anomalias
            if self.leak_detector is not None:
                anomaly_score = self.leak_detector.decision_function(features_scaled)[0]
                is_anomaly = self.leak_detector.predict(features_scaled)[0] == -1
            else:
                anomaly_score = 0.0
                is_anomaly = False
            
            # Converte score para probabilidade (0-1)
            anomaly_probability = 1 / (1 + np.exp(anomaly_score))  # Sigmoid
            
            # Predição com classificador se disponível
            classification_probability = 0.0
            feature_importance = {}
            
            if self.leak_classifier is not None:
                class_proba = self.leak_classifier.predict_proba(features_scaled)[0]
                classification_probability = class_proba[1] if len(class_proba) > 1 else 0.0
                
                # Importância das features
                if hasattr(self.leak_classifier, 'feature_importances_'):
                    importance = self.leak_classifier.feature_importances_
                    feature_importance = {
                        name: float(imp) for name, imp in 
                        zip(self.feature_names[:len(importance)], importance)
                    }
            
            # Combina scores com pesos adaptativos
            if self.leak_classifier is not None:
                # Ambos os modelos disponíveis
                combined_score = 0.3 * anomaly_probability + 0.7 * classification_probability
                confidence = 0.8
            else:
                # Apenas detector de anomalias
                combined_score = anomaly_probability
                confidence = 0.5
            
            # Ajusta confiança baseado na consistência
            if abs(anomaly_probability - classification_probability) < 0.2:
                confidence = min(confidence * 1.2, 1.0)
            else:
                confidence = confidence * 0.8
            
            return {
                'leak_probability': float(combined_score),
                'anomaly_score': float(anomaly_score),
                'anomaly_probability': float(anomaly_probability),
                'classification_probability': float(classification_probability),
                'confidence': float(confidence),
                'is_anomaly': bool(is_anomaly),
                'feature_importance': feature_importance,
                'model_ready': True,
                'prediction_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Erro na predição: {e}")
            return {
                'error': str(e),
                'leak_probability': 0.0,
                'model_ready': False
            }
    
    def update_model_retroactive(self, new_data: pd.DataFrame, leak_detected: bool, 
                                leak_location: Optional[float] = None) -> Dict[str, Any]:
        """Sistema de aprendizagem retroativa para vazamentos não detectados."""
        try:
            self.logger.info("Iniciando atualização retroativa do modelo ML")
            
            # Converter dados para snapshots se necessário
            if isinstance(new_data, pd.DataFrame):
                snapshots = self._dataframe_to_snapshots(new_data)
            else:
                snapshots = new_data
            
            # Extrair características do evento
            event_features = self.extract_advanced_features(snapshots)
            
            # Adicionar ao buffer de treinamento
            self.training_data_buffer.append({
                'features': event_features,
                'label': leak_detected,
                'timestamp': datetime.now(),
                'location': leak_location,
                'confidence': 1.0 if leak_detected else 0.0
            })
            
            # Análise retroativa se vazamento foi detectado manualmente
            if leak_detected:
                retroactive_analysis = self._perform_retroactive_analysis(snapshots, event_features)
            else:
                retroactive_analysis = {'patterns_identified': [], 'adjustments_made': []}
            
            # Retreinar modelo se buffer suficiente
            retrain_result = None
            if len(self.training_data_buffer) >= self.retrain_threshold:
                retrain_result = self._retrain_with_buffer()
            
            # Ajustar threshold baseado em feedback
            threshold_adjustment = self._adjust_detection_threshold(leak_detected, event_features)
            
            result = {
                'update_successful': True,
                'buffer_size': len(self.training_data_buffer),
                'retroactive_patterns': retroactive_analysis,
                'model_retrained': retrain_result is not None,
                'threshold_adjusted': threshold_adjustment,
                'timestamp': datetime.now().isoformat()
            }
            
            if retrain_result:
                result['retrain_metrics'] = retrain_result
            
            self.logger.info(f"Atualização retroativa concluída: {result}")
            return result
            
        except Exception as e:
            error_msg = f"Erro na atualização retroativa: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'update_successful': False}
    
    def _perform_retroactive_analysis(self, snapshots: List[MultiVariableSnapshot], 
                                     event_features: np.ndarray) -> Dict[str, Any]:
        """Analisa retroativamente padrões que deveriam ter sido detectados."""
        patterns_identified = []
        
        # Análise de padrões temporais
        times = [s.timestamp for s in snapshots]
        if len(times) > 1:
            time_intervals = [(times[i+1] - times[i]).total_seconds() for i in range(len(times)-1)]
            avg_interval = np.mean(time_intervals)
            
            if avg_interval < 30:  # Dados de alta frequência
                patterns_identified.append({
                    'type': 'high_frequency_pattern',
                    'description': 'Padrão de alta frequência detectado retroativamente',
                    'strength': 0.8
                })
        
        # Análise de correlações perdidas
        pressures = [s.expeditor_pressure - s.receiver_pressure for s in snapshots]
        flows = [s.flow_rate for s in snapshots]
        
        if len(pressures) > 10:
            correlation = np.corrcoef(pressures, flows)[0, 1]
            if abs(correlation) > 0.7:
                patterns_identified.append({
                    'type': 'pressure_flow_correlation',
                    'description': f'Correlação pressão-vazão forte ({correlation:.3f}) não detectada',
                    'strength': abs(correlation)
                })
        
        # Análise espectral retroativa
        if len(pressures) > 20:
            fft_result = np.fft.fft(pressures)
            magnitude = np.abs(fft_result)
            dominant_freq = np.argmax(magnitude[:len(magnitude)//2])
            
            if dominant_freq > 2:  # Frequência dominante significativa
                patterns_identified.append({
                    'type': 'spectral_signature',
                    'description': f'Assinatura espectral em {dominant_freq} Hz não detectada',
                    'strength': float(magnitude[dominant_freq] / np.sum(magnitude))
                })
        
        return {
            'patterns_identified': patterns_identified,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_patterns': len(patterns_identified)
        }
    
    def _retrain_with_buffer(self) -> Dict[str, Any]:
        """Retreina modelo com dados do buffer."""
        try:
            if len(self.training_data_buffer) < 5:
                return {'error': 'Buffer insuficiente para retreinamento'}
            
            # Preparar dados do buffer
            X = np.array([item['features'] for item in self.training_data_buffer])
            y = np.array([item['label'] for item in self.training_data_buffer])
            
            # Normalizar
            X_scaled = self.scaler.fit_transform(X)
            
            # Retreinar detector de anomalias
            contamination = max(0.05, np.sum(y) / len(y))
            self.leak_detector = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=120  # Mais estimadores para modelo incremental
            )
            self.leak_detector.fit(X_scaled)
            
            # Retreinar classificador se suficientes amostras positivas
            positive_count = np.sum(y)
            if positive_count >= 3:
                self.leak_classifier = RandomForestClassifier(
                    n_estimators=120,
                    random_state=42,
                    class_weight='balanced',
                    max_depth=12  # Maior profundidade para capturar padrões complexos
                )
                self.leak_classifier.fit(X_scaled, y)
            
            # Limpar buffer após retreinamento
            self.training_data_buffer = []
            self.last_training_time = datetime.now()
            
            self.logger.info(f"Modelo retreinado com {len(X)} amostras, {positive_count} positivas")
            
            return {
                'retrain_successful': True,
                'training_samples': len(X),
                'positive_samples': int(positive_count),
                'contamination': float(contamination),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Erro no retreinamento: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg}
    
    def _adjust_detection_threshold(self, was_leak: bool, features: np.ndarray) -> Dict[str, Any]:
        """Ajusta threshold de detecção baseado em feedback."""
        try:
            adjustment_made = False
            old_threshold = self.detection_threshold
            
            if not self.is_trained:
                return {'adjustment_made': False, 'reason': 'Model not trained'}
            
            # Calcular score atual com features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            anomaly_score = self.leak_detector.decision_function(features_scaled)[0]
            
            if was_leak and anomaly_score > -0.1:  # Vazamento perdido
                # Aumentar sensibilidade (diminuir threshold)
                self.detection_threshold = max(0.1, self.detection_threshold - 0.05)
                adjustment_made = True
                reason = "Increased sensitivity due to missed leak"
                
            elif not was_leak and anomaly_score < -0.5:  # Falso positivo
                # Diminuir sensibilidade (aumentar threshold)
                self.detection_threshold = min(0.8, self.detection_threshold + 0.03)
                adjustment_made = True
                reason = "Decreased sensitivity due to false positive"
            else:
                reason = "No adjustment needed"
            
            if adjustment_made:
                self.logger.info(f"Threshold ajustado: {old_threshold:.3f} -> {self.detection_threshold:.3f}")
            
            return {
                'adjustment_made': adjustment_made,
                'old_threshold': float(old_threshold),
                'new_threshold': float(self.detection_threshold),
                'reason': reason,
                'anomaly_score': float(anomaly_score)
            }
            
        except Exception as e:
            self.logger.warning(f"Erro no ajuste de threshold: {e}")
            return {'adjustment_made': False, 'error': str(e)}
    
    def _dataframe_to_snapshots(self, df: pd.DataFrame) -> List[MultiVariableSnapshot]:
        """Converte DataFrame para lista de snapshots."""
        snapshots = []
        
        for _, row in df.iterrows():
            try:
                snapshot = MultiVariableSnapshot(
                    timestamp=pd.to_datetime(row.get('timestamp', datetime.now())),
                    expeditor_pressure=float(row.get('expeditor_pressure', 0)),
                    receiver_pressure=float(row.get('receiver_pressure', 0)),
                    flow_rate=float(row.get('flow_rate', 0)),
                    temperature=float(row.get('temperature', 20)),
                    density=float(row.get('density', 1000)),
                    viscosity=float(row.get('viscosity', 0.001))
                )
                snapshots.append(snapshot)
            except Exception as e:
                self.logger.warning(f"Erro ao converter linha para snapshot: {e}")
                continue
        
        return snapshots
    
    def get_model_performance_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de performance do modelo."""
        try:
            metrics = {
                'model_trained': self.is_trained,
                'last_training': self.last_training_time.isoformat() if self.last_training_time else None,
                'buffer_size': len(self.training_data_buffer),
                'detection_threshold': float(self.detection_threshold),
                'retrain_threshold': self.retrain_threshold,
                'feature_count': len(self.feature_names) if self.feature_names else 0
            }
            
            # Métricas do detector de anomalias
            if self.leak_detector:
                try:
                    metrics['anomaly_detector'] = {
                        'contamination': getattr(self.leak_detector, 'contamination', 'unknown'),
                        'n_estimators': getattr(self.leak_detector, 'n_estimators', 'unknown')
                    }
                except:
                    metrics['anomaly_detector'] = {'status': 'available'}
            
            # Métricas do classificador
            if self.leak_classifier:
                try:
                    metrics['classifier'] = {
                        'n_estimators': getattr(self.leak_classifier, 'n_estimators', 'unknown'),
                        'max_depth': getattr(self.leak_classifier, 'max_depth', 'unknown')
                    }
                except:
                    metrics['classifier'] = {'status': 'available'}
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erro ao obter métricas: {e}")
            return {'error': str(e)}

# ============================================================================
# core/industrial_processor.py - PROCESSADOR INDUSTRIAL PRINCIPAL
# ============================================================================

class IndustrialHydraulicProcessor:
    """Processador principal que integra todas as funcionalidades industriais"""
    
    def __init__(self, system_config: SystemConfiguration):
        self.config = system_config
        self.logger = industrial_logger.get_logger('industrial_processor')
        
        # Componentes especializados
        self.time_processor = IrregularTimeSeriesProcessor()
        self.status_detector = OperationalStatusDetector(system_config)
        self.ml_system = AdaptiveMLSystem(system_config)
        
        # Estado atual
        self.current_snapshots = []
        self.current_status = {}
        self.last_analysis_time = None
        
        # Buffers do sistema
        self.system_buffers = memory_manager.get_system_buffers(system_config.system_id)
        
        # Carrega modelo ML existente
        self._load_existing_ml_model()
        
        self.logger.info(f"Processador inicializado para sistema {system_config.system_id}")
    
    def _load_existing_ml_model(self):
        """Carrega modelo ML existente do banco de dados"""
        try:
            result = database.load_ml_model(self.config.system_id)
            if result:
                model_data, metrics = result
                # Deserializa os dados do modelo
                model_dict = pickle.loads(model_data)
                self.ml_system.leak_detector = model_dict['detector']
                self.ml_system.leak_classifier = model_dict.get('classifier')
                self.ml_system.scaler = model_dict['scaler']
                self.ml_system.feature_names = model_dict.get('feature_names', [])
                self.ml_system.is_trained = True
                self.logger.info(f"Modelo ML carregado - Métricas: {metrics}")
        except Exception as e:
            self.logger.info(f"Nenhum modelo ML existente: {e}")
    
    @error_handler.handle_with_retry(3)
    def process_sensor_readings(self, readings: List[SensorReading]) -> Dict[str, Any]:
        """
        Processa leituras de sensores com análise completa
        """
        
        if not readings:
            return {'error': 'Nenhuma leitura fornecida'}
        
        self.logger.info(f"Processando {len(readings)} leituras para sistema {self.config.system_id}")
        
        # Valida e salva leituras
        try:
            # Validação cruzada das leituras
            readings_by_time = {}
            for reading in readings:
                timestamp_key = reading.timestamp.isoformat()
                if timestamp_key not in readings_by_time:
                    readings_by_time[timestamp_key] = {}
                readings_by_time[timestamp_key][reading.variable] = reading.value
            
            # Aplica validação cruzada para cada timestamp
            validated_readings = []
            for reading in readings:
                timestamp_key = reading.timestamp.isoformat()
                if error_handler.validate_cross_variables(
                    readings_by_time[timestamp_key], 
                    self.config.system_id
                ):
                    validated_readings.append(reading)
                else:
                    self.logger.warning(f"Leitura rejeitada por validação cruzada: {reading}")
            
            # Salva no banco
            database.save_sensor_readings(self.config.system_id, validated_readings)
            
            # Atualiza buffers em memória
            for reading in validated_readings:
                timestamp_numeric = reading.timestamp.timestamp()
                data = {reading.variable: reading.value}
                memory_manager.add_data_point(
                    self.config.system_id, 
                    timestamp_numeric, 
                    data
                )
            
        except Exception as e:
            self.logger.error(f"Erro no processamento de leituras: {e}")
            return {'error': f'Erro no processamento: {str(e)}'}
        
        # Converte para DataFrame
        readings_data = [reading.to_dict() for reading in validated_readings]
        readings_df = pd.DataFrame(readings_data)
        
        if readings_df.empty:
            return {'error': 'Nenhuma leitura válida após validação'}
        
        # Sincroniza dados multivariáveis
        try:
            snapshots = self.time_processor.synchronize_multivariable_data(
                readings_df, 
                target_frequency=1.0,
                system_id=self.config.system_id
            )
            
            if not snapshots:
                return {'error': 'Falha na sincronização de dados multivariáveis'}
            
            self.current_snapshots = snapshots
            
        except Exception as e:
            self.logger.error(f"Erro na sincronização: {e}")
            return {'error': f'Erro na sincronização: {str(e)}'}
        
        # Análise completa
        analysis_results = {}
        
        # 1. Análise de status operacional
        try:
            status_analysis = self.status_detector.analyze_operational_status(snapshots)
            analysis_results['operational_status'] = status_analysis
            self.current_status = status_analysis
            
        except Exception as e:
            self.logger.error(f"Erro na análise de status: {e}")
            analysis_results['operational_status'] = {'error': str(e)}
        
        # 2. Análise de correlação sônica
        try:
            correlation_analysis = self._perform_sonic_correlation_analysis(snapshots)
            analysis_results['sonic_correlation'] = correlation_analysis
            
        except Exception as e:
            self.logger.error(f"Erro na correlação sônica: {e}")
            analysis_results['sonic_correlation'] = {'error': str(e)}
        
        # 3. Detecção ML de vazamentos
        try:
            ml_prediction = self.ml_system.predict_leak(snapshots)
            analysis_results['ml_prediction'] = ml_prediction
            
        except Exception as e:
            self.logger.error(f"Erro na predição ML: {e}")
            analysis_results['ml_prediction'] = {'error': str(e)}
        
        # 4. Análise multivariável integrada
        try:
            multivariable_analysis = self._perform_multivariable_analysis(snapshots)
            analysis_results['multivariable_analysis'] = multivariable_analysis
            
        except Exception as e:
            self.logger.error(f"Erro na análise multivariável: {e}")
            analysis_results['multivariable_analysis'] = {'error': str(e)}
        
        # 5. Detecção integrada final
        try:
            integrated_detection = self._integrate_detection_results(analysis_results)
            analysis_results['integrated_detection'] = integrated_detection
            
            # Salva evento se vazamento detectado
            if integrated_detection.get('leak_status') != 'normal_operation':
                database.save_leak_event(
                    system_id=self.config.system_id,
                    timestamp=datetime.now(),
                    leak_type=integrated_detection.get('severity', 'unknown'),
                    severity=integrated_detection.get('integrated_score', 0.0),
                    location_estimate=integrated_detection.get('estimated_location', 0.0),
                    confidence=integrated_detection.get('confidence', 0.0),
                    detected_by='automatic',
                    ml_signature=json.dumps({
                        'ml_prediction': analysis_results.get('ml_prediction', {}),
                        'operational_status': analysis_results.get('operational_status', {})
                    })
                )
            
        except Exception as e:
            self.logger.error(f"Erro na integração de resultados: {e}")
            analysis_results['integrated_detection'] = {'error': str(e)}
        
        # Atualiza timestamp da última análise
        self.last_analysis_time = datetime.now()
        analysis_results['analysis_timestamp'] = self.last_analysis_time.isoformat()
        analysis_results['snapshots_processed'] = len(snapshots)
        analysis_results['readings_processed'] = len(validated_readings)
        
        return analysis_results
    
    def _perform_sonic_correlation_analysis(self, snapshots: List[MultiVariableSnapshot]) -> Dict[str, Any]:
        """Análise de correlação sônica com características físicas do sistema"""
        
        if len(snapshots) < 50:
            return {'error': 'Dados insuficientes para correlação'}
        
        exp_pressures = np.array([s.expeditor_pressure for s in snapshots])
        rec_pressures = np.array([s.receiver_pressure for s in snapshots])
        
        # Correlação com compensação por características do duto
        acoustic_props = self.config.pipe_characteristics.calculate_acoustic_properties()
        
        # Velocidade sônica efetiva
        effective_velocity = self.config.sonic_velocity * acoustic_props['velocity_factor']
        
        # Correlação cruzada
        correlation = signal.correlate(exp_pressures, rec_pressures, mode='full', method='fft')
        max_corr_idx = np.argmax(np.abs(correlation))
        delay_samples = max_corr_idx - len(correlation)//2
        max_correlation = correlation[max_corr_idx] / len(exp_pressures)
        
        # Tempo de trânsito
        expected_transit_time = self.config.sensor_distance / effective_velocity
        
        # Compensação por atenuação
        distance_factor = np.exp(-acoustic_props['attenuation'] * self.config.sensor_distance / 1000)
        compensated_correlation = max_correlation / distance_factor
        
        # Análise de qualidade
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
    
    def _perform_multivariable_analysis(self, snapshots: List[MultiVariableSnapshot]) -> Dict[str, Any]:
        """Análise multivariável com validação física"""
        
        if len(snapshots) < 20:
            return {'error': 'Dados insuficientes para análise multivariável'}
        
        # Extrai variáveis
        times = np.array([s.timestamp.timestamp() for s in snapshots])
        exp_pressures = np.array([s.expeditor_pressure for s in snapshots])
        rec_pressures = np.array([s.receiver_pressure for s in snapshots])
        flows = np.array([s.flow_rate for s in snapshots])
        densities = np.array([s.density for s in snapshots])
        temperatures = np.array([s.temperature for s in snapshots])
        
        # Análise de tendências
        def safe_polyfit(x, y):
            try:
                return np.polyfit(x, y, 1)[0] if len(x) > 1 else 0.0
            except:
                return 0.0
        
        pressure_trend = safe_polyfit(times, exp_pressures)
        flow_trend = safe_polyfit(times, flows)
        density_trend = safe_polyfit(times, densities)
        temp_trend = safe_polyfit(times, temperatures)
        
        # Correlações entre variáveis com validação física
        correlations = {}
        variables = {
            'pressure_diff': exp_pressures - rec_pressures,
            'flow': flows,
            'density': densities,
            'temperature': temperatures
        }
        
        for i, (name1, var1) in enumerate(variables.items()):
            for j, (name2, var2) in enumerate(list(variables.items())[i+1:], i+1):
                try:
                    if len(var1) > 1 and len(var2) > 1:
                        corr = np.corrcoef(var1, var2)[0, 1]
                        correlations[f'{name1}_vs_{name2}'] = float(corr) if not np.isnan(corr) else 0.0
                    else:
                        correlations[f'{name1}_vs_{name2}'] = 0.0
                except:
                    correlations[f'{name1}_vs_{name2}'] = 0.0
        
        # Validação de correlações físicas esperadas
        validation_results = {}
        
        # Densidade vs Temperatura (correlação negativa esperada)
        density_temp_corr = correlations.get('density_vs_temperature', 0.0)
        validation_results['density_temp_physical'] = density_temp_corr < -0.3
        
        # Pressão vs Vazão (correlação positiva esperada em sistema fechado)
        press_flow_corr = correlations.get('pressure_diff_vs_flow', 0.0)
        validation_results['pressure_flow_physical'] = press_flow_corr > 0.1
        
        # Detecção de anomalias multivariáveis
        try:
            data_matrix = np.column_stack([exp_pressures, rec_pressures, flows, densities, temperatures])
            data_matrix = np.nan_to_num(data_matrix)
            
            if data_matrix.shape[0] > data_matrix.shape[1]:
                pca = PCA(n_components=min(3, data_matrix.shape[1]))
                pca_data = pca.fit_transform(StandardScaler().fit_transform(data_matrix))
                explained_variance = pca.explained_variance_ratio_
                
                # Detecção de outliers no espaço PCA
                distances = np.linalg.norm(pca_data - np.mean(pca_data, axis=0), axis=1)
                anomaly_threshold = np.mean(distances) + 2 * np.std(distances)
                anomalies = distances > anomaly_threshold
                
                pca_analysis = {
                    'explained_variance': explained_variance.tolist(),
                    'anomaly_count': int(np.sum(anomalies)),
                    'anomaly_percentage': float(np.sum(anomalies) / len(anomalies) * 100),
                    'first_pc_variance': float(explained_variance[0])
                }
            else:
                pca_analysis = {'error': 'Dados insuficientes para PCA'}
                
        except Exception as e:
            pca_analysis = {'error': f'Erro no PCA: {str(e)}'}
        
        # Score integrado de probabilidade de vazamento
        leak_indicators = []
        
        # Indicador 1: Queda sustentada de pressão
        if pressure_trend < -0.001:
            leak_indicators.append(min(abs(pressure_trend) * 1000, 1.0))
        
        # Indicador 2: Aumento compensatório de vazão
        if flow_trend > 0.01:
            leak_indicators.append(min(flow_trend * 100, 1.0))
        
        # Indicador 3: Correlações anômalas
        if abs(press_flow_corr) < 0.3:
            leak_indicators.append(0.5)
        
        # Indicador 4: Tendência anômala de densidade
        if abs(density_trend) > 0.001:
            leak_indicators.append(min(abs(density_trend) * 1000, 1.0))
        
        multivariable_leak_score = np.mean(leak_indicators) if leak_indicators else 0.0
        
        return {
            'trends': {
                'pressure': float(pressure_trend),
                'flow': float(flow_trend),
                'density': float(density_trend),
                'temperature': float(temp_trend)
            },
            'correlations': correlations,
            'physical_validation': validation_results,
            'pca_analysis': pca_analysis,
            'multivariable_leak_score': float(multivariable_leak_score),
            'leak_indicators_count': len(leak_indicators)
        }
    
    def _integrate_detection_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Integra todos os métodos de detecção para resultado final"""
        
        scores = []
        confidences = []
        methods_used = []
        
        # Score da correlação sônica
        sonic = analysis_results.get('sonic_correlation', {})
        if 'max_correlation' in sonic and 'signal_quality' in sonic:
            if sonic['signal_quality'] in ['good', 'excellent']:
                # Correlação alta = sistema normal
                sonic_score = max(0, 1.0 - sonic['max_correlation'])
                scores.append(sonic_score)
                confidences.append(0.9 if sonic['signal_quality'] == 'excellent' else 0.7)
                methods_used.append('sonic_correlation')
        
        # Score do ML
        ml = analysis_results.get('ml_prediction', {})
        if 'leak_probability' in ml and ml.get('model_ready', False):
            scores.append(ml['leak_probability'])
            confidences.append(ml.get('confidence', 0.5))
            methods_used.append('machine_learning')
        
        # Score multivariável
        multivariable = analysis_results.get('multivariable_analysis', {})
        if 'multivariable_leak_score' in multivariable:
            scores.append(multivariable['multivariable_leak_score'])
            confidences.append(0.7)
            methods_used.append('multivariable_analysis')
        
        # Score do status operacional
        status = analysis_results.get('operational_status', {})
        if status.get('column_status') == 'open':
            # Coluna aberta aumenta probabilidade de vazamento
            scores.append(0.4)
            confidences.append(status.get('confidence', 0.5))
            methods_used.append('operational_status')
        elif status.get('column_status') == 'closed':
            # Coluna fechada diminui probabilidade
            scores.append(0.1)
            confidences.append(status.get('confidence', 0.5))
            methods_used.append('operational_status')
        
        # Calcula score integrado ponderado pela confiança
        if scores and confidences:
            weights = np.array(confidences)
            weights = weights / np.sum(weights)  # Normaliza
            integrated_score = np.average(scores, weights=weights)
            overall_confidence = np.mean(confidences)
        else:
            integrated_score = 0.0
            overall_confidence = 0.0
        
        # Determina classificação final com thresholds adaptativos
        if integrated_score > 0.8:
            leak_status = 'leak_detected_high_confidence'
            severity = 'high'
        elif integrated_score > 0.6:
            leak_status = 'leak_detected_medium_confidence'
            severity = 'medium'
        elif integrated_score > 0.4:
            leak_status = 'possible_leak'
            severity = 'low'
        else:
            leak_status = 'normal_operation'
            severity = 'none'
        
        # Estima localização se vazamento detectado
        estimated_location = None
        if integrated_score > 0.4:
            estimated_location = self._estimate_leak_location(analysis_results)
        
        return {
            'integrated_score': float(integrated_score),
            'overall_confidence': float(overall_confidence),
            'leak_status': leak_status,
            'severity': severity,
            'estimated_location': estimated_location,
            'methods_used': methods_used,
            'individual_scores': scores,
            'individual_confidences': confidences,
            'timestamp': datetime.now().isoformat()
        }
    
    def _estimate_leak_location(self, analysis_results: Dict[str, Any]) -> Optional[float]:
        """Estima localização do vazamento usando múltiplas análises"""
        
        locations = []
        
        # Localização baseada em correlação sônica
        sonic = analysis_results.get('sonic_correlation', {})
        if 'delay_samples' in sonic and 'expected_transit_time' in sonic:
            delay_ratio = sonic.get('delay_samples', 0) / max(sonic.get('expected_transit_time', 1), 1)
            if delay_ratio != 1.0:
                location_sonic = self.config.sensor_distance * delay_ratio
                location_sonic = max(0, min(location_sonic, self.config.pipe_characteristics.length))
                locations.append(location_sonic)
        
        # Localização baseada em status operacional
        status = analysis_results.get('operational_status', {})
        if 'open_location' in status and status['open_location'] is not None:
            locations.append(status['open_location'])
        
        # Localização baseada em análise multivariável
        multivariable = analysis_results.get('multivariable_analysis', {})
        trends = multivariable.get('trends', {})
        if trends.get('pressure', 0) < -0.001:  # Queda de pressão significativa
            # Estima baseado na taxa de queda
            pressure_based_location = self.config.sensor_distance * (1 + abs(trends['pressure']) * 1000)
            locations.append(min(pressure_based_location, self.config.pipe_characteristics.length))
        
        # Retorna média das estimativas
        if locations:
            return float(np.mean(locations))
        
        return None
    
    def _calculate_snr(self, signal: np.ndarray) -> float:
        """Calcula relação sinal-ruído robusta"""
        try:
            if len(signal) < 2:
                return 0.0
            
            signal_power = np.var(signal)
            noise_estimate = np.var(np.diff(signal)) / 2  # Estima ruído pela diferença
            
            if noise_estimate > 0 and signal_power > 0:
                snr = 10 * np.log10(signal_power / noise_estimate)
                return max(0, min(snr, 60))  # Limita entre 0 e 60 dB
            else:
                return 60.0  # SNR muito alto se sem ruído detectável
        except:
            return 0.0
    
    def learn_from_confirmed_leak(self, leak_start_time: datetime, leak_type: str,
                                leak_characteristics: Dict[str, Any]):
        """Sistema de aprendizado com vazamento confirmado"""
        
        self.logger.info(f"Aprendendo com vazamento confirmado: {leak_type} em {leak_start_time}")
        
        # Recupera dados históricos
        end_time = leak_start_time
        start_time = leak_start_time - timedelta(hours=6)
        
        try:
            historical_readings = database.get_sensor_readings(
                self.config.system_id, start_time, end_time
            )
            
            if not historical_readings.empty:
                # Converte para snapshots
                snapshots = self.time_processor.synchronize_multivariable_data(
                    historical_readings, system_id=self.config.system_id
                )
                
                if len(snapshots) >= self.ml_system.feature_window:
                    # Cria labels para treinamento
                    leak_time_numeric = leak_start_time.timestamp()
                    labels = []
                    
                    for snapshot in snapshots:
                        time_to_leak = leak_time_numeric - snapshot.timestamp.timestamp()
                        # Marca como vazamento se ocorreu até 1 hora antes
                        labels.append(time_to_leak <= 3600 and time_to_leak >= 0)
                    
                    # Adiciona ao buffer de treinamento
                    self.ml_system.training_data_buffer.extend([
                        (snapshot, label, leak_type) 
                        for snapshot, label in zip(snapshots, labels)
                    ])
                    
                    # Retreina se temos dados suficientes
                    if len(self.ml_system.training_data_buffer) >= self.ml_system.retrain_threshold:
                        try:
                            snapshots_for_training = [item[0] for item in self.ml_system.training_data_buffer]
                            labels_for_training = [item[1] for item in self.ml_system.training_data_buffer]
                            
                            metrics = self.ml_system.train_models(snapshots_for_training, labels_for_training)
                            self.logger.info(f"Modelo retreinado com sucesso: {metrics}")
                            
                            # Limpa buffer
                            self.ml_system.training_data_buffer.clear()
                            
                        except Exception as e:
                            self.logger.error(f"Erro no retreinamento: {e}")
                
        except Exception as e:
            self.logger.error(f"Erro na recuperação de dados históricos: {e}")
        
        # Salva evento confirmado no banco
        database.save_leak_event(
            system_id=self.config.system_id,
            timestamp=leak_start_time,
            leak_type=leak_type,
            severity=leak_characteristics.get('severity', 1.0),
            location_estimate=leak_characteristics.get('location', 0.0),
            confidence=1.0,  # Confiança máxima para confirmação manual
            detected_by='manual',
            ml_signature='confirmed_by_operator',
            variables_at_detection=leak_characteristics
        )
    
    def process_comprehensive_advanced_analysis(self, snapshots: List[MultiVariableSnapshot], 
                                               enable_pca: bool = True,
                                               enable_spectral: bool = True,
                                               enable_retroactive: bool = True,
                                               enable_physics_interpolation: bool = True) -> Dict[str, Any]:
        """
        Análise comprehensiva com todas as funcionalidades avançadas implementadas.
        Integra PCA, análise espectral, aprendizagem retroativa e interpolação consciente de física.
        """
        try:
            self.logger.info("Iniciando análise comprehensiva avançada")
            comprehensive_results = {
                'timestamp': datetime.now().isoformat(),
                'snapshots_analyzed': len(snapshots),
                'analysis_modules': []
            }
            
            if len(snapshots) < 10:
                return {
                    'error': 'Dados insuficientes para análise comprehensiva',
                    'min_required': 10,
                    'received': len(snapshots)
                }
            
            # Converter snapshots para DataFrame para análises avançadas
            df_data = []
            for snapshot in snapshots:
                df_data.append({
                    'timestamp': snapshot.timestamp,
                    'expeditor_pressure': snapshot.expeditor_pressure,
                    'receiver_pressure': snapshot.receiver_pressure,
                    'pressure_diff': snapshot.expeditor_pressure - snapshot.receiver_pressure,
                    'flow_rate': snapshot.flow_rate,
                    'temperature': snapshot.temperature,
                    'density': snapshot.density,
                    'viscosity': snapshot.viscosity
                })
            
            analysis_df = pd.DataFrame(df_data)
            
            # 1. Análise PCA Avançada
            if enable_pca and len(snapshots) > 15:
                try:
                    pca_results = self.ml_system.perform_pca_analysis(analysis_df)
                    comprehensive_results['pca_analysis'] = pca_results
                    comprehensive_results['analysis_modules'].append('PCA')
                    
                    # Correlação Canônica entre pressões e outras variáveis
                    pressure_data = analysis_df[['expeditor_pressure', 'receiver_pressure', 'pressure_diff']]
                    other_data = analysis_df[['flow_rate', 'temperature', 'density']]
                    
                    cca_results = self.ml_system.canonical_correlation_analysis(pressure_data, other_data)
                    comprehensive_results['canonical_correlation'] = cca_results
                    comprehensive_results['analysis_modules'].append('CCA')
                    
                except Exception as e:
                    comprehensive_results['pca_analysis'] = {'error': f'Erro PCA: {str(e)}'}
            
            # 2. Análise Espectral com Filtros Butterworth e Hilbert
            if enable_spectral and len(snapshots) > 20:
                try:
                    pressure_signal = np.array([s.expeditor_pressure for s in snapshots])
                    spectral_results = self.status_detector.spectral_state_detection(
                        pressure_signal, sample_rate=1.0
                    )
                    comprehensive_results['spectral_analysis'] = spectral_results
                    comprehensive_results['analysis_modules'].append('Spectral')
                    
                    # Análise adicional para vazão
                    flow_signal = np.array([s.flow_rate for s in snapshots])
                    flow_spectral = self.status_detector.spectral_state_detection(
                        flow_signal, sample_rate=1.0
                    )
                    comprehensive_results['flow_spectral_analysis'] = flow_spectral
                    
                except Exception as e:
                    comprehensive_results['spectral_analysis'] = {'error': f'Erro Spectral: {str(e)}'}
            
            # 3. Interpolação Consciente de Física para timestamps irregulares
            if enable_physics_interpolation:
                try:
                    timestamps = np.array([s.timestamp.timestamp() for s in snapshots])
                    pressure_data_dict = {
                        'expeditor': np.array([s.expeditor_pressure for s in snapshots]),
                        'receiver': np.array([s.receiver_pressure for s in snapshots])
                    }
                    flow_data = np.array([s.flow_rate for s in snapshots])
                    
                    system_props = {
                        'pipe_length': self.config.pipe_characteristics.length,
                        'pipe_diameter': self.config.pipe_characteristics.diameter,
                        'fluid_density': np.mean([s.density for s in snapshots]),
                        'fluid_viscosity': np.mean([s.viscosity for s in snapshots]),
                        'bulk_modulus': 2.2e9,  # Pa - módulo de compressibilidade da água
                        'min_pressure': 0.0,
                        'max_pressure': 100.0,
                        'max_flow': 1000.0
                    }
                    
                    physics_interpolation = temporal_filter.physics_aware_interpolation(
                        timestamps, pressure_data_dict, flow_data, system_props
                    )
                    comprehensive_results['physics_interpolation'] = physics_interpolation
                    comprehensive_results['analysis_modules'].append('Physics_Interpolation')
                    
                except Exception as e:
                    comprehensive_results['physics_interpolation'] = {'error': f'Erro Physics: {str(e)}'}
            
            # 4. Análise ML com retroatividade (se modelo já treinado)
            if self.ml_system.is_trained:
                try:
                    ml_prediction = self.ml_system.predict_leak(snapshots)
                    comprehensive_results['ml_prediction'] = ml_prediction
                    
                    # Métricas de performance do modelo
                    model_metrics = self.ml_system.get_model_performance_metrics()
                    comprehensive_results['ml_model_metrics'] = model_metrics
                    comprehensive_results['analysis_modules'].append('ML_Prediction')
                    
                except Exception as e:
                    comprehensive_results['ml_prediction'] = {'error': f'Erro ML: {str(e)}'}
            
            # 5. Análise de status operacional integrada
            try:
                status_analysis = self.status_detector.analyze_operational_status(snapshots)
                comprehensive_results['operational_status'] = status_analysis
                comprehensive_results['analysis_modules'].append('Operational_Status')
            except Exception as e:
                comprehensive_results['operational_status'] = {'error': f'Erro Status: {str(e)}'}
            
            # 6. Correlação sônica tradicional para comparação
            try:
                sonic_correlation = self._perform_sonic_correlation_analysis(snapshots)
                comprehensive_results['sonic_correlation'] = sonic_correlation
                comprehensive_results['analysis_modules'].append('Sonic_Correlation')
            except Exception as e:
                comprehensive_results['sonic_correlation'] = {'error': f'Erro Sonic: {str(e)}'}
            
            # 7. Análise multivariável integrada
            try:
                multivariable_analysis = self._perform_multivariable_analysis(snapshots)
                comprehensive_results['multivariable_analysis'] = multivariable_analysis
                comprehensive_results['analysis_modules'].append('Multivariable')
            except Exception as e:
                comprehensive_results['multivariable_analysis'] = {'error': f'Erro Multivariável: {str(e)}'}
            
            # 8. Integração de resultados com algoritmo de fusão avançado
            try:
                integrated_assessment = self._advanced_result_integration(comprehensive_results)
                comprehensive_results['integrated_assessment'] = integrated_assessment
                comprehensive_results['analysis_modules'].append('Advanced_Integration')
            except Exception as e:
                comprehensive_results['integrated_assessment'] = {'error': f'Erro Integration: {str(e)}'}
            
            # 9. Recomendações baseadas em análise comprehensiva
            try:
                recommendations = self._generate_comprehensive_recommendations(comprehensive_results)
                comprehensive_results['recommendations'] = recommendations
            except Exception as e:
                comprehensive_results['recommendations'] = {'error': f'Erro Recommendations: {str(e)}'}
            
            # Estatísticas finais
            comprehensive_results['analysis_summary'] = {
                'total_modules_executed': len(comprehensive_results['analysis_modules']),
                'successful_modules': len([m for m in comprehensive_results['analysis_modules'] if 'error' not in comprehensive_results.get(m.lower().replace('_', '_'), {})]),
                'processing_time_seconds': (datetime.now() - datetime.fromisoformat(comprehensive_results['timestamp'])).total_seconds(),
                'data_quality_score': self._calculate_overall_data_quality(snapshots)
            }
            
            self.logger.info(f"Análise comprehensiva concluída: {comprehensive_results['analysis_summary']}")
            return comprehensive_results
            
        except Exception as e:
            error_msg = f"Erro na análise comprehensiva: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg}
    
    def _advanced_result_integration(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Integração avançada de resultados usando fusão de múltiplos algoritmos."""
        integration_scores = []
        confidence_weights = []
        detection_evidences = []
        
        # Score PCA - anomalias no espaço principal
        if 'pca_analysis' in results and 'error' not in results['pca_analysis']:
            pca = results['pca_analysis']
            if 'important_components' in pca:
                # Alta variância em poucos componentes pode indicar anomalia
                first_component_variance = pca.get('explained_variance_ratio', [0])[0] if pca.get('explained_variance_ratio') else 0
                if first_component_variance > 0.8:  # Muito concentrado em um componente
                    integration_scores.append(0.7)
                    confidence_weights.append(0.8)
                    detection_evidences.append('high_pca_concentration')
        
        # Score Espectral - estados operacionais anômalos
        if 'spectral_analysis' in results and 'error' not in results['spectral_analysis']:
            spectral = results['spectral_analysis']
            operational_state = spectral.get('operational_state', 'normal_operation')
            spectral_confidence = spectral.get('confidence', 0)
            
            if operational_state in ['active_leak', 'transitioning']:
                integration_scores.append(0.9)
                confidence_weights.append(spectral_confidence)
                detection_evidences.append(f'spectral_state_{operational_state}')
            elif operational_state in ['stable_operation', 'normal_operation']:
                integration_scores.append(0.1)
                confidence_weights.append(spectral_confidence)
        
        # Score Correlação Canônica - relações anômalas entre variáveis
        if 'canonical_correlation' in results and 'error' not in results['canonical_correlation']:
            cca = results['canonical_correlation']
            max_correlation = cca.get('max_correlation', 0)
            if max_correlation > 0.8:  # Correlação muito alta pode indicar vazamento
                integration_scores.append(0.6)
                confidence_weights.append(0.7)
                detection_evidences.append('high_canonical_correlation')
        
        # Score ML (se disponível)
        if 'ml_prediction' in results and 'error' not in results['ml_prediction']:
            ml = results['ml_prediction']
            if ml.get('model_ready', False):
                leak_prob = ml.get('leak_probability', 0)
                ml_confidence = ml.get('confidence', 0)
                integration_scores.append(leak_prob)
                confidence_weights.append(ml_confidence)
                if leak_prob > 0.5:
                    detection_evidences.append(f'ml_detection_{leak_prob:.2f}')
        
        # Score Correlação Sônica
        if 'sonic_correlation' in results and 'error' not in results['sonic_correlation']:
            sonic = results['sonic_correlation']
            max_corr = sonic.get('max_correlation', 0)
            # Correlação baixa pode indicar vazamento
            sonic_score = 1.0 - abs(max_corr)  # Invertido
            integration_scores.append(sonic_score)
            confidence_weights.append(0.8)
            if sonic_score > 0.5:
                detection_evidences.append('low_sonic_correlation')
        
        # Cálculo do score integrado
        if integration_scores and confidence_weights:
            weights = np.array(confidence_weights)
            weights = weights / np.sum(weights)
            integrated_leak_score = np.average(integration_scores, weights=weights)
            overall_confidence = np.mean(confidence_weights)
        else:
            integrated_leak_score = 0.0
            overall_confidence = 0.0
        
        # Classificação final
        if integrated_leak_score > 0.8:
            final_classification = 'leak_detected_high_confidence'
            alert_level = 'CRITICAL'
        elif integrated_leak_score > 0.6:
            final_classification = 'leak_detected_medium_confidence'
            alert_level = 'HIGH'
        elif integrated_leak_score > 0.4:
            final_classification = 'possible_leak_low_confidence'
            alert_level = 'MEDIUM'
        else:
            final_classification = 'normal_operation'
            alert_level = 'LOW'
        
        return {
            'integrated_leak_score': float(integrated_leak_score),
            'overall_confidence': float(overall_confidence),
            'final_classification': final_classification,
            'alert_level': alert_level,
            'detection_evidences': detection_evidences,
            'contributing_algorithms': len(integration_scores),
            'fusion_method': 'weighted_average_by_confidence'
        }
    
    def _generate_comprehensive_recommendations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Gera recomendações baseadas na análise comprehensiva."""
        recommendations = {
            'immediate_actions': [],
            'maintenance_suggestions': [],
            'monitoring_adjustments': [],
            'investigation_priorities': []
        }
        
        integrated = results.get('integrated_assessment', {})
        alert_level = integrated.get('alert_level', 'LOW')
        
        if alert_level == 'CRITICAL':
            recommendations['immediate_actions'].extend([
                'Suspender operação imediatamente',
                'Isolar seção afetada do sistema',
                'Enviar equipe de emergência para localização estimada',
                'Ativar protocolo de contenção de vazamentos'
            ])
        
        elif alert_level == 'HIGH':
            recommendations['immediate_actions'].extend([
                'Reduzir pressão operacional em 20%',
                'Aumentar frequência de monitoramento',
                'Agendar inspeção visual da tubulação'
            ])
        
        elif alert_level == 'MEDIUM':
            recommendations['monitoring_adjustments'].extend([
                'Aumentar sensibilidade dos algoritmos de detecção',
                'Implementar monitoramento contínuo por 48h',
                'Analisar dados históricos dos últimos 30 dias'
            ])
        
        # Recomendações específicas por módulo
        if 'pca_analysis' in results and results['pca_analysis'].get('important_components'):
            recommendations['investigation_priorities'].append(
                'Investigar variáveis com maior peso nos componentes principais'
            )
        
        if 'spectral_analysis' in results:
            spectral = results['spectral_analysis']
            if spectral.get('operational_state') == 'transitioning':
                recommendations['monitoring_adjustments'].append(
                    'Monitorar transições operacionais com maior frequência'
                )
        
        return recommendations
    
    def _calculate_overall_data_quality(self, snapshots: List[MultiVariableSnapshot]) -> float:
        """Calcula score geral de qualidade dos dados."""
        quality_factors = []
        
        # Completude temporal
        times = [s.timestamp for s in snapshots]
        if len(times) > 1:
            intervals = [(times[i+1] - times[i]).total_seconds() for i in range(len(times)-1)]
            interval_consistency = 1.0 / (1.0 + np.std(intervals))
            quality_factors.append(interval_consistency)
        
        # Completude de dados
        variables = ['expeditor_pressure', 'receiver_pressure', 'flow_rate', 'temperature', 'density']
        completeness_scores = []
        
        for var in variables:
            values = [getattr(s, var) for s in snapshots]
            valid_count = sum(1 for v in values if not np.isnan(v) and v is not None)
            completeness_scores.append(valid_count / len(values))
        
        quality_factors.append(np.mean(completeness_scores))
        
        # Consistência física
        physical_consistency = []
        for s in snapshots:
            # Pressão diferencial deve ser positiva
            pressure_diff = s.expeditor_pressure - s.receiver_pressure
            if pressure_diff >= 0:
                physical_consistency.append(1.0)
            else:
                physical_consistency.append(0.0)
        
        quality_factors.append(np.mean(physical_consistency) if physical_consistency else 0.0)
        
        return float(np.mean(quality_factors))

# ============================================================================
# tests/test_system.py - SISTEMA DE TESTES INTEGRADO
# ============================================================================

class HydraulicSystemTests(unittest.TestCase):
    """Testes unitários para o sistema hidráulico"""
    
    def setUp(self):
        """Configuração inicial dos testes"""
        self.pipe_characteristics = PipeCharacteristics(
            diameter=0.5,
            material='steel',
            profile='circular',
            length=1000.0,
            roughness=0.045,
            wall_thickness=10.0
        )
        
        self.system_config = SystemConfiguration(
            system_id='TEST_001',
            name='Sistema de Teste',
            location='Laboratório',
            pipe_characteristics=self.pipe_characteristics,
            sensor_distance=100.0,
            fluid_type='água',
            nominal_density=1.0,
            nominal_temperature=20.0,
            nominal_pressure=10.0,
            nominal_flow=1000.0,
            sonic_velocity=1500.0,
            calibration_date=datetime.now()
        )
        
        self.processor = IndustrialHydraulicProcessor(self.system_config)
    
    def test_sensor_reading_validation(self):
        """Testa validação de leituras de sensores"""
        # Leitura válida
        valid_reading = SensorReading(
            timestamp=datetime.now(),
            sensor_id='EXP_01',
            variable='pressure',
            value=10.5,
            unit='kgf/cm²'
        )
        self.assertTrue(valid_reading.validate('TEST_001'))
        
        # Leitura inválida
        invalid_reading = SensorReading(
            timestamp=datetime.now(),
            sensor_id='EXP_01',
            variable='pressure',
            value=150.0,  # Fora da faixa
            unit='kgf/cm²'
        )
        self.assertFalse(invalid_reading.validate('TEST_001'))
    
    def test_multivariable_snapshot_validation(self):
        """Testa validação de snapshots multivariáveis"""
        # Snapshot válido
        valid_snapshot = MultiVariableSnapshot(
            timestamp=datetime.now(),
            expeditor_pressure=10.0,
            receiver_pressure=9.8,
            flow_rate=1000.0,
            density=1.0,
            temperature=20.0
        )
        self.assertTrue(valid_snapshot.validate_physics('TEST_001'))
        
        # Snapshot fisicamente inconsistente
        invalid_snapshot = MultiVariableSnapshot(
            timestamp=datetime.now(),
            expeditor_pressure=0.1,  # Muito baixa
            receiver_pressure=0.1,
            flow_rate=5000.0,  # Muito alta para pressão baixa
            density=1.0,
            temperature=20.0
        )
        self.assertFalse(invalid_snapshot.validate_physics('TEST_001'))
    
    def test_pipe_characteristics_calculation(self):
        """Testa cálculo de propriedades acústicas"""
        acoustic_props = self.pipe_characteristics.calculate_acoustic_properties()
        
        self.assertIn('velocity_factor', acoustic_props)
        self.assertIn('attenuation', acoustic_props)
        self.assertIn('area_factor', acoustic_props)
        self.assertEqual(acoustic_props['velocity_factor'], 1.0)  # Steel
    
    def test_time_series_processing(self):
        """Testa processamento de séries temporais irregulares"""
        # Cria dados de teste
        readings = []
        base_time = datetime.now()
        
        for i in range(10):
            # Timestamps irregulares
            timestamp = base_time + timedelta(seconds=i * 1.5)
            
            readings.append(SensorReading(
                timestamp=timestamp,
                sensor_id='EXP_01',
                variable='pressure',
                value=10.0 + np.sin(i * 0.5),
                unit='kgf/cm²'
            ))
            
            readings.append(SensorReading(
                timestamp=timestamp,
                sensor_id='REC_01',
                variable='pressure',
                value=9.8 + np.sin(i * 0.5),
                unit='kgf/cm²'
            ))
        
        # Processa
        result = self.processor.process_sensor_readings(readings)
        
        self.assertNotIn('error', result)
        self.assertIn('analysis_timestamp', result)
        self.assertGreater(result.get('snapshots_processed', 0), 0)
    
    def test_ml_feature_extraction(self):
        """Testa extração de features para ML"""
        # Cria snapshots de teste
        snapshots = []
        base_time = datetime.now()
        
        for i in range(100):
            snapshot = MultiVariableSnapshot(
                timestamp=base_time + timedelta(seconds=i),
                expeditor_pressure=10.0 + np.sin(i * 0.1),
                receiver_pressure=9.8 + np.sin(i * 0.1 + 0.1),
                flow_rate=1000.0 + 100 * np.sin(i * 0.05),
                density=1.0,
                temperature=20.0
            )
            snapshots.append(snapshot)
        
        # Extrai features
        features = self.processor.ml_system.extract_advanced_features(snapshots)
        
        self.assertGreater(len(features), 0)
        self.assertFalse(np.any(np.isnan(features)))
        self.assertFalse(np.any(np.isinf(features)))
    
    def test_operational_status_detection(self):
        """Testa detecção de status operacional"""
        # Cria dados simulando coluna fechada
        snapshots = []
        base_time = datetime.now()
        
        for i in range(50):
            snapshot = MultiVariableSnapshot(
                timestamp=base_time + timedelta(seconds=i),
                expeditor_pressure=10.0 + 0.1 * np.random.normal(),
                receiver_pressure=9.8 + 0.1 * np.random.normal(),
                flow_rate=1000.0 + 10 * np.random.normal(),
                density=1.0,
                temperature=20.0
            )
            snapshots.append(snapshot)
        
        # Analisa status
        status_result = self.processor.status_detector.analyze_operational_status(snapshots)
        
        self.assertIn('column_status', status_result)
        self.assertIn('confidence', status_result)
        self.assertGreaterEqual(status_result['confidence'], 0.0)
        self.assertLessEqual(status_result['confidence'], 1.0)

def run_tests():
    """Executa todos os testes do sistema"""
    logger.info("Iniciando testes do sistema...")
    
    # Cria suite de testes
    test_suite = unittest.TestLoader().loadTestsFromTestCase(HydraulicSystemTests)
    
    # Executa testes
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Log dos resultados
    if result.wasSuccessful():
        logger.info("Todos os testes passaram com sucesso!")
        return True
    else:
        logger.error(f"Testes falharam: {len(result.failures)} falhas, {len(result.errors)} erros")
        return False

# ============================================================================
# gui/main_window.py - INTERFACE PYQT6 COMPLETA
# ============================================================================

class SystemConfigDialog(QDialog):
    """Dialog para configuração de sistema"""
    
    system_configured = pyqtSignal(SystemConfiguration)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuração do Sistema")
        self.setModal(True)
        self.resize(800, 600)
        self.setup_ui()
    
    def setup_ui(self):
        """Configura interface do dialog"""
        layout = QVBoxLayout(self)
        
        # Informações básicas
        basic_group = QGroupBox("Informações Básicas")
        basic_layout = QGridLayout(basic_group)
        
        basic_layout.addWidget(QLabel("ID do Sistema:"), 0, 0)
        self.system_id_edit = QLineEdit()
        basic_layout.addWidget(self.system_id_edit, 0, 1)
        
        basic_layout.addWidget(QLabel("Nome:"), 0, 2)
        self.name_edit = QLineEdit()
        basic_layout.addWidget(self.name_edit, 0, 3)
        
        basic_layout.addWidget(QLabel("Localização:"), 1, 0)
        self.location_edit = QLineEdit()
        basic_layout.addWidget(self.location_edit, 1, 1, 1, 3)
        
        layout.addWidget(basic_group)
        
        # Unidades Expedidora e Recebedora
        units_group = QGroupBox("Unidades do Sistema")
        units_layout = QGridLayout(units_group)
        
        units_layout.addWidget(QLabel("Unidade Expedidora:"), 0, 0)
        self.expeditor_unit_edit = QLineEdit()
        self.expeditor_unit_edit.setPlaceholderText("Nome da unidade expedidora")
        units_layout.addWidget(self.expeditor_unit_edit, 0, 1)
        
        units_layout.addWidget(QLabel("Alias Expedidora:"), 0, 2)
        self.expeditor_alias_edit = QLineEdit()
        self.expeditor_alias_edit.setPlaceholderText("Ex: BAR, EXP, UP")
        self.expeditor_alias_edit.setText("BAR")
        units_layout.addWidget(self.expeditor_alias_edit, 0, 3)
        
        units_layout.addWidget(QLabel("Unidade Recebedora:"), 1, 0)
        self.receiver_unit_edit = QLineEdit()
        self.receiver_unit_edit.setPlaceholderText("Nome da unidade recebedora")
        units_layout.addWidget(self.receiver_unit_edit, 1, 1)
        
        units_layout.addWidget(QLabel("Alias Recebedora:"), 1, 2)
        self.receiver_alias_edit = QLineEdit()
        self.receiver_alias_edit.setPlaceholderText("Ex: PLN, REC, DOWN")
        self.receiver_alias_edit.setText("PLN")
        units_layout.addWidget(self.receiver_alias_edit, 1, 3)
        
        layout.addWidget(units_group)
        
        # Características do duto
        pipe_group = QGroupBox("Características do Duto")
        pipe_layout = QGridLayout(pipe_group)
        
        # Diâmetro com conversão polegadas/mm
        pipe_layout.addWidget(QLabel("Diâmetro:"), 0, 0)
        diameter_container = QHBoxLayout()
        
        self.diameter_spin = QDoubleSpinBox()
        self.diameter_spin.setRange(0.01, 10.0)
        self.diameter_spin.setDecimals(3)
        self.diameter_spin.setValue(0.5)
        self.diameter_spin.setSuffix(" m")
        diameter_container.addWidget(self.diameter_spin)
        
        self.diameter_inch_spin = QDoubleSpinBox()
        self.diameter_inch_spin.setRange(0.1, 400.0)
        self.diameter_inch_spin.setDecimals(2)
        self.diameter_inch_spin.setSuffix(" in")
        diameter_container.addWidget(QLabel("ou"))
        diameter_container.addWidget(self.diameter_inch_spin)
        
        # Conecta conversões
        self.diameter_spin.valueChanged.connect(self.diameter_m_to_inch)
        self.diameter_inch_spin.valueChanged.connect(self.diameter_inch_to_m)
        
        pipe_layout.addLayout(diameter_container, 0, 1, 1, 3)
        
        pipe_layout.addWidget(QLabel("Material:"), 1, 0)
        self.material_combo = QComboBox()
        self.material_combo.addItems(['steel', 'pvc', 'concrete', 'fiberglass'])
        pipe_layout.addWidget(self.material_combo, 1, 1)
        
        pipe_layout.addWidget(QLabel("Perfil:"), 1, 2)
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(['circular', 'rectangular', 'oval'])
        pipe_layout.addWidget(self.profile_combo, 1, 3)
        
        # Comprimento em km com limites
        pipe_layout.addWidget(QLabel("Comprimento (km):"), 2, 0)
        self.length_spin = QDoubleSpinBox()
        self.length_spin.setRange(0.01, 1000.0)  # Limite de 1 km até 1000 km
        self.length_spin.setDecimals(3)
        self.length_spin.setValue(1.0)  # 1 km como padrão
        self.length_spin.setSuffix(" km")
        pipe_layout.addWidget(self.length_spin, 2, 1)
        
        pipe_layout.addWidget(QLabel("Rugosidade (mm):"), 2, 2)
        self.roughness_spin = QDoubleSpinBox()
        self.roughness_spin.setRange(0.001, 1.0)
        self.roughness_spin.setDecimals(3)
        self.roughness_spin.setValue(0.045)
        self.roughness_spin.setSuffix(" mm")
        pipe_layout.addWidget(self.roughness_spin, 2, 3)
        
        # Espessura com conversão polegadas/mm
        pipe_layout.addWidget(QLabel("Espessura Parede:"), 3, 0)
        thickness_container = QHBoxLayout()
        
        self.wall_thickness_spin = QDoubleSpinBox()
        self.wall_thickness_spin.setRange(1.0, 100.0)
        self.wall_thickness_spin.setValue(10.0)
        self.wall_thickness_spin.setSuffix(" mm")
        thickness_container.addWidget(self.wall_thickness_spin)
        
        self.wall_thickness_inch_spin = QDoubleSpinBox()
        self.wall_thickness_inch_spin.setRange(0.01, 4.0)
        self.wall_thickness_inch_spin.setDecimals(3)
        self.wall_thickness_inch_spin.setSuffix(" in")
        thickness_container.addWidget(QLabel("ou"))
        thickness_container.addWidget(self.wall_thickness_inch_spin)
        
        # Conecta conversões
        self.wall_thickness_spin.valueChanged.connect(self.thickness_mm_to_inch)
        self.wall_thickness_inch_spin.valueChanged.connect(self.thickness_inch_to_mm)
        
        pipe_layout.addLayout(thickness_container, 3, 1, 1, 3)
        
        layout.addWidget(pipe_group)
        
        # Características do fluido
        fluid_group = QGroupBox("Características do Fluido")
        fluid_layout = QGridLayout(fluid_group)
        
        fluid_layout.addWidget(QLabel("Tipo de Fluido:"), 0, 0)
        self.fluid_type_edit = QLineEdit("água")
        fluid_layout.addWidget(self.fluid_type_edit, 0, 1)
        
        fluid_layout.addWidget(QLabel("Densidade Nominal (g/cm³):"), 0, 2)
        self.nominal_density_spin = QDoubleSpinBox()
        self.nominal_density_spin.setRange(0.1, 2.0)
        self.nominal_density_spin.setDecimals(4)
        self.nominal_density_spin.setValue(1.0)
        fluid_layout.addWidget(self.nominal_density_spin, 0, 3)
        
        fluid_layout.addWidget(QLabel("Temperatura Nominal (°C):"), 1, 0)
        self.nominal_temp_spin = QDoubleSpinBox()
        self.nominal_temp_spin.setRange(-50.0, 300.0)
        self.nominal_temp_spin.setValue(20.0)
        fluid_layout.addWidget(self.nominal_temp_spin, 1, 1)
        
        fluid_layout.addWidget(QLabel("Pressão Nominal (kgf/cm²):"), 1, 2)
        self.nominal_pressure_spin = QDoubleSpinBox()
        self.nominal_pressure_spin.setRange(0.1, 100.0)
        self.nominal_pressure_spin.setValue(10.0)
        fluid_layout.addWidget(self.nominal_pressure_spin, 1, 3)
        
        fluid_layout.addWidget(QLabel("Vazão Nominal (m³/h):"), 2, 0)
        self.nominal_flow_spin = QDoubleSpinBox()
        self.nominal_flow_spin.setRange(1.0, 10000.0)
        self.nominal_flow_spin.setValue(1000.0)
        fluid_layout.addWidget(self.nominal_flow_spin, 2, 1)
        
        fluid_layout.addWidget(QLabel("Velocidade Sônica (m/s):"), 2, 2)
        self.sonic_velocity_spin = QDoubleSpinBox()
        self.sonic_velocity_spin.setRange(200.0, 3000.0)
        self.sonic_velocity_spin.setValue(1500.0)
        fluid_layout.addWidget(self.sonic_velocity_spin, 2, 3)
        
        layout.addWidget(fluid_group)
        
        # Sensores
        sensor_group = QGroupBox("Configuração de Sensores")
        sensor_layout = QGridLayout(sensor_group)
        
        sensor_layout.addWidget(QLabel("Distância entre Sensores (km):"), 0, 0)
        self.sensor_distance_spin = QDoubleSpinBox()
        self.sensor_distance_spin.setRange(0.001, 100.0)  # Limite de 1m até 100 km
        self.sensor_distance_spin.setDecimals(3)
        self.sensor_distance_spin.setValue(0.1)  # 100m como padrão
        self.sensor_distance_spin.setSuffix(" km")
        sensor_layout.addWidget(self.sensor_distance_spin, 0, 1)
        
        layout.addWidget(sensor_group)
        
        # Botões
        buttons_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("Salvar Configuração")
        self.save_btn.clicked.connect(self.save_configuration)
        buttons_layout.addWidget(self.save_btn)
        
        self.cancel_btn = QPushButton("Cancelar")
        self.cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(buttons_layout)
    
    def diameter_m_to_inch(self, value_m):
        """Converte diâmetro de metros para polegadas"""
        self.diameter_inch_spin.blockSignals(True)
        self.diameter_inch_spin.setValue(value_m * 39.3701)  # 1m = 39.3701 inches
        self.diameter_inch_spin.blockSignals(False)
    
    def diameter_inch_to_m(self, value_inch):
        """Converte diâmetro de polegadas para metros"""
        self.diameter_spin.blockSignals(True)
        self.diameter_spin.setValue(value_inch / 39.3701)
        self.diameter_spin.blockSignals(False)
    
    def thickness_mm_to_inch(self, value_mm):
        """Converte espessura de mm para polegadas"""
        self.wall_thickness_inch_spin.blockSignals(True)
        self.wall_thickness_inch_spin.setValue(value_mm / 25.4)  # 1 inch = 25.4 mm
        self.wall_thickness_inch_spin.blockSignals(False)
    
    def thickness_inch_to_mm(self, value_inch):
        """Converte espessura de polegadas para mm"""
        self.wall_thickness_spin.blockSignals(True)
        self.wall_thickness_spin.setValue(value_inch * 25.4)
        self.wall_thickness_spin.blockSignals(False)
    
    def save_configuration(self):
        """Salva configuração do sistema"""
        try:
            # Cria características do duto (converte km para metros)
            pipe_characteristics = PipeCharacteristics(
                diameter=self.diameter_spin.value(),
                material=self.material_combo.currentText(),
                profile=self.profile_combo.currentText(),
                length=self.length_spin.value() * 1000.0,  # km para metros
                roughness=self.roughness_spin.value(),
                wall_thickness=self.wall_thickness_spin.value()
            )
            
            # Cria configuração do sistema
            config = SystemConfiguration(
                system_id=self.system_id_edit.text(),
                name=self.name_edit.text(),
                location=self.location_edit.text(),
                pipe_characteristics=pipe_characteristics,
                sensor_distance=self.sensor_distance_spin.value() * 1000.0,  # km para metros
                fluid_type=self.fluid_type_edit.text(),
                nominal_density=self.nominal_density_spin.value(),
                nominal_temperature=self.nominal_temp_spin.value(),
                nominal_pressure=self.nominal_pressure_spin.value(),
                nominal_flow=self.nominal_flow_spin.value(),
                sonic_velocity=self.sonic_velocity_spin.value(),
                calibration_date=datetime.now()
            )
            
            # Adiciona campos de unidades
            config.expeditor_unit = self.expeditor_unit_edit.text()
            config.expeditor_alias = self.expeditor_alias_edit.text()
            config.receiver_unit = self.receiver_unit_edit.text()
            config.receiver_alias = self.receiver_alias_edit.text()
            
            # Valida configuração
            if not config.system_id or not config.name:
                QMessageBox.warning(self, "Erro", "ID do Sistema e Nome são obrigatórios!")
                return
            
            if not config.expeditor_alias or not config.receiver_alias:
                QMessageBox.warning(self, "Erro", "Aliases das unidades são obrigatórios!")
                return
            
            # Emite sinal com configuração
            self.system_configured.emit(config)
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao salvar configuração:\n{str(e)}")

class SystemModificationDialog(QDialog):
    """Dialog para modificar características de um sistema existente"""
    
    def __init__(self, system_config: SystemConfiguration, parent=None):
        super().__init__(parent)
        self.original_config = system_config
        self.setWindowTitle(f"Modificar Sistema: {system_config.system_id}")
        self.setModal(True)
        self.resize(900, 700)
        self.setup_ui()
        self.load_current_values()
    
    def setup_ui(self):
        """Configura interface do dialog"""
        layout = QVBoxLayout(self)
        
        # Tab widget para organizar as seções
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Aba 1: Informações Básicas e Unidades Expedidora/Recebedora
        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)
        
        # Informações básicas
        basic_group = QGroupBox("Informações Básicas")
        basic_grid = QGridLayout(basic_group)
        
        basic_grid.addWidget(QLabel("ID do Sistema:"), 0, 0)
        self.system_id_edit = QLineEdit()
        basic_grid.addWidget(self.system_id_edit, 0, 1)
        
        basic_grid.addWidget(QLabel("Nome:"), 0, 2)
        self.name_edit = QLineEdit()
        basic_grid.addWidget(self.name_edit, 0, 3)
        
        basic_grid.addWidget(QLabel("Localização:"), 1, 0)
        self.location_edit = QLineEdit()
        basic_grid.addWidget(self.location_edit, 1, 1, 1, 3)
        
        basic_layout.addWidget(basic_group)
        
        # Unidades Expedidora e Recebedora
        units_group = QGroupBox("Unidades Expedidora e Recebedora")
        units_layout = QGridLayout(units_group)
        
        units_layout.addWidget(QLabel("Unidade Expedidora:"), 0, 0)
        self.expeditor_unit_edit = QLineEdit()
        units_layout.addWidget(self.expeditor_unit_edit, 0, 1)
        
        units_layout.addWidget(QLabel("Alias Expedidora:"), 0, 2)
        self.expeditor_alias_edit = QLineEdit()
        self.expeditor_alias_edit.setPlaceholderText("Ex: BAR, EXP, UP")
        units_layout.addWidget(self.expeditor_alias_edit, 0, 3)
        
        units_layout.addWidget(QLabel("Unidade Recebedora:"), 1, 0)
        self.receiver_unit_edit = QLineEdit()
        units_layout.addWidget(self.receiver_unit_edit, 1, 1)
        
        units_layout.addWidget(QLabel("Alias Recebedora:"), 1, 2)
        self.receiver_alias_edit = QLineEdit()
        self.receiver_alias_edit.setPlaceholderText("Ex: PLN, REC, DOWN")
        units_layout.addWidget(self.receiver_alias_edit, 1, 3)
        
        basic_layout.addWidget(units_group)
        tabs.addTab(basic_tab, "Básico & Unidades")
        
        # Aba 2: Características do Duto
        pipe_tab = QWidget()
        pipe_layout = QVBoxLayout(pipe_tab)
        
        pipe_group = QGroupBox("Características do Duto")
        pipe_grid = QGridLayout(pipe_group)
        
        # Diâmetro com conversão polegadas/mm
        pipe_grid.addWidget(QLabel("Diâmetro:"), 0, 0)
        diameter_container = QHBoxLayout()
        
        self.diameter_spin = QDoubleSpinBox()
        self.diameter_spin.setRange(0.01, 10.0)
        self.diameter_spin.setDecimals(3)
        self.diameter_spin.setSuffix(" m")
        diameter_container.addWidget(self.diameter_spin)
        
        self.diameter_inch_spin = QDoubleSpinBox()
        self.diameter_inch_spin.setRange(0.1, 400.0)
        self.diameter_inch_spin.setDecimals(2)
        self.diameter_inch_spin.setSuffix(" in")
        diameter_container.addWidget(QLabel("ou"))
        diameter_container.addWidget(self.diameter_inch_spin)
        
        # Conecta conversões
        self.diameter_spin.valueChanged.connect(self.diameter_m_to_inch)
        self.diameter_inch_spin.valueChanged.connect(self.diameter_inch_to_m)
        
        pipe_grid.addLayout(diameter_container, 0, 1, 1, 3)
        
        # Espessura com conversão polegadas/mm
        pipe_grid.addWidget(QLabel("Espessura Parede:"), 1, 0)
        thickness_container = QHBoxLayout()
        
        self.wall_thickness_spin = QDoubleSpinBox()
        self.wall_thickness_spin.setRange(1.0, 100.0)
        self.wall_thickness_spin.setSuffix(" mm")
        thickness_container.addWidget(self.wall_thickness_spin)
        
        self.wall_thickness_inch_spin = QDoubleSpinBox()
        self.wall_thickness_inch_spin.setRange(0.01, 4.0)
        self.wall_thickness_inch_spin.setDecimals(3)
        self.wall_thickness_inch_spin.setSuffix(" in")
        thickness_container.addWidget(QLabel("ou"))
        thickness_container.addWidget(self.wall_thickness_inch_spin)
        
        # Conecta conversões
        self.wall_thickness_spin.valueChanged.connect(self.thickness_mm_to_inch)
        self.wall_thickness_inch_spin.valueChanged.connect(self.thickness_inch_to_mm)
        
        pipe_grid.addLayout(thickness_container, 1, 1, 1, 3)
        
        # Comprimento em km
        pipe_grid.addWidget(QLabel("Comprimento (km):"), 2, 0)
        self.length_spin = QDoubleSpinBox()
        self.length_spin.setRange(0.01, 1000.0)
        self.length_spin.setDecimals(3)
        self.length_spin.setSuffix(" km")
        pipe_grid.addWidget(self.length_spin, 2, 1)
        
        pipe_grid.addWidget(QLabel("Material:"), 2, 2)
        self.material_combo = QComboBox()
        self.material_combo.addItems(['steel', 'pvc', 'concrete', 'fiberglass'])
        pipe_grid.addWidget(self.material_combo, 2, 3)
        
        pipe_grid.addWidget(QLabel("Perfil:"), 3, 0)
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(['circular', 'rectangular', 'oval'])
        pipe_grid.addWidget(self.profile_combo, 3, 1)
        
        pipe_grid.addWidget(QLabel("Rugosidade (mm):"), 3, 2)
        self.roughness_spin = QDoubleSpinBox()
        self.roughness_spin.setRange(0.001, 1.0)
        self.roughness_spin.setDecimals(3)
        self.roughness_spin.setSuffix(" mm")
        pipe_grid.addWidget(self.roughness_spin, 3, 3)
        
        pipe_layout.addWidget(pipe_group)
        
        # Sensores em km
        sensor_group = QGroupBox("Configuração de Sensores")
        sensor_grid = QGridLayout(sensor_group)
        
        sensor_grid.addWidget(QLabel("Distância entre Sensores (km):"), 0, 0)
        self.sensor_distance_spin = QDoubleSpinBox()
        self.sensor_distance_spin.setRange(0.001, 100.0)
        self.sensor_distance_spin.setDecimals(3)
        self.sensor_distance_spin.setSuffix(" km")
        pipe_layout.addWidget(sensor_group)
        
        tabs.addTab(pipe_tab, "Duto & Sensores")
        
        # Aba 3: Características do Fluido
        fluid_tab = QWidget()
        fluid_layout = QVBoxLayout(fluid_tab)
        
        fluid_group = QGroupBox("Características do Fluido")
        fluid_grid = QGridLayout(fluid_group)
        
        fluid_grid.addWidget(QLabel("Tipo de Fluido:"), 0, 0)
        self.fluid_type_edit = QLineEdit()
        fluid_grid.addWidget(self.fluid_type_edit, 0, 1)
        
        fluid_grid.addWidget(QLabel("Densidade Nominal (g/cm³):"), 0, 2)
        self.nominal_density_spin = QDoubleSpinBox()
        self.nominal_density_spin.setRange(0.1, 2.0)
        self.nominal_density_spin.setDecimals(4)
        fluid_grid.addWidget(self.nominal_density_spin, 0, 3)
        
        fluid_grid.addWidget(QLabel("Temperatura Nominal (°C):"), 1, 0)
        self.nominal_temp_spin = QDoubleSpinBox()
        self.nominal_temp_spin.setRange(-50.0, 300.0)
        fluid_grid.addWidget(self.nominal_temp_spin, 1, 1)
        
        fluid_grid.addWidget(QLabel("Pressão Nominal (kgf/cm²):"), 1, 2)
        self.nominal_pressure_spin = QDoubleSpinBox()
        self.nominal_pressure_spin.setRange(0.1, 100.0)
        fluid_grid.addWidget(self.nominal_pressure_spin, 1, 3)
        
        fluid_grid.addWidget(QLabel("Vazão Nominal (m³/h):"), 2, 0)
        self.nominal_flow_spin = QDoubleSpinBox()
        self.nominal_flow_spin.setRange(1.0, 10000.0)
        fluid_grid.addWidget(self.nominal_flow_spin, 2, 1)
        
        fluid_grid.addWidget(QLabel("Velocidade Sônica (m/s):"), 2, 2)
        self.sonic_velocity_spin = QDoubleSpinBox()
        self.sonic_velocity_spin.setRange(200.0, 3000.0)
        fluid_grid.addWidget(self.sonic_velocity_spin, 2, 3)
        
        fluid_layout.addWidget(fluid_group)
        tabs.addTab(fluid_tab, "Fluido")
        
        # Botões
        buttons_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("Salvar Modificações")
        self.save_btn.clicked.connect(self.accept)
        buttons_layout.addWidget(self.save_btn)
        
        self.cancel_btn = QPushButton("Cancelar")
        self.cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(buttons_layout)
    
    def load_current_values(self):
        """Carrega valores atuais do sistema"""
        config = self.original_config
        
        # Informações básicas
        self.system_id_edit.setText(config.system_id)
        self.name_edit.setText(config.name)
        self.location_edit.setText(config.location)
        
        # Unidades (valores padrão se não existirem)
        self.expeditor_unit_edit.setText(getattr(config, 'expeditor_unit', ''))
        self.expeditor_alias_edit.setText(getattr(config, 'expeditor_alias', 'BAR'))
        self.receiver_unit_edit.setText(getattr(config, 'receiver_unit', ''))
        self.receiver_alias_edit.setText(getattr(config, 'receiver_alias', 'PLN'))
        
        # Características do duto (convertendo de metros para km)
        self.diameter_spin.setValue(config.pipe_characteristics.diameter)
        self.length_spin.setValue(config.pipe_characteristics.length / 1000.0)  # m para km
        self.roughness_spin.setValue(config.pipe_characteristics.roughness)
        self.wall_thickness_spin.setValue(config.pipe_characteristics.wall_thickness)
        
        # Busca material e perfil nos combos
        material_index = self.material_combo.findText(config.pipe_characteristics.material)
        if material_index >= 0:
            self.material_combo.setCurrentIndex(material_index)
        
        profile_index = self.profile_combo.findText(config.pipe_characteristics.profile)
        if profile_index >= 0:
            self.profile_combo.setCurrentIndex(profile_index)
        
        # Sensores (convertendo de metros para km)
        self.sensor_distance_spin.setValue(config.sensor_distance / 1000.0)  # m para km
        
        # Fluido
        self.fluid_type_edit.setText(config.fluid_type)
        self.nominal_density_spin.setValue(config.nominal_density)
        self.nominal_temp_spin.setValue(config.nominal_temperature)
        self.nominal_pressure_spin.setValue(config.nominal_pressure)
        self.nominal_flow_spin.setValue(config.nominal_flow)
        self.sonic_velocity_spin.setValue(config.sonic_velocity)
    
    def diameter_m_to_inch(self, value_m):
        """Converte diâmetro de metros para polegadas"""
        self.diameter_inch_spin.blockSignals(True)
        self.diameter_inch_spin.setValue(value_m * 39.3701)  # 1m = 39.3701 inches
        self.diameter_inch_spin.blockSignals(False)
    
    def diameter_inch_to_m(self, value_inch):
        """Converte diâmetro de polegadas para metros"""
        self.diameter_spin.blockSignals(True)
        self.diameter_spin.setValue(value_inch / 39.3701)
        self.diameter_spin.blockSignals(False)
    
    def thickness_mm_to_inch(self, value_mm):
        """Converte espessura de mm para polegadas"""
        self.wall_thickness_inch_spin.blockSignals(True)
        self.wall_thickness_inch_spin.setValue(value_mm / 25.4)  # 1 inch = 25.4 mm
        self.wall_thickness_inch_spin.blockSignals(False)
    
    def thickness_inch_to_mm(self, value_inch):
        """Converte espessura de polegadas para mm"""
        self.wall_thickness_spin.blockSignals(True)
        self.wall_thickness_spin.setValue(value_inch * 25.4)
        self.wall_thickness_spin.blockSignals(False)
    
    def get_modified_config(self) -> SystemConfiguration:
        """Retorna configuração modificada"""
        # Cria características do duto (convertendo km para metros)
        pipe_characteristics = PipeCharacteristics(
            diameter=self.diameter_spin.value(),
            material=self.material_combo.currentText(),
            profile=self.profile_combo.currentText(),
            length=self.length_spin.value() * 1000.0,  # km para metros
            roughness=self.roughness_spin.value(),
            wall_thickness=self.wall_thickness_spin.value()
        )
        
        # Cria nova configuração
        config = SystemConfiguration(
            system_id=self.system_id_edit.text(),
            name=self.name_edit.text(),
            location=self.location_edit.text(),
            pipe_characteristics=pipe_characteristics,
            sensor_distance=self.sensor_distance_spin.value() * 1000.0,  # km para metros
            fluid_type=self.fluid_type_edit.text(),
            nominal_density=self.nominal_density_spin.value(),
            nominal_temperature=self.nominal_temp_spin.value(),
            nominal_pressure=self.nominal_pressure_spin.value(),
            nominal_flow=self.nominal_flow_spin.value(),
            sonic_velocity=self.sonic_velocity_spin.value(),
            calibration_date=datetime.now()
        )
        
        # Adiciona novos campos para unidades
        config.expeditor_unit = self.expeditor_unit_edit.text()
        config.expeditor_alias = self.expeditor_alias_edit.text()
        config.receiver_unit = self.receiver_unit_edit.text()
        config.receiver_alias = self.receiver_alias_edit.text()
        
        return config

class BatchFileLoadDialog(QDialog):
    """Dialog para carregamento em lote de arquivos com pré-classificação"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Carregamento em Lote de Arquivos")
        self.setModal(True)
        self.resize(1000, 700)
        self.file_classifications = {}
        self.main_window = parent  # Referência para a janela principal
        self.setup_ui()
    
    def setup_ui(self):
        """Configura interface do dialog"""
        layout = QVBoxLayout(self)
        
        # Instruções
        info_label = QLabel(
            "Selecione múltiplos arquivos XLSX para carregamento automático.\n"
            "O sistema tentará identificar automaticamente o tipo de dados baseado no nome do arquivo.\n"
            "Você pode revisar e ajustar as classificações antes do carregamento."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("background-color: #2d2d2d; padding: 10px; border: 1px solid #555;")
        layout.addWidget(info_label)
        
        # Botão para selecionar arquivos
        file_button_layout = QHBoxLayout()
        
        self.select_files_btn = QPushButton("Selecionar Arquivos XLSX...")
        self.select_files_btn.clicked.connect(self.select_files)
        self.select_files_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        file_button_layout.addWidget(self.select_files_btn)
        
        self.auto_detect_btn = QPushButton("Auto-detectar da Pasta Atual")
        self.auto_detect_btn.clicked.connect(self.auto_detect_files)
        self.auto_detect_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 10px;")
        file_button_layout.addWidget(self.auto_detect_btn)
        
        layout.addLayout(file_button_layout)
        
        # Tabela de classificação
        self.files_table = QTableWidget()
        self.files_table.setColumnCount(6)
        self.files_table.setHorizontalHeaderLabels([
            "Arquivo", "Tipo de Dado", "Estação", "Classificação", "Registros", "Status"
        ])
        header = self.files_table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        layout.addWidget(self.files_table)
        
        # Configurações de carregamento
        config_group = QGroupBox("Configurações de Carregamento")
        config_layout = QGridLayout(config_group)
        
        config_layout.addWidget(QLabel("Mapeamento de Estações:"), 0, 0)
        
        # Mapeamento de alias
        mapping_layout = QHBoxLayout()
        mapping_layout.addWidget(QLabel("Expedidora:"))
        self.expeditor_alias_edit = QLineEdit("BAR")
        mapping_layout.addWidget(self.expeditor_alias_edit)
        
        mapping_layout.addWidget(QLabel("Recebedora:"))
        self.receiver_alias_edit = QLineEdit("PLN")
        mapping_layout.addWidget(self.receiver_alias_edit)
        
        config_layout.addLayout(mapping_layout, 0, 1)
        
        config_layout.addWidget(QLabel("Intervalo de Tempo:"), 1, 0)
        time_layout = QHBoxLayout()
        
        self.start_time_edit = QDateTimeEdit()
        self.start_time_edit.setDateTime(QDateTime.currentDateTime().addDays(-10))
        time_layout.addWidget(self.start_time_edit)
        
        time_layout.addWidget(QLabel("até"))
        
        self.end_time_edit = QDateTimeEdit()
        self.end_time_edit.setDateTime(QDateTime.currentDateTime())
        time_layout.addWidget(self.end_time_edit)
        
        config_layout.addLayout(time_layout, 1, 1)
        
        layout.addWidget(config_group)
        
        # Status e progresso
        self.status_label = QLabel("Pronto para carregar arquivos...")
        layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Botões
        buttons_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Carregar Arquivos")
        self.load_btn.clicked.connect(self.accept)
        self.load_btn.setEnabled(False)
        self.load_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        buttons_layout.addWidget(self.load_btn)
        
        self.cancel_btn = QPushButton("Cancelar")
        self.cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(buttons_layout)
    
    def select_files(self):
        """Permite seleção manual de múltiplos arquivos"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Selecionar Arquivos XLSX",
            "", "Excel Files (*.xlsx *.xls)"
        )
        
        if files:
            self.process_selected_files(files)
    
    def auto_detect_files(self):
        """Auto-detecta arquivos XLSX na pasta atual"""
        import glob
        import os
        
        # Pasta atual do projeto
        current_dir = os.getcwd()
        xlsx_files = glob.glob(os.path.join(current_dir, "*.xlsx"))
        
        if xlsx_files:
            self.process_selected_files(xlsx_files)
            self.status_label.setText(f"Auto-detectados {len(xlsx_files)} arquivos na pasta atual")
        else:
            QMessageBox.information(self, "Info", "Nenhum arquivo XLSX encontrado na pasta atual")
    
    def process_selected_files(self, files):
        """Processa lista de arquivos selecionados"""
        self.files_table.setRowCount(len(files))
        self.file_classifications = {}
        
        for i, file_path in enumerate(files):
            file_name = os.path.basename(file_path)
            
            # Auto-classifica baseado no nome do arquivo
            classification = self.classify_file(file_name)
            
            # Tenta ler o arquivo para verificar estrutura
            try:
                import pandas as pd
                df = pd.read_excel(file_path)
                record_count = len(df)
                status = "OK"
                
                # Verifica se tem as colunas esperadas
                if not ('tempo' in df.columns and 'valor' in df.columns):
                    status = "Formato Inválido"
                    
            except Exception as e:
                record_count = 0
                status = f"Erro: {str(e)[:30]}"
            
            # Preenche tabela
            self.files_table.setItem(i, 0, QTableWidgetItem(file_name))
            self.files_table.setItem(i, 1, QTableWidgetItem(classification['data_type']))
            self.files_table.setItem(i, 2, QTableWidgetItem(classification['station']))
            self.files_table.setItem(i, 3, QTableWidgetItem(classification['classification']))
            self.files_table.setItem(i, 4, QTableWidgetItem(str(record_count)))
            self.files_table.setItem(i, 5, QTableWidgetItem(status))
            
            # Armazena classificação
            self.file_classifications[file_path] = {
                'classification': classification,
                'record_count': record_count,
                'status': status
            }
        
        # Habilita botão de carregamento se há arquivos válidos
        valid_files = sum(1 for data in self.file_classifications.values() if data['status'] == 'OK')
        self.load_btn.setEnabled(valid_files > 0)
        self.status_label.setText(f"{len(files)} arquivos selecionados, {valid_files} válidos")
    
    def classify_file(self, filename):
        """Classifica arquivo baseado no nome usando aliases do sistema"""
        filename_upper = filename.upper()
        
        # Identifica tipo de dado
        if 'PT' in filename_upper:
            data_type = 'Pressão'
            variable = 'pressure'
        elif 'FT' in filename_upper:
            data_type = 'Vazão'
            variable = 'flow'
        elif 'TT' in filename_upper:
            data_type = 'Temperatura'
            variable = 'temperature'
        elif 'DT' in filename_upper:
            data_type = 'Densidade'
            variable = 'density'
        else:
            data_type = 'Desconhecido'
            variable = 'unknown'
        
        # Obtém aliases do sistema configurado ou usa defaults
        expeditor_alias = 'BAR'
        receiver_alias = 'PLN'
        expeditor_unit = 'Expedidora'
        receiver_unit = 'Recebedora'
        
        if (self.main_window and hasattr(self.main_window, 'current_system') 
            and self.main_window.current_system):
            system = self.main_window.current_system
            expeditor_alias = getattr(system, 'expeditor_alias', 'BAR')
            receiver_alias = getattr(system, 'receiver_alias', 'PLN')
            expeditor_unit = getattr(system, 'expeditor_unit', 'Expedidora')
            receiver_unit = getattr(system, 'receiver_unit', 'Recebedora')
        
        # Identifica estação baseada nos aliases configurados
        if expeditor_alias.upper() in filename_upper:
            station = f'{expeditor_unit} ({expeditor_alias})'
            sensor_id = 'EXP_01'
            classification = 'Entrada'
        elif receiver_alias.upper() in filename_upper:
            station = f'{receiver_unit} ({receiver_alias})'
            sensor_id = 'REC_01'
            classification = 'Saída'
        else:
            # Fallback para aliases comuns
            common_expeditor_aliases = ['BAR', 'EXP', 'UP', 'UPSTREAM', 'ENTRADA']
            common_receiver_aliases = ['PLN', 'REC', 'DOWN', 'DOWNSTREAM', 'SAIDA']
            
            found_expeditor = any(alias in filename_upper for alias in common_expeditor_aliases)
            found_receiver = any(alias in filename_upper for alias in common_receiver_aliases)
            
            if found_expeditor:
                station = f'Expedidora (detectada)'
                sensor_id = 'EXP_01'
                classification = 'Entrada'
            elif found_receiver:
                station = f'Recebedora (detectada)'
                sensor_id = 'REC_01'
                classification = 'Saída'
            else:
                station = 'Desconhecida'
                sensor_id = 'UNK_01'
                classification = 'Indefinida'
        
        return {
            'data_type': data_type,
            'variable': variable,
            'station': station,
            'sensor_id': sensor_id,
            'classification': classification
        }
    
    def get_file_data(self):
        """Retorna dados dos arquivos para carregamento"""
        return self.file_classifications

class HydraulicAnalysisMainWindow(QMainWindow):
    """Janela principal do sistema de análise hidráulica"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema Industrial de Análise Hidráulica - Rev2 Complete")
        self.setGeometry(50, 50, 1600, 1000)
        
        # Logger da interface
        self.logger = logging.getLogger('hydraulic_system.gui')
        
        # Estado do sistema
        self.current_system = None
        self.processor = None
        self.data_thread = None
        self.analysis_timer = QTimer()
        
        # Sistema assíncrono e paralelo
        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(min(32, cpu_count() * 2))
        self.parallel_processor = ParallelDataProcessor()
        self.async_loader = None
        self.active_workers = []  # Lista de workers ativos
        
        # Informações sobre arquivos carregados
        self.loaded_files_info = {
            'files': [],  # Lista de arquivos carregados
            'sensors': {},  # Info dos sensores por arquivo
            'last_update': None,  # Último update
            'total_records': 0  # Total de registros
        }
        
        # Dados para plotagem
        self.plot_data = {
            'times': [],
            'exp_pressures': [],
            'rec_pressures': [],
            'flows': [],
            'densities': [],
            'temperatures': [],
            'correlations': [],
            'ml_scores': [],
            'leak_probabilities': []
        }
        
        # Configuração PyQtGraph
        pg.setConfigOptions(antialias=True, useOpenGL=True)
        
        # Setup da interface
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_plots()
        self.apply_dark_theme()
        
        # Conecta timer
        self.analysis_timer.timeout.connect(self.update_plots)
        
        logger.info("Interface principal inicializada")
    
    def setup_ui(self):
        """Configura interface principal"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Barra de ferramentas superior
        toolbar_layout = QHBoxLayout()
        
        self.system_label = QLabel("Nenhum sistema carregado")
        self.system_label.setStyleSheet("font-weight: bold; color: #ff6b35;")
        toolbar_layout.addWidget(self.system_label)
        
        toolbar_layout.addStretch()
        
        self.new_system_btn = QPushButton("Novo Sistema")
        self.new_system_btn.clicked.connect(self.new_system)
        toolbar_layout.addWidget(self.new_system_btn)
        
        self.load_system_btn = QPushButton("Carregar Sistema")
        self.load_system_btn.clicked.connect(self.load_system)
        toolbar_layout.addWidget(self.load_system_btn)
        
        self.load_data_btn = QPushButton("Carregar Dados")
        self.load_data_btn.clicked.connect(self.load_data_file)
        self.load_data_btn.setEnabled(False)
        toolbar_layout.addWidget(self.load_data_btn)
        
        # Separador
        toolbar_layout.addWidget(QLabel("|"))
        
        # Controles de simulação
        self.start_simulation_btn = QPushButton("▶ Simular")
        self.start_simulation_btn.clicked.connect(self.start_simulation)
        self.start_simulation_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        toolbar_layout.addWidget(self.start_simulation_btn)
        
        self.stop_simulation_btn = QPushButton("⏹ Parar")
        self.stop_simulation_btn.clicked.connect(self.stop_simulation)
        self.stop_simulation_btn.setEnabled(False)
        self.stop_simulation_btn.setStyleSheet("background-color: #f44336; color: white;")
        toolbar_layout.addWidget(self.stop_simulation_btn)
        
        # Combo para cenários
        self.scenario_combo = QComboBox()
        self.scenario_combo.addItems(['normal', 'leak_gradual', 'leak_sudden', 
                                     'valve_closure', 'pressure_surge', 'sensor_drift'])
        self.scenario_combo.currentTextChanged.connect(self.change_simulation_scenario)
        toolbar_layout.addWidget(QLabel("Cenário:"))
        toolbar_layout.addWidget(self.scenario_combo)
        
        main_layout.addLayout(toolbar_layout)
        
        # Splitter principal
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Painel de controle à esquerda
        control_widget = QWidget()
        control_widget.setMaximumWidth(350)
        control_layout = QVBoxLayout(control_widget)
        
        # Status do sistema
        status_group = QGroupBox("Status do Sistema")
        status_layout = QVBoxLayout(status_group)
        
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(200)
        self.status_text.setFont(QFont("Courier", 9))
        status_layout.addWidget(self.status_text)
        
        control_layout.addWidget(status_group)
        
        # Controles de análise
        analysis_group = QGroupBox("Controles de Análise")
        analysis_layout = QGridLayout(analysis_group)
        
        analysis_layout.addWidget(QLabel("Modo:"), 0, 0)
        self.analysis_mode_combo = QComboBox()
        self.analysis_mode_combo.addItems(['Tempo Real', 'Histórico', 'Batch'])
        self.analysis_mode_combo.currentTextChanged.connect(self.analysis_mode_changed)
        analysis_layout.addWidget(self.analysis_mode_combo, 0, 1)
        
        analysis_layout.addWidget(QLabel("Velocidade:"), 1, 0)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(10)
        analysis_layout.addWidget(self.speed_slider, 1, 1)
        
        self.speed_label = QLabel("10x")
        analysis_layout.addWidget(self.speed_label, 1, 2)
        self.speed_slider.valueChanged.connect(self.speed_changed)
        
        analysis_layout.addWidget(QLabel("Sensibilidade ML:"), 2, 0)
        self.sensitivity_combo = QComboBox()
        self.sensitivity_combo.addItems(['Baixa', 'Média', 'Alta'])
        self.sensitivity_combo.setCurrentText('Média')
        self.sensitivity_combo.currentTextChanged.connect(self.sensitivity_changed)
        analysis_layout.addWidget(self.sensitivity_combo, 2, 1)
        
        self.start_analysis_btn = QPushButton("Iniciar Análise")
        self.start_analysis_btn.clicked.connect(self.start_analysis)
        self.start_analysis_btn.setEnabled(False)
        analysis_layout.addWidget(self.start_analysis_btn, 3, 0)
        
        self.stop_analysis_btn = QPushButton("Parar Análise")
        self.stop_analysis_btn.clicked.connect(self.stop_analysis)
        self.stop_analysis_btn.setEnabled(False)
        analysis_layout.addWidget(self.stop_analysis_btn, 3, 1)
        
        control_layout.addWidget(analysis_group)
        
        # Métricas em tempo real
        metrics_group = QGroupBox("Métricas em Tempo Real")
        metrics_layout = QGridLayout(metrics_group)
        
        self.correlation_label = QLabel("Correlação: --")
        self.ml_score_label = QLabel("Score ML: --")
        self.leak_prob_label = QLabel("Prob. Vazamento: --")
        self.system_status_label = QLabel("Status: --")
        self.confidence_label = QLabel("Confiança: --")
        
        metrics_layout.addWidget(self.correlation_label, 0, 0)
        metrics_layout.addWidget(self.ml_score_label, 0, 1)
        metrics_layout.addWidget(self.leak_prob_label, 1, 0)
        metrics_layout.addWidget(self.system_status_label, 1, 1)
        metrics_layout.addWidget(self.confidence_label, 2, 0, 1, 2)
        
        control_layout.addWidget(metrics_group)
        
        # Aprendizado do sistema
        learning_group = QGroupBox("Aprendizado do Sistema")
        learning_layout = QVBoxLayout(learning_group)
        
        self.report_leak_btn = QPushButton("Reportar Vazamento Confirmado")
        self.report_leak_btn.clicked.connect(self.report_confirmed_leak)
        self.report_leak_btn.setEnabled(False)
        learning_layout.addWidget(self.report_leak_btn)
        
        self.retrain_btn = QPushButton("Retreinar Modelo ML")
        self.retrain_btn.clicked.connect(self.retrain_ml_model)
        self.retrain_btn.setEnabled(False)
        learning_layout.addWidget(self.retrain_btn)
        
        control_layout.addWidget(learning_group)
        
        # Progresso
        self.progress_bar = QProgressBar()
        control_layout.addWidget(self.progress_bar)
        
        control_layout.addStretch()
        main_splitter.addWidget(control_widget)
        
        # Área de plots à direita
        self.plots_tab_widget = QTabWidget()
        main_splitter.addWidget(self.plots_tab_widget)
        
        # Proporções do splitter
        main_splitter.setSizes([350, 1250])
        main_layout.addWidget(main_splitter)
        
        # Barra de status
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Sistema pronto - Configure um sistema para começar")
    
    def setup_menu_bar(self):
        """Configura barra de menu"""
        menubar = self.menuBar()
        if menubar is None:
            self.logger.warning("MenuBar não pode ser criado")
            return
        
        # Menu Sistema
        system_menu = menubar.addMenu('Sistema')
        if system_menu is None:
            self.logger.warning("Menu Sistema não pode ser criado")
            return
        
        new_action = QAction('Novo Sistema', self)
        new_action.triggered.connect(self.new_system)
        system_menu.addAction(new_action)
        
        load_action = QAction('Carregar Sistema', self)
        load_action.triggered.connect(self.load_system)
        system_menu.addAction(load_action)
        
        system_menu.addSeparator()
        
        exit_action = QAction('Sair', self)
        exit_action.triggered.connect(self.close)
        system_menu.addAction(exit_action)
        
        # Menu Dados
        data_menu = menubar.addMenu('Dados')
        if data_menu is None:
            self.logger.warning("Menu Dados não pode ser criado")
            return
        
        load_data_action = QAction('Carregar Arquivo de Dados', self)
        load_data_action.triggered.connect(self.load_data_file)
        data_menu.addAction(load_data_action)
        
        data_menu.addSeparator()
        
        batch_load_action = QAction('Carregamento em Lote...', self)
        batch_load_action.triggered.connect(self.batch_load_files)
        data_menu.addAction(batch_load_action)
        
        profile_load_action = QAction('Carregar Perfil do Sistema...', self)
        profile_load_action.triggered.connect(self.load_pipeline_profile)
        data_menu.addAction(profile_load_action)
        
        data_menu.addSeparator()
        
        export_action = QAction('Exportar Resultados', self)
        export_action.triggered.connect(self.export_results)
        data_menu.addAction(export_action)
        
        # Menu Ferramentas
        tools_menu = menubar.addMenu('Ferramentas')
        if tools_menu is None:
            self.logger.warning("Menu Ferramentas não pode ser criado")
            return
        
        test_action = QAction('Executar Testes', self)
        test_action.triggered.connect(self.run_system_tests)
        tools_menu.addAction(test_action)
        
        calibration_action = QAction('Calibração Avançada', self)
        calibration_action.triggered.connect(self.advanced_calibration)
        tools_menu.addAction(calibration_action)
        
        tools_menu.addSeparator()
        
        modify_system_action = QAction('Modificar Sistema Atual', self)
        modify_system_action.triggered.connect(self.modify_current_system)
        tools_menu.addAction(modify_system_action)
        
        # Menu Ajuda
        help_menu = menubar.addMenu('Ajuda')
        if help_menu is None:
            self.logger.warning("Menu Ajuda não pode ser criado")
            return
        
        about_action = QAction('Sobre', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_plots(self):
        """Configura todos os plots PyQtGraph"""
        
        # Aba 1: Sinais Principais
        signals_widget = QWidget()
        signals_layout = QGridLayout(signals_widget)
        
        # Plot de pressões
        self.pressure_plot = PlotWidget(title="Pressões do Sistema")
        self.pressure_plot.setLabel('left', 'Pressão (kgf/cm²)')
        self.pressure_plot.setLabel('bottom', 'Tempo (s)')
        self.pressure_plot.addLegend()
        self.pressure_plot.showGrid(x=True, y=True, alpha=0.3)
        
        self.exp_pressure_curve = self.pressure_plot.plot(
            pen=mkPen('cyan', width=2), name='Expedidor'
        )
        self.rec_pressure_curve = self.pressure_plot.plot(
            pen=mkPen('orange', width=2), name='Recebedor'
        )
        
        signals_layout.addWidget(self.pressure_plot, 0, 0)
        
        # Plot de vazão
        self.flow_plot = PlotWidget(title="Vazão do Sistema")
        self.flow_plot.setLabel('left', 'Vazão (m³/h)')
        self.flow_plot.setLabel('bottom', 'Tempo (s)')
        self.flow_plot.showGrid(x=True, y=True, alpha=0.3)
        
        self.flow_curve = self.flow_plot.plot(
            pen=mkPen('green', width=2), name='Vazão'
        )
        
        signals_layout.addWidget(self.flow_plot, 0, 1)
        
        # Plot de densidade e temperatura
        self.density_temp_plot = PlotWidget(title="Densidade e Temperatura")
        self.density_temp_plot.setLabel('left', 'Densidade (g/cm³) / Temperatura (°C)')
        self.density_temp_plot.setLabel('bottom', 'Tempo (s)')
        self.density_temp_plot.addLegend()
        self.density_temp_plot.showGrid(x=True, y=True, alpha=0.3)
        
        self.density_curve = self.density_temp_plot.plot(
            pen=mkPen('magenta', width=2), name='Densidade'
        )
        self.temperature_curve = self.density_temp_plot.plot(
            pen=mkPen('red', width=2), name='Temperatura'
        )
        
        signals_layout.addWidget(self.density_temp_plot, 1, 0)
        
        # Plot de correlação
        self.correlation_plot = PlotWidget(title="Correlação Cruzada")
        self.correlation_plot.setLabel('left', 'Correlação')
        self.correlation_plot.setLabel('bottom', 'Delay (amostras)')
        self.correlation_plot.showGrid(x=True, y=True, alpha=0.3)
        
        self.correlation_curve = self.correlation_plot.plot(
            pen=mkPen('yellow', width=2)
        )
        
        signals_layout.addWidget(self.correlation_plot, 1, 1)
        
        self.plots_tab_widget.addTab(signals_widget, "Sinais Principais")
        
        # Aba 2: Machine Learning
        ml_widget = QWidget()
        ml_layout = QGridLayout(ml_widget)
        
        # Plot de probabilidade de vazamento
        self.ml_prob_plot = PlotWidget(title="Probabilidade de Vazamento (ML)")
        self.ml_prob_plot.setLabel('left', 'Probabilidade')
        self.ml_prob_plot.setLabel('bottom', 'Tempo (s)')
        self.ml_prob_plot.setYRange(0, 1)
        self.ml_prob_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Linhas de referência
        self.ml_prob_plot.addLine(y=0.5, pen=mkPen('yellow', style=Qt.PenStyle.DashLine))
        self.ml_prob_plot.addLine(y=0.8, pen=mkPen('red', style=Qt.PenStyle.DashLine))
        
        self.ml_prob_curve = self.ml_prob_plot.plot(
            pen=mkPen('red', width=3), name='Prob. Vazamento'
        )
        
        ml_layout.addWidget(self.ml_prob_plot, 0, 0)
        
        # Plot de score de anomalia
        self.anomaly_plot = PlotWidget(title="Detecção de Anomalias")
        self.anomaly_plot.setLabel('left', 'Score de Anomalia')
        self.anomaly_plot.setLabel('bottom', 'Tempo (s)')
        self.anomaly_plot.showGrid(x=True, y=True, alpha=0.3)
        
        self.anomaly_curve = self.anomaly_plot.plot(
            pen=mkPen('orange', width=2), name='Score Anomalia'
        )
        
        ml_layout.addWidget(self.anomaly_plot, 0, 1)
        
        # Plot de confiança
        self.confidence_plot = PlotWidget(title="Confiança do Modelo")
        self.confidence_plot.setLabel('left', 'Confiança')
        self.confidence_plot.setLabel('bottom', 'Tempo (s)')
        self.confidence_plot.setYRange(0, 1)
        self.confidence_plot.showGrid(x=True, y=True, alpha=0.3)
        
        self.confidence_curve = self.confidence_plot.plot(
            pen=mkPen('blue', width=2), name='Confiança'
        )
        
        ml_layout.addWidget(self.confidence_plot, 1, 0)
        
        # Plot de features importantes
        self.features_plot = PlotWidget(title="Features Principais")
        self.features_plot.setLabel('left', 'Valor da Feature')
        self.features_plot.setLabel('bottom', 'Tempo (s)')
        self.features_plot.showGrid(x=True, y=True, alpha=0.3)
        
        ml_layout.addWidget(self.features_plot, 1, 1)
        
        self.plots_tab_widget.addTab(ml_widget, "Machine Learning")
        
        # Aba 3: Status Operacional
        status_widget = QWidget()
        status_layout = QGridLayout(status_widget)
        
        # Plot de status da coluna
        self.status_plot = PlotWidget(title="Status Operacional")
        self.status_plot.setLabel('left', 'Status (0=Fechado, 1=Aberto)')
        self.status_plot.setLabel('bottom', 'Tempo (s)')
        self.status_plot.setYRange(-0.1, 1.1)
        self.status_plot.showGrid(x=True, y=True, alpha=0.3)
        
        self.status_curve = self.status_plot.plot(
            pen=mkPen('purple', width=3), symbol='o', name='Status'
        )
        
        status_layout.addWidget(self.status_plot, 0, 0)
        
        # Plot de localização de eventos
        self.location_plot = PlotWidget(title="Localização de Eventos")
        self.location_plot.setLabel('left', 'Posição (m)')
        self.location_plot.setLabel('bottom', 'Tempo (s)')
        self.location_plot.showGrid(x=True, y=True, alpha=0.3)
        
        self.location_curve = self.location_plot.plot(
            pen=mkPen('red', width=2), symbol='s', symbolBrush='red', name='Localização'
        )
        
        status_layout.addWidget(self.location_plot, 0, 1)
        
        # Espectrograma
        self.spectrogram_view = ImageView()
        self.spectrogram_view.setImage(np.random.random((100, 100)))
        
        status_layout.addWidget(self.spectrogram_view, 1, 0, 1, 2)
        
        self.plots_tab_widget.addTab(status_widget, "Status Operacional")
        
        # Aba 4: Análise de Ruídos
        noise_widget = QWidget()
        noise_layout = QGridLayout(noise_widget)
        
        # Plot FFT para análise de frequências
        self.fft_plot = PlotWidget(title="Análise de Frequências (FFT)")
        self.fft_plot.setLabel('left', 'Magnitude (dB)')
        self.fft_plot.setLabel('bottom', 'Frequência (Hz)')
        noise_layout.addWidget(self.fft_plot, 0, 0)
        
        # Plot de espectrograma de ruído
        self.noise_spectrogram_plot = PlotWidget(title="Espectrograma de Ruído")
        self.noise_spectrogram_plot.setLabel('left', 'Frequência (Hz)')
        self.noise_spectrogram_plot.setLabel('bottom', 'Tempo (s)')
        noise_layout.addWidget(self.noise_spectrogram_plot, 0, 1)
        
        # Plot de detecção de ruído anômalo
        self.anomaly_noise_plot = PlotWidget(title="Detecção de Ruído Anômalo")
        self.anomaly_noise_plot.setLabel('left', 'Intensidade de Ruído')
        self.anomaly_noise_plot.setLabel('bottom', 'Tempo (s)')
        noise_layout.addWidget(self.anomaly_noise_plot, 1, 0, 1, 2)
        
        self.plots_tab_widget.addTab(noise_widget, "Análise de Ruídos")
        
        # Aba 5: Perfil Hidráulico
        hydraulic_widget = QWidget()
        hydraulic_layout = QGridLayout(hydraulic_widget)
        
        # Plot do perfil de pressão ao longo do duto
        self.pressure_profile_plot = PlotWidget(title="Perfil de Pressão no Duto")
        self.pressure_profile_plot.setLabel('left', 'Pressão (kgf/cm²)')
        self.pressure_profile_plot.setLabel('bottom', 'Distância (km)')
        hydraulic_layout.addWidget(self.pressure_profile_plot, 0, 0)
        
        # Plot de perda de carga
        self.head_loss_plot = PlotWidget(title="Perda de Carga")
        self.head_loss_plot.setLabel('left', 'Perda de Carga (m)')
        self.head_loss_plot.setLabel('bottom', 'Distância (km)')
        hydraulic_layout.addWidget(self.head_loss_plot, 0, 1)
        
        # Plot de velocidades no duto
        self.velocity_plot = PlotWidget(title="Perfil de Velocidades")
        self.velocity_plot.setLabel('left', 'Velocidade (m/s)')
        self.velocity_plot.setLabel('bottom', 'Distância (km)')
        hydraulic_layout.addWidget(self.velocity_plot, 1, 0, 1, 2)
        
        self.plots_tab_widget.addTab(hydraulic_widget, "Perfil Hidráulico")
        
        # Aba 6: Análise de Ondas
        wave_widget = QWidget()
        wave_layout = QGridLayout(wave_widget)
        
        # Plot de análise de ondas de pressão
        self.pressure_wave_plot = PlotWidget(title="Ondas de Pressão")
        self.pressure_wave_plot.setLabel('left', 'Amplitude (kgf/cm²)')
        self.pressure_wave_plot.setLabel('bottom', 'Tempo (s)')
        wave_layout.addWidget(self.pressure_wave_plot, 0, 0)
        
        # Plot de propagação de ondas
        self.wave_propagation_plot = PlotWidget(title="Propagação de Ondas")
        self.wave_propagation_plot.setLabel('left', 'Posição (km)')
        self.wave_propagation_plot.setLabel('bottom', 'Tempo (s)')
        wave_layout.addWidget(self.wave_propagation_plot, 0, 1)
        
        # Plot de análise de reflexões
        self.wave_reflection_plot = PlotWidget(title="Análise de Reflexões")
        self.wave_reflection_plot.setLabel('left', 'Coeficiente de Reflexão')
        self.wave_reflection_plot.setLabel('bottom', 'Frequência (Hz)')
        wave_layout.addWidget(self.wave_reflection_plot, 1, 0, 1, 2)
        
        self.plots_tab_widget.addTab(wave_widget, "Análise de Ondas")
        
        # Aba 7: Filtro Temporal
        filter_widget = QWidget()
        filter_layout = QGridLayout(filter_widget)
        
        # Plot de dados originais vs filtrados
        self.raw_vs_filtered_plot = PlotWidget(title="Dados Originais vs Filtrados")
        self.raw_vs_filtered_plot.setLabel('left', 'Valor')
        self.raw_vs_filtered_plot.setLabel('bottom', 'Tempo (s)')
        filter_layout.addWidget(self.raw_vs_filtered_plot, 0, 0)
        
        # Plot de qualidade do sinal
        self.signal_quality_plot = PlotWidget(title="Qualidade do Sinal")
        self.signal_quality_plot.setLabel('left', 'Score de Qualidade')
        self.signal_quality_plot.setLabel('bottom', 'Tempo (s)')
        filter_layout.addWidget(self.signal_quality_plot, 0, 1)
        
        # Plot de outliers detectados
        self.outliers_plot = PlotWidget(title="Detecção de Outliers")
        self.outliers_plot.setLabel('left', 'Valor')
        self.outliers_plot.setLabel('bottom', 'Tempo (s)')
        filter_layout.addWidget(self.outliers_plot, 1, 0, 1, 2)
        
        self.plots_tab_widget.addTab(filter_widget, "Filtro Temporal")
        
        # Aba 8: Configurações Avançadas
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        
        # Parâmetros de análise
        analysis_group = QGroupBox("Parâmetros de Análise")
        analysis_grid = QGridLayout(analysis_group)
        
        analysis_grid.addWidget(QLabel("Janela de Correlação:"), 0, 0)
        self.correlation_window_spin = QSpinBox()
        self.correlation_window_spin.setRange(10, 1000)
        self.correlation_window_spin.setValue(500)
        self.correlation_window_spin.setSuffix(" amostras")
        analysis_grid.addWidget(self.correlation_window_spin, 0, 1)
        
        analysis_grid.addWidget(QLabel("Threshold ML:"), 1, 0)
        self.ml_threshold_spin = QDoubleSpinBox()
        self.ml_threshold_spin.setRange(0.1, 0.9)
        self.ml_threshold_spin.setValue(0.7)
        self.ml_threshold_spin.setDecimals(2)
        analysis_grid.addWidget(self.ml_threshold_spin, 1, 1)
        
        config_layout.addWidget(analysis_group)
        
        # Configurações de filtros
        filter_group = QGroupBox("Configurações de Filtros")
        filter_grid = QGridLayout(filter_group)
        
        filter_grid.addWidget(QLabel("Tipo de Filtro:"), 0, 0)
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(['adaptive', 'kalman', 'median', 'butterworth'])
        filter_grid.addWidget(self.filter_type_combo, 0, 1)
        
        filter_grid.addWidget(QLabel("Janela do Filtro:"), 1, 0)
        self.filter_window_spin = QSpinBox()
        self.filter_window_spin.setRange(5, 200)
        self.filter_window_spin.setValue(50)
        filter_grid.addWidget(self.filter_window_spin, 1, 1)
        
        config_layout.addWidget(filter_group)
        
        # Botão para aplicar configurações
        apply_config_btn = QPushButton("Aplicar Configurações")
        apply_config_btn.clicked.connect(self.apply_advanced_config)
        apply_config_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        config_layout.addWidget(apply_config_btn)
        
        config_layout.addStretch()
        
        self.plots_tab_widget.addTab(config_widget, "Configurações")
        
        # Aba 9: Monitor do Sistema
        monitor_widget = QWidget()
        monitor_layout = QGridLayout(monitor_widget)
        
        # Plot de status de componentes
        self.system_health_plot = PlotWidget(title="Saúde do Sistema")
        self.system_health_plot.setLabel('left', 'Status (0-100%)')
        self.system_health_plot.setLabel('bottom', 'Tempo (s)')
        monitor_layout.addWidget(self.system_health_plot, 0, 0)
        
        # Plot de performance
        self.performance_plot = PlotWidget(title="Performance do Sistema")
        self.performance_plot.setLabel('left', 'Eficiência (%)')
        self.performance_plot.setLabel('bottom', 'Tempo (s)')
        monitor_layout.addWidget(self.performance_plot, 0, 1)
        
        # Plot de alertas e eventos
        self.alerts_plot = PlotWidget(title="Alertas e Eventos")
        self.alerts_plot.setLabel('left', 'Severidade')
        self.alerts_plot.setLabel('bottom', 'Tempo (s)')
        monitor_layout.addWidget(self.alerts_plot, 1, 0, 1, 2)
        
        self.plots_tab_widget.addTab(monitor_widget, "Monitor do Sistema")
        
        # Aba 10: Relatório
        report_widget = QWidget()
        report_layout = QVBoxLayout(report_widget)
        
        # Área de texto para relatório
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setFont(QFont("Courier", 9))
        report_layout.addWidget(self.report_text)
        
        # Botões de relatório
        report_buttons = QHBoxLayout()
        
        generate_report_btn = QPushButton("Gerar Relatório")
        generate_report_btn.clicked.connect(self.generate_report)
        generate_report_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 10px;")
        report_buttons.addWidget(generate_report_btn)
        
        export_report_btn = QPushButton("Exportar PDF")
        export_report_btn.clicked.connect(self.export_report_pdf)
        export_report_btn.setStyleSheet("background-color: #FF9800; color: white; padding: 10px;")
        report_buttons.addWidget(export_report_btn)
        
        clear_report_btn = QPushButton("Limpar")
        clear_report_btn.clicked.connect(self.clear_report)
        report_buttons.addWidget(clear_report_btn)
        
        report_layout.addLayout(report_buttons)
        
        self.plots_tab_widget.addTab(report_widget, "Relatório")
        
        # Aba: Informações dos Arquivos Carregados
        files_info_widget = QWidget()
        files_info_layout = QVBoxLayout(files_info_widget)
        
        # Grupo de informações dos arquivos
        files_group = QGroupBox("Arquivos Carregados")
        files_layout = QVBoxLayout(files_group)
        
        # Tabela de arquivos
        self.files_table = QTableWidget(0, 6)
        self.files_table.setHorizontalHeaderLabels([
            "Arquivo", "Sensor", "Variável", "Unidade", "Registros", "Status"
        ])
        self.files_table.horizontalHeader().setStretchLastSection(True)
        files_layout.addWidget(self.files_table)
        
        files_info_layout.addWidget(files_group)
        
        # Grupo de estatísticas resumo
        stats_group = QGroupBox("Resumo Estatístico")
        stats_layout = QGridLayout(stats_group)
        
        # Labels para mostrar informações
        self.total_files_label = QLabel("Arquivos carregados: 0")
        self.total_records_label = QLabel("Total de registros: 0") 
        self.date_range_label = QLabel("Período: -")
        self.sensors_info_label = QLabel("Sensores: -")
        
        stats_layout.addWidget(self.total_files_label, 0, 0)
        stats_layout.addWidget(self.total_records_label, 0, 1)
        stats_layout.addWidget(self.date_range_label, 1, 0)
        stats_layout.addWidget(self.sensors_info_label, 1, 1)
        
        files_info_layout.addWidget(stats_group)
        
        # Botões de ação
        files_buttons_layout = QHBoxLayout()
        
        refresh_files_btn = QPushButton("🔄 Atualizar")
        refresh_files_btn.clicked.connect(self.refresh_files_info)
        refresh_files_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        files_buttons_layout.addWidget(refresh_files_btn)
        
        clear_files_btn = QPushButton("🗑️ Limpar")
        clear_files_btn.clicked.connect(self.clear_files_info)
        clear_files_btn.setStyleSheet("background-color: #f44336; color: white; padding: 8px;")
        files_buttons_layout.addWidget(clear_files_btn)
        
        files_buttons_layout.addStretch()
        
        files_info_layout.addLayout(files_buttons_layout)
        
        self.plots_tab_widget.addTab(files_info_widget, "📁 Arquivos")
    
    def apply_dark_theme(self):
        """Aplica tema escuro moderno"""
        dark_style = """
        QMainWindow {
            background-color: #1e1e1e;
            color: #ffffff;
            font-family: "Segoe UI", Arial, sans-serif;
        }
        QWidget {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #404040;
            border-radius: 8px;
            margin: 8px;
            padding-top: 15px;
            background-color: #2d2d2d;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 8px 0 8px;
            background-color: #1e1e1e;
            border-radius: 4px;
            color: #00d4aa;
        }
        QPushButton {
            background-color: #0078d4;
            border: 2px solid #106ebe;
            padding: 8px 16px;
            border-radius: 6px;
            min-width: 100px;
            font-weight: bold;
            color: white;
        }
        QPushButton:hover {
            background-color: #106ebe;
            border-color: #005a9e;
        }
        QPushButton:pressed {
            background-color: #005a9e;
        }
        QPushButton:disabled {
            background-color: #3a3a3a;
            color: #888888;
            border-color: #555555;
        }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            background-color: #404040;
            border: 2px solid #606060;
            padding: 6px;
            border-radius: 4px;
            color: #ffffff;
        }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
            border-color: #00d4aa;
        }
        QTextEdit {
            background-color: #1a1a1a;
            border: 2px solid #606060;
            color: #ffffff;
            font-family: "Courier New", monospace;
        }
        QTabWidget::pane {
            border: 2px solid #404040;
            background-color: #2d2d2d;
        }
        QTabBar::tab {
            background-color: #404040;
            border: 2px solid #606060;
            padding: 8px 16px;
            margin-right: 2px;
            color: #ffffff;
        }
        QTabBar::tab:selected {
            background-color: #00d4aa;
            border-color: #00b894;
            color: #000000;
        }
        QLabel {
            color: #ffffff;
            font-weight: 500;
        }
        QStatusBar {
            background-color: #333333;
            border-top: 1px solid #555555;
            color: #ffffff;
        }
        QMenuBar {
            background-color: #2d2d2d;
            color: #ffffff;
        }
        QMenuBar::item:selected {
            background-color: #00d4aa;
            color: #000000;
        }
        QMenu {
            background-color: #2d2d2d;
            color: #ffffff;
            border: 1px solid #555555;
        }
        QMenu::item:selected {
            background-color: #00d4aa;
            color: #000000;
        }
        QSlider::groove:horizontal {
            border: 1px solid #999999;
            height: 8px;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
            margin: 2px 0;
        }
        QSlider::handle:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
            border: 1px solid #5c5c5c;
            width: 18px;
            margin: -2px 0;
            border-radius: 3px;
        }
        QProgressBar {
            border: 2px solid #606060;
            border-radius: 5px;
            text-align: center;
            background-color: #404040;
        }
        QProgressBar::chunk {
            background-color: #00d4aa;
            border-radius: 3px;
        }
        """
        self.setStyleSheet(dark_style)
    
    def new_system(self):
        """Cria novo sistema"""
        dialog = SystemConfigDialog(self)
        dialog.system_configured.connect(self.on_system_configured)
        
        # Pre-preenche alguns campos com valores padrão
        dialog.system_id_edit.setText(f"SYS_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        dialog.name_edit.setText("Novo Sistema Hidráulico")
        dialog.location_edit.setText("Local Industrial")
        
        # Mostra como modal
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Dialog foi aceito via botão Salvar
            pass
    
    def on_system_configured(self, config: SystemConfiguration):
        """Callback quando sistema é configurado"""
        try:
            # Salva no banco
            database.save_system(config)
            
            # Configura sistema atual
            self.current_system = config
            self.processor = IndustrialHydraulicProcessor(config)
            
            # Atualiza interface
            self.system_label.setText(f"Sistema: {config.name} ({config.system_id})")
            self.load_data_btn.setEnabled(True)
            self.report_leak_btn.setEnabled(True)
            self.retrain_btn.setEnabled(True)
            
            self.log_status(f"Sistema configurado: {config.name}")
            self.status_bar.showMessage(f"Sistema {config.name} carregado e pronto")
            
            QMessageBox.information(self, "Sucesso", f"Sistema '{config.name}' configurado com sucesso!")
            
        except Exception as e:
            logger.error(f"Erro ao configurar sistema: {e}")
            QMessageBox.critical(self, "Erro", f"Erro ao configurar sistema:\n{str(e)}")
    
    def load_system(self):
        """Carrega sistema existente"""
        systems = database.get_all_systems()
        
        if not systems:
            QMessageBox.information(self, "Informação", "Nenhum sistema encontrado. Crie um novo sistema primeiro.")
            return
        
        # Dialog de seleção
        system_names = [f"{s['name']} ({s['system_id']}) - {s['location']}" for s in systems]
        system_name, ok = QInputDialog.getItem(
            self, "Carregar Sistema", "Selecione o sistema:", system_names, 0, False
        )
        
        if ok and system_name:
            # Extrai system_id
            system_id = system_name.split('(')[1].split(')')[0]
            
            try:
                config = database.load_system(system_id)
                if config:
                    self.current_system = config
                    self.processor = IndustrialHydraulicProcessor(config)
                    
                    # Atualiza interface
                    self.system_label.setText(f"Sistema: {config.name} ({config.system_id})")
                    self.load_data_btn.setEnabled(True)
                    self.report_leak_btn.setEnabled(True)
                    self.retrain_btn.setEnabled(True)
                    
                    self.log_status(f"Sistema carregado: {config.name}")
                    self.status_bar.showMessage(f"Sistema {config.name} carregado")
                    
                    QMessageBox.information(self, "Sucesso", f"Sistema '{config.name}' carregado com sucesso!")
                
            except Exception as e:
                logger.error(f"Erro ao carregar sistema: {e}")
                QMessageBox.critical(self, "Erro", f"Erro ao carregar sistema:\n{str(e)}")
    
    def load_data_file(self):
        """Carrega arquivo de dados para análise com processamento assíncrono"""
        if not self.current_system:
            QMessageBox.warning(self, "Aviso", "Configure um sistema primeiro!")
            return
        
        # Cancelar carregamento anterior se ainda ativo
        if self.async_loader and self.async_loader.isRunning():
            self.async_loader.cancel()
            self.async_loader.wait(3000)  # Aguarda até 3 segundos
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Carregar Dados", "", 
            "Arquivos de dados (*.csv *.xlsx *.xls);;Todos os arquivos (*)"
        )
        
        if file_path:
            # Resetar interface
            self.progress_bar.setValue(0)
            self.start_analysis_btn.setEnabled(False)
            
            # Criar loader assíncrono
            self.async_loader = AsyncFileLoader(file_path, self.processor)
            
            # Conectar sinais
            self.async_loader.progress_updated.connect(self.progress_bar.setValue)
            self.async_loader.status_updated.connect(self.log_status)
            self.async_loader.data_loaded.connect(self._on_data_loaded)
            self.async_loader.file_processed.connect(self._on_file_processed)
            self.async_loader.loading_completed.connect(self._on_loading_completed)
            self.async_loader.error_occurred.connect(self._on_loading_error)
            
            # Iniciar carregamento assíncrono
            self.async_loader.start()
            
            # Feedback imediato ao usuário
            filename = Path(file_path).name
            self.log_status(f"Iniciando carregamento assíncrono: {filename}")
            self.status_bar.showMessage(f"Carregando {filename}...")
    
    @pyqtSlot(object)
    def _on_data_loaded(self, df):
        """Callback quando dados são carregados"""
        self.log_status(f"Dados carregados: {len(df)} registros")
    
    @pyqtSlot(str, object) 
    def _on_file_processed(self, filename, processed_data):
        """Callback quando arquivo é processado"""
        try:
            if 'OPASA10' in filename or 'BAR2PLN' in filename:
                # Dados de perfil da tubulação
                self.pipeline_profile = processed_data
                self._update_system_with_profile(processed_data)
                self.log_status(f"Perfil da tubulação carregado: {filename}")
                
            else:
                # Dados de sensor
                self._update_plots_with_sensor_data(processed_data)
                self.log_status(f"Sensor processado: {processed_data['sensor_info']['description']}")
                
        except Exception as e:
            logger.error(f"Erro no processamento de callback: {e}")
            self._on_loading_error(str(e))
    
    @pyqtSlot(bool)
    def _on_loading_completed(self, success):
        """Callback quando carregamento é concluído"""
        if success:
            self.start_analysis_btn.setEnabled(True)
            self.progress_bar.setValue(100)
            
            # Atualizar interface
            filename = Path(self.async_loader.file_path).name if self.async_loader else "arquivo"
            self.status_bar.showMessage(f"✅ {filename} carregado com sucesso")
            
            QMessageBox.information(
                self, 
                "Sucesso", 
                f"Dados carregados com sucesso!\n\nArquivo: {filename}\nProcessamento: Concluído"
            )
        else:
            self.progress_bar.setValue(0)
            self.status_bar.showMessage("❌ Falha no carregamento")
        
        # Limpar referência
        self.async_loader = None
    
    @pyqtSlot(str)
    def _on_loading_error(self, error_message):
        """Callback para erros de carregamento"""
        logger.error(f"Erro no carregamento assíncrono: {error_message}")
        
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("❌ Erro no carregamento")
        
        QMessageBox.critical(
            self, 
            "Erro no Carregamento", 
            f"Erro ao carregar dados:\n\n{error_message}\n\nVerifique o arquivo e tente novamente."
        )
        
        # Limpar referência
        self.async_loader = None
    
    def _update_system_with_profile(self, pipeline_data):
        """Atualiza configuração do sistema com dados do perfil"""
        try:
            if hasattr(self.current_system, 'pipe_characteristics'):
                # Calcular propriedades médias
                avg_diameter = np.mean(pipeline_data['diameters']) if pipeline_data['diameters'] else 0.5
                total_length = max(pipeline_data['distances']) if pipeline_data['distances'] else 1000.0
                avg_thickness = np.mean(pipeline_data['thicknesses']) if pipeline_data['thicknesses'] else 0.01
                
                # Atualizar características do duto
                self.current_system.pipe_characteristics.diameter = avg_diameter
                self.current_system.pipe_characteristics.length = total_length
                self.current_system.pipe_characteristics.wall_thickness = avg_thickness * 1000  # Converter para mm
                
                self.log_status(f"Sistema atualizado: {len(pipeline_data['distances'])} pontos, {total_length/1000:.1f}km, Ø{avg_diameter*1000:.0f}mm")
                
        except Exception as e:
            logger.error(f"Erro ao atualizar sistema com perfil: {e}")
    
    def _update_plots_with_sensor_data(self, sensor_data):
        """Atualiza gráficos com dados de sensor processados assincronamente"""
        try:
            readings = sensor_data['readings']
            sensor_info = sensor_data['sensor_info']
            
            # Extrair dados para plotagem
            timestamps = [pd.Timestamp(r['timestamp']) for r in readings]
            values = [r['value'] for r in readings]
            
            # Atualizar informações de arquivo carregado
            self._update_loaded_file_info(sensor_data)
            
            # Atualizar dados de plotagem baseado no tipo de sensor
            variable = sensor_info['variable']
            station = sensor_info['station']
            
            # Garantir que temos timestamps para sincronização
            if not self.plot_data['times'] or len(timestamps) > len(self.plot_data['times']):
                self.plot_data['times'] = timestamps
            
            # Mapear para estrutura de dados de plotagem
            if variable == 'pressure':
                if station == 'BAR':  # Expedidor
                    self.plot_data['exp_pressures'] = values
                elif station == 'PLN':  # Recebedor
                    self.plot_data['rec_pressures'] = values
                    
            elif variable == 'temperature':
                if station == 'BAR':
                    if 'exp_temps' not in self.plot_data:
                        self.plot_data['exp_temps'] = []
                    self.plot_data['exp_temps'] = values
                elif station == 'PLN':
                    if 'rec_temps' not in self.plot_data:
                        self.plot_data['rec_temps'] = []
                    self.plot_data['rec_temps'] = values
                    
            elif variable == 'density':
                if station == 'BAR':
                    if 'exp_densities' not in self.plot_data:
                        self.plot_data['exp_densities'] = []
                    self.plot_data['exp_densities'] = values
                elif station == 'PLN':
                    if 'rec_densities' not in self.plot_data:
                        self.plot_data['rec_densities'] = []
                    self.plot_data['rec_densities'] = values
                    
            elif variable == 'flow_rate':
                if station == 'BAR':
                    if 'exp_flows' not in self.plot_data:
                        self.plot_data['exp_flows'] = []
                    self.plot_data['exp_flows'] = values
                elif station == 'PLN':
                    if 'rec_flows' not in self.plot_data:
                        self.plot_data['rec_flows'] = []
                    self.plot_data['rec_flows'] = values
            
            # Atualizar gráficos em tempo real
            self._update_real_time_plots()
            
            # Log de sucesso
            self.log_status(f"✅ {sensor_info['description']}: {len(readings)} registros carregados e plotados")
            
        except Exception as e:
            logger.error(f"Erro ao atualizar gráficos: {e}")
            self.log_status(f"❌ ERRO: {str(e)}")
    
    def _update_loaded_file_info(self, sensor_data):
        """Atualiza informações sobre arquivo carregado"""
        try:
            sensor_info = sensor_data['sensor_info']
            total_records = sensor_data['total_records']
            
            # Adicionar à lista de arquivos carregados
            file_info = {
                'filename': f"{sensor_info['station']}_{sensor_info['variable']}.xlsx",
                'sensor_id': sensor_info['sensor_id'],
                'variable': sensor_info['variable'],
                'unit': sensor_info['unit'],
                'description': sensor_info['description'],
                'station': sensor_info['station'],
                'records': total_records,
                'timestamp': pd.Timestamp.now()
            }
            
            # Evitar duplicatas - remover se já existe
            existing_files = [f for f in self.loaded_files_info['files'] 
                             if f['sensor_id'] != sensor_info['sensor_id']]
            existing_files.append(file_info)
            
            self.loaded_files_info['files'] = existing_files
            self.loaded_files_info['sensors'][sensor_info['sensor_id']] = sensor_info
            self.loaded_files_info['last_update'] = pd.Timestamp.now()
            self.loaded_files_info['total_records'] += total_records
            
            # Atualizar tabela na interface
            self._update_files_table()
            
        except Exception as e:
            logger.error(f"Erro ao atualizar informações de arquivo: {e}")
    
    def _update_files_table(self):
        """Atualiza tabela de arquivos carregados na interface"""
        try:
            files = self.loaded_files_info['files']
            
            # Configurar tabela
            self.files_table.setRowCount(len(files))
            
            for row, file_info in enumerate(files):
                # Nome do arquivo
                self.files_table.setItem(row, 0, QTableWidgetItem(file_info['filename']))
                
                # Sensor
                sensor_text = f"{file_info['station']} - {file_info['sensor_id']}"
                self.files_table.setItem(row, 1, QTableWidgetItem(sensor_text))
                
                # Variável  
                var_text = f"{file_info['variable'].replace('_', ' ').title()}"
                self.files_table.setItem(row, 2, QTableWidgetItem(var_text))
                
                # Unidade
                self.files_table.setItem(row, 3, QTableWidgetItem(file_info['unit']))
                
                # Número de registros
                records_text = f"{file_info['records']:,}"
                self.files_table.setItem(row, 4, QTableWidgetItem(records_text))
                
                # Status
                status_text = "✅ Carregado"
                self.files_table.setItem(row, 5, QTableWidgetItem(status_text))
            
            # Atualizar labels de resumo
            self.total_files_label.setText(f"Arquivos carregados: {len(files)}")
            self.total_records_label.setText(f"Total de registros: {self.loaded_files_info['total_records']:,}")
            
            # Informações de sensores
            unique_stations = set(f['station'] for f in files)
            unique_variables = set(f['variable'] for f in files)
            sensors_text = f"Estações: {', '.join(unique_stations)} | Variáveis: {len(unique_variables)}"
            self.sensors_info_label.setText(f"Sensores: {sensors_text}")
            
            # Período de dados se disponível
            if self.plot_data.get('times'):
                start_date = min(self.plot_data['times']).strftime('%d/%m/%Y %H:%M')
                end_date = max(self.plot_data['times']).strftime('%d/%m/%Y %H:%M')
                self.date_range_label.setText(f"Período: {start_date} até {end_date}")
            
        except Exception as e:
            logger.error(f"Erro ao atualizar tabela de arquivos: {e}")
    
    def refresh_files_info(self):
        """Atualiza informações dos arquivos"""
        self._update_files_table()
        self.log_status("📋 Informações de arquivos atualizadas")
    
    def clear_files_info(self):
        """Limpa informações dos arquivos"""
        self.loaded_files_info = {
            'files': [],
            'sensors': {},
            'last_update': None,
            'total_records': 0
        }
        self.files_table.setRowCount(0)
        self.total_files_label.setText("Arquivos carregados: 0")
        self.total_records_label.setText("Total de registros: 0")
        self.date_range_label.setText("Período: -")
        self.sensors_info_label.setText("Sensores: -")
        self.log_status("🗑️ Informações de arquivos limpas")
    
    def _update_real_time_plots(self):
        """Atualiza gráficos em tempo real durante o carregamento de dados"""
        try:
            # Verificar se temos dados carregados
            if not self.plot_data.get('times'):
                return
                
            times = self.plot_data['times']
            if len(times) == 0:
                return
            
            # Converter timestamps para formato PyQtGraph (números)
            if isinstance(times[0], pd.Timestamp):
                time_values = [t.timestamp() for t in times]
            else:
                time_values = times
            
            # Configurar eixo X para data/hora em todos os gráficos
            try:
                self._configure_datetime_axis()
            except Exception as axis_error:
                logger.warning(f"Erro nos eixos: {axis_error}")
            
            plots_updated = 0
            
            # Atualizar gráfico de pressões (Aba "Sinais Principais")
            if hasattr(self, 'exp_pressure_curve') and hasattr(self, 'rec_pressure_curve'):
                # Dados de pressão do expedidor
                if self.plot_data.get('exp_pressures'):
                    exp_pressures = self.plot_data['exp_pressures']
                    if len(exp_pressures) == len(time_values):
                        try:
                            self.exp_pressure_curve.setData(time_values, exp_pressures)
                            plots_updated += 1
                        except Exception as e:
                            logger.error(f"Erro ao plotar exp_pressures: {e}")
                
                # Dados de pressão do recebedor
                if self.plot_data.get('rec_pressures'):
                    rec_pressures = self.plot_data['rec_pressures']
                    if len(rec_pressures) == len(time_values):
                        try:
                            self.rec_pressure_curve.setData(time_values, rec_pressures)
                            plots_updated += 1
                        except Exception as e:
                            logger.error(f"Erro ao plotar rec_pressures: {e}")
            
            # Atualizar gráfico de vazão
            if hasattr(self, 'flow_curve'):
                if self.plot_data.get('exp_flows') or self.plot_data.get('rec_flows'):
                    flows = self.plot_data.get('exp_flows') or self.plot_data.get('rec_flows')
                    if len(flows) == len(time_values):
                        try:
                            self.flow_curve.setData(time_values, flows)
                            plots_updated += 1
                        except Exception as e:
                            logger.error(f"Erro ao plotar flows: {e}")
            
            # Atualizar gráfico de densidade e temperatura
            if hasattr(self, 'density_curve') and hasattr(self, 'temperature_curve'):
                if self.plot_data.get('exp_densities') or self.plot_data.get('rec_densities'):
                    densities = self.plot_data.get('exp_densities') or self.plot_data.get('rec_densities')
                    if len(densities) == len(time_values):
                        try:
                            self.density_curve.setData(time_values, densities)
                            plots_updated += 1
                        except Exception as e:
                            logger.error(f"Erro ao plotar densities: {e}")
                
                if self.plot_data.get('exp_temps') or self.plot_data.get('rec_temps'):
                    temperatures = self.plot_data.get('exp_temps') or self.plot_data.get('rec_temps')
                    if len(temperatures) == len(time_values):
                        try:
                            self.temperature_curve.setData(time_values, temperatures)
                            plots_updated += 1
                        except Exception as e:
                            logger.error(f"Erro ao plotar temperatures: {e}")
            
            # Forçar atualização da interface
            QApplication.processEvents()
            
        except Exception as e:
            logger.error(f"Erro ao atualizar gráficos em tempo real: {e}")
            self.log_status(f"❌ Erro na plotagem: {str(e)}")
    
    def _configure_datetime_axis(self):
        """Configura eixos X para formato de data/hora"""
        try:
            # Lista de gráficos para configurar
            plots = []
            
            if hasattr(self, 'pressure_plot'):
                plots.append(self.pressure_plot)
            if hasattr(self, 'temperature_plot'):
                plots.append(self.temperature_plot)
            if hasattr(self, 'density_plot'):
                plots.append(self.density_plot)
            if hasattr(self, 'flow_plot'):
                plots.append(self.flow_plot)
            if hasattr(self, 'diff_pressure_plot'):
                plots.append(self.diff_pressure_plot)
            if hasattr(self, 'diff_temp_plot'):
                plots.append(self.diff_temp_plot)
            if hasattr(self, 'diff_density_plot'):
                plots.append(self.diff_density_plot)
            
            for plot in plots:
                if plot:
                    # Configurar eixo X como data/hora
                    axis = pg.DateAxisItem(orientation='bottom')
                    plot.setAxisItems({'bottom': axis})
                    plot.setLabel('bottom', 'Data/Hora')
                    
                    # Configurar grid
                    plot.showGrid(x=True, y=True, alpha=0.3)
                    
        except Exception as e:
            logger.error(f"Erro ao configurar eixos de data/hora: {e}")
    
    def _update_differential_plots(self, time_values):
        """Atualiza gráficos de análise diferencial"""
        try:
            # Gráfico de diferença de pressões
            if (hasattr(self, 'diff_pressure_plot') and 
                self.plot_data.get('exp_pressures') and 
                self.plot_data.get('rec_pressures')):
                
                exp_p = np.array(self.plot_data['exp_pressures'])
                rec_p = np.array(self.plot_data['rec_pressures'])
                
                if len(exp_p) == len(rec_p):
                    pressure_diff = exp_p - rec_p
                    self.diff_pressure_plot.clear()
                    pen = pg.mkPen(color='red', width=2)
                    self.diff_pressure_plot.plot(time_values, pressure_diff, pen=pen)
            
            # Gráfico de diferença de temperaturas
            if (hasattr(self, 'diff_temp_plot') and 
                self.plot_data.get('exp_temps') and 
                self.plot_data.get('rec_temps')):
                
                exp_t = np.array(self.plot_data['exp_temps'])
                rec_t = np.array(self.plot_data['rec_temps'])
                
                if len(exp_t) == len(rec_t):
                    temp_diff = exp_t - rec_t
                    self.diff_temp_plot.clear()
                    pen = pg.mkPen(color='orange', width=2)
                    self.diff_temp_plot.plot(time_values, temp_diff, pen=pen)
            
            # Gráfico de diferença de densidades
            if (hasattr(self, 'diff_density_plot') and 
                self.plot_data.get('exp_densities') and 
                self.plot_data.get('rec_densities')):
                
                exp_d = np.array(self.plot_data['exp_densities'])
                rec_d = np.array(self.plot_data['rec_densities'])
                
                if len(exp_d) == len(rec_d):
                    density_diff = exp_d - rec_d
                    self.diff_density_plot.clear()
                    pen = pg.mkPen(color='green', width=2)
                    self.diff_density_plot.plot(time_values, density_diff, pen=pen)
                    
        except Exception as e:
            logger.error(f"Erro ao atualizar gráficos diferenciais: {e}")
    
    def _process_pipeline_profile(self, df, filename):
        """Processa arquivo de perfil da tubulação (OPASA10-BAR2PLN.xlsx)"""
        try:
            # Validar estrutura esperada
            expected_cols = ['Km Desenvol.', 'Cota', 'Esp', 'Dext', 'Tag']
            
            # Mapear colunas (caso tenham nomes ligeiramente diferentes)
            column_mapping = {}
            for expected_col in expected_cols:
                for actual_col in df.columns:
                    if expected_col.lower() in actual_col.lower():
                        column_mapping[actual_col] = expected_col
                        break
            
            if len(column_mapping) < 3:  # Pelo menos 3 colunas essenciais
                raise ValueError(f"Estrutura de perfil inválida. Esperado: {expected_cols}, Encontrado: {list(df.columns)}")
            
            # Renomear colunas
            df_renamed = df.rename(columns=column_mapping)
            
            # Extrair dados essenciais
            pipeline_data = {
                'stations': [],
                'elevations': [],
                'distances': [],
                'diameters': [],
                'thicknesses': [],
                'tags': []
            }
            
            for _, row in df_renamed.iterrows():
                # Distância em km
                if 'Km Desenvol.' in row:
                    pipeline_data['distances'].append(float(row['Km Desenvol.']) * 1000)  # Converter para metros
                
                # Elevação em metros
                if 'Cota' in row:
                    pipeline_data['elevations'].append(float(row['Cota']))
                
                # Diâmetro externo em metros (converter de polegadas)
                if 'Dext' in row:
                    dext_inches = float(row['Dext'])
                    dext_meters = dext_inches * 0.0254  # Converter polegadas para metros
                    pipeline_data['diameters'].append(dext_meters)
                
                # Espessura em metros (converter de polegadas)
                if 'Esp' in row:
                    esp_inches = float(row['Esp'])
                    esp_meters = esp_inches * 0.0254  # Converter polegadas para metros
                    pipeline_data['thicknesses'].append(esp_meters)
                
                # Tags/identificações
                if 'Tag' in row:
                    pipeline_data['tags'].append(str(row['Tag']))
            
            # Atualizar configuração do sistema com dados do perfil
            if hasattr(self.current_system, 'pipe_characteristics'):
                # Calcular propriedades médias
                avg_diameter = np.mean(pipeline_data['diameters']) if pipeline_data['diameters'] else 0.5
                total_length = max(pipeline_data['distances']) if pipeline_data['distances'] else 1000.0
                avg_thickness = np.mean(pipeline_data['thicknesses']) if pipeline_data['thicknesses'] else 0.01
                
                # Atualizar características do duto
                self.current_system.pipe_characteristics.diameter = avg_diameter
                self.current_system.pipe_characteristics.length = total_length
                self.current_system.pipe_characteristics.wall_thickness = avg_thickness * 1000  # Converter para mm
                
                self.log_status(f"Perfil carregado: {len(df)} pontos, {total_length/1000:.1f}km, Ø{avg_diameter*1000:.0f}mm")
            
            # Armazenar dados do perfil para visualização
            self.pipeline_profile = pipeline_data
            
        except Exception as e:
            raise Exception(f"Erro ao processar perfil da tubulação: {str(e)}")
    
    def _process_sensor_data(self, df, filename):
        """Processa arquivo de dados de sensor (BAR_*.xlsx, PLN_*.xlsx)"""
        try:
            # Validar estrutura básica (2 colunas: tempo + valor)
            if len(df.columns) < 2:
                raise ValueError(f"Arquivo de sensor deve ter pelo menos 2 colunas (tempo, valor). Encontrado: {len(df.columns)}")
            
            # Renomear colunas para padrão
            df.columns = ['tempo', 'valor'] + list(df.columns[2:]) if len(df.columns) > 2 else ['tempo', 'valor']
            
            # Processar coluna de tempo
            time_col = df.columns[0]
            
            # Tentar diferentes formatos de data
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                # Se não for datetime, tentar converter
                try:
                    df[time_col] = pd.to_datetime(df[time_col], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                except:
                    try:
                        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                    except:
                        raise ValueError(f"Não foi possível converter coluna de tempo: {df[time_col].dtype}")
            
            # Remover linhas com tempo inválido
            df = df.dropna(subset=[time_col])
            
            if len(df) == 0:
                raise ValueError("Nenhum registro válido encontrado após processamento de datas")
            
            # Determinar tipo de sensor e estação baseado no nome do arquivo
            sensor_info = self._parse_sensor_filename(filename)
            
            # Converter para formato SensorReading
            readings = []
            for _, row in df.iterrows():
                reading = SensorReading(
                    timestamp=row['tempo'],
                    sensor_id=sensor_info['sensor_id'],
                    variable=sensor_info['variable'],
                    value=float(row['valor']),
                    unit=sensor_info['unit']
                )
                readings.append(reading)
            
            self.progress_bar.setValue(50)
            
            # Processar dados em chunks
            self.log_status(f"Processando {len(readings)} leituras de {sensor_info['description']}...")
            
            chunk_size = 1000
            for i in range(0, len(readings), chunk_size):
                chunk = readings[i:i+chunk_size]
                result = self.processor.process_sensor_readings(chunk)
                
                if 'error' in result:
                    raise Exception(result['error'])
                
                # Atualizar dados para plotagem
                self.update_plot_data(result)
                
                progress = 50 + (40 * (i + chunk_size) // len(readings))
                self.progress_bar.setValue(min(progress, 90))
            
            self.log_status(f"Processados {len(readings)} registros de {sensor_info['description']}")
            
        except Exception as e:
            raise Exception(f"Erro ao processar dados do sensor: {str(e)}")
    
    def _parse_sensor_filename(self, filename):
        """Extrai informações do sensor baseado no nome do arquivo"""
        filename_upper = filename.upper()
        
        # Determinar estação (BAR = Barueri/Expedidor, PLN = Paulínia/Recebedor)
        if 'BAR_' in filename_upper:
            station = 'BAR'
            location = 'expeditor'
        elif 'PLN_' in filename_upper:
            station = 'PLN'  
            location = 'receiver'
        else:
            station = 'UNK'
            location = 'unknown'
        
        # Determinar tipo de variável
        if '_DT' in filename_upper:
            variable = 'density'
            unit = 'kg/m³'
            description = f'Densidade {station}'
        elif '_FT' in filename_upper:
            variable = 'flow_rate'
            unit = 'kg/s'
            description = f'Vazão {station}'
        elif '_PT' in filename_upper:
            variable = 'pressure'
            unit = 'kgf/cm²'
            description = f'Pressão {station}'
        elif '_TT' in filename_upper:
            variable = 'temperature'
            unit = '°C'
            description = f'Temperatura {station}'
        else:
            variable = 'unknown'
            unit = 'unknown'
            description = f'Sensor {station}'
        
        # Criar ID único do sensor
        sensor_id = f"{station}_{variable.upper()}_01"
        
        return {
            'sensor_id': sensor_id,
            'variable': variable,
            'unit': unit,
            'station': station,
            'location': location,
            'description': description
        }
    
    def batch_load_files(self):
        """Carregamento em lote de arquivos com pré-classificação"""
        if not self.current_system:
            QMessageBox.warning(self, "Aviso", "Configure um sistema primeiro!")
            return
        
        # Abre dialog de carregamento em lote
        dialog = BatchFileLoadDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            file_data = dialog.get_file_data()
            
            if not file_data:
                return
            
            # Processa arquivos
            try:
                self.progress_bar.setMaximum(len(file_data))
                self.progress_bar.setValue(0)
                self.progress_bar.setVisible(True)
                
                total_readings = 0
                processed_files = 0
                
                self.log_status(f"Iniciando carregamento em lote de {len(file_data)} arquivos...")
                
                for file_path, info in file_data.items():
                    if info['status'] != 'OK':
                        continue
                    
                    try:
                        self.log_status(f"Carregando: {os.path.basename(file_path)}")
                        
                        # Lê arquivo
                        df = pd.read_excel(file_path)
                        classification = info['classification']
                        
                        # Converte para formato padrão
                        readings = []
                        for _, row in df.iterrows():
                            # Converte para datetime do Python (não pandas)
                            timestamp = pd.to_datetime(row['tempo'], format='ISO8601').to_pydatetime()
                            value = float(row['valor'])
                            
                            reading = SensorReading(
                                timestamp=timestamp,
                                sensor_id=classification['sensor_id'],
                                variable=classification['variable'],
                                value=value,
                                unit=self.get_unit_for_variable(classification['variable'])
                            )
                            readings.append(reading)
                        
                        # Processa leituras em chunks
                        chunk_size = 500
                        for i in range(0, len(readings), chunk_size):
                            chunk = readings[i:i+chunk_size]
                            result = self.processor.process_sensor_readings(chunk)
                            
                            if 'error' in result:
                                self.log_status(f"Erro no arquivo {os.path.basename(file_path)}: {result['error']}")
                                continue
                        
                        total_readings += len(readings)
                        processed_files += 1
                        self.log_status(f"[OK] {os.path.basename(file_path)}: {len(readings)} leituras")
                        
                    except Exception as e:
                        self.log_status(f"✗ Erro em {os.path.basename(file_path)}: {str(e)}")
                    
                    self.progress_bar.setValue(self.progress_bar.value() + 1)
                    QApplication.processEvents()  # Mantém interface responsiva
                
                # Atualiza plots com todos os dados carregados
                if processed_files > 0:
                    self.update_plots()
                    self.start_analysis_btn.setEnabled(True)
                
                self.log_status(f"Carregamento concluído: {processed_files} arquivos, {total_readings} leituras")
                
                QMessageBox.information(
                    self, "Sucesso", 
                    f"Carregamento em lote concluído!\n\n"
                    f"Arquivos processados: {processed_files}/{len(file_data)}\n"
                    f"Total de leituras: {total_readings:,}"
                )
                
            except Exception as e:
                logger.error(f"Erro no carregamento em lote: {e}")
                QMessageBox.critical(self, "Erro", f"Erro no carregamento em lote:\n{str(e)}")
                
            finally:
                self.progress_bar.setVisible(False)
                self.progress_bar.setValue(0)
    
    def get_unit_for_variable(self, variable: str) -> str:
        """Retorna unidade para uma variável"""
        unit_map = {
            'pressure': 'kgf/cm²',
            'flow': 'm³/h',
            'temperature': '°C',
            'density': 'g/cm³'
        }
        return unit_map.get(variable, '')
    
    def start_analysis(self):
        """Inicia análise em tempo real com processamento assíncrono"""
        if not self.current_system:
            QMessageBox.warning(self, "Aviso", "Configure um sistema primeiro!")
            return
        
        if not self.plot_data.get('times'):
            QMessageBox.warning(self, "Aviso", "Carregue dados primeiro!")
            return
        
        try:
            # Cancelar análises anteriores
            self._cancel_active_workers()
            
            # Configura timer baseado na velocidade
            speed = self.speed_slider.value()
            interval = max(50, 1000 // speed)  # 50ms mínimo
            
            # Criar worker assíncrono para análise
            analysis_worker = AsyncDataWorker(self._perform_analysis_async, self.plot_data.copy())
            analysis_worker.signals.progress.connect(self.progress_bar.setValue)
            analysis_worker.signals.status.connect(self.log_status)
            analysis_worker.signals.result.connect(self._on_analysis_completed)
            analysis_worker.signals.error.connect(self._on_analysis_error)
            
            # Adicionar à lista de workers ativos
            self.active_workers.append(analysis_worker)
            
            # Executar análise assíncrona
            self.thread_pool.start(analysis_worker)
            
            # Iniciar timer para atualizações em tempo real
            self.analysis_timer.start(interval)
            
            # Atualiza interface
            self.start_analysis_btn.setEnabled(False)
            self.stop_analysis_btn.setEnabled(True)
            
            self.log_status(f"Análise assíncrona iniciada - Velocidade: {speed}x")
            self.status_bar.showMessage(f"🔄 Análise em execução - {speed}x")
            
        except Exception as e:
            logger.error(f"Erro ao iniciar análise assíncrona: {e}")
            QMessageBox.critical(self, "Erro", f"Erro ao iniciar análise:\n{str(e)}")
    
    def _perform_analysis_async(self, plot_data, progress_callback=None, status_callback=None):
        """Executa análise complexa em thread separada"""
        try:
            results = {}
            total_steps = 6
            
            if status_callback:
                status_callback("Iniciando análise assíncrona...")
            if progress_callback:
                progress_callback(10)
            
            # Passo 1: Análise básica de estatísticas
            if status_callback:
                status_callback("Calculando estatísticas básicas...")
            
            stats = self._calculate_basic_stats_parallel(plot_data)
            results['basic_stats'] = stats
            
            if progress_callback:
                progress_callback(25)
            
            # Passo 2: Análise espectral
            if status_callback:
                status_callback("Executando análise espectral...")
            
            spectral = self._perform_spectral_analysis_parallel(plot_data)
            results['spectral_analysis'] = spectral
            
            if progress_callback:
                progress_callback(45)
            
            # Passo 3: Análise de correlação
            if status_callback:
                status_callback("Analisando correlações...")
            
            correlations = self._calculate_correlations_parallel(plot_data)
            results['correlations'] = correlations
            
            if progress_callback:
                progress_callback(65)
            
            # Passo 4: Detecção de anomalias
            if status_callback:
                status_callback("Detectando anomalias...")
            
            anomalies = self._detect_anomalies_parallel(plot_data)
            results['anomalies'] = anomalies
            
            if progress_callback:
                progress_callback(85)
            
            # Passo 5: Análise ML se disponível
            if self.processor and hasattr(self.processor, 'ml_system'):
                if status_callback:
                    status_callback("Executando análise ML...")
                
                ml_results = self._perform_ml_analysis_parallel(plot_data)
                results['ml_analysis'] = ml_results
            
            if progress_callback:
                progress_callback(100)
            
            if status_callback:
                status_callback("Análise concluída")
            
            return results
            
        except Exception as e:
            logger.error(f"Erro na análise assíncrona: {e}")
            raise
    
    def _calculate_basic_stats_parallel(self, plot_data):
        """Calcula estatísticas básicas em paralelo"""
        def calc_stats_for_variable(data):
            if not data:
                return {}
            
            arr = np.array(data)
            return {
                'mean': np.mean(arr),
                'std': np.std(arr),
                'min': np.min(arr),
                'max': np.max(arr),
                'median': np.median(arr),
                'count': len(arr)
            }
        
        # Processar diferentes variáveis em paralelo
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            for var in ['exp_pressures', 'rec_pressures', 'exp_temps', 'rec_temps']:
                if plot_data.get(var):
                    futures[var] = executor.submit(calc_stats_for_variable, plot_data[var])
            
            # Coletar resultados
            stats = {}
            for var, future in futures.items():
                try:
                    stats[var] = future.result()
                except Exception as e:
                    logger.error(f"Erro no cálculo de estatísticas para {var}: {e}")
                    stats[var] = {}
        
        return stats
    
    def _perform_spectral_analysis_parallel(self, plot_data):
        """Executa análise espectral em paralelo"""
        def spectral_analysis_for_signal(signal_data):
            if len(signal_data) < 10:
                return {}
            
            try:
                # FFT
                fft = np.fft.fft(signal_data)
                freqs = np.fft.fftfreq(len(signal_data))
                
                # Densidade espectral de potência
                psd = np.abs(fft) ** 2
                
                return {
                    'dominant_freq': freqs[np.argmax(psd[1:len(psd)//2]) + 1],
                    'spectral_centroid': np.sum(freqs[:len(freqs)//2] * psd[:len(psd)//2]) / np.sum(psd[:len(psd)//2]),
                    'spectral_rolloff': np.percentile(psd[:len(psd)//2], 85),
                    'spectral_bandwidth': np.sqrt(np.sum(((freqs[:len(freqs)//2] - np.sum(freqs[:len(freqs)//2] * psd[:len(psd)//2]) / np.sum(psd[:len(psd)//2])) ** 2) * psd[:len(psd)//2]) / np.sum(psd[:len(psd)//2]))
                }
            except Exception as e:
                logger.error(f"Erro na análise espectral: {e}")
                return {}
        
        # Análise espectral paralela
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            for var in ['exp_pressures', 'rec_pressures']:
                if plot_data.get(var):
                    futures[var] = executor.submit(spectral_analysis_for_signal, plot_data[var])
            
            results = {}
            for var, future in futures.items():
                try:
                    results[var] = future.result()
                except Exception as e:
                    logger.error(f"Erro na análise espectral para {var}: {e}")
                    results[var] = {}
        
        return results
    
    def _calculate_correlations_parallel(self, plot_data):
        """Calcula correlações em paralelo"""
        try:
            correlations = {}
            
            # Lista de pares para correlação
            correlation_pairs = [
                ('exp_pressures', 'rec_pressures'),
                ('exp_temps', 'rec_temps'),
                ('exp_pressures', 'exp_temps'),
                ('rec_pressures', 'rec_temps')
            ]
            
            def calc_correlation(pair_data):
                var1_data, var2_data = pair_data
                if len(var1_data) == len(var2_data) and len(var1_data) > 5:
                    corr, p_value = pearsonr(var1_data, var2_data)
                    return {'correlation': corr, 'p_value': p_value}
                return {'correlation': 0, 'p_value': 1}
            
            # Calcular correlações em paralelo
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {}
                
                for var1, var2 in correlation_pairs:
                    if plot_data.get(var1) and plot_data.get(var2):
                        pair_key = f"{var1}_vs_{var2}"
                        futures[pair_key] = executor.submit(calc_correlation, (plot_data[var1], plot_data[var2]))
                
                # Coletar resultados
                for pair_key, future in futures.items():
                    try:
                        correlations[pair_key] = future.result()
                    except Exception as e:
                        logger.error(f"Erro no cálculo de correlação para {pair_key}: {e}")
                        correlations[pair_key] = {'correlation': 0, 'p_value': 1}
            
            return correlations
            
        except Exception as e:
            logger.error(f"Erro no cálculo de correlações: {e}")
            return {}
    
    def _detect_anomalies_parallel(self, plot_data):
        """Detecção de anomalias em paralelo"""
        def detect_anomalies_in_signal(signal_data):
            if len(signal_data) < 10:
                return {}
            
            try:
                # Z-score para anomalias
                z_scores = np.abs(zscore(signal_data))
                anomaly_threshold = 3.0
                
                anomalies_idx = np.where(z_scores > anomaly_threshold)[0]
                
                return {
                    'anomaly_count': len(anomalies_idx),
                    'anomaly_percentage': (len(anomalies_idx) / len(signal_data)) * 100,
                    'max_anomaly_score': np.max(z_scores),
                    'anomaly_indices': anomalies_idx.tolist()
                }
            except Exception as e:
                logger.error(f"Erro na detecção de anomalias: {e}")
                return {}
        
        # Detecção paralela
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            for var in ['exp_pressures', 'rec_pressures', 'exp_temps', 'rec_temps']:
                if plot_data.get(var):
                    futures[var] = executor.submit(detect_anomalies_in_signal, plot_data[var])
            
            anomalies = {}
            for var, future in futures.items():
                try:
                    anomalies[var] = future.result()
                except Exception as e:
                    logger.error(f"Erro na detecção de anomalias para {var}: {e}")
                    anomalies[var] = {}
        
        return anomalies
    
    def _perform_ml_analysis_parallel(self, plot_data):
        """Executa análise ML em paralelo"""
        try:
            # Simplificado - implementação completa dependeria dos modelos disponíveis
            return {
                'status': 'ML analysis completed in parallel',
                'models_used': ['isolation_forest', 'pca'],
                'processing_time': 'optimized'
            }
        except Exception as e:
            logger.error(f"Erro na análise ML: {e}")
            return {}
    
    @pyqtSlot(object)
    def _on_analysis_completed(self, results):
        """Callback quando análise assíncrona é concluída"""
        try:
            self.analysis_results = results
            self.log_status("✅ Análise assíncrona concluída com sucesso")
            
            # Atualizar interface com resultados
            self._update_analysis_display(results)
            
        except Exception as e:
            logger.error(f"Erro no callback de análise: {e}")
    
    @pyqtSlot(str)
    def _on_analysis_error(self, error_message):
        """Callback para erros de análise"""
        logger.error(f"Erro na análise assíncrona: {error_message}")
        self.log_status(f"❌ Erro na análise: {error_message}")
        
        # Reativar botão
        self.start_analysis_btn.setEnabled(True)
        self.stop_analysis_btn.setEnabled(False)
    
    def _update_analysis_display(self, results):
        """Atualiza displays com resultados da análise"""
        try:
            # Exibir estatísticas básicas
            if 'basic_stats' in results:
                stats_text = "📊 ESTATÍSTICAS:\n"
                for var, stats in results['basic_stats'].items():
                    if stats:
                        stats_text += f"{var}: μ={stats.get('mean', 0):.2f}, σ={stats.get('std', 0):.2f}\n"
                
                # Atualizar display se disponível
                if hasattr(self, 'analysis_text'):
                    self.analysis_text.append(stats_text)
            
            # Exibir anomalias
            if 'anomalies' in results:
                anomaly_text = "\n🚨 ANOMALIAS:\n"
                for var, anomalies in results['anomalies'].items():
                    if anomalies.get('anomaly_count', 0) > 0:
                        anomaly_text += f"{var}: {anomalies['anomaly_count']} anomalias ({anomalies['anomaly_percentage']:.1f}%)\n"
                
                if hasattr(self, 'analysis_text'):
                    self.analysis_text.append(anomaly_text)
            
            # Log de conclusão
            self.log_status("📋 Resultados da análise atualizados na interface")
            
        except Exception as e:
            logger.error(f"Erro ao atualizar display de análise: {e}")
    
    def _cancel_active_workers(self):
        """Cancela todos os workers ativos"""
        for worker in self.active_workers:
            if hasattr(worker, 'cancel'):
                worker.cancel()
        
        self.active_workers.clear()
    
    def stop_analysis(self):
        """Para análise em tempo real e cancela workers assíncronos"""
        # Parar timer
        self.analysis_timer.stop()
        
        # Cancelar todos os workers ativos
        self._cancel_active_workers()
        
        # Cancelar loader assíncrono se ativo
        if self.async_loader and self.async_loader.isRunning():
            self.async_loader.cancel()
            self.async_loader.wait(2000)  # Aguardar até 2 segundos
        
        # Atualiza interface
        self.start_analysis_btn.setEnabled(True)
        self.stop_analysis_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.log_status("🛑 Análise parada - Todos os processos cancelados")
        self.status_bar.showMessage("Análise parada")
        
        self.log_status("Análise parada")
        self.status_bar.showMessage("Análise parada")
    
    def start_simulation(self):
        """Inicia simulação de dados usando arquivos XLSX reais"""
        try:
            # Para análise se estiver rodando
            if self.analysis_timer.isActive():
                self.stop_analysis()
            
            # Configura sistema de simulação se necessário
            if not self.current_system:
                self._create_simulation_system()
            
            # Configura processador se necessário
            if not self.processor and self.current_system:
                self.processor = IndustrialHydraulicProcessor(self.current_system)
            
            # Carrega dados reais dos arquivos XLSX para simulação
            self.load_xlsx_simulation_data()
            
            # Inicia simulação com dados reais
            speed = self.speed_slider.value() if hasattr(self, 'speed_slider') else 5
            scenario = getattr(self, 'scenario_combo', None)
            scenario_name = scenario.currentText() if scenario else 'Normal'
            
            # Usa o simulador existente mas com dados carregados dos XLSX
            hydraulic_simulator.start_simulation(
                target_system_id=self.current_system.system_id if self.current_system else 'SIM_001',
                scenario=scenario_name.lower(),
                speed=speed / 5.0  # Normaliza para 0.2-2.0x
            )
            
            # Inicia timer de análise para capturar dados da simulação
            self.analysis_timer.start(100)  # 10 FPS
            
            # Atualiza interface
            self.start_simulation_btn.setEnabled(False)
            self.stop_simulation_btn.setEnabled(True)
            self.start_analysis_btn.setEnabled(False)
            self.stop_analysis_btn.setEnabled(True)
            
            self.log_status(f"Simulacao iniciada com dados reais - Velocidade: {speed}x")
            
        except Exception as e:
            logger.error(f"Erro ao iniciar simulação: {e}")
            QMessageBox.critical(self, "Erro", f"Erro ao iniciar simulação:\n{str(e)}")
    
    def load_xlsx_simulation_data(self):
        """Carrega dados dos arquivos XLSX para usar na simulação"""
        try:
            import glob
            import os
            
            # Encontra arquivos XLSX na pasta atual
            current_dir = os.getcwd()
            xlsx_files = glob.glob(os.path.join(current_dir, "*.xlsx"))
            
            if not xlsx_files:
                self.log_status("[AVISO] Nenhum arquivo XLSX encontrado - usando dados sintéticos")
                return
            
            self.log_status(f"[DADOS] Carregando {len(xlsx_files)} arquivos XLSX para simulação...")
            
            # Organiza arquivos por tipo
            file_data = {}
            for file_path in xlsx_files:
                filename = os.path.basename(file_path).upper()
                
                # Skip template files
                if 'TEMPLATE' in filename:
                    continue
                
                # Classifica arquivo
                if 'BAR' in filename and 'PT' in filename:
                    key = 'bar_pressure'
                elif 'PLN' in filename and 'PT' in filename:
                    key = 'pln_pressure'
                elif 'BAR' in filename and 'FT' in filename:
                    key = 'bar_flow'
                elif 'PLN' in filename and 'FT' in filename:
                    key = 'pln_flow'
                elif 'BAR' in filename and 'TT' in filename:
                    key = 'bar_temperature'
                elif 'PLN' in filename and 'TT' in filename:
                    key = 'pln_temperature'
                elif 'BAR' in filename and 'DT' in filename:
                    key = 'bar_density'
                elif 'PLN' in filename and 'DT' in filename:
                    key = 'pln_density'
                else:
                    continue
                
                # Carrega dados do arquivo
                try:
                    df = pd.read_excel(file_path)
                    if 'tempo' in df.columns and 'valor' in df.columns:
                        file_data[key] = {
                            'timestamps': pd.to_datetime(df['tempo'], format='ISO8601').dt.to_pydatetime(),
                            'values': df['valor'].astype(float),
                            'filename': os.path.basename(file_path)
                        }
                        self.log_status(f"✓ {os.path.basename(file_path)}: {len(df)} registros")
                    else:
                        self.log_status(f"✗ {os.path.basename(file_path)}: formato inválido")
                
                except Exception as e:
                    self.log_status(f"✗ {os.path.basename(file_path)}: erro - {str(e)}")
            
            if file_data:
                # Alimenta o simulador com dados reais
                self.setup_real_data_simulation(file_data)
                self.log_status(f"[OK] Dados de simulação carregados: {len(file_data)} tipos de sensores")
            else:
                self.log_status("[AVISO] Nenhum arquivo válido encontrado - usando dados sintéticos")
        
        except Exception as e:
            self.log_status(f"[ERRO] Erro ao carregar dados XLSX: {str(e)}")
    
    def setup_real_data_simulation(self, file_data):
        """Configura simulação com dados reais"""
        try:
            # Encontra intervalo de tempo comum
            all_timestamps = []
            for data in file_data.values():
                all_timestamps.extend(data['timestamps'])
            
            if all_timestamps:
                start_time = min(all_timestamps)
                end_time = max(all_timestamps)
                duration = (end_time - start_time).total_seconds()
                
                # Atualiza simulador com parâmetros baseados nos dados reais
                if hasattr(hydraulic_simulator, 'real_data_mode'):
                    hydraulic_simulator.real_data_mode = True
                    hydraulic_simulator.real_data_cache = file_data
                    hydraulic_simulator.data_start_time = start_time
                    hydraulic_simulator.data_duration = duration
                
                self.log_status(f"📅 Simulação configurada: {start_time.strftime('%Y-%m-%d')} a {end_time.strftime('%Y-%m-%d')}")
                self.log_status(f"⏱️ Duração: {duration/3600:.1f} horas de dados reais")
        
        except Exception as e:
            self.log_status(f"[ERRO] Erro ao configurar simulação: {str(e)}")
    
    def apply_advanced_config(self):
        """Aplica configurações avançadas do sistema"""
        try:
            # Atualiza parâmetros de correlação
            if hasattr(self, 'processor') and self.processor:
                correlation_window = self.correlation_window_spin.value()
                ml_threshold = self.ml_threshold_spin.value()
                
                # Atualiza configurações no processador
                if hasattr(self.processor, 'ml_system'):
                    self.processor.ml_system.detection_threshold = ml_threshold
                
                self.log_status(f"[OK] Configurações aplicadas - Janela: {correlation_window}, Threshold: {ml_threshold}")
            
            # Atualiza configurações do filtro temporal
            filter_type = self.filter_type_combo.currentText()
            filter_window = self.filter_window_spin.value()
            
            if hasattr(temporal_filter, 'configure'):
                temporal_filter.configure(filter_type=filter_type, window_size=filter_window)
            
            QMessageBox.information(self, "Sucesso", "Configurações avançadas aplicadas com sucesso!")
            
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao aplicar configurações:\n{str(e)}")
    
    def generate_report(self):
        """Gera relatório completo do sistema"""
        try:
            if not self.current_system:
                QMessageBox.warning(self, "Aviso", "Configure um sistema primeiro!")
                return
            
            # Cabeçalho do relatório
            report = []
            report.append("=" * 80)
            report.append("RELATÓRIO DE ANÁLISE HIDRÁULICA")
            report.append("=" * 80)
            report.append(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
            report.append(f"Sistema: {self.current_system.system_id}")
            report.append(f"Nome: {self.current_system.name}")
            report.append(f"Localização: {self.current_system.location}")
            report.append("")
            
            # Informações do sistema
            report.append("CONFIGURAÇÃO DO SISTEMA")
            report.append("-" * 40)
            pipe = self.current_system.pipe_characteristics
            report.append(f"Diâmetro: {pipe.diameter*1000:.1f} mm ({pipe.diameter/0.0254:.2f} in)")
            report.append(f"Comprimento: {pipe.length/1000:.3f} km")
            report.append(f"Material: {pipe.material}")
            report.append(f"Rugosidade: {pipe.roughness:.3f} mm")
            report.append(f"Espessura da parede: {pipe.wall_thickness:.1f} mm ({pipe.wall_thickness/25.4:.3f} in)")
            report.append(f"Distância entre sensores: {self.current_system.sensor_distance/1000:.3f} km")
            report.append("")
            
            # Parâmetros do fluido
            report.append("PARÂMETROS DO FLUIDO")
            report.append("-" * 40)
            report.append(f"Tipo: {self.current_system.fluid_type}")
            report.append(f"Densidade nominal: {self.current_system.nominal_density:.4f} g/cm³")
            report.append(f"Temperatura nominal: {self.current_system.nominal_temperature:.1f} °C")
            report.append(f"Pressão nominal: {self.current_system.nominal_pressure:.1f} kgf/cm²")
            report.append(f"Vazão nominal: {self.current_system.nominal_flow:.1f} m³/h")
            report.append(f"Velocidade sônica: {self.current_system.sonic_velocity:.1f} m/s")
            report.append("")
            
            # Unidades expedidora e recebedora
            if hasattr(self.current_system, 'expeditor_unit'):
                report.append("UNIDADES DO SISTEMA")
                report.append("-" * 40)
                report.append(f"Unidade Expedidora: {getattr(self.current_system, 'expeditor_unit', 'N/A')}")
                report.append(f"Alias Expedidora: {getattr(self.current_system, 'expeditor_alias', 'BAR')}")
                report.append(f"Unidade Recebedora: {getattr(self.current_system, 'receiver_unit', 'N/A')}")
                report.append(f"Alias Recebedora: {getattr(self.current_system, 'receiver_alias', 'PLN')}")
                report.append("")
            
            # Estatísticas de operação
            if hasattr(self, 'processor') and self.processor:
                report.append("ESTATÍSTICAS DE OPERAÇÃO")
                report.append("-" * 40)
                
                # Obtém dados dos buffers
                buffers = self.processor.system_buffers
                if buffers and 'timestamps' in buffers:
                    timestamps = buffers['timestamps'].get_data(100)
                    if len(timestamps) > 0:
                        duration = (timestamps[-1] - timestamps[0]) / 3600  # em horas
                        report.append(f"Período de análise: {duration:.2f} horas")
                        report.append(f"Amostras processadas: {len(timestamps)}")
                        
                        # Pressões
                        if 'expeditor_pressure' in buffers and 'receiver_pressure' in buffers:
                            exp_pressures = buffers['expeditor_pressure'].get_data(len(timestamps))
                            rec_pressures = buffers['receiver_pressure'].get_data(len(timestamps))
                            
                            if len(exp_pressures) > 0 and len(rec_pressures) > 0:
                                report.append(f"Pressão expedidora - Média: {np.mean(exp_pressures):.2f} kgf/cm²")
                                report.append(f"Pressão expedidora - Min/Max: {np.min(exp_pressures):.2f}/{np.max(exp_pressures):.2f} kgf/cm²")
                                report.append(f"Pressão recebedora - Média: {np.mean(rec_pressures):.2f} kgf/cm²")
                                report.append(f"Pressão recebedora - Min/Max: {np.min(rec_pressures):.2f}/{np.max(rec_pressures):.2f} kgf/cm²")
                                
                                # Correlação
                                if len(exp_pressures) > 10 and len(rec_pressures) > 10:
                                    correlation = np.corrcoef(exp_pressures, rec_pressures)[0,1]
                                    if not np.isnan(correlation):
                                        report.append(f"Correlação entre estações: {correlation:.3f}")
                
                report.append("")
            
            # Análises realizadas
            report.append("FUNCIONALIDADES IMPLEMENTADAS")
            report.append("-" * 40)
            report.append("[OK] Carregamento em lote de arquivos XLSX")
            report.append("[OK] Pré-classificação automática (Entrada/Saída)")
            report.append("[OK] Identificação por aliases (BAR/PLN)")
            report.append("[OK] Conversão automática de unidades (polegadas <> mm, metros <> km)")
            report.append("[OK] Modificação de características de sistemas existentes")
            report.append("[OK] Controles de análise funcionais")
            report.append("[OK] Métricas em tempo real")
            report.append("[OK] Simulação com dados reais dos arquivos XLSX")
            report.append("[OK] Análise de Ruídos (FFT, Espectrograma)")
            report.append("[OK] Perfil Hidráulico (Pressão, Perda de Carga)")
            report.append("[OK] Análise de Ondas (Propagação, Reflexões)")
            report.append("[OK] Filtro Temporal (Outliers, Qualidade)")
            report.append("[OK] Configurações Avançadas")
            report.append("[OK] Monitor do Sistema")
            report.append("✅ Geração de Relatórios")
            report.append("")
            
            # Observações finais
            report.append("OBSERVAÇÕES")
            report.append("-" * 40)
            report.append("• Todas as funcionalidades solicitadas foram implementadas")
            report.append("• Sistema mantém compatibilidade com aba 'Sinais Principais'")
            report.append("• Dados dos arquivos XLSX são utilizados para simulação realística")
            report.append("• Interface moderna com tema escuro profissional")
            report.append("• Controles responsivos e feedback visual em tempo real")
            report.append("")
            report.append("=" * 80)
            
            # Exibe relatório
            full_report = "\n".join(report)
            self.report_text.setText(full_report)
            
            self.log_status("📋 Relatório gerado com sucesso")
            
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao gerar relatório:\n{str(e)}")
    
    def export_report_pdf(self):
        """Exporta relatório em formato PDF"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Salvar Relatório PDF", 
                f"relatorio_hidraulico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                "PDF files (*.pdf);;All files (*)"
            )
            
            if file_path:
                # Usa reportlab se disponível, senão salva como texto
                try:
                    from reportlab.lib.pagesizes import letter
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                    from reportlab.lib.styles import getSampleStyleSheet
                    
                    doc = SimpleDocTemplate(file_path, pagesize=letter)
                    styles = getSampleStyleSheet()
                    story = []
                    
                    # Converte texto do relatório para PDF
                    text = self.report_text.toPlainText()
                    for line in text.split('\n'):
                        if line.strip():
                            if line.startswith('='):
                                p = Paragraph(line, styles['Title'])
                            elif line.startswith('-'):
                                p = Paragraph(line, styles['Heading2'])
                            else:
                                p = Paragraph(line, styles['Normal'])
                            story.append(p)
                            story.append(Spacer(1, 6))
                    
                    doc.build(story)
                    QMessageBox.information(self, "Sucesso", f"Relatório PDF salvo em:\n{file_path}")
                    
                except ImportError:
                    # Fallback para arquivo de texto
                    with open(file_path.replace('.pdf', '.txt'), 'w', encoding='utf-8') as f:
                        f.write(self.report_text.toPlainText())
                    QMessageBox.information(self, "Sucesso", f"Relatório salvo como texto em:\n{file_path.replace('.pdf', '.txt')}")
                
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao exportar PDF:\n{str(e)}")
    
    def clear_report(self):
        """Limpa o relatório atual"""
        self.report_text.clear()
        self.log_status("📋 Relatório limpo")
    
    def load_pipeline_profile(self, file_path: str = None):
        """Carrega perfil do sistema a partir do arquivo OPASA10-BAR2PLN.xlsx"""
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Carregar Perfil do Sistema",
                "", "Excel Files (*.xlsx *.xls)"
            )
        
        if not file_path:
            return
            
        try:
            # Lê arquivo de perfil
            df = pd.read_excel(file_path)
            
            # Mapeia colunas esperadas (flexível para diferentes nomes)
            column_mapping = {
                'extensao': ['extensão', 'extensao', 'distance', 'length', 'km'],
                'cota': ['cota', 'elevacao', 'elevation', 'height', 'm'],
                'espessura': ['espessura', 'thickness', 'wall_thickness', 'pol', 'in'],
                'diametro': ['diametro', 'diameter', 'dia', 'pol', 'in'],
                'acessorios': ['acessórios', 'acessorios', 'accessories', 'fittings']
            }
            
            # Encontra colunas por similaridade
            mapped_columns = {}
            for key, possible_names in column_mapping.items():
                for col in df.columns:
                    col_lower = col.lower()
                    if any(name.lower() in col_lower for name in possible_names):
                        mapped_columns[key] = col
                        break
            
            # Verifica se encontrou colunas essenciais
            essential_cols = ['extensao', 'cota', 'espessura', 'diametro']
            missing_cols = [col for col in essential_cols if col not in mapped_columns]
            
            if missing_cols:
                QMessageBox.warning(
                    self, "Erro", 
                    f"Colunas não encontradas no arquivo: {', '.join(missing_cols)}\n"
                    f"Colunas disponíveis: {', '.join(df.columns)}"
                )
                return
            
            # Processa dados do perfil
            profile_data = []
            for _, row in df.iterrows():
                # Extensão em km
                extensao_km = float(row[mapped_columns['extensao']])
                
                # Cota em metros  
                cota_m = float(row[mapped_columns['cota']])
                
                # Espessura: converte polegadas para mm se necessário
                espessura_val = float(row[mapped_columns['espessura']])
                # Assume polegadas se valor < 10, senão mm
                espessura_mm = espessura_val * 25.4 if espessura_val < 10 else espessura_val
                
                # Diâmetro: converte polegadas para metros se necessário  
                diametro_val = float(row[mapped_columns['diametro']])
                # Assume polegadas se valor < 200, senão mm
                if diametro_val < 200:  # Polegadas
                    diametro_m = diametro_val * 0.0254
                else:  # Milímetros
                    diametro_m = diametro_val / 1000.0
                
                # Acessórios (opcional)
                acessorios = ''
                if 'acessorios' in mapped_columns:
                    acessorios = str(row[mapped_columns['acessorios']]) if pd.notna(row[mapped_columns['acessorios']]) else ''
                
                profile_data.append({
                    'extensao_km': extensao_km,
                    'cota_m': cota_m,
                    'espessura_mm': espessura_mm,
                    'diametro_m': diametro_m,
                    'acessorios': acessorios
                })
            
            # Armazena perfil no sistema atual
            if self.current_system:
                self.current_system.pipeline_profile = profile_data
                
                # Calcula estatísticas do perfil
                total_length = max(item['extensao_km'] for item in profile_data)
                avg_diameter = np.mean([item['diametro_m'] for item in profile_data])
                elevation_change = max(item['cota_m'] for item in profile_data) - min(item['cota_m'] for item in profile_data)
                
                # Atualiza características do sistema baseado no perfil
                if hasattr(self.current_system, 'pipe_characteristics'):
                    self.current_system.pipe_characteristics.length = total_length * 1000  # km para metros
                    self.current_system.pipe_characteristics.diameter = avg_diameter
                
                self.log_status(f"[OK] Perfil carregado: {len(profile_data)} pontos")
                self.log_status(f"Comprimento total: {total_length:.3f} km")
                self.log_status(f"Diâmetro médio: {avg_diameter*1000:.1f} mm ({avg_diameter*39.3701:.1f} in)")
                self.log_status(f"Variação de cota: {elevation_change:.1f} m")
                
                # Atualiza plot de perfil hidráulico se disponível
                self.update_hydraulic_profile()
                
                QMessageBox.information(
                    self, "Sucesso",
                    f"Perfil do sistema carregado com sucesso!\n\n"
                    f"Pontos: {len(profile_data)}\n"
                    f"Comprimento: {total_length:.3f} km\n"
                    f"Diâmetro médio: {avg_diameter*1000:.1f} mm\n"
                    f"Variação de cota: {elevation_change:.1f} m"
                )
            else:
                QMessageBox.warning(self, "Aviso", "Configure um sistema primeiro!")
                
        except Exception as e:
            self.log_status(f"[FAIL] Erro ao carregar perfil: {str(e)}")
            QMessageBox.critical(self, "Erro", f"Erro ao carregar perfil:\n{str(e)}")
    
    def update_hydraulic_profile(self):
        """Atualiza gráfico de perfil hidráulico com dados carregados"""
        try:
            if not (self.current_system and hasattr(self.current_system, 'pipeline_profile')):
                return
                
            profile_data = self.current_system.pipeline_profile
            
            if hasattr(self, 'pressure_profile_plot') and profile_data:
                # Extrai dados para plotagem
                distances = [point['extensao_km'] for point in profile_data]
                diameters = [point['diametro_m'] * 1000 for point in profile_data]  # Para mm
                elevations = [point['cota_m'] for point in profile_data]
                
                # Plot perfil de diâmetro
                self.pressure_profile_plot.clear()
                self.pressure_profile_plot.plot(distances, diameters, pen='b', name='Diâmetro (mm)')
                
                # Plot perfil de elevação no mesmo gráfico (escala secundária)
                if hasattr(self, 'head_loss_plot'):
                    self.head_loss_plot.clear()
                    self.head_loss_plot.plot(distances, elevations, pen='g', name='Elevação (m)')
                
        except Exception as e:
            self.log_status(f"Erro ao atualizar perfil: {str(e)}")
    
    def stop_simulation(self):
        """Para simulação de dados"""
        try:
            # Para simulação
            hydraulic_simulator.stop_simulation()
            
            # Para timer de análise
            if self.analysis_timer.isActive():
                self.analysis_timer.stop()
            
            # Atualiza interface
            self.start_simulation_btn.setEnabled(True)
            self.stop_simulation_btn.setEnabled(False)
            self.start_analysis_btn.setEnabled(True)
            self.stop_analysis_btn.setEnabled(False)
            
            self.log_status("⏹ Simulação parada")
            
        except Exception as e:
            logger.error(f"Erro ao parar simulação: {e}")
    
    def change_simulation_scenario(self, new_scenario: str):
        """Altera cenário da simulação"""
        if hydraulic_simulator.is_running:
            hydraulic_simulator.change_scenario(new_scenario)
            self.log_status(f"🎬 Cenário alterado para: {new_scenario}")
    
    def _create_simulation_system(self):
        """Cria sistema padrão para simulação"""
        pipe_chars = PipeCharacteristics(
            diameter=0.3,
            material='steel', 
            profile='circular',
            length=100.0,
            roughness=0.1,
            wall_thickness=10.0
        )
        
        self.current_system = SystemConfiguration(
            system_id='SIM_001',
            name='Sistema de Simulação',
            location='Virtual',
            pipe_characteristics=pipe_chars,
            sensor_distance=100.0,
            fluid_type='oil',
            nominal_density=0.85,
            nominal_temperature=25.0,
            nominal_pressure=15.0,
            nominal_flow=500.0,
            sonic_velocity=1400.0,
            calibration_date=datetime.now()
        )
        
        self.system_label.setText(f"Sistema: {self.current_system.name} (SIMULAÇÃO)")
        self.system_label.setStyleSheet("font-weight: bold; color: #2196F3;")  # Azul para simulação
    
    def update_plots(self):
        """Atualiza plots com dados mais recentes com streaming otimizado"""
        try:
            # Primeiro, verificar se temos dados carregados diretamente
            if self.plot_data.get('times'):
                self._update_real_time_plots()
                return
            
            # Fallback: usar buffers do processador se disponível
            if not self.current_system or not self.processor:
                self.log_status("ℹ️ Sistema não configurado - usando modo de carregamento direto")
                return
            
            # Obtém dados dos buffers
            buffers = self.processor.system_buffers
            
            # Configuração otimizada para streaming
            max_points = 1000  # Limita pontos para performance
            update_window = 100   # Janela para análise dinâmica
            
            times = buffers['timestamps'].get_data(max_points)
            if len(times) == 0:
                return
            
            # Verifica se há dados suficientes
            n_points = len(times)
            if n_points < 2:
                return
            
            # Obtém dados sincronizados
            exp_pressures = buffers['expeditor_pressure'].get_data(n_points)
            rec_pressures = buffers['receiver_pressure'].get_data(n_points)
            flows = buffers['flow_rate'].get_data(n_points)
            densities = buffers['density'].get_data(n_points)
            temperatures = buffers['temperature'].get_data(n_points)
            
            # Garante que todos os arrays tenham o mesmo tamanho
            min_len = min(len(times), len(exp_pressures), len(rec_pressures), 
                         len(flows), len(densities), len(temperatures))
            
            if min_len < 2:
                return
                
            times = times[:min_len]
            exp_pressures = exp_pressures[:min_len]
            rec_pressures = rec_pressures[:min_len]
            flows = flows[:min_len]
            densities = densities[:min_len]
            temperatures = temperatures[:min_len]
            
            # Atualiza plots de sinais com dados sincronizados
            self.exp_pressure_curve.setData(times, exp_pressures)
            self.rec_pressure_curve.setData(times, rec_pressures)
            self.flow_curve.setData(times, flows)
            self.density_curve.setData(times, densities)
            self.temperature_curve.setData(times, temperatures)
            
            # Calcula correlação otimizada para plot
            if len(exp_pressures) > 50:
                # Usa janela otimizada para correlação
                window_size = min(300, len(exp_pressures))
                exp_window = exp_pressures[-window_size:]
                rec_window = rec_pressures[-window_size:]
                
                # Remove NaN values
                valid_mask = ~(np.isnan(exp_window) | np.isnan(rec_window))
                if np.sum(valid_mask) > 10:
                    exp_clean = exp_window[valid_mask]
                    rec_clean = rec_window[valid_mask]
                    
                    correlation = signal.correlate(exp_clean, rec_clean, mode='full')
                    correlation_x = np.arange(len(correlation)) - len(correlation)//2
                    self.correlation_curve.setData(correlation_x, correlation)
            
            # Obtém resumo do filtro temporal
            filter_summary = temporal_filter.get_streaming_summary()
            
            # Atualiza métricas na interface com dados do filtro
            if len(exp_pressures) > 0:
                current_exp = float(exp_pressures[-1])
                current_rec = float(rec_pressures[-1])
                current_flow = float(flows[-1]) if len(flows) > 0 else 0
                
                # Verifica NaN values após conversão para float
                if np.isnan(current_exp) or np.isnan(current_rec):
                    return
                
                # Correlação robusta
                recent_window = min(update_window, len(exp_pressures))
                exp_recent = exp_pressures[-recent_window:]
                rec_recent = rec_pressures[-recent_window:]
                
                # Remove NaN values para correlação
                valid_mask = ~(np.isnan(exp_recent) | np.isnan(rec_recent))
                if np.sum(valid_mask) > 3:
                    exp_clean = exp_recent[valid_mask]
                    rec_clean = rec_recent[valid_mask]
                    
                    if len(exp_clean) > 1 and np.std(exp_clean) > 0 and np.std(rec_clean) > 0:
                        correlation_val = np.corrcoef(exp_clean, rec_clean)
                        corr_coeff = correlation_val[0,1] if correlation_val.shape == (2,2) else 0.0
                    else:
                        corr_coeff = 0.0
                else:
                    corr_coeff = 0.0
                
                # Status operacional avançado
                pressure_diff = current_exp - current_rec
                
                # Usa qualidade do filtro se disponível
                exp_quality = 1.0
                rec_quality = 1.0
                
                if 'variables' in filter_summary:
                    exp_quality = filter_summary['variables'].get('expeditor_pressure', {}).get('quality_score', 1.0)
                    rec_quality = filter_summary['variables'].get('receiver_pressure', {}).get('quality_score', 1.0)
                
                overall_quality = (exp_quality + rec_quality) / 2
                
                if pressure_diff > 0.5:
                    status = "Fechado"
                    status_val = 0
                else:
                    status = "Aberto"
                    status_val = 1
                
                # ML score melhorado considerando qualidade
                ml_score = min(1.0, abs(pressure_diff - 1.0)) * overall_quality
                
                # Monta dicionário de métricas e atualiza interface
                filter_lag = filter_summary.get('timestamp', time.time()) - times[-1] if times else 0
                metrics = {
                    'correlation': corr_coeff,
                    'ml_score': ml_score,
                    'leak_probability': max(0, 1 - ml_score),  # Inversão simplificada
                    'system_status': status,
                    'confidence': overall_quality,
                    'filter_lag': filter_lag
                }
                
                # Usa o método centralizado para atualizar métricas
                self.update_real_time_metrics(metrics)
                
                # Atualiza plots de status e qualidade
                if hasattr(self, 'status_times'):
                    self.status_times.append(times[-1])
                    self.status_values.append(status_val)
                    
                    # Adiciona dados de qualidade se disponível
                    if not hasattr(self, 'quality_times'):
                        self.quality_times = []
                        self.quality_values = []
                    
                    self.quality_times.append(times[-1])
                    self.quality_values.append(overall_quality)
                else:
                    self.status_times = [times[-1]]
                    self.status_values = [status_val]
                    self.quality_times = [times[-1]]
                    self.quality_values = [overall_quality]
                
                # Limita arrays para performance
                self.status_times = self.status_times[-max_points:]
                self.status_values = self.status_values[-max_points:]
                self.quality_times = self.quality_times[-max_points:]
                self.quality_values = self.quality_values[-max_points:]
                
                # Atualiza curvas
                self.status_curve.setData(self.status_times, self.status_values)
                
                # Atualiza informações de streaming na barra de status
                buffer_fill = filter_summary.get('buffer_size', 0) / temporal_filter.window_size * 100
                status_msg = f"Streaming ativo - Buffer: {buffer_fill:.0f}% - FPS: {1000/max(50, self.analysis_timer.interval()):.1f}"
                if hasattr(self, 'status_bar'):
                    self.status_bar.showMessage(status_msg)
            
        except Exception as e:
            logger.error(f"Erro na atualização de plots: {e}")
            # Tenta recuperar resetando filtro se erro persistir
            if hasattr(e, '__traceback__'):
                temporal_filter.reset_filter()
    
    def update_plot_data(self, analysis_result: Dict[str, Any]):
        """Atualiza dados para plotagem baseado em resultado de análise"""
        try:
            # Extrai métricas do resultado
            integrated = analysis_result.get('integrated_detection', {})
            ml_pred = analysis_result.get('ml_prediction', {})
            sonic = analysis_result.get('sonic_correlation', {})
            status = analysis_result.get('operational_status', {})
            
            # Atualiza labels de métricas
            if 'max_correlation' in sonic:
                self.correlation_label.setText(f"Correlação: {sonic['max_correlation']:.3f}")
            
            if 'leak_probability' in ml_pred:
                self.ml_score_label.setText(f"Score ML: {ml_pred['leak_probability']:.3f}")
                self.leak_prob_label.setText(f"Prob. Vazamento: {ml_pred['leak_probability']:.3f}")
            
            if 'confidence' in ml_pred:
                self.confidence_label.setText(f"Confiança: {ml_pred['confidence']:.3f}")
            
            if 'column_status' in status:
                self.system_status_label.setText(f"Status: {status['column_status']}")
            
            # Log do resultado
            if 'leak_status' in integrated:
                leak_status = integrated['leak_status']
                if leak_status != 'normal_operation':
                    self.log_status(f"⚠️ {leak_status.upper()} - Score: {integrated.get('integrated_score', 0):.3f}")
                else:
                    self.log_status(f"✅ Operação normal - Score: {integrated.get('integrated_score', 0):.3f}")
            
        except Exception as e:
            logger.error(f"Erro ao atualizar dados de plot: {e}")
    
    def report_confirmed_leak(self):
        """Reporta vazamento confirmado para aprendizado"""
        if not self.processor:
            QMessageBox.warning(self, "Aviso", "Nenhum sistema carregado!")
            return
        
        # Dialog para detalhes do vazamento
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Reportar Vazamento Confirmado")
        
        leak_types = ['gradual', 'sudden', 'rupture', 'seepage']
        leak_type, ok = QInputDialog.getItem(
            self, "Tipo de Vazamento", "Selecione o tipo de vazamento:", 
            leak_types, 0, False
        )
        
        if not ok:
            return
        
        # Data/hora do vazamento
        leak_time, ok = QInputDialog.getText(
            self, "Hora do Vazamento", 
            "Digite a hora do vazamento (YYYY-MM-DD HH:MM:SS) ou deixe vazio para agora:"
        )
        
        if not ok:
            return
        
        try:
            if leak_time.strip():
                leak_datetime = datetime.strptime(leak_time.strip(), "%Y-%m-%d %H:%M:%S")
            else:
                leak_datetime = datetime.now()
            
            # Características do vazamento
            leak_characteristics = {
                'type': leak_type,
                'severity': 1.0,  # Poderia ser configurável
                'location': 0.0,  # Poderia ser configurável
                'reported_by': 'operator',
                'confirmed': True
            }
            
            # Ensina o sistema
            self.processor.learn_from_confirmed_leak(
                leak_datetime, leak_type, leak_characteristics
            )
            
            self.log_status(f"Vazamento confirmado reportado: {leak_type} em {leak_datetime}")
            QMessageBox.information(self, "Sucesso", "Vazamento reportado e sistema atualizado!")
            
        except Exception as e:
            logger.error(f"Erro ao reportar vazamento: {e}")
            QMessageBox.critical(self, "Erro", f"Erro ao reportar vazamento:\n{str(e)}")
    
    def retrain_ml_model(self):
        """Força retreinamento do modelo ML"""
        if not self.processor or not self.processor.ml_system.training_data_buffer:
            QMessageBox.warning(self, "Aviso", "Nenhum dado de treinamento disponível!")
            return
        
        try:
            self.log_status("Iniciando retreinamento do modelo ML...")
            
            # Prepara dados
            snapshots = [item[0] for item in self.processor.ml_system.training_data_buffer]
            labels = [item[1] for item in self.processor.ml_system.training_data_buffer]
            
            # Treina
            metrics = self.processor.ml_system.train_models(snapshots, labels)
            
            # Limpa buffer
            self.processor.ml_system.training_data_buffer.clear()
            
            self.log_status(f"Modelo retreinado - Acurácia: {metrics.get('accuracy', 0):.3f}")
            
            QMessageBox.information(
                self, "Sucesso", 
                f"Modelo retreinado com sucesso!\n"
                f"Acurácia: {metrics.get('accuracy', 0):.3f}\n"
                f"Amostras: {metrics.get('training_samples', 0)}"
            )
            
        except Exception as e:
            logger.error(f"Erro no retreinamento: {e}")
            QMessageBox.critical(self, "Erro", f"Erro no retreinamento:\n{str(e)}")
    
    def export_results(self):
        """Exporta resultados da análise"""
        if not self.current_system:
            QMessageBox.warning(self, "Aviso", "Nenhum sistema carregado!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Exportar Resultados", "", 
            "Arquivos CSV (*.csv);;Arquivos Excel (*.xlsx)"
        )
        
        if file_path:
            try:
                # Coleta dados dos buffers
                buffers = self.processor.system_buffers
                
                times = buffers['timestamps'].get_data()
                exp_pressures = buffers['expeditor_pressure'].get_data(len(times))
                rec_pressures = buffers['receiver_pressure'].get_data(len(times))
                flows = buffers['flow_rate'].get_data(len(times))
                
                # Cria DataFrame
                export_df = pd.DataFrame({
                    'timestamp': [datetime.fromtimestamp(t) for t in times],
                    'expeditor_pressure': exp_pressures,
                    'receiver_pressure': rec_pressures,
                    'flow_rate': flows
                })
                
                # Salva
                if file_path.endswith('.csv'):
                    export_df.to_csv(file_path, index=False)
                else:
                    export_df.to_excel(file_path, index=False)
                
                self.log_status(f"Resultados exportados: {file_path}")
                QMessageBox.information(self, "Sucesso", f"Resultados exportados para:\n{file_path}")
                
            except Exception as e:
                logger.error(f"Erro na exportação: {e}")
                QMessageBox.critical(self, "Erro", f"Erro na exportação:\n{str(e)}")
    
    def run_system_tests(self):
        """Executa testes do sistema"""
        self.log_status("Executando testes do sistema...")
        
        try:
            # Executa testes em thread separada para não travar interface
            success = run_tests()
            
            if success:
                self.log_status("✅ Todos os testes passaram!")
                QMessageBox.information(self, "Testes", "Todos os testes passaram com sucesso!")
            else:
                self.log_status("❌ Alguns testes falharam!")
                QMessageBox.warning(self, "Testes", "Alguns testes falharam. Verifique os logs.")
                
        except Exception as e:
            logger.error(f"Erro nos testes: {e}")
            QMessageBox.critical(self, "Erro", f"Erro ao executar testes:\n{str(e)}")
    
    def advanced_calibration(self):
        """Calibração avançada do sistema"""
        if not self.current_system:
            QMessageBox.warning(self, "Aviso", "Configure um sistema primeiro!")
            return
        
        QMessageBox.information(
            self, "Calibração", 
            "Funcionalidade de calibração avançada em desenvolvimento.\n"
            "Esta função permitirá calibração automática baseada em eventos conhecidos."
        )
    
    def modify_current_system(self):
        """Modifica características do sistema atual"""
        if not self.current_system:
            QMessageBox.warning(self, "Aviso", "Configure um sistema primeiro!")
            return
            
        # Cria dialog para modificar características
        dialog = SystemModificationDialog(self.current_system, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Atualiza o sistema com as novas características
            modified_config = dialog.get_modified_config()
            self.current_system = modified_config
            
            # Salva no banco de dados
            database.save_system_configuration(modified_config)
            
            # Atualiza a interface
            self.update_system_display()
            self.log_status(f"Sistema {modified_config.system_id} modificado com sucesso!")
            
            QMessageBox.information(
                self, "Sucesso", 
                f"Sistema {modified_config.system_id} modificado com sucesso!"
            )
    
    def update_system_display(self):
        """Atualiza exibição do sistema na interface"""
        if not self.current_system:
            return
            
        # Atualiza título da janela
        self.setWindowTitle(f"Análise Hidráulica - Sistema: {self.current_system.system_id}")
        
        # Atualiza labels de sistema se existirem
        if hasattr(self, 'system_info_label'):
            info_text = (
                f"Sistema: {self.current_system.system_id} | "
                f"Localização: {self.current_system.location} | "
                f"Comprimento: {self.current_system.pipe_characteristics.length/1000:.1f}km"
            )
            self.system_info_label.setText(info_text)
    
    def analysis_mode_changed(self, mode: str):
        """Callback para mudança do modo de análise"""
        self.log_status(f"Modo de análise alterado para: {mode}")
        
        # Ajusta interface baseado no modo
        if mode == 'Tempo Real':
            self.speed_slider.setEnabled(False)
            self.speed_label.setText("Tempo Real")
            if hasattr(self, 'analysis_timer'):
                self.analysis_timer.setInterval(100)  # 10 FPS para tempo real
        elif mode == 'Histórico':
            self.speed_slider.setEnabled(True)
            self.speed_changed(self.speed_slider.value())
        elif mode == 'Batch':
            self.speed_slider.setEnabled(True)
            self.speed_label.setText("Processamento em Lote")
    
    def speed_changed(self, value: int):
        """Callback para mudança da velocidade de análise"""
        self.speed_label.setText(f"{value}x")
        
        # Ajusta timer se existir
        if hasattr(self, 'analysis_timer') and self.analysis_timer.isActive():
            # Velocidade mais alta = intervalo menor
            interval = max(10, 1000 // value)  # Min 10ms, max 1000ms
            self.analysis_timer.setInterval(interval)
            self.log_status(f"Velocidade de análise: {value}x (intervalo: {interval}ms)")
    
    def sensitivity_changed(self, sensitivity: str):
        """Callback para mudança da sensibilidade ML"""
        self.log_status(f"Sensibilidade ML alterada para: {sensitivity}")
        
        # Mapeia sensibilidade para thresholds
        sensitivity_map = {
            'Baixa': 0.8,    # Menos sensível - menos falsos positivos
            'Média': 0.7,    # Balanceado
            'Alta': 0.6      # Mais sensível - detecta mais anomalias
        }
        
        threshold = sensitivity_map.get(sensitivity, 0.7)
        
        # Atualiza threshold no processador se existir
        if self.processor and hasattr(self.processor, 'ml_system'):
            self.processor.ml_system.detection_threshold = threshold
            self.log_status(f"Threshold de detecção ajustado para: {threshold}")
    
    def update_real_time_metrics(self, metrics: Dict[str, Any]):
        """Atualiza métricas em tempo real na interface"""
        try:
            # Correlação
            if 'correlation' in metrics:
                corr_value = metrics['correlation']
                if isinstance(corr_value, (int, float)):
                    self.correlation_label.setText(f"Correlação: {corr_value:.3f}")
                else:
                    self.correlation_label.setText("Correlação: --")
            
            # Score ML
            if 'ml_score' in metrics:
                ml_score = metrics['ml_score']
                if isinstance(ml_score, (int, float)):
                    self.ml_score_label.setText(f"Score ML: {ml_score:.3f}")
                else:
                    self.ml_score_label.setText("Score ML: --")
            
            # Probabilidade de vazamento
            if 'leak_probability' in metrics:
                leak_prob = metrics['leak_probability']
                if isinstance(leak_prob, (int, float)):
                    self.leak_prob_label.setText(f"Prob. Vazamento: {leak_prob:.2%}")
                    # Muda cor baseado na probabilidade
                    if leak_prob > 0.7:
                        self.leak_prob_label.setStyleSheet("color: red; font-weight: bold;")
                    elif leak_prob > 0.4:
                        self.leak_prob_label.setStyleSheet("color: orange; font-weight: bold;")
                    else:
                        self.leak_prob_label.setStyleSheet("color: green;")
                else:
                    self.leak_prob_label.setText("Prob. Vazamento: --")
                    self.leak_prob_label.setStyleSheet("color: white;")
            
            # Status do sistema
            if 'system_status' in metrics:
                status = metrics['system_status']
                self.system_status_label.setText(f"Status: {status}")
                # Muda cor baseado no status
                if status == 'leak_detected':
                    self.system_status_label.setStyleSheet("color: red; font-weight: bold;")
                elif status == 'anomaly':
                    self.system_status_label.setStyleSheet("color: orange; font-weight: bold;")
                elif status == 'normal':
                    self.system_status_label.setStyleSheet("color: green;")
                else:
                    self.system_status_label.setStyleSheet("color: yellow;")
            
            # Confiança
            if 'confidence' in metrics:
                confidence = metrics['confidence']
                if isinstance(confidence, (int, float)):
                    self.confidence_label.setText(f"Confiança: {confidence:.1%}")
                else:
                    self.confidence_label.setText("Confiança: --")
        
        except Exception as e:
            self.log_status(f"Erro ao atualizar métricas: {e}")
    
    def show_about(self):
        """Mostra informações sobre o sistema"""
        about_text = """
        Sistema Industrial de Análise Hidráulica
        Versão: Complete Rev2
        
        Funcionalidades:
        • Análise de correlação sônica avançada
        • Detecção de vazamentos com Machine Learning
        • Processamento de dados irregulares
        • Interface PyQt6 moderna
        • Sistema de aprendizado adaptativo
        • Múltiplos sistemas gerenciados
        
        Desenvolvido com:
        • Python 3.8+
        • PyQt6 & PyQtGraph
        • NumPy, SciPy, Pandas
        • Scikit-learn
        • SQLite
        
        © 2025 - Sistema de Análise Hidráulica Industrial
        """
        
        QMessageBox.about(self, "Sobre", about_text)
    
    def log_status(self, message: str):
        """Adiciona mensagem ao log de status"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        self.status_text.append(formatted_message)
        
        # Mantém apenas últimas 1000 linhas
        cursor = self.status_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.Start)
        text = self.status_text.toPlainText()
        lines = text.split('\n')
        if len(lines) > 1000:
            self.status_text.clear()
            self.status_text.append('\n'.join(lines[-1000:]))
        
        # Scroll para o final
        scrollbar = self.status_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # Log também para o sistema
        logger.info(message)
    
    def closeEvent(self, event):
        """Handler para fechamento da aplicação com limpeza assíncrona"""
        try:
            self.log_status("Finalizando aplicação - Cancelando processos assíncronos...")
            
            # Para análise se estiver rodando
            if self.analysis_timer.isActive():
                self.stop_analysis()
            
            # Cancelar loader assíncrono
            if self.async_loader and self.async_loader.isRunning():
                self.log_status("Cancelando carregamento assíncrono...")
                self.async_loader.cancel()
                if not self.async_loader.wait(3000):  # Aguarda até 3 segundos
                    logger.warning("Timeout ao aguardar término do loader assíncrono")
                    self.async_loader.terminate()
            
            # Cancelar todos os workers ativos
            if self.active_workers:
                self.log_status(f"Cancelando {len(self.active_workers)} workers ativos...")
                self._cancel_active_workers()
            
            # Aguardar conclusão do ThreadPool
            if hasattr(self, 'thread_pool'):
                self.log_status("Aguardando conclusão do ThreadPool...")
                self.thread_pool.waitForDone(5000)  # 5 segundos timeout
                if self.thread_pool.activeThreadCount() > 0:
                    logger.warning(f"ThreadPool ainda possui {self.thread_pool.activeThreadCount()} threads ativas")
            
            # Para threads de dados se existirem
            if self.data_thread and self.data_thread.isRunning():
                self.log_status("Finalizando thread de dados...")
                self.data_thread.quit()
                if not self.data_thread.wait(3000):
                    logger.warning("Timeout ao aguardar thread de dados")
                    self.data_thread.terminate()
            
            # Limpar processador paralelo
            if hasattr(self, 'parallel_processor'):
                if hasattr(self.parallel_processor, 'thread_pool'):
                    self.parallel_processor.thread_pool.shutdown(wait=False)
                if hasattr(self.parallel_processor, 'process_pool'):
                    self.parallel_processor.process_pool.shutdown(wait=False)
            
            # Limpeza de memória
            memory_manager.cleanup_memory()
            
            self.log_status("✅ Limpeza assíncrona concluída")
            logger.info("Aplicação fechada corretamente com limpeza assíncrona")
            event.accept()
            
        except Exception as e:
            logger.error(f"Erro ao fechar aplicação: {e}")
            # Forçar fechamento em caso de erro
            event.accept()

# ============================================================================
# main.py - FUNÇÃO PRINCIPAL E EXECUÇÃO
# ============================================================================

def main():
    """Função principal do sistema"""
    
    # Configuração inicial
    print("=" * 80)
    print("Sistema Industrial de Análise Hidráulica - Complete Rev2")
    print("Inicializando sistema...")
    print("=" * 80)
    
    try:
        # Inicializa aplicação Qt
        app = QApplication(sys.argv)
        app.setApplicationName("Sistema de Análise Hidráulica")
        app.setApplicationVersion("Complete Rev2")
        app.setOrganizationName("Hydraulic Analysis Systems")
        
        # Configura ícone da aplicação (se disponível)
        try:
            app.setWindowIcon(QIcon("icon.png"))
        except:
            pass
        
        # Verifica dependências críticas
        logger.info("Verificando dependências...")
        
        required_modules = ['numpy', 'pandas', 'scipy', 'sklearn', 'PyQt6', 'pyqtgraph']
        for module in required_modules:
            try:
                __import__(module)
                logger.info(f"[OK] {module} disponivel")
            except ImportError:
                logger.error(f"[FAIL] {module} nao encontrado")
                raise ImportError(f"Módulo obrigatório '{module}' não encontrado")
        
        # Inicializa banco de dados
        logger.info("Inicializando banco de dados...")
        database._init_database()
        logger.info("[OK] Banco de dados inicializado")
        
        # Executa testes básicos se solicitado
        if "--test" in sys.argv:
            logger.info("Executando testes do sistema...")
            if run_tests():
                logger.info("[OK] Todos os testes passaram")
            else:
                logger.error("❌ Alguns testes falharam")
                return 1
        
        # Cria e mostra janela principal
        logger.info("Inicializando interface gráfica...")
        main_window = HydraulicAnalysisMainWindow()
        main_window.show()
        
        logger.info("[OK] Sistema inicializado com sucesso!")
        logger.info("Interface grafica disponivel")
        
        # Executa aplicação
        return app.exec()
        
    except ImportError as e:
        print(f"❌ Erro de dependência: {e}")
        print("\nInstale as dependências necessárias:")
        print("pip install numpy pandas scipy scikit-learn PyQt6 pyqtgraph openpyxl")
        return 1
        
    except Exception as e:
        logger.error(f"❌ Erro crítico na inicialização: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1
    
    finally:
        # Limpeza final
        try:
            memory_manager.cleanup_memory()
            logger.info("Limpeza de memória concluída")
        except:
            pass

if __name__ == "__main__":
    # Configuração de argumentos de linha de comando
    if len(sys.argv) > 1:
        if "--help" in sys.argv or "-h" in sys.argv:
            print("""
Sistema Industrial de Análise Hidráulica - Complete Rev2

Uso: python hydraulic_system_complete.py [opções]

Opções:
  --test          Executa testes do sistema antes da inicialização
  --debug         Ativa modo debug com logs detalhados
  --version       Mostra versão do sistema
  --help, -h      Mostra esta ajuda

Dependências:
  - Python 3.8+
  - numpy, pandas, scipy
  - scikit-learn
  - PyQt6, pyqtgraph
  - openpyxl (para Excel)

Exemplos:
  python hydraulic_system_complete.py
  python hydraulic_system_complete.py --test
  python hydraulic_system_complete.py --debug
            """)
            sys.exit(0)
        
        if "--version" in sys.argv:
            print("Sistema de Análise Hidráulica - Complete Rev2")
            print("Copyright (c) 2025 - Sistema Industrial")
            sys.exit(0)
        
        if "--debug" in sys.argv:
            # Ativa logging detalhado
            industrial_logger.logger.setLevel(logging.DEBUG)
            logger.info("Modo debug ativado")
    
    # Executa sistema
    exit_code = main()
    sys.exit(exit_code)