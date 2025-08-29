"""
🎮 GPU Optimizer para Camera-Renni
Detecta y optimiza el uso de GPU para máximo rendimiento
"""

import torch
import subprocess
import psutil
import os

class GPUOptimizer:
    @staticmethod
    def detect_gpu():
        """🔍 DETECTAR GPU DISPONIBLE Y SUS CAPACIDADES"""
        gpu_info = {
            'available': False,
            'name': None,
            'memory_gb': 0,
            'compute_capability': None,
            'driver_version': None,
            'cuda_version': None,
            'recommended_settings': {}
        }
        
        try:
            if torch.cuda.is_available():
                gpu_info['available'] = True
                gpu_info['name'] = torch.cuda.get_device_name(0)
                gpu_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                # Obtener compute capability
                props = torch.cuda.get_device_properties(0)
                gpu_info['compute_capability'] = f"{props.major}.{props.minor}"
                
                # Obtener versiones
                gpu_info['cuda_version'] = torch.version.cuda
                
                # Detectar driver version si es posible
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        gpu_info['driver_version'] = result.stdout.strip()
                except:
                    pass
                
                # Configuraciones recomendadas basadas en GPU
                gpu_info['recommended_settings'] = GPUOptimizer._get_gpu_recommendations(gpu_info)
                
        except Exception as e:
            print(f"⚠️ Error detectando GPU: {e}")
            
        return gpu_info
    
    @staticmethod
    def _get_gpu_recommendations(gpu_info):
        """🎯 OBTENER CONFIGURACIONES RECOMENDADAS BASADAS EN GPU"""
        memory_gb = gpu_info['memory_gb']
        
        if memory_gb >= 8:
            # GPU de gama alta
            return {
                'batch_size': 4,
                'image_size': 640,
                'half_precision': True,
                'max_det': 30,
                'confidence_threshold': 0.5
            }
        elif memory_gb >= 4:
            # GPU de gama media
            return {
                'batch_size': 2,
                'image_size': 480,
                'half_precision': True,
                'max_det': 20,
                'confidence_threshold': 0.55
            }
        else:
            # GPU de gama baja
            return {
                'batch_size': 1,
                'image_size': 416,
                'half_precision': True,
                'max_det': 15,
                'confidence_threshold': 0.6
            }
    
    @staticmethod
    def optimize_for_gpu():
        """⚡ CONFIGURAR OPTIMIZACIONES DE GPU"""
        if not torch.cuda.is_available():
            return False
            
        try:
            # Limpiar cache de GPU
            torch.cuda.empty_cache()
            
            # Configurar memory fraction (80% del total)
            torch.cuda.set_per_process_memory_fraction(0.8)
            
            # Habilitar cuDNN benchmark para optimización automática
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = False  # Para máxima velocidad
            
            # Configurar caching allocator
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            print("✅ GPU optimizada para inference")
            return True
            
        except Exception as e:
            print(f"❌ Error optimizando GPU: {e}")
            return False

    @staticmethod
    def get_optimal_batch_size(gpu_memory_gb):
        """📊 DETERMINAR BATCH SIZE ÓPTIMO"""
        if gpu_memory_gb >= 8:
            return 4
        elif gpu_memory_gb >= 4:
            return 2
        else:
            return 1

    @staticmethod
    def monitor_gpu_usage():
        """📊 MONITOREAR USO DE GPU"""
        if not torch.cuda.is_available():
            return None
            
        try:
            # Obtener estadísticas de memoria
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # Obtener utilización si nvidia-ml-py está disponible
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
                memory_util = utilization.memory
            except:
                gpu_util = None
                memory_util = None
            
            return {
                'memory_allocated_gb': memory_allocated,
                'memory_reserved_gb': memory_reserved,
                'memory_total_gb': memory_total,
                'memory_usage_percent': (memory_allocated / memory_total) * 100,
                'gpu_utilization_percent': gpu_util,
                'memory_utilization_percent': memory_util
            }
            
        except Exception as e:
            print(f"⚠️ Error monitoreando GPU: {e}")
            return None

    @staticmethod
    def print_gpu_info():
        """🖥️ IMPRIMIR INFORMACIÓN COMPLETA DE GPU"""
        gpu_info = GPUOptimizer.detect_gpu()
        
        print("\n" + "="*60)
        print("🎮 INFORMACIÓN DE GPU")
        print("="*60)
        
        if gpu_info['available']:
            print(f"✅ GPU Disponible: {gpu_info['name']}")
            print(f"🔧 VRAM: {gpu_info['memory_gb']:.1f}GB")
            print(f"⚡ Compute Capability: {gpu_info['compute_capability']}")
            print(f"🚀 CUDA Version: {gpu_info['cuda_version']}")
            if gpu_info['driver_version']:
                print(f"🔌 Driver Version: {gpu_info['driver_version']}")
            
            print("\n📊 CONFIGURACIONES RECOMENDADAS:")
            rec = gpu_info['recommended_settings']
            print(f"   • Batch Size: {rec['batch_size']}")
            print(f"   • Image Size: {rec['image_size']}px")
            print(f"   • Half Precision: {'Sí' if rec['half_precision'] else 'No'}")
            print(f"   • Max Detections: {rec['max_det']}")
            print(f"   • Confidence: {rec['confidence_threshold']}")
            
            # Mostrar uso actual si está disponible
            usage = GPUOptimizer.monitor_gpu_usage()
            if usage:
                print(f"\n💾 USO ACTUAL DE GPU:")
                print(f"   • VRAM Usada: {usage['memory_allocated_gb']:.1f}GB / {usage['memory_total_gb']:.1f}GB")
                print(f"   • VRAM %: {usage['memory_usage_percent']:.1f}%")
                if usage['gpu_utilization_percent'] is not None:
                    print(f"   • GPU Utilization: {usage['gpu_utilization_percent']}%")
        else:
            print("❌ GPU No Disponible - Usando CPU")
            print("💡 Para mejor rendimiento, considera:")
            print("   • Instalar drivers NVIDIA")
            print("   • Instalar CUDA toolkit")
            print("   • Usar GPU compatible")
        
        print("="*60)

    @staticmethod
    def benchmark_inference(model_path=None, iterations=10):
        """🏁 BENCHMARK DE INFERENCIA GPU vs CPU"""
        print("\n🏁 BENCHMARK GPU vs CPU")
        print("-" * 40)
        
        # Test con tensor dummy si no hay modelo
        import time
        import numpy as np
        
        # Crear tensor de prueba
        test_tensor = torch.randn(1, 3, 416, 416)
        
        results = {}
        
        # Test CPU
        print("⏱️ Testeando CPU...")
        test_tensor_cpu = test_tensor.to('cpu')
        start_time = time.time()
        for _ in range(iterations):
            # Simular operación de inference
            _ = torch.nn.functional.conv2d(test_tensor_cpu, torch.randn(64, 3, 3, 3))
        cpu_time = (time.time() - start_time) / iterations
        results['cpu_ms'] = cpu_time * 1000
        
        # Test GPU si está disponible
        if torch.cuda.is_available():
            print("⏱️ Testeando GPU...")
            test_tensor_gpu = test_tensor.to('cuda')
            conv_filter = torch.randn(64, 3, 3, 3).to('cuda')
            
            # Warm up GPU
            for _ in range(3):
                _ = torch.nn.functional.conv2d(test_tensor_gpu, conv_filter)
            torch.cuda.synchronize()
            
            start_time = time.time()
            for _ in range(iterations):
                _ = torch.nn.functional.conv2d(test_tensor_gpu, conv_filter)
            torch.cuda.synchronize()
            gpu_time = (time.time() - start_time) / iterations
            results['gpu_ms'] = gpu_time * 1000
            results['speedup'] = cpu_time / gpu_time
        
        # Mostrar resultados
        print(f"🖥️ CPU Time: {results['cpu_ms']:.1f}ms")
        if 'gpu_ms' in results:
            print(f"🎮 GPU Time: {results['gpu_ms']:.1f}ms")
            print(f"🚀 Speedup: {results['speedup']:.1f}x")
        else:
            print("❌ GPU no disponible para benchmark")
        
        return results

# Función de utilidad para usar en otros módulos
def setup_optimal_device():
    """🚀 CONFIGURAR DEVICE ÓPTIMO AUTOMÁTICAMENTE"""
    gpu_info = GPUOptimizer.detect_gpu()
    
    if gpu_info['available']:
        GPUOptimizer.optimize_for_gpu()
        return 'cuda', gpu_info['recommended_settings']
    else:
        # Optimizar CPU
        import psutil
        cpu_count = psutil.cpu_count()
        torch.set_num_threads(min(8, cpu_count // 2) if cpu_count > 4 else 2)
        torch.set_float32_matmul_precision('medium')
        
        cpu_settings = {
            'batch_size': 1,
            'image_size': 416,
            'half_precision': False,
            'max_det': 15,
            'confidence_threshold': 0.65
        }
        
        return 'cpu', cpu_settings

if __name__ == "__main__":
    # Test del optimizador
    GPUOptimizer.print_gpu_info()
    GPUOptimizer.benchmark_inference()