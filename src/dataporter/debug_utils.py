"""
Advanced debugging utilities for DataPorter.
Only activated during error conditions to maintain zero overhead during normal operation.
"""

import sys
import traceback
import inspect
import psutil
import torch
import gc
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from datetime import datetime
import threading

def get_advanced_debug_info() -> Dict[str, Any]:
    """
    Collect comprehensive debug information when an error occurs.
    This is expensive but only runs during error handling.
    
    Returns:
        Dictionary with system state, memory info, call stack, etc.
    """
    debug_info = {}
    
    # 1. Full stack trace with locals
    debug_info['stack_trace'] = get_detailed_stack_trace()
    
    # 2. Memory information
    debug_info['memory'] = get_memory_info()
    
    # 3. Process information
    debug_info['process'] = get_process_info()
    
    # 4. Thread information
    debug_info['threads'] = get_thread_info()
    
    # 5. PyTorch-specific info if available
    if torch.cuda.is_available():
        debug_info['cuda'] = get_cuda_info()
    
    # 6. Python garbage collection info
    debug_info['gc'] = get_gc_info()
    
    # 7. Environment variables (filtered for relevant ones)
    debug_info['environment'] = get_relevant_env_vars()
    
    # 8. Timestamp and context
    debug_info['timestamp'] = datetime.now().isoformat()
    debug_info['python_version'] = sys.version
    
    return debug_info


def get_detailed_stack_trace(max_frames: int = 20) -> List[Dict[str, Any]]:
    """
    Get detailed stack trace including local variables for each frame.
    
    Args:
        max_frames: Maximum number of frames to capture
        
    Returns:
        List of frame information dictionaries
    """
    frames = []
    
    # Get the current stack
    current_frame = sys._getframe()
    frame_records = inspect.getouterframes(current_frame)
    
    for i, frame_info in enumerate(frame_records[:max_frames]):
        frame_obj = frame_info.frame
        
        frame_data = {
            'file': frame_info.filename,
            'line': frame_info.lineno,
            'function': frame_info.function,
            'code': frame_info.code_context[0] if frame_info.code_context else '',
            'locals': {}
        }
        
        # Capture local variables (with size limits to prevent huge outputs)
        for var_name, var_value in frame_obj.f_locals.items():
            try:
                # Skip certain variable types that are too large or not useful
                if var_name.startswith('__') or var_name in ['self', 'cls']:
                    continue
                    
                # Represent tensors specially
                if torch.is_tensor(var_value):
                    frame_data['locals'][var_name] = {
                        'type': 'tensor',
                        'shape': str(var_value.shape),
                        'dtype': str(var_value.dtype),
                        'device': str(var_value.device),
                        'requires_grad': var_value.requires_grad,
                        'has_nan': bool(var_value.isnan().any()) if var_value.dtype.is_floating_point else False,
                        'has_inf': bool(var_value.isinf().any()) if var_value.dtype.is_floating_point else False,
                    }
                # Limit string representation
                elif isinstance(var_value, (str, int, float, bool, type(None))):
                    frame_data['locals'][var_name] = repr(var_value)[:200]
                elif isinstance(var_value, (list, tuple)):
                    frame_data['locals'][var_name] = f"{type(var_value).__name__}(len={len(var_value)})"
                elif isinstance(var_value, dict):
                    frame_data['locals'][var_name] = f"dict(keys={list(var_value.keys())[:5]}...)" if len(var_value) > 5 else f"dict(keys={list(var_value.keys())})"
                else:
                    frame_data['locals'][var_name] = f"{type(var_value).__name__} object"
            except:
                frame_data['locals'][var_name] = "<error getting value>"
        
        frames.append(frame_data)
    
    return frames


def get_memory_info() -> Dict[str, Any]:
    """Get detailed memory information."""
    process = psutil.Process()
    vm = psutil.virtual_memory()
    
    memory_info = {
        'system': {
            'total_gb': vm.total / (1024**3),
            'available_gb': vm.available / (1024**3),
            'used_gb': vm.used / (1024**3),
            'percent': vm.percent
        },
        'process': {
            'rss_mb': process.memory_info().rss / (1024**2),
            'vms_mb': process.memory_info().vms / (1024**2),
            'percent': process.memory_percent(),
            'num_fds': process.num_fds() if hasattr(process, 'num_fds') else None
        }
    }
    
    # Add swap info if available
    if hasattr(psutil, 'swap_memory'):
        swap = psutil.swap_memory()
        memory_info['swap'] = {
            'total_gb': swap.total / (1024**3),
            'used_gb': swap.used / (1024**3),
            'percent': swap.percent
        }
    
    return memory_info


def get_process_info() -> Dict[str, Any]:
    """Get current process information."""
    process = psutil.Process()
    
    return {
        'pid': process.pid,
        'ppid': process.ppid(),
        'name': process.name(),
        'status': process.status(),
        'num_threads': process.num_threads(),
        'cpu_percent': process.cpu_percent(interval=0.1),
        'create_time': datetime.fromtimestamp(process.create_time()).isoformat(),
        'cwd': os.getcwd(),
        'num_open_files': len(process.open_files()) if hasattr(process, 'open_files') else None
    }


def get_thread_info() -> List[Dict[str, Any]]:
    """Get information about all threads in the current process."""
    threads = []
    
    for thread in threading.enumerate():
        thread_info = {
            'name': thread.name,
            'id': thread.ident,
            'daemon': thread.daemon,
            'is_alive': thread.is_alive()
        }
        threads.append(thread_info)
    
    return threads


def get_cuda_info() -> Dict[str, Any]:
    """Get CUDA/GPU information if available."""
    cuda_info = {
        'available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        try:
            current_device = torch.cuda.current_device()
            cuda_info.update({
                'current_device': current_device,
                'device_name': torch.cuda.get_device_name(current_device),
                'memory_allocated_mb': torch.cuda.memory_allocated(current_device) / (1024**2),
                'memory_reserved_mb': torch.cuda.memory_reserved(current_device) / (1024**2),
                'memory_cached_mb': torch.cuda.memory_reserved(current_device) / (1024**2),
                'max_memory_allocated_mb': torch.cuda.max_memory_allocated(current_device) / (1024**2),
            })
            
            # Get memory stats if available (PyTorch 1.4+)
            if hasattr(torch.cuda, 'memory_stats'):
                stats = torch.cuda.memory_stats(current_device)
                cuda_info['memory_stats'] = {
                    'num_alloc_retries': stats.get('num_alloc_retries', 0),
                    'num_ooms': stats.get('num_ooms', 0),
                    'active_bytes': stats.get('active_bytes.all.current', 0) / (1024**2)
                }
        except Exception as e:
            cuda_info['error'] = str(e)
    
    return cuda_info


def get_gc_info() -> Dict[str, Any]:
    """Get garbage collection information."""
    gc_stats = gc.get_stats()
    
    return {
        'enabled': gc.isenabled(),
        'threshold': gc.get_threshold(),
        'count': gc.get_count(),
        'stats': gc_stats[-1] if gc_stats else {},  # Most recent generation
        'num_tracked_objects': len(gc.get_objects())
    }


def get_relevant_env_vars() -> Dict[str, str]:
    """Get environment variables relevant to debugging."""
    relevant_patterns = [
        'CUDA', 'TORCH', 'PYTHON', 'OMP', 'MKL', 'DATALOADER',
        'PATH', 'LD_LIBRARY_PATH', 'NCCL', 'PYTORCH'
    ]
    
    env_vars = {}
    for key, value in os.environ.items():
        if any(pattern in key.upper() for pattern in relevant_patterns):
            # Truncate long values
            env_vars[key] = value[:200] + '...' if len(value) > 200 else value
    
    return env_vars


def format_debug_report(debug_info: Dict[str, Any], error: Exception) -> str:
    """
    Format debug information into a readable report.
    
    Args:
        debug_info: Debug information dictionary
        error: The exception that triggered the debug report
        
    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("ADVANCED DEBUG REPORT")
    lines.append("=" * 80)
    
    # Error information
    lines.append(f"\nERROR: {type(error).__name__}: {str(error)}")
    lines.append("\nTRACEBACK:")
    lines.append(traceback.format_exc())
    
    # Memory status
    mem = debug_info['memory']
    lines.append("\nMEMORY STATUS:")
    lines.append(f"  System: {mem['system']['used_gb']:.1f}/{mem['system']['total_gb']:.1f} GB ({mem['system']['percent']:.1f}%)")
    lines.append(f"  Process: {mem['process']['rss_mb']:.1f} MB ({mem['process']['percent']:.1f}%)")
    
    # CUDA status
    if 'cuda' in debug_info:
        cuda = debug_info['cuda']
        if cuda['available']:
            lines.append(f"\nCUDA STATUS:")
            lines.append(f"  Device: {cuda.get('device_name', 'Unknown')}")
            lines.append(f"  Memory: {cuda.get('memory_allocated_mb', 0):.1f}/{cuda.get('memory_reserved_mb', 0):.1f} MB")
            if 'memory_stats' in cuda:
                lines.append(f"  OOMs: {cuda['memory_stats'].get('num_ooms', 0)}")
    
    # Process info
    proc = debug_info['process']
    lines.append(f"\nPROCESS INFO:")
    lines.append(f"  PID: {proc['pid']} (Parent: {proc['ppid']})")
    lines.append(f"  Threads: {proc['num_threads']}")
    lines.append(f"  CPU: {proc['cpu_percent']:.1f}%")
    
    # Stack frames with locals
    lines.append(f"\nSTACK TRACE WITH LOCALS:")
    for i, frame in enumerate(debug_info['stack_trace'][-10:]):  # Last 10 frames
        lines.append(f"\n  Frame {i}: {frame['function']} in {frame['file']}:{frame['line']}")
        if frame['code']:
            lines.append(f"    Code: {frame['code'].strip()}")
        if frame['locals']:
            lines.append(f"    Locals:")
            for var_name, var_value in list(frame['locals'].items())[:5]:  # First 5 locals
                if isinstance(var_value, dict) and var_value.get('type') == 'tensor':
                    lines.append(f"      {var_name}: tensor{var_value['shape']} {var_value['dtype']} nan={var_value['has_nan']} inf={var_value['has_inf']}")
                else:
                    lines.append(f"      {var_name}: {var_value}")
    
    lines.append("\n" + "=" * 80)
    
    return "\n".join(lines)


def save_debug_dump(debug_info: Dict[str, Any], error: Exception, dump_dir: Optional[Path] = None):
    """
    Save debug information to a file for later analysis.
    
    Args:
        debug_info: Debug information dictionary
        error: The exception that triggered the dump
        dump_dir: Directory to save dump files (default: ./debug_dumps/)
    """
    if dump_dir is None:
        dump_dir = Path("./debug_dumps")
    
    dump_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dump_file = dump_dir / f"debug_dump_{timestamp}.json"
    
    # Add error info to the dump
    debug_info['error'] = {
        'type': type(error).__name__,
        'message': str(error),
        'traceback': traceback.format_exc()
    }
    
    # Convert non-serializable objects
    def make_serializable(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        else:
            return str(obj)
    
    serializable_info = make_serializable(debug_info)
    
    with open(dump_file, 'w') as f:
        json.dump(serializable_info, f, indent=2)
    
    return dump_file


# Context manager for automatic debug collection
class DebugOnError:
    """
    Context manager that automatically collects debug info on exceptions.
    
    Usage:
        with DebugOnError(save_dump=True):
            # Your code here
            risky_operation()
    """
    
    def __init__(self, save_dump: bool = False, print_report: bool = True):
        self.save_dump = save_dump
        self.print_report = print_report
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            # Collect debug info
            debug_info = get_advanced_debug_info()
            
            # Print report if requested
            if self.print_report:
                report = format_debug_report(debug_info, exc_val)
                print(report, file=sys.stderr)
            
            # Save dump if requested
            if self.save_dump:
                dump_file = save_debug_dump(debug_info, exc_val)
                print(f"Debug dump saved to: {dump_file}", file=sys.stderr)
        
        # Don't suppress the exception
        return False