"""
Worker Queue System for concurrent toxicity checking and rephrasing.
Uses Python's queue and threading for simple, efficient job processing.
"""

import queue
import threading
import uuid
import time
from typing import Dict, Optional, Callable, Any
from datetime import datetime, timedelta

# Job statuses
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

class JobResult:
    """Container for job results"""
    def __init__(self, job_id: str, job_type: str):
        self.job_id = job_id
        self.job_type = job_type
        self.status = STATUS_PENDING
        self.result = None
        self.error = None
        self.created_at = datetime.now()
        self.completed_at = None
    
    def to_dict(self):
        """Convert to dictionary for JSON response"""
        return {
            'job_id': self.job_id,
            'job_type': self.job_type,
            'status': self.status,
            'result': self.result,
            'error': self.error,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }

class WorkerQueue:
    """Queue-based worker system for processing jobs"""
    
    def __init__(self, num_workers: int = 4):
        """
        Initialize worker queue system.
        
        Args:
            num_workers: Number of worker threads to spawn
        """
        self.num_workers = num_workers
        self.job_queue = queue.Queue()
        self.job_results: Dict[str, JobResult] = {}
        self.workers = []
        self.running = False
        self.lock = threading.Lock()
        
        # Cleanup old jobs every 5 minutes
        self.last_cleanup = datetime.now()
        self.cleanup_interval = timedelta(minutes=5)
        self.job_retention = timedelta(hours=1)  # Keep completed jobs for 1 hour
    
    def start_workers(self, process_function: Callable[[str, Any], Any]):
        """
        Start worker threads.
        
        Args:
            process_function: Function to process jobs. Should accept (job_type, data) and return result.
        """
        if self.running:
            return
        
        self.running = True
        
        def worker():
            """Worker thread function"""
            while self.running:
                try:
                    # Get job from queue (timeout to allow checking self.running)
                    try:
                        job_id, job_type, data = self.job_queue.get(timeout=1)
                    except queue.Empty:
                        continue
                    
                    # Update job status
                    with self.lock:
                        if job_id in self.job_results:
                            self.job_results[job_id].status = STATUS_PROCESSING
                    
                    try:
                        # Process the job
                        result = process_function(job_type, data)
                        
                        # Update job result
                        with self.lock:
                            if job_id in self.job_results:
                                self.job_results[job_id].status = STATUS_COMPLETED
                                self.job_results[job_id].result = result
                                self.job_results[job_id].completed_at = datetime.now()
                    
                    except Exception as e:
                        # Update job with error
                        with self.lock:
                            if job_id in self.job_results:
                                self.job_results[job_id].status = STATUS_FAILED
                                self.job_results[job_id].error = str(e)
                                self.job_results[job_id].completed_at = datetime.now()
                    
                    finally:
                        self.job_queue.task_done()
                        
                        # Periodic cleanup
                        self._cleanup_old_jobs()
                
                except Exception as e:
                    print(f"Worker error: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Start worker threads
        for i in range(self.num_workers):
            worker_thread = threading.Thread(target=worker, daemon=True, name=f"Worker-{i+1}")
            worker_thread.start()
            self.workers.append(worker_thread)
        
        print(f"Started {self.num_workers} worker threads")
    
    def submit_job(self, job_type: str, data: Any) -> str:
        """
        Submit a job to the queue.
        
        Args:
            job_type: Type of job ('toxicity' or 'rephrase')
            data: Data for the job (e.g., text to process)
        
        Returns:
            job_id: Unique identifier for the job
        """
        job_id = str(uuid.uuid4())
        
        # Create job result entry
        job_result = JobResult(job_id, job_type)
        with self.lock:
            self.job_results[job_id] = job_result
        
        # Add job to queue
        self.job_queue.put((job_id, job_type, data))
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """
        Get the status of a job.
        
        Args:
            job_id: Job identifier
        
        Returns:
            Job status dictionary or None if job not found
        """
        with self.lock:
            if job_id in self.job_results:
                return self.job_results[job_id].to_dict()
        return None
    
    def _cleanup_old_jobs(self):
        """Remove old completed/failed jobs to free memory"""
        now = datetime.now()
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        self.last_cleanup = now
        
        with self.lock:
            jobs_to_remove = []
            for job_id, job_result in self.job_results.items():
                if job_result.completed_at:
                    age = now - job_result.completed_at
                    if age > self.job_retention:
                        jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self.job_results[job_id]
            
            if jobs_to_remove:
                print(f"Cleaned up {len(jobs_to_remove)} old jobs")
    
    def stop_workers(self):
        """Stop all worker threads"""
        self.running = False
        # Wait for workers to finish current jobs
        for worker in self.workers:
            worker.join(timeout=5)
        print("Workers stopped")
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.job_queue.qsize()
    
    def get_active_jobs(self) -> int:
        """Get number of active (processing) jobs"""
        with self.lock:
            return sum(1 for jr in self.job_results.values() if jr.status == STATUS_PROCESSING)


