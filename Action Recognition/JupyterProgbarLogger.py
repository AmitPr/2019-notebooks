from tensorflow.keras.callbacks import Callback
from collections import OrderedDict
import numpy as np
import os
import psutil
import platform
import time
from distutils import spawn
from subprocess import Popen, PIPE


class JupyterProgbarLogger(Callback):
    """Callback that prints metrics to stdout. -- Fixed for Jupyter Notebooks
    # Arguments
        count_mode: One of "steps" or "samples".
            Whether the progress bar should
            count samples seen or steps (batches) seen.
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over an epoch.
            Metrics in this list will be logged as-is.
            All others will be averaged over time (e.g. loss, etc).
    # Raises
        ValueError: In case of invalid `count_mode`.
    """

    def __init__(self, count_mode='samples',
                 stateful_metrics=None,
                 notebook=True,
                 measure_gpu=True):
        super(JupyterProgbarLogger,self).__init__()
        if notebook:
            from tqdm import tqdm_notebook as tqdm
        else:
            from tqdm import tqdm
        self.tqdm = tqdm
        self._values = OrderedDict()
        self._seen_so_far = 0
        self._epoch = 0
        self.cpu_usage = []
        self.gpu_usage = []
        self.nvidia_smi= self._find_nvidia_smi()
        self.has_gpu = True and measure_gpu
        self.subproc = None
        self.last_time = time.time()
        if count_mode == 'samples':
            self.use_steps = False
        elif count_mode == 'steps':
            self.use_steps = True
        else:
            raise ValueError('Unknown `count_mode`: ' + str(count_mode))
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()
            
    def on_train_begin(self, logs=None):
        self.verbose = True
        self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        self._values = OrderedDict()
        self._epoch = epoch
        self._seen_so_far = 0
        if self.verbose:
            #print()
            if self.use_steps:
                target = self.params['steps']
            else:
                target = self.params['samples']
            self.target = target
            self.progbar = self.tqdm(total=self.target,desc = self._get_desc_str())
        self.seen = 0
        self.gpu_usage = []
        self.cpu_usage = []

    def on_batch_begin(self, batch, logs=None):
        if self.seen < self.target:
            self.log_values = []

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        if self.use_steps:
            self.seen = 1
        else:
            self.seen = batch_size

        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        if self.verbose and self.seen < self.target:
            self.progbar.set_postfix(self.log_values,refresh=False)
            if time.time()-self.last_time>=1:
                self.progbar.set_description(self._get_desc_str(),refresh=False)
                self.last_time=time.time()
            self.progbar.update(self.seen)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
        if self.verbose:
            self.progbar.update(self.seen)
            self.progbar.set_description(self._get_desc_str())
            self.progbar.set_postfix(self.log_values)
        self.progbar.close()
        
    def update_vals(self,values,step_amt):
        current=self._seen_so_far+step_amt
        for k, v in values:
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                # Stateful metrics output a numeric value.  This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        info=''
        first = True
        for k in self._values:
            if not first:
                info += ' - %s:' % k
            else:
                info += 'Metrics: %s:' % k
                first = False
            if isinstance(self._values[k], list):
                avg = np.mean(
                    self._values[k][0] / max(1, self._values[k][1]))
                if abs(avg) > 1e-3:
                    info += ' %.4f' % avg
                else:
                    info += ' %.4e' % avg
            else:
                info += ' %s' % self._values[k]
        self._seen_so_far =current
        info += '           '
        return info
    
    def _get_desc_str(self):
        """Adapted from Keras-Team/keras-tuner on Github"""
        description = 'Epoch %d/%d' % (self._epoch + 1, self.epochs)
        if self.has_gpu:
            self._get_gpu_usage()
            if len(self.gpu_usage):
                description += '[GPU:%3s%%]' % int(np.average(self.gpu_usage))
        self.cpu_usage.append(int(psutil.cpu_percent(interval=None)))
        description += '[CPU:%3s%%]' % int(np.average(self.cpu_usage))
        return description
            
    def _find_nvidia_smi(self):
        """Find nvidia-smi program used to query the gpu"""
        if platform.system() == "Windows":
            # If the platform is Windows and nvidia-smi
            # could not be found from the environment path,
            # try to find it from system drive with default installation path
            nvidia_smi = spawn.find_executable('nvidia-smi')
            if nvidia_smi is None:
                nvidia_smi = "%s\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe" % os.environ['systemdrive']
        else:
            nvidia_smi = "nvidia-smi"
        return nvidia_smi
    
    def _get_gpu_usage(self):
        """gpu usage"""
        if not self.nvidia_smi:
            return []
        if self.subproc is None:
            self.subproc = Popen([self.nvidia_smi, "--query-gpu=utilization.gpu",
                      "--format=csv,noheader,nounits"], stdout=PIPE)
            return []
        try:
            if self.subproc.poll() is None:
                return []
            stdout = self.subproc.stdout
        except:
            return []
        info = stdout.read().decode('UTF-8')
        #print('Info:', info, 'Truth: ', "Failed" in info)
        if not "Failed" in info:
            self.subproc = None
            self.gpu_usage.append(int(np.average(np.array(info.split('\n')[:-1]).astype(np.uint8))))
        else:
            self.has_gpu=False