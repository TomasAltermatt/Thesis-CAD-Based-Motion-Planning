import multiprocessing as mp
from tqdm import tqdm
import traceback


def parallel_worker(worker, args, kwargs, queue, proc_idx):
    result = worker(*args, **kwargs)
    queue.put([result, proc_idx, args, kwargs])


def parallel_execute(worker, args, kwargs=None, num_proc=1, show_progress=True, desc=None, terminate_func=None, return_args=False, raise_exception=True):
    '''
    Tool for parallel execution
    '''
    if kwargs is not None:
        assert len(args) == len(kwargs)
        has_kwargs = True
    else:
        kwargs = [{}] * len(args)
        has_kwargs = False
    
    if show_progress:
        pbar = tqdm(total=len(args), desc=desc)

    queue = mp.Queue()
    procs = {}
    n_active_proc = 0

    try:

        # loop over arguments for all processes
        for proc_idx, (arg, kwarg) in enumerate(zip(args, kwargs)):

            if num_proc > 1:
                proc = mp.Process(target=parallel_worker, args=(worker, arg, kwarg, queue, proc_idx))
                proc.start()
                procs[proc_idx] = proc
                n_active_proc += 1

                if n_active_proc >= num_proc: # launch a new process after an existing one finishes
                    result, proc_idx, arg, kwarg = queue.get()
                    procs.pop(proc_idx)
                    if return_args:
                        if has_kwargs:
                            yield result, arg, kwarg
                        else:
                            yield result, arg
                    else:
                        yield result

                    if terminate_func and terminate_func(result): # terminate condition meets
                        for p in procs.values(): # terminate all running processes
                            p.terminate()
                        if show_progress:
                            pbar.update(pbar.total - pbar.last_print_n)
                            pbar.close()
                        return
                    
                    n_active_proc -= 1

                    if show_progress:
                        pbar.update(1)
            else:
                result = worker(*arg, **kwarg) # no need to use mp.Process when serial
                if return_args:
                    if has_kwargs:
                        yield result, arg, kwarg
                    else:
                        yield result, arg
                else:
                    yield result

                if terminate_func and terminate_func(result): # terminate condition meets
                    if show_progress:
                        pbar.update(pbar.total - pbar.last_print_n)
                        pbar.close()
                    return

                if show_progress:
                    pbar.update(1)

        for _ in range(n_active_proc): # wait for existing processes to finish
            result, proc_idx, arg, kwarg = queue.get()
            procs.pop(proc_idx)
            if return_args:
                if has_kwargs:
                    yield result, arg, kwarg
                else:
                    yield result, arg
            else:
                yield result

            if terminate_func and terminate_func(result): # terminate condition meets
                for p in procs.values(): # terminate all running processes
                    p.terminate()
                if show_progress:
                    pbar.update(pbar.total - pbar.last_print_n)
                    pbar.close()
                return

            if show_progress:
                pbar.update(1)

    except (Exception, KeyboardInterrupt) as e:
        if type(e) == KeyboardInterrupt:
            print('[parallel_execute] interrupt')
        else:
            print('[parallel_execute] exception:', e)
            print(traceback.format_exc())
        for proc in procs.values():
            proc.terminate()
        if raise_exception:
            raise e

    if show_progress:
        pbar.close()

def fast_parallel_worker(worker, args, queue, proc_idx):
    """Bare-metal worker with zero overhead."""
    result = worker(*args)
    queue.put([result, proc_idx])

def fast_parallel_execute(worker, args, num_proc=1):
    """Stripped-down executor for micro-tasks (no UI, no timeouts)."""
    import multiprocessing as mp
    
    queue = mp.Queue()
    procs = {}
    n_active_proc = 0

    try:
        for proc_idx, arg in enumerate(args):
            if num_proc > 1:
                proc = mp.Process(target=fast_parallel_worker, args=(worker, arg, queue, proc_idx))
                proc.start()
                procs[proc_idx] = proc
                n_active_proc += 1

                if n_active_proc >= num_proc: 
                    result, returned_proc_idx = queue.get()
                    procs.pop(returned_proc_idx)
                    yield result
                    n_active_proc -= 1
            else:
                yield worker(*arg)

        for _ in range(n_active_proc): 
            result, returned_proc_idx = queue.get()
            procs.pop(returned_proc_idx)
            yield result

    except (Exception, KeyboardInterrupt) as e:
        for proc in procs.values():
            proc.terminate()
        raise e