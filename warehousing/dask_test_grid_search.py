from grid_search_experiment import grid_search_experiment
from distributed import LocalCluster, Client

from dask import config as cfg

if __name__ == '__main__':

    cfg.set({'distributed.scheduler.worker-ttl': '60 minutes'})
    # Set up a local cluster with 4 workers (each worker is like a core)
    n_workers = 4
    threads_per_worker = 1
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
    client = Client(cluster)

    # Define tasks for each instance of the script
    tasks = [client.submit(grid_search_experiment, "very_low", f"mdp_config_very_low.json", 0, 20)]
    tasks.extend([client.submit(grid_search_experiment, "low", f"mdp_config_low.json", 0, 20)])
    tasks.extend([client.submit(grid_search_experiment, "medium", f"mdp_config_medium.json", 0, 20)])
    tasks.extend([client.submit(grid_search_experiment, "high", f"mdp_config_high.json", 0, 20)])

    # Gather results (if you need them)
    results = client.gather(tasks)

    # Optionally, shut down the client when done
    client.close()
