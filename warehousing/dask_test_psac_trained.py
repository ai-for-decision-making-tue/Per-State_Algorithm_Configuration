from psac_trained_experiment import psac_trained_experiment

from dask.distributed import LocalCluster, Client

import numpy as np

from dask import config as cfg

if __name__ == '__main__':
    cfg.set({'distributed.scheduler.worker-ttl': '60 minutes'})
    n_workers = 4
    threads_per_worker = 1
    # Set up a local cluster with 4 workers (each worker is like a core)
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
    client = Client(cluster)

    # Define tasks for each instance of the script
    tasks = [client.submit(psac_trained_experiment, "mdp_config_non_stat.json", "trained_policies/psac_non_stat_coll_pick.pt", 5 * i, 5 * i + 5) for i in range(n_workers)]

    # Gather results
    results = client.gather(tasks)

    rewards = [item for res in results for item in res]

    # We can take the mean since all the lists have the same length
    print(f"Mean reward: {np.mean(rewards).round(2)} +- {np.std(rewards).round(2)}")

    # Shut down the client
    client.close()
