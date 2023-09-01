import numpy as np
from datasets.iterable_dataset import _BaseExamplesIterable, HasNextIterator, deepcopy, IterableDataset, Features, DatasetInfo
from typing import Optional, List, Any, Dict, Optional, Union, Tuple, Iterator
from datasets import load_dataset, IterableDataset, set_caching_enabled


class MultiSourcesExamplesIterable(_BaseExamplesIterable):
    def __init__(
        self,
        ex_iterables,
        generator: np.random.Generator,
        probabilities: Optional[List[float]] = None
    ):
        self.ex_iterables = ex_iterables
        self.generator = deepcopy(generator)
        self.probabilities = probabilities
        

    @staticmethod
    def _iter_random_indices(
        rng: np.random.Generator,
        num_sources: int,
        random_batch_size=1000,
        p: Optional[List[float]] = None,
    ) -> Iterator[int]:
        """Get an infinite iterator that randomly samples the index of the source to pick examples from."""
        if p is None:
            while True:
                yield from (int(i) for i in rng.integers(0, num_sources, size=random_batch_size))
        else:
            while True:
                yield from (int(i) for i in rng.choice(num_sources, size=random_batch_size, p=p))

    def _give_indice_iterator(self):
        rng = deepcopy(self.generator)
        # this is an infinite iterator that randomly samples the index of the source to pick examples from
        return self._iter_random_indices(rng, len(self.ex_iterables), p=self.probabilities)


    def __iter__(self):
        iterators = [[HasNextIterator(ex) for ex in ex_iterable] for ex_iterable in self.ex_iterables]
        ex_idxs =  [0 for _ in self.ex_iterables]

        indices_iterator = self._give_indice_iterator()

        for i in indices_iterator:

            j = ex_idxs[i]

            try:  # let's pick one example from the iterator at index i
                yield next(iterators[i][j])
                # it will resume from the yield at the next call so that we can directly test if the iterable is exhausted and if we need to break out of the loop
                if not iterators[i][j].hasnext():
                    iterators[i][j] = HasNextIterator(self.ex_iterables[i][j])

            except StopIteration:
                iterators[i][j] = HasNextIterator(self.ex_iterables[i][j])
            
            ex_idxs[i] = (j + 1) % len(iterators[i])


    def shuffle_data_sources(self, generator: np.random.Generator) -> "MultiSourcesExamplesIterable":
        """Shuffle the data sources of each wrapped examples iterable."""
        ex_iterables = [[ex.shuffle_data_sources(generator) for ex in ex_iterable] for ex_iterable in self.ex_iterables]
        return MultiSourcesExamplesIterable(
            ex_iterables, generator=generator, probabilities=self.probabilities
        )

    def shard_data_sources(self, shard_idx: int) -> "MultiSourcesExamplesIterable":
        """Either keep only the requested shard, or propagate the request to the underlying iterable."""
        raise NotImplementedError("Sharding a RandomlyCyclingMultiSourcesExamplesIterable is not implemented")

def mkdir_json_dataset(
    json_data_paths: List[str], 
    probabilities: Optional[List[float]] = None,
    seed: Optional[int] = None,
    features: Optional[Features] = None,
):
    generator = np.random.default_rng(seed)

    assert len(json_data_paths) == len(probabilities)

    for json_data_path in json_data_paths:
        assert len(json_data_path)


    json_datasets = [
        [
            load_dataset(
                "json", 
                data_files=data_path, 
                streaming=True,
                split="train",
                features=features
            )
            for data_path in json_data_path
        ]
        for json_data_path in json_data_paths
    ]

    ex_iterables = [[d._ex_iterable for d in json_dataset] for json_dataset in json_datasets]

    ex_iterable = MultiSourcesExamplesIterable(
        ex_iterables, 
        generator=generator, 
        probabilities=probabilities
    )

    flatten_json_datasets = []
    for item in json_datasets:
        flatten_json_datasets.extend(item)

    info = DatasetInfo.from_merge([d.info for d in flatten_json_datasets])

    token_per_repo_id = {
        repo_id: token for dataset in flatten_json_datasets for repo_id, token in dataset._token_per_repo_id.items()
    }

    # Return new daset
    return IterableDataset(ex_iterable=ex_iterable, info=info, split=None, token_per_repo_id=token_per_repo_id)


