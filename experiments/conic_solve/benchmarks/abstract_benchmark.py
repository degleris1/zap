import os
from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional
import cvxpy as cp


class AbstractBenchmarkSet(ABC):
    def __init__(self, data_dir: Optional[str] = None, num_problems: Optional[int] = None):
        """
        Args:
            data_dir: Path to the data directory (for file-based problems). For
                      generated problems, this can be None.
            num_problems: For generated problems, the number to produce.
        """
        self.data_dir = data_dir
        self.num_problems = num_problems

    @abstractmethod
    def get_data(self, identifier: Any) -> Any:
        """
        Given an identifier (e.g. a file path for file-based problems, or an index for random ones),
        we load (or generate) problem data
        """
        pass

    @abstractmethod
    def create_problem(self, data: Any) -> cp.Problem:
        """
        Create and return a CVXPY problem.
        """
        pass

    def get_problem_identifiers(self) -> Iterator[Any]:
        """
        Returns an iterator of problem identifiers.
        For file-based sets, this is a list of file paths.
        For generated sets, this is a range of numbers.
        """
        if self.data_dir is not None:
            for fname in os.listdir(self.data_dir):
                # Currently only supporting .mat files (Maros and Netlib both use .mat)
                if fname.endswith(".mat"):
                    yield os.path.join(self.data_dir, fname)
        elif self.num_problems is not None:
            yield from range(self.num_problems)
        else:
            raise ValueError("Must specify either a data_dir or num_problems.")

    def __iter__(self) -> Iterator[cp.Problem]:
        """
        Iterate over the benchmark problems.
        """
        for ident in self.get_problem_identifiers():
            data = self.get_data(ident)
            problem = self.create_problem(data)
            yield problem
