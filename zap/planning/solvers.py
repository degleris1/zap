import torch
import dataclasses


@dataclasses.dataclass(kw_only=True)
class GradientDescent:
    """Parameters for gradient descent."""

    step_size: float = 1e-1
    clip: float = 1e3

    def step(self, state: dict, grad: dict):
        grad_norms = [torch.linalg.vector_norm(grad[param], ord=2) for param in state.keys()]
        total_grad_norm = torch.linalg.vector_norm(torch.stack(grad_norms), ord=2)

        for param in state.keys():
            if total_grad_norm > self.clip:
                clipped_grad = (self.clip / total_grad_norm) * grad[param]
            else:
                clipped_grad = grad[param]

            if isinstance(state[param], torch.Tensor):
                state[param] = state[param].detach() - self.step_size * clipped_grad

            else:
                state[param] -= self.step_size * clipped_grad.numpy()

        return state
