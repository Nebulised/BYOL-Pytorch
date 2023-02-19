import torch


class Lars(torch.optim.Optimizer):

    def __init__(self,
                 params,
                 lr: float,
                 nesterov: bool = True,
                 weight_decay: float = 0.0,
                 momentum: float = 0.9,
                 eta: float = 0.001,
                 weight_decay_filter=None,
                 lars_adaptation_filter=None):
        self._weight_decay_filter = weight_decay_filter
        self._lars_adaptation_filter = lars_adaptation_filter
        defaults = dict(lr=lr,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        eta=eta,
                        nesterov=nesterov)
        super().__init__(params,
                         defaults)

    @torch.no_grad()
    def step(self,
             closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                self._add_weight_decay(param=param,
                                       weight_decay=group["weight_decay"],
                                       weight_decay_filter=self._weight_decay_filter)
                self._scale_by_lars(param=param,
                                    momentum=group["momentum"],
                                    eta=group["eta"],
                                    filter_fn=self._lars_adaptation_filter,
                                    nesterov=group["nesterov"])
                param.data.add_(param.grad,
                                alpha=-group["lr"])
        return loss

    def _add_weight_decay(self,
                          param,
                          weight_decay: float,
                          weight_decay_filter=None):
        if weight_decay_filter is None or weight_decay_filter(param):
            param.grad.add_(param,
                            alpha=weight_decay)

    def _scale_by_lars(self,
                       param,
                       momentum: float = 0.9,
                       eta: float = 0.001,
                       filter_fn=None,
                       nesterov=True):
        def lars_adaption(param):
            param_norm = torch.linalg.vector_norm(param,
                                                  ord=2)
            grad_norm = torch.linalg.vector_norm(param.grad,
                                                 ord=2)
            return param.grad * torch.where(param_norm > 0.0,
                                            torch.where(grad_norm > 0.0,
                                                        (eta * param_norm / grad_norm),
                                                        1.0),
                                            1.0)

        if filter_fn is None or filter_fn(param):
            lars_adaption(param=param)
        if momentum != 0:
            state = self.state[param]
            if 'momentum_buffer' not in state:
                buf = torch.clone(param.grad).detach()
                state["momentum_buffer"] = buf
            else:
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(param.grad,
                                        alpha=1)

            if nesterov:
                param.grad = param.grad.add(buf,
                                            alpha=momentum)
            else:
                param.grad = buf
