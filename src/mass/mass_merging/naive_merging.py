def format_parameters(
    parameters: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
    train: bool,
    device: torch.device
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, int]]:
    tasks = list(parameters.keys())
    layers = parameters[tasks[0]].keys()
    state: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer in layers:
        if not is_supported_layer_svd(layer):
            continue
        u_list, s_list, v_list = [], [], []
        for task in tasks:
            svd = parameters[task][layer]
            u_list.append(svd['u'])
            s_list.append(svd['s'])
            v_list.append(svd['v'])

        u = torch.stack(u_list).to(device)
        s = torch.stack(s_list).to(device)
        v = torch.stack(v_list).to(device)

        state[layer] = {
            'u': nn.Parameter(u.clone().detach(), requires_grad=train),
            's': nn.Parameter(s.clone().detach(), requires_grad=train),
            'v': nn.Parameter(v.clone().detach(), requires_grad=train),
        }
    idx_map = {get_dataset_label(task): i for i, task in enumerate(tasks)}
    return state, idx_map

class LayerOptimProblem:
    def __init__(
        self,
        cfg,
        parameters: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
        dataset,
        device,
    ) -> None:
        self.device = device
        self.params, self.idx_map = format_parameters(parameters, train=True, device=self.device)
        self.original, _         = format_parameters(parameters, train=False, device=self.device)
        for layer in self.params:
            for name in ('u','s','v'):
                self.params[layer][name]     = self.params[layer][name].to(self.device)
                self.original[layer][name]   = self.original[layer][name].to(self.device).detach()
        self.dataset = dataset
        optim_params = []
        for layer_state in self.params.values():
            optim_params += [
                layer_state['u'],
                layer_state['s'],
                layer_state['v'],
            ]
        self.optimizer = instantiate(cfg.optim, params=optim_params)

    def _reconstruct(self, layer_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        u = layer_params['u']
        s = torch.diag_embed(layer_params['s'])
        vT = layer_params['v']
        return torch.einsum('tir,trr,trm->tim', u, s, vT)

    def _recon_diff_loss(self, layer: str) -> torch.Tensor:
        '''
        it's the part of the loss corresponding to: 
        \sum_t\lVert U_t^l-U_t^{l'}\rVert_2 + \lVert \Sigma_t^l-\Sigma_t^{l'}\rVert_2 + \lVert V_t^l-V_t^{l'}\rVert_2
        i.e. the distance from the original parameters

        we use this term to avoid sampling out of distribution weights
        '''
        delta       = self.params[layer]
        delta_prime = self.original[layer]
        return (
            torch.sum((delta['u'] - delta_prime['u']) ** 2) +
            torch.sum((delta['s'] - delta_prime['s']) ** 2) +
            torch.sum((delta['v'] - delta_prime['v']) ** 2)
        )
    def _interference_loss(self, layer: str, x: torch.Tensor, t_x: torch.Tensor) -> torch.Tensor:
        '''
        It's the part of the loss corresponding to:
        \sum_{x^{l-1}\in \mathcal{D}', t_x \neq t}\lVert(U_t^{l'}\Sigma_t^{l'} (V_t^{l'})^T)x^{l-1} \rVert_2 
        i.e. is the magnitude of the output of the models not corresponding to the "correct one"

        we use this term to shrink out the infterference
        '''
        delta = self._reconstruct(self.params[layer]) # t, output, embedding
        y = torch.einsum('tod,bpd->tbpo', delta, x) # t, output, embedding @ batch, patch, embedding
        norms = torch.norm(y, dim=-1)
        norms = norms.sum(dim=-1)       
        T = norms.size(0)
        task_indices = torch.tensor([self.idx_map[int(label)] for label in t_x.tolist()], device=self.device)

        # only where the task index doesn't coincide with the label
        mask = torch.arange(T, device=self.device).unsqueeze(1) != task_indices.unsqueeze(0)
        return norms[mask].mean()

    def _signal_loss(self, layer: str, x: torch.Tensor, t_x: torch.Tensor) -> torch.Tensor:
        '''
        It's the part of the loss corresponding to:
        \sum_{x^{l-1}\in \mathcal{D}', t_x = t} \lVert ((U_t^l\Sigma_t^l (V_t^l)^T)- (U_t^{l'}\Sigma_t^{l'} (V_t^{l'})^T)x^{l-1})\rVert_2
        i.e. is the difference between the deltas of the tasks corresponding to the label of x times x

        it is a term we use to avoid shrking also the signal of finetuned when optimising
        '''
        delta_prime = self._reconstruct(self.params[layer])
        delta = self._reconstruct(self.original[layer])
        diff  = delta_prime - delta
        y = torch.einsum('tod,bpd->tbpo', diff, x)
        norms = torch.norm(y, dim=-1)
        norms = norms.sum(dim=-1)  
        T = norms.size(0)
        task_indices = torch.tensor([self.idx_map[int(label)] for label in t_x.tolist()], device=self.device)

        # only where the task index coincide with the label
        mask = torch.arange(T, device=self.device).unsqueeze(1) == task_indices.unsqueeze(0)
        return norms[mask].mean()
        

    def fit(
        self,
        max_epochs: int = 10,
        tol: float = 1e-4
    ) -> None:
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0

            
            layer_bar = tqdm(
                self.params,
                desc=f"Epoch {epoch}",
                unit="layer",
                position=0,
                leave=False,
                dynamic_ncols=True
            )
            for layer in layer_bar:
                layer_bar.set_description(f"Epoch {epoch} | Layer {layer}")

                dataloader = DataLoader(
                    self.dataset[svd_key_to_router_key(layer)],
                    shuffle=True,
                    batch_size=32 # TODO: remove this hard coding
                )

                batch_bar = tqdm(
                    dataloader,
                    desc="  Batch",
                    unit="batch",
                    position=0,
                    leave=True,
                    dynamic_ncols=True
                )
                for x, t_x in batch_bar:
                    x, t_x = x.to(self.device), t_x.to(self.device)

                    self.optimizer.zero_grad()

                    # l_diff = self._recon_diff_loss(layer)
                    l_int  = self._interference_loss(layer, x, t_x)
                    l_sig  = self._signal_loss(layer, x, t_x)
                    # batch_loss = l_diff + l_int + l_sig
                    batch_loss = l_int + l_sig
                    batch_loss.backward()
                    self.optimizer.step()

                    total_loss += batch_loss.item()

                    batch_bar.set_postfix({
                        # "l_diff": f"{l_diff.item():.4f}",
                        "l_int":  f"{l_int.item():.4f}",
                        "l_sig":  f"{l_sig.item():.4f}",
                        "loss":   f"{batch_loss.item():.4f}"
                    })

                layer_bar.set_postfix(last_loss=f"{batch_loss.item():.4f}")

            avg_loss = total_loss / len(dataloader)
            pylogger.info(
                f"Epoch {epoch:3d} — Avg Loss: {avg_loss:.4f} "
                # f"-- l_recon: {l_diff.item():.4f} "
                f"-- l_sig:   {l_sig.item():.4f} "
                f"-- l_int:   {l_int.item():.4f}"
            )
            if avg_loss < tol:
                break

# TODO: put this inside the class
def merge_parameters(zeroshot, problem: LayerOptimProblem) -> Dict[str, torch.Tensor]:
    def _reconstruct(layer_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        u = layer_params['u']                                  # [T, O, R]
        s = torch.diag_embed(layer_params['s'])               # [T, R, R]
        vT = layer_params['v']                                # [T, I, R]
        return torch.einsum('tor,trr,tri->toi', u, s, vT)

    new_state = copy.deepcopy(zeroshot.state_dict())

    for layer_key, layer_params in problem.params.items():
        if add_transformer_key(layer_key) not in new_state:
            pylogger.warning(f"Skipping layer {layer_key}")
        delta_t = _reconstruct(layer_params)    # [T, out, in]
        delta   = delta_t.sum(dim=0)            # [out, in]

        delta = delta.to(new_state[add_transformer_key(layer_key)].device).type_as(new_state[add_transformer_key(layer_key)])
        new_state[add_transformer_key(layer_key)] = new_state[add_transformer_key(layer_key)] + delta

    return new_state