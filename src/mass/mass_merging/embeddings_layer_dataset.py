class LayerHook:

    def __init__(self, model: torch.nn.Module):
        self.middle_features: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.hooks = []

        pylogger.info(f"Registering hooks...")
        for name, module in model.named_modules():
            if not is_supported_layer(name):
                continue
            handle = module.register_forward_hook(self._hook_fn(name))
            self.hooks.append(handle)

    def _hook_fn(self, name: str):
        def hook(module, inputs, outputs):
            data = inputs[0] if isinstance(inputs, tuple) else inputs
            if isinstance(data, torch.Tensor):
                self.middle_features[name].append(data.permute(1, 0, 2).detach().cpu())
            else:
                pylogger.warning(f"Unexpected input type {type(data)} at layer '{name}'")
        return hook

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()


class EmbeddingsDataset(Dataset):
    def __init__(
        self,
        finetuned_models: Dict[str, torch.nn.Module],
        datasets: Dict[str, pl.LightningDataModule],
        n_batches,
        cfg: dict,
        callbacks: List = None
    ):
        super().__init__()
        self.finetuned_models = finetuned_models
        self.datasets = datasets
        self.cfg = cfg
        self.n_batches = n_batches
        self.callbacks = callbacks or []

        self.loggers: Dict[str, LayerHook] = {}
        self.layer_datasets: Dict[str, TensorDataset] = {}

    def generate_layer_datasets(self) -> Dict[str, TensorDataset]:
        temp_feats: Dict[str, List[torch.Tensor]] = defaultdict(list)
        temp_labels: Dict[str, List[torch.Tensor]] = defaultdict(list)

        for task, model in self.finetuned_models.items():
            
            pylogger.info(f"Instantiating finetuned model for task: '{task}'")
            finetuned_encoder: ImageEncoder = instantiate(
                cfg.nn.module.encoder
            ) 

            finetuned_encoder.load_state_dict(model, strict=False)
                

            hook = LayerHook(finetuned_encoder)
            self.loggers[task] = hook

            lt_encoder: EncoderWrapper = instantiate(
                cfg.nn.module,
                encoder = finetuned_encoder,
                _recursive_=False,
            )
            
            label = get_dataset_label(task)

            trainer = pl.Trainer(
                default_root_dir=cfg.core.storage_dir,
                plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
                logger=logger,
                callbacks=self.callbacks,
                limit_test_batches=self.n_batches,
                **cfg.train.trainer,
            )

            dataloader = self.datasets[task].train_loader
            pylogger.info(f"Generating embeddings for task '{task}' with label {label}")
            trainer.test(model=lt_encoder, dataloaders=dataloader)


            hook.remove_hooks()
    
            for layer_name, feats in hook.middle_features.items():
                for batch_feats in feats:
                    batch_size = batch_feats.size(0)
                    temp_feats[layer_name].append(batch_feats)
                
                    temp_labels[layer_name].append(
                        torch.full((batch_size,), label, dtype=torch.long)
                    )
            del hook

        for layer_name in temp_feats:
            all_feats = torch.cat(temp_feats[layer_name], dim=0)
            all_labels = torch.cat(temp_labels[layer_name], dim=0)
            self.layer_datasets[layer_name] = TensorDataset(all_feats, all_labels)

        return self.layer_datasets