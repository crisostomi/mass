from mass.merger.merger import TaskVectorBasedMerger


class DummyMerger(TaskVectorBasedMerger):

    def __init__(self):
        super().__init__()

    def merge(self, base_model, finetuned_models):

        return base_model
