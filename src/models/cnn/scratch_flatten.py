class ScratchFlatten:
    def forward(self, x):
        return x.reshape(x.shape[0], -1)