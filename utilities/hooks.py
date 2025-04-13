import torch
import torch.nn as nn

class BaseHook:
    """ Hook base class for PyTorch Models. """
    def __init__(self):
        self.reset()

    def reset(self):
        pass

    def __call__(self, module, input, output):
        pass

class LogitsHook(BaseHook):
    """ Hook to extract logits from a PyTorch model"""
    def reset(self):
        self.logits = []
    
    def __call__(self, module, input, output):
        self.logits.append(output.detach())



# Test code 
if __name__ == "__main__":
    """ Test Code for LogitsHook """
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    # Instantiate the model and hook
    model = SimpleModel()
    hook = LogitsHook()

    # Register the hook to the model's fully connected layer
    model.fc.register_forward_hook(hook)

    # Create dummy input and pass it through the model
    dummy_input = torch.randn(5, 10)  # Batch of 5, 10 features each
    output = model(dummy_input)

    # Print the captured logits
    print("Captured logits:")
    for logits in hook.logits:
        print(logits)
