"""
Node 4: AceStep Loss Graph (Observer)
A passive observer node that displays a real-time loss curve during training.
The frontend JS widget receives WebSocket events from the training loop.
"""


class AceStepLossGraph:
    """Real-time training loss graph. Connect the Trainer's 'status' output to visualize loss."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "training_status": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect the 'status' output from the LoRA Trainer node. "
                               "This keeps the graph in the execution chain so it runs "
                               "after training. The value is passed through to the output.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "observe"
    CATEGORY = "AceStep/Training"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def observe(self, training_status=None):
        return (training_status or "",)
