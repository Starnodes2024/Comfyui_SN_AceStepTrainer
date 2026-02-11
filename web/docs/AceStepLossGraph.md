# ⭐SN AceStep Loss Graph

## Description
A real-time training loss visualization node. Displays a live-updating loss curve during LoRA training using HTML5 Canvas with a dark neon aesthetic. The graph updates every single optimizer step via WebSocket — no polling or page reloads needed.

## Inputs

### Optional
- **training_status**: Connect the `status` output from the **⭐SN AceStep 1.5 LoRA Trainer** node. This keeps the graph in the execution chain so it runs after training. The value is passed through to the output.

## Outputs
- **status**: Pass-through of the training status string from the Trainer node.

## Visual Features
- **Neon green loss curve** with glow effect on a dark semi-transparent background
- **Auto-scaling Y-axis** that adjusts as loss decreases
- **Subtle grid lines** for easy reading
- **Stats overlay** showing: Current Step, Loss, Learning Rate, and ETA
- **Status pill**: Green "TRAINING" during training, blue "DONE" when finished
- **Pulsing dot** on the latest data point
- **Responsive**: Resizes with the node when dragged

## Usage
1. Add the **⭐SN AceStep Loss Graph** node to your workflow
2. Connect the `status` output from the **⭐SN AceStep 1.5 LoRA Trainer** to the `training_status` input
3. Queue the workflow — the graph updates live as training progresses
4. Resize the node by dragging its edges for a larger or smaller view

## Notes
- The graph receives data via WebSocket events (`ace_step_training_update`) and does **not** require the node to be connected — it will display data from any active training run
- Up to 4000 data points are stored; older points are dropped for very long runs
- The graph resets automatically when a new training run starts
- When no training is active, the node shows "Waiting for training…"
