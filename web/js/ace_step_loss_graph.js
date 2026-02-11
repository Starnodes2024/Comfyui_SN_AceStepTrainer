/**
 * AceStep Loss Graph — Real-time training loss visualization node.
 *
 * Listens for "ace_step_training_update" WebSocket events from the server
 * and renders a live-updating loss curve using HTML5 Canvas inside LiteGraph.
 *
 * Dark-mode neon aesthetic: dark background, subtle grid, neon-green line.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "AceStep.LossGraph",

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== "AceStep_Loss_Graph") return;

        // ── colours & style constants ──────────────────────────────
        const BG           = "rgba(13, 17, 23, 0.92)";
        const GRID_COLOR   = "rgba(48, 54, 61, 0.55)";
        const AXIS_COLOR   = "rgba(139, 148, 158, 0.6)";
        const LINE_COLOR   = "#39ff14";          // neon green
        const GLOW_COLOR   = "rgba(57, 255, 20, 0.25)";
        const TEXT_COLOR    = "#c9d1d9";
        const ACCENT_COLOR = "#39ff14";
        const LABEL_FONT   = "11px 'Segoe UI', Consolas, monospace";
        const STAT_FONT    = "bold 13px 'Segoe UI', Consolas, monospace";
        const TITLE_FONT   = "bold 11px 'Segoe UI', Consolas, monospace";

        // ── per-instance state (stored on each node) ───────────────
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origOnNodeCreated?.apply(this, arguments);

            this._lossData   = [];   // [{step, loss}]
            this._maxPoints  = 4000;
            this._minLoss    = Infinity;
            this._maxLoss    = -Infinity;
            this._curStep    = 0;
            this._curLoss    = 0;
            this._totalSteps = 0;
            this._lr         = 0;
            this._eta        = "";
            this._running    = false;
            this._finished   = false;

            // Ensure a reasonable minimum size
            this.size = this.size || [420, 280];
            if (this.size[0] < 300) this.size[0] = 300;
            if (this.size[1] < 220) this.size[1] = 220;
            this.resizable = true;

            // ── WebSocket listener ─────────────────────────────────
            this._handler = (event) => {
                const d = event.detail;
                if (!d) return;

                if (d.type === "start") {
                    this._lossData   = [];
                    this._minLoss    = Infinity;
                    this._maxLoss    = -Infinity;
                    this._totalSteps = d.max_steps || 0;
                    this._running    = true;
                    this._finished   = false;
                } else if (d.type === "step") {
                    const pt = { step: d.step, loss: d.loss };
                    this._lossData.push(pt);
                    if (this._lossData.length > this._maxPoints) {
                        this._lossData.shift();
                    }
                    if (d.loss < this._minLoss) this._minLoss = d.loss;
                    if (d.loss > this._maxLoss) this._maxLoss = d.loss;
                    this._curStep  = d.step;
                    this._curLoss  = d.loss;
                    this._lr       = d.lr  || 0;
                    this._eta      = d.eta || "";
                } else if (d.type === "done") {
                    this._running  = false;
                    this._finished = true;
                }

                // Request a canvas redraw
                app.graph.setDirtyCanvas(true, false);
            };
            api.addEventListener("ace_step_training_update", this._handler);
        };

        // Clean up listener on removal
        const origOnRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            if (this._handler) {
                api.removeEventListener("ace_step_training_update", this._handler);
            }
            origOnRemoved?.apply(this, arguments);
        };

        // ── Drawing ────────────────────────────────────────────────
        nodeType.prototype.onDrawForeground = function (ctx) {
            if (this.flags.collapsed) return;

            const PAD_L = 58;   // left padding (Y-axis labels)
            const PAD_R = 16;
            const PAD_T = 10;
            const PAD_B = 36;   // bottom padding (X-axis labels)
            const STAT_H = 52;  // height of the stats overlay band

            const w = this.size[0];
            const h = this.size[1];

            const plotL = PAD_L;
            const plotR = w - PAD_R;
            const plotT = PAD_T;
            const plotB = h - PAD_B - STAT_H;
            const plotW = plotR - plotL;
            const plotH = plotB - plotT;

            if (plotW < 20 || plotH < 20) return;

            // ── background ──
            ctx.save();
            ctx.fillStyle = BG;
            ctx.beginPath();
            roundRect(ctx, 0, 0, w, h, 6);
            ctx.fill();

            // ── grid ──
            const data = this._lossData;
            const hasData = data.length > 1;

            // Y range (auto-scale with 10 % padding)
            let yMin = this._minLoss;
            let yMax = this._maxLoss;
            if (!hasData || yMin === yMax) {
                yMin = 0;
                yMax = 1;
            }
            const yPad = (yMax - yMin) * 0.10 || 0.05;
            yMin = Math.max(0, yMin - yPad);
            yMax = yMax + yPad;

            // X range
            let xMin = hasData ? data[0].step : 0;
            let xMax = hasData ? data[data.length - 1].step : (this._totalSteps || 100);
            if (xMin === xMax) xMax = xMin + 1;

            // Grid lines
            ctx.strokeStyle = GRID_COLOR;
            ctx.lineWidth = 1;
            const yTicks = niceTickCount(plotH, 40);
            const xTicks = niceTickCount(plotW, 70);

            // Y grid + labels
            ctx.font = LABEL_FONT;
            ctx.textAlign = "right";
            ctx.textBaseline = "middle";
            for (let i = 0; i <= yTicks; i++) {
                const frac = i / yTicks;
                const y = plotB - frac * plotH;
                const val = yMin + frac * (yMax - yMin);
                ctx.beginPath();
                ctx.moveTo(plotL, y);
                ctx.lineTo(plotR, y);
                ctx.stroke();
                ctx.fillStyle = AXIS_COLOR;
                ctx.fillText(formatLoss(val), plotL - 6, y);
            }

            // X grid + labels
            ctx.textAlign = "center";
            ctx.textBaseline = "top";
            for (let i = 0; i <= xTicks; i++) {
                const frac = i / xTicks;
                const x = plotL + frac * plotW;
                const val = xMin + frac * (xMax - xMin);
                ctx.beginPath();
                ctx.moveTo(x, plotT);
                ctx.lineTo(x, plotB);
                ctx.stroke();
                ctx.fillStyle = AXIS_COLOR;
                ctx.fillText(Math.round(val).toString(), x, plotB + 4);
            }

            // ── loss curve ──
            if (hasData) {
                // Glow layer
                ctx.save();
                ctx.beginPath();
                ctx.strokeStyle = GLOW_COLOR;
                ctx.lineWidth = 6;
                ctx.lineJoin = "round";
                plotLine(ctx, data, xMin, xMax, yMin, yMax, plotL, plotT, plotW, plotH);
                ctx.stroke();
                ctx.restore();

                // Main line
                ctx.beginPath();
                ctx.strokeStyle = LINE_COLOR;
                ctx.lineWidth = 2;
                ctx.lineJoin = "round";
                plotLine(ctx, data, xMin, xMax, yMin, yMax, plotL, plotT, plotW, plotH);
                ctx.stroke();

                // Latest-point dot
                const lastPt = data[data.length - 1];
                const dotX = plotL + ((lastPt.step - xMin) / (xMax - xMin)) * plotW;
                const dotY = plotB - ((lastPt.loss - yMin) / (yMax - yMin)) * plotH;
                ctx.beginPath();
                ctx.arc(dotX, dotY, 4, 0, Math.PI * 2);
                ctx.fillStyle = LINE_COLOR;
                ctx.fill();
                ctx.beginPath();
                ctx.arc(dotX, dotY, 7, 0, Math.PI * 2);
                ctx.strokeStyle = GLOW_COLOR;
                ctx.lineWidth = 2;
                ctx.stroke();
            }

            // ── empty-state message ──
            if (!hasData && !this._running) {
                ctx.fillStyle = AXIS_COLOR;
                ctx.font = "13px 'Segoe UI', sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(
                    this._finished ? "Training complete" : "Waiting for training\u2026",
                    plotL + plotW / 2,
                    plotT + plotH / 2,
                );
            }

            // ── stats overlay bar ──
            const statsY = h - STAT_H;
            ctx.fillStyle = "rgba(22, 27, 34, 0.85)";
            ctx.beginPath();
            roundRect(ctx, 4, statsY, w - 8, STAT_H - 4, 4);
            ctx.fill();

            // Divider line
            ctx.strokeStyle = GRID_COLOR;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(8, statsY);
            ctx.lineTo(w - 8, statsY);
            ctx.stroke();

            // Stats text
            const col1 = 14;
            const col2 = w * 0.5;

            ctx.textAlign = "left";
            ctx.textBaseline = "top";

            // Row 1
            ctx.font = TITLE_FONT;
            ctx.fillStyle = AXIS_COLOR;
            ctx.fillText("Step", col1, statsY + 6);
            ctx.fillStyle = ACCENT_COLOR;
            ctx.font = STAT_FONT;
            const stepStr = this._totalSteps
                ? `${this._curStep} / ${this._totalSteps}`
                : `${this._curStep}`;
            ctx.fillText(stepStr, col1 + 40, statsY + 5);

            ctx.font = TITLE_FONT;
            ctx.fillStyle = AXIS_COLOR;
            ctx.fillText("Loss", col2, statsY + 6);
            ctx.fillStyle = TEXT_COLOR;
            ctx.font = STAT_FONT;
            ctx.fillText(
                hasData ? this._curLoss.toFixed(6) : "—",
                col2 + 38, statsY + 5,
            );

            // Row 2
            ctx.font = TITLE_FONT;
            ctx.fillStyle = AXIS_COLOR;
            ctx.fillText("LR", col1, statsY + 26);
            ctx.fillStyle = TEXT_COLOR;
            ctx.font = STAT_FONT;
            ctx.fillText(
                this._lr ? this._lr.toExponential(2) : "—",
                col1 + 40, statsY + 25,
            );

            ctx.font = TITLE_FONT;
            ctx.fillStyle = AXIS_COLOR;
            ctx.fillText("ETA", col2, statsY + 26);
            ctx.fillStyle = TEXT_COLOR;
            ctx.font = STAT_FONT;
            ctx.fillText(this._eta || "—", col2 + 38, statsY + 25);

            // Status pill
            if (this._running || this._finished) {
                const pill = this._running ? "TRAINING" : "DONE";
                const pillColor = this._running ? LINE_COLOR : "#58a6ff";
                const pillW = ctx.measureText(pill).width + 16;
                const pillX = w - pillW - 10;
                const pillY = statsY + 12;
                ctx.fillStyle = this._running
                    ? "rgba(57,255,20,0.12)"
                    : "rgba(88,166,255,0.12)";
                ctx.beginPath();
                roundRect(ctx, pillX, pillY, pillW, 20, 4);
                ctx.fill();
                ctx.fillStyle = pillColor;
                ctx.font = "bold 11px 'Segoe UI', Consolas, monospace";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(pill, pillX + pillW / 2, pillY + 10);
            }

            ctx.restore();
        };
    },
});

// ── helpers ────────────────────────────────────────────────────────
function plotLine(ctx, data, xMin, xMax, yMin, yMax, plotL, plotT, plotW, plotH) {
    const plotB = plotT + plotH;
    for (let i = 0; i < data.length; i++) {
        const px = plotL + ((data[i].step - xMin) / (xMax - xMin)) * plotW;
        const py = plotB - ((data[i].loss - yMin) / (yMax - yMin)) * plotH;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    }
}

function roundRect(ctx, x, y, w, h, r) {
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.arcTo(x + w, y, x + w, y + r, r);
    ctx.lineTo(x + w, y + h - r);
    ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
    ctx.lineTo(x + r, y + h);
    ctx.arcTo(x, y + h, x, y + h - r, r);
    ctx.lineTo(x, y + r);
    ctx.arcTo(x, y, x + r, y, r);
    ctx.closePath();
}

function niceTickCount(px, minGap) {
    return Math.max(2, Math.min(10, Math.floor(px / minGap)));
}

function formatLoss(v) {
    if (v >= 1)   return v.toFixed(2);
    if (v >= 0.01) return v.toFixed(3);
    return v.toFixed(4);
}
