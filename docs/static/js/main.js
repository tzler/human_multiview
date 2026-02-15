/* ============================================================
   Project Page: Interactive elements
   ============================================================ */

// ---------- Oddity demo ----------

let demoTrials = [];
let currentTrialIdx = 0;
let currentDisplayOrder = [];

async function initOddityDemo() {
    const grid = document.getElementById('demo-trial');
    if (!grid) return;

    // Load trial manifest (cache-bust to ensure fresh data)
    try {
        const resp = await fetch('static/data/demo_trials/manifest.json?v=' + Date.now());
        demoTrials = await resp.json();
        console.log('Demo trials loaded:', demoTrials.map(t => `${t.trial}: oddity=${t.oddity_index}`));
    } catch (e) {
        console.warn('Could not load demo trials', e);
        return;
    }

    // Set up condition tab handlers
    const tabs = document.querySelectorAll('.condition-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const idx = parseInt(tab.dataset.trial);
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            loadTrial(idx);
        });
    });

    // Load first trial
    loadTrial(0);
}

function loadTrial(idx) {
    currentTrialIdx = idx;
    const trial = demoTrials[idx];
    if (!trial) return;

    const grid = document.getElementById('demo-trial');
    const feedback = document.getElementById('demo-feedback');
    feedback.classList.add('hidden');

    // Build shuffled display order
    const order = Array.from({ length: trial.n_images }, (_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
    }
    currentDisplayOrder = order;

    // Generate cards
    grid.innerHTML = '';
    order.forEach((imgIdx, pos) => {
        const card = document.createElement('div');
        card.className = 'oddity-card';
        card.dataset.choice = pos;

        const img = document.createElement('img');
        img.src = `static/data/demo_trials/${trial.trial}/img_${imgIdx}.png?v=2`;
        img.alt = `Object ${pos + 1}`;
        card.appendChild(img);

        card.addEventListener('click', () => handleOddityChoice(card, grid));
        grid.appendChild(card);
    });
}

function handleOddityChoice(card, grid) {
    const trial = demoTrials[currentTrialIdx];
    const choicePos = parseInt(card.dataset.choice);
    const chosenImgIdx = currentDisplayOrder[choicePos];
    const isCorrect = chosenImgIdx === trial.oddity_index;
    console.log(`Choice: pos=${choicePos}, imgIdx=${chosenImgIdx}, oddity=${trial.oddity_index}, correct=${isCorrect}, order=${currentDisplayOrder}`);

    const allCards = grid.querySelectorAll('.oddity-card');
    allCards.forEach(c => {
        c.style.pointerEvents = 'none';
        const imgIdx = currentDisplayOrder[parseInt(c.dataset.choice)];
        if (imgIdx === trial.oddity_index) {
            c.classList.add(c === card ? 'correct' : 'revealed');
        } else if (c === card && !isCorrect) {
            c.classList.add('incorrect');
        }
    });

    const feedback = document.getElementById('demo-feedback');
    const result = document.getElementById('demo-result');
    const stats = document.getElementById('demo-stats');

    feedback.classList.remove('hidden');
    result.textContent = isCorrect
        ? 'Correct! You identified the different object.'
        : 'Not quite\u2014the highlighted image is the odd one out.';
    stats.textContent = `On this trial, human participants were ${Math.round(trial.human_accuracy * 100)}% accurate.`;
}


// ---------- Results charts (Canvas-based, no dependencies) ----------

function drawBarChart(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    const dpr = window.devicePixelRatio || 1;

    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    ctx.scale(dpr, dpr);

    const pad = { top: 30, right: 20, bottom: 50, left: 55 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    // Axes
    ctx.strokeStyle = '#ccc';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.stroke();

    // Y-axis ticks and grid
    ctx.fillStyle = '#888';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'right';
    for (let v = 0; v <= 1; v += 0.2) {
        const y = pad.top + plotH - (v * plotH);
        ctx.fillText((v * 100).toFixed(0) + '%', pad.left - 8, y + 4);
        if (v > 0) {
            ctx.strokeStyle = '#eee';
            ctx.beginPath();
            ctx.moveTo(pad.left, y);
            ctx.lineTo(pad.left + plotW, y);
            ctx.stroke();
        }
    }

    // Y-axis label
    ctx.save();
    ctx.translate(14, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillStyle = '#666';
    ctx.font = '12px Inter, sans-serif';
    ctx.fillText('Normalized accuracy', 0, 0);
    ctx.restore();

    // Chance line
    ctx.strokeStyle = '#ccc';
    ctx.setLineDash([4, 4]);
    const chanceY = pad.top + plotH - (0.0 * plotH);
    ctx.beginPath();
    ctx.moveTo(pad.left, chanceY);
    ctx.lineTo(pad.left + plotW, chanceY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Bars
    const barW = plotW / data.length * 0.55;
    const gap = plotW / data.length;

    data.forEach((d, i) => {
        const x = pad.left + gap * i + (gap - barW) / 2;
        const barH = d.value * plotH;
        const y = pad.top + plotH - barH;

        ctx.fillStyle = d.color;
        ctx.beginPath();
        roundRect(ctx, x, y, barW, barH, 3);
        ctx.fill();

        // Error bars
        if (d.sem) {
            const semPx = d.sem * plotH;
            ctx.strokeStyle = '#555';
            ctx.lineWidth = 1.5;
            const cx = x + barW / 2;
            ctx.beginPath();
            ctx.moveTo(cx, y - semPx);
            ctx.lineTo(cx, y + semPx);
            ctx.moveTo(cx - 4, y - semPx);
            ctx.lineTo(cx + 4, y - semPx);
            ctx.moveTo(cx - 4, y + semPx);
            ctx.lineTo(cx + 4, y + semPx);
            ctx.stroke();
        }

        // Label
        ctx.fillStyle = '#333';
        ctx.textAlign = 'center';
        ctx.font = '11px Inter, sans-serif';
        ctx.fillText(d.label, x + barW / 2, pad.top + plotH + 18);

        // Value on top
        ctx.fillStyle = '#666';
        ctx.font = '10px Inter, sans-serif';
        ctx.fillText(Math.round(d.value * 100) + '%', x + barW / 2, y - 8);
    });
}

function drawScatterChart(canvasId, data, options) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    const dpr = window.devicePixelRatio || 1;

    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    ctx.scale(dpr, dpr);

    const pad = { top: 20, right: 20, bottom: 50, left: 55 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    const xMin = options.xMin ?? Math.min(...data.map(d => d.x));
    const xMax = options.xMax ?? Math.max(...data.map(d => d.x));
    const yMin = options.yMin ?? 0.55;
    const yMax = options.yMax ?? 0.95;

    function toX(v) { return pad.left + ((v - xMin) / (xMax - xMin)) * plotW; }
    function toY(v) { return pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH; }

    // Grid
    ctx.strokeStyle = '#eee';
    ctx.lineWidth = 1;
    const gridStep = options.yFormat === 'ms' ? (yMax - yMin > 2000 ? 500 : 200) : 0.05;
    for (let v = yMin; v <= yMax; v += gridStep) {
        ctx.beginPath();
        ctx.moveTo(pad.left, toY(v));
        ctx.lineTo(pad.left + plotW, toY(v));
        ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = '#ccc';
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = '#666';
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(options.xLabel || '', pad.left + plotW / 2, H - 6);

    ctx.save();
    ctx.translate(14, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(options.yLabel || '', 0, 0);
    ctx.restore();

    // Y-axis ticks
    ctx.textAlign = 'right';
    ctx.font = '10px Inter, sans-serif';
    ctx.fillStyle = '#888';
    const yRange = yMax - yMin;
    const yTickStep = options.yFormat === 'ms' ? (yRange > 2000 ? 500 : 200) : 0.1;
    for (let v = yMin; v <= yMax + yTickStep * 0.01; v += yTickStep) {
        const label = options.yFormat === 'ms' ? Math.round(v) : (v * 100).toFixed(0) + '%';
        ctx.fillText(label, pad.left - 6, toY(v) + 3);
    }

    // X-axis ticks
    ctx.textAlign = 'center';
    const xStep = (xMax - xMin) / 5;
    for (let v = xMin; v <= xMax; v += xStep) {
        const label = options.xIsLayer ? Math.round(v) : v.toFixed(0);
        ctx.fillText(label, toX(v), pad.top + plotH + 16);
    }

    // Regression line
    if (data.length > 2) {
        const n = data.length;
        const sx = data.reduce((s, d) => s + d.x, 0);
        const sy = data.reduce((s, d) => s + d.y, 0);
        const sxy = data.reduce((s, d) => s + d.x * d.y, 0);
        const sxx = data.reduce((s, d) => s + d.x * d.x, 0);
        const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        const intercept = (sy - slope * sx) / n;

        ctx.strokeStyle = 'rgba(192, 48, 48, 0.4)';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(toX(xMin), toY(slope * xMin + intercept));
        ctx.lineTo(toX(xMax), toY(slope * xMax + intercept));
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Points
    data.forEach(d => {
        const x = toX(d.x);
        const y = toY(d.y);

        // Error bars
        if (d.yerr) {
            ctx.strokeStyle = 'rgba(192, 48, 48, 0.3)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x, toY(d.y - d.yerr));
            ctx.lineTo(x, toY(d.y + d.yerr));
            ctx.stroke();
        }

        ctx.fillStyle = 'rgba(192, 48, 48, 0.7)';
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = 'rgba(192, 48, 48, 0.9)';
        ctx.lineWidth = 1;
        ctx.stroke();
    });
}

function roundRect(ctx, x, y, w, h, r) {
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
}


// ---------- Initialize charts ----------

async function initCharts() {
    // Load real data from JSON
    let chartData;
    try {
        const resp = await fetch('static/data/chart_data.json');
        chartData = await resp.json();
    } catch (e) {
        console.warn('Could not load chart data, using defaults', e);
        chartData = null;
    }

    // Accuracy bar chart (Fig 3 left)
    const stats = chartData?.accuracy_stats || { human: 0.777, human_sem: 0.035, vggt: 0.790, vggt_sem: 0.040, dinov2: 0.261, dinov2_sem: 0.056 };
    drawBarChart('chart-accuracy', [
        { label: 'Humans', value: stats.human, sem: stats.human_sem, color: '#e8a838' },
        { label: 'VGGT', value: stats.vggt, sem: stats.vggt_sem, color: '#c03030' },
        { label: 'DINOv2', value: stats.dinov2, sem: stats.dinov2_sem, color: '#888' },
    ]);

    // Confidence vs. human accuracy scatter (Fig 3 middle)
    const confidenceData = chartData?.confidence_vs_human_accuracy || [];
    drawScatterChart('chart-confidence', confidenceData, {
        xLabel: 'Model confidence (low \u2192 high)',
        yLabel: 'Human accuracy',
        xMin: 0, xMax: 30,
        yMin: 0.60, yMax: 0.95,
    });

    // Solution layer vs. human RT scatter (Fig 3 right)
    // Convert RT from ms to ms scale for the chart
    const rtRaw = chartData?.solution_layer_vs_human_rt || [];
    drawScatterChart('chart-rt', rtRaw, {
        xLabel: 'Model solution layer',
        yLabel: 'Human RT (ms)',
        xMin: 0, xMax: 24,
        yMin: 2800, yMax: 4500,
        xIsLayer: true,
        yFormat: 'ms',
    });
}


// ---------- Attention visualization ----------

// Plasma colormap (64 entries, sampled from matplotlib)
const PLASMA = [
    [13,8,135],[27,6,141],[40,5,147],[51,5,151],[62,5,155],[72,5,157],
    [82,6,158],[91,10,159],[99,15,159],[107,21,159],[115,27,158],[122,33,157],
    [129,39,155],[135,45,153],[141,51,151],[147,57,148],[153,63,145],
    [158,69,142],[163,75,138],[168,81,135],[173,87,131],[178,93,127],
    [182,99,123],[187,105,119],[191,111,115],[195,117,110],[199,123,106],
    [203,129,102],[206,135,97],[210,141,93],[213,147,88],[216,153,84],
    [219,159,79],[222,165,75],[224,171,70],[227,177,66],[229,183,61],
    [231,189,57],[233,195,52],[234,201,48],[236,207,44],[237,213,40],
    [237,219,37],[237,225,34],[237,231,32],[236,237,31],[235,243,31],
    [240,249,33]
];

const POINT_COLORS = ['#e63946', '#457b9d', '#2a9d8f'];

let attnManifest = [];
let attnData = {};       // { trialId: Uint8Array }
let attnImages = {};     // { trialId: { A: Image, Aprime: Image, B: Image } }
let attnMasks = {};      // { trialId: { Aprime: Uint8Array, B: Uint8Array } }
let attnCurrentTrial = 0;
let attnCurrentLayer = 12;
let attnMaskEnabled = false;

async function initAttentionViz() {
    const select = document.getElementById('attn-trial-select');
    if (!select) return;

    // Load manifest
    try {
        const resp = await fetch('static/data/attention/manifest.json');
        attnManifest = await resp.json();
    } catch (e) {
        console.warn('Could not load attention manifest', e);
        return;
    }

    // Populate trial selector
    attnManifest.forEach((t, i) => {
        const opt = document.createElement('option');
        opt.value = i;
        opt.textContent = `${t.trial_name} (${t.condition})`;
        select.appendChild(opt);
    });

    select.addEventListener('change', () => {
        attnCurrentTrial = parseInt(select.value);
        loadAttnTrial(attnCurrentTrial);
    });

    // Layer slider
    const slider = document.getElementById('layer-slider');
    const layerLabel = document.getElementById('layer-label');
    slider.value = 12;
    layerLabel.textContent = '12';

    slider.addEventListener('input', () => {
        attnCurrentLayer = parseInt(slider.value);
        layerLabel.textContent = slider.value;
        renderAttnHeatmaps();
    });

    // Mask toggle
    const maskToggle = document.getElementById('attn-mask-toggle');
    if (maskToggle) {
        maskToggle.addEventListener('change', () => {
            attnMaskEnabled = maskToggle.checked;
            renderAttnHeatmaps();
        });
    }

    // Load first trial
    await loadAttnTrial(0);
}

async function loadAttnTrial(idx) {
    const trial = attnManifest[idx];
    if (!trial) return;

    // Load binary heatmap data if not cached
    if (!attnData[idx]) {
        const resp = await fetch(`static/data/attention/trial_${trial.id}/heatmaps.bin`);
        attnData[idx] = new Uint8Array(await resp.arrayBuffer());
    }

    // Load images if not cached
    if (!attnImages[idx]) {
        attnImages[idx] = {};
        const names = { A: 'img_A.png', Aprime: 'img_A_prime.png', B: 'img_B.png' };
        const promises = Object.entries(names).map(([key, fname]) => {
            return new Promise(resolve => {
                const img = new Image();
                img.onload = () => { attnImages[idx][key] = img; resolve(); };
                img.onerror = () => resolve();
                img.src = `static/data/attention/trial_${trial.id}/${fname}`;
            });
        });
        await Promise.all(promises);
    }

    // Compute object masks if not cached
    if (!attnMasks[idx]) {
        try {
            attnMasks[idx] = {
                Aprime: computeObjectMask(attnImages[idx]?.Aprime),
                B: computeObjectMask(attnImages[idx]?.B),
            };
        } catch (e) {
            console.warn('Could not compute object masks', e);
            attnMasks[idx] = { Aprime: null, B: null };
        }
    }

    // Build heatmap column DOM (once per trial change)
    buildAttnColumns(trial);

    // Draw source image with query points
    drawSourceImage(trial);

    // Draw reference images (A' and B)
    drawRefImage('attn-ref-match', attnImages[idx]?.Aprime);
    drawRefImage('attn-ref-nonmatch', attnImages[idx]?.B);

    // Render heatmaps at current layer
    renderAttnHeatmaps();
}

function buildAttnColumns(trial) {
    const container = document.getElementById('attn-heatmap-cols');
    container.innerHTML = '';

    for (let p = 0; p < trial.n_points; p++) {
        const col = document.createElement('div');
        col.className = 'attn-point-col';

        const header = document.createElement('div');
        header.className = 'attn-point-header';
        header.innerHTML = `<span style="color:${POINT_COLORS[p]}">&#9679;</span> Point ${p + 1}`;
        col.appendChild(header);

        // Match canvas (A')
        const matchCanvas = document.createElement('canvas');
        matchCanvas.id = `attn-match-${p}`;
        matchCanvas.className = 'attn-heatmap-canvas';
        col.appendChild(matchCanvas);

        // Nonmatch canvas (B)
        const nonmatchCanvas = document.createElement('canvas');
        nonmatchCanvas.id = `attn-nonmatch-${p}`;
        nonmatchCanvas.className = 'attn-heatmap-canvas';
        col.appendChild(nonmatchCanvas);

        container.appendChild(col);
    }
}

function drawSourceImage(trial) {
    const canvas = document.getElementById('attn-source');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const displayW = 180, displayH = 180;

    canvas.width = displayW * dpr;
    canvas.height = displayH * dpr;
    canvas.style.width = displayW + 'px';
    canvas.style.height = displayH + 'px';
    ctx.scale(dpr, dpr);

    const img = attnImages[attnCurrentTrial]?.A;
    if (img) {
        ctx.drawImage(img, 0, 0, displayW, displayH);
    }

    // Draw query points using patch indices (ground truth positions)
    const pH = trial.patch_h, pW = trial.patch_w;
    trial.patch_indices.forEach((patchIdx, i) => {
        const row = Math.floor(patchIdx / pW);
        const col = patchIdx % pW;
        const x = ((col + 0.5) / pW) * displayW;
        const y = ((row + 0.5) / pH) * displayH;

        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fillStyle = POINT_COLORS[i];
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
    });
}

function drawRefImage(canvasId, img) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !img) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const displayW = 180, displayH = 180;

    canvas.width = displayW * dpr;
    canvas.height = displayH * dpr;
    canvas.style.width = displayW + 'px';
    canvas.style.height = displayH + 'px';
    ctx.scale(dpr, dpr);
    ctx.drawImage(img, 0, 0, displayW, displayH);
}

// Morphological operations for mask cleanup (matches manuscript's scipy approach)
function morphErode(mask, w, h) {
    const out = new Uint8Array(w * h);
    for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
            // 3x3 structuring element: all neighbors must be 1
            out[y * w + x] = (
                mask[(y-1)*w+x-1] && mask[(y-1)*w+x] && mask[(y-1)*w+x+1] &&
                mask[y*w+x-1]     && mask[y*w+x]     && mask[y*w+x+1] &&
                mask[(y+1)*w+x-1] && mask[(y+1)*w+x] && mask[(y+1)*w+x+1]
            ) ? 1 : 0;
        }
    }
    return out;
}

function morphDilate(mask, w, h) {
    const out = new Uint8Array(w * h);
    for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
            // 3x3 structuring element: any neighbor is 1
            out[y * w + x] = (
                mask[(y-1)*w+x-1] || mask[(y-1)*w+x] || mask[(y-1)*w+x+1] ||
                mask[y*w+x-1]     || mask[y*w+x]     || mask[y*w+x+1] ||
                mask[(y+1)*w+x-1] || mask[(y+1)*w+x] || mask[(y+1)*w+x+1]
            ) ? 1 : 0;
        }
    }
    return out;
}

function morphOpen(mask, w, h, iterations) {
    let m = mask;
    for (let i = 0; i < iterations; i++) m = morphErode(m, w, h);
    for (let i = 0; i < iterations; i++) m = morphDilate(m, w, h);
    return m;
}

function morphClose(mask, w, h, iterations) {
    let m = mask;
    for (let i = 0; i < iterations; i++) m = morphDilate(m, w, h);
    for (let i = 0; i < iterations; i++) m = morphErode(m, w, h);
    return m;
}

function computeObjectMask(img) {
    // Pixel-level mask with morphological cleanup (matches manuscript's approach:
    //   binary_opening(iterations=2) then binary_closing(iterations=3))
    if (!img) return null;
    try {
        const w = img.naturalWidth || img.width;
        const h = img.naturalHeight || img.height;
        if (!w || !h) return null;
        const maskSize = 180;
        const c = document.createElement('canvas');
        c.width = maskSize;
        c.height = maskSize;
        const ctx = c.getContext('2d');
        ctx.drawImage(img, 0, 0, maskSize, maskSize);
        const imgData = ctx.getImageData(0, 0, maskSize, maskSize);
        const px = imgData.data;

        // Build binary mask from threshold
        let mask = new Uint8Array(maskSize * maskSize);
        for (let i = 0; i < mask.length; i++) {
            const gray = (px[i*4] + px[i*4+1] + px[i*4+2]) / 3;
            mask[i] = gray < 230 ? 1 : 0;
        }

        // Morphological opening (removes noise) then closing (fills holes)
        mask = morphOpen(mask, maskSize, maskSize, 2);
        mask = morphClose(mask, maskSize, maskSize, 3);

        // Write back to image data as alpha channel
        for (let i = 0; i < mask.length; i++) {
            px[i*4 + 3] = mask[i] ? 255 : 0;
        }

        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = maskSize;
        maskCanvas.height = maskSize;
        const maskCtx = maskCanvas.getContext('2d');
        maskCtx.putImageData(imgData, 0, 0);
        return maskCanvas;
    } catch (e) {
        console.warn('computeObjectMask failed:', e);
        return null;
    }
}

function renderAttnHeatmaps() {
    const trial = attnManifest[attnCurrentTrial];
    const data = attnData[attnCurrentTrial];
    if (!trial || !data) return;

    const pH = trial.patch_h, pW = trial.patch_w;
    const patchCount = pH * pW;  // 1369
    const nLayers = trial.n_layers;  // 24

    for (let p = 0; p < trial.n_points; p++) {
        // Get both heatmaps for per-point normalization (matching manuscript)
        const matchOffset = (p * nLayers * 2 + attnCurrentLayer * 2 + 0) * patchCount;
        const nonmatchOffset = (p * nLayers * 2 + attnCurrentLayer * 2 + 1) * patchCount;
        const matchHeatmap = data.slice(matchOffset, matchOffset + patchCount);
        const nonmatchHeatmap = data.slice(nonmatchOffset, nonmatchOffset + patchCount);

        // Per-point normalization: find max across both targets
        let pointMax = 0;
        for (let i = 0; i < patchCount; i++) {
            if (matchHeatmap[i] > pointMax) pointMax = matchHeatmap[i];
            if (nonmatchHeatmap[i] > pointMax) pointMax = nonmatchHeatmap[i];
        }
        if (pointMax === 0) pointMax = 1;

        for (let t = 0; t < 2; t++) {
            const canvasId = t === 0 ? `attn-match-${p}` : `attn-nonmatch-${p}`;
            const canvas = document.getElementById(canvasId);
            if (!canvas) continue;

            const heatmap = t === 0 ? matchHeatmap : nonmatchHeatmap;
            const bgImg = t === 0
                ? attnImages[attnCurrentTrial]?.Aprime
                : attnImages[attnCurrentTrial]?.B;
            const mask = attnMaskEnabled
                ? (t === 0 ? attnMasks[attnCurrentTrial]?.Aprime : attnMasks[attnCurrentTrial]?.B)
                : null;

            drawHeatmapOnCanvas(canvas, heatmap, pH, pW, bgImg, pointMax, mask);
        }
    }
}

function drawHeatmapOnCanvas(canvas, heatmapData, pH, pW, bgImg, pointMax, mask) {
    const displayW = 180, displayH = 180;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = displayW * dpr;
    canvas.height = displayH * dpr;
    canvas.style.width = displayW + 'px';
    canvas.style.height = displayH + 'px';

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    // Draw background image dimly (manuscript uses alpha=0.2)
    if (bgImg) {
        ctx.globalAlpha = 0.2;
        ctx.drawImage(bgImg, 0, 0, displayW, displayH);
        ctx.globalAlpha = 1.0;
    }

    // Build plasma-colormapped heatmap at 37x37 (full alpha), then bilinear upscale
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = pW;
    tmpCanvas.height = pH;
    const tmpCtx = tmpCanvas.getContext('2d');
    const tmpImg = tmpCtx.createImageData(pW, pH);
    const tmpPx = tmpImg.data;

    for (let py = 0; py < pH; py++) {
        for (let px = 0; px < pW; px++) {
            const patchIdx = py * pW + px;
            const rawVal = heatmapData[patchIdx];
            const norm = Math.min(rawVal / pointMax, 1.0);
            const boosted = Math.pow(norm, 0.7);
            const val = (boosted * 255) | 0;
            const ci = Math.min((val * (PLASMA.length - 1) / 255) | 0, PLASMA.length - 1);
            const idx = patchIdx << 2;
            tmpPx[idx]     = PLASMA[ci][0];
            tmpPx[idx + 1] = PLASMA[ci][1];
            tmpPx[idx + 2] = PLASMA[ci][2];
            tmpPx[idx + 3] = 255;
        }
    }
    tmpCtx.putImageData(tmpImg, 0, 0);

    // Bilinear upscale from 37x37 to display resolution
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.globalAlpha = 0.85;
    ctx.drawImage(tmpCanvas, 0, 0, displayW, displayH);
    ctx.globalAlpha = 1.0;

    // Apply pixel-level mask: erase background pixels using destination-in compositing
    if (mask) {
        ctx.globalCompositeOperation = 'destination-in';
        ctx.drawImage(mask, 0, 0, displayW, displayH);
        ctx.globalCompositeOperation = 'source-over';
    }
}


// ---------- Lazy-load videos (play when scrolled into view) ----------

function initLazyVideos() {
    const videos = document.querySelectorAll('video.lazy-video');
    if (!videos.length) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Start loading and play
                video.preload = 'auto';
                video.load();
                video.play().catch(() => {});
            } else {
                video.pause();
            }
        });
    }, { threshold: 0.25 });

    videos.forEach(v => observer.observe(v));
}


// ---------- Boot ----------

document.addEventListener('DOMContentLoaded', () => {
    initLazyVideos();
    initOddityDemo();
    initCharts();
    initAttentionViz();
    if (typeof initCameraViz === 'function') initCameraViz();
});
