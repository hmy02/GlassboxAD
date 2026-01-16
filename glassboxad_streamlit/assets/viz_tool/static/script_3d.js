import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

console.log("HYDRA-AD script starting to load...");

let data_name = "";

let scene, camera, renderer, controls;
let nodes = [];
let nodeMeshesById = new Map();
let repLines = [];
let layerEdgeLines = [];
let pathLines = [];
let ghostPoints = [];
let ghostLines = [];
let nnLines = [];
let selectedNode = null;
let data = null;
let raycaster, mouse;
let currentZoomRange = null;
let isZooming = false;

// Score layer selection: "ens" or numeric layer index
let scoreLayerMode = "ens";


// Layout constants
const LAYER_SPACING = 3.0;

async function hydraStartApp() {
    console.log("hydraStartApp called - fetching data...");
    const response = await fetch('/api/data');
    data = await response.json();
    data_name = data.data_name;
    // Populate score-layer dropdown
    initScoreLayerSelector(data);

    document.getElementById('landing-page').style.display = 'none';
    document.getElementById('container').style.display = 'flex';
    document.getElementById('data-name-display').innerText = `Dataset: ${data_name}`;

    if (!renderer) {
        init();
    } else {
        // Reset scene for new data
        scene.clear();
        nodes = [];
        nodeMeshesById.clear();
        repLines = [];
        layerEdgeLines = [];
        pathLines = [];
        ghostPoints = [];
        ghostLines = [];
        nnLines = [];
        selectedNode = null;

        // Re-run setup
        const numLevels = data.levels.length;
        const centerY = (numLevels - 1) * LAYER_SPACING / 2;
        camera.position.set(20, centerY + 10, 20);
        camera.lookAt(0, centerY, 0);
        controls.target.set(0, centerY, 0);
        controls.update();

        renderHierarchy(data);
        renderTimeSeries(data);
        renderAnomalyScore(data);
        initHistogram(data);
        initScoreLayerSelector(data);
        onWindowResize();
    }
    if (window.hideLoading) window.hideLoading();
}

function resetZoom() {
    currentZoomRange = null;
    document.getElementById('reset-zoom-btn').style.display = 'none';
    renderTimeSeries(data, selectedNode ? selectedNode.global_idx : null);
    renderAnomalyScore(data, selectedNode ? selectedNode.global_idx : null);
}

function initScoreLayerSelector(data) {
    const sel = document.getElementById('score-layer-select');
    if (!sel) return;

    // Determine number of layers from hierarchy levels (preferred) or ts_scores
    const nLevels = (data && data.levels && Array.isArray(data.levels)) ? data.levels.length :
                    (data && data.ts_scores && Array.isArray(data.ts_scores)) ? data.ts_scores.length : 0;
    console.log("initScoreLayerSelector called", data?.levels?.length, data?.ts_scores?.length);
    // Rebuild options
    sel.innerHTML = "";

    const optEns = document.createElement('option');
    optEns.value = "ens";
    optEns.textContent = "Ensemble";
    sel.appendChild(optEns);

    for (let i = 0; i < nLevels; i++) {
        const opt = document.createElement('option');
        opt.value = String(i);
        opt.textContent = `Layer L${i}`;
        sel.appendChild(opt);
    }

    // Default selection
    sel.value = "ens";
    scoreLayerMode = "ens";

    sel.onchange = () => {
        const v = sel.value;
        scoreLayerMode = (v === "ens") ? "ens" : parseInt(v, 10);
        renderAnomalyScore(data, selectedNode ? selectedNode.global_idx : null);
    };
}


function init() {
    if (!data) return;

    // Setup Three.js scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);


    const container = document.getElementById('hierarchy-chart');
    let width = container.clientWidth;
    let height = container.clientHeight;

    if (width === 0 || height === 0) {
        width = window.innerWidth * 0.7;
        height = window.innerHeight * 0.6;
    }

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    // Initialize camera with isometric position
    const numLevels = data.levels.length;
    const centerY = (numLevels - 1) * LAYER_SPACING / 2;
    camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 1, 1000);
    camera.position.set(20, centerY + 10, 20);
    camera.lookAt(0, centerY, 0);

    // Initialize OrbitControls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, centerY, 0);
    controls.enableRotate = true; // Allow rotation in 3D
    controls.enableZoom = true;
    controls.enablePan = true;
    controls.update();

    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    console.log("Initializing hierarchy rendering...");
    try {
        renderHierarchy(data);
    } catch (e) {
        console.error("Error rendering hierarchy:", e);
    }

    // Render Time Series (2D D3)
    renderTimeSeries(data);
    renderAnomalyScore(data);

    // Initialize Histogram
    initHistogram(data);
    initScoreLayerSelector(data);

    // Force resize to ensure correct frustum
    onWindowResize();

    // Event listeners
    window.addEventListener('resize', onWindowResize, false);
    renderer.domElement.addEventListener('click', onMouseClick, false);
    renderer.domElement.addEventListener('mousemove', onMouseMove, false);

    animate();
}

let hoveredNode = null;
function onMouseMove(event) {
    if (!renderer) return;
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(nodes);

    if (intersects.length > 0) {
        renderer.domElement.style.cursor = 'pointer';
        const object = intersects[0].object;
        if (hoveredNode !== object) {
            if (hoveredNode) hoveredNode.material.opacity = 1.0;
            hoveredNode = object;
            hoveredNode.material.transparent = true;
            hoveredNode.material.opacity = 0.7;
        }
    } else {
        renderer.domElement.style.cursor = 'default';
        if (hoveredNode) {
            hoveredNode.material.opacity = 1.0;
            hoveredNode = null;
        }
    }
}

function createLevelLabel(text, y) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = 256;
    canvas.height = 128;

    // Draw background (optional, for visibility)
    context.fillStyle = 'rgba(0, 0, 0, 0.4)';
    context.roundRect(0, 0, 256, 128, 20);
    context.fill();

    context.fillStyle = 'white';
    context.font = 'bold 80px Arial';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText(text, 128, 64);

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(material);

    // Position it at the front-left corner of the grid
    sprite.position.set(-2.5, y, 2.5);
    sprite.scale.set(1.5, 0.75, 1);

    return sprite;
}

function renderHierarchy(data) {
    const numLevels = data.levels.length;

    // Create planes for each level
    data.levels.forEach((levelData, i) => {
        const y = i * LAYER_SPACING;

        // Grid helper
        const gridHelper = new THREE.GridHelper(4, 10, 0x888888, 0xeeeeee);
        gridHelper.position.y = y;
        scene.add(gridHelper);

        // Level Label (L0, L1, etc.)
        const label = createLevelLabel(`L${i}`, y);
        scene.add(label);

        // Render nodes
        const geometry = new THREE.SphereGeometry(0.15, 16, 16);

        levelData.nodes.forEach(node => {
            const maxScore = data.global_max_nn || 2.0;
            const t = Math.min(node.score / maxScore, 1.0);
            const color = new THREE.Color().setHSL(0.7 - t * 0.7, 1.0, 0.5);

            const material = new THREE.MeshBasicMaterial({ color: color });
            const sphere = new THREE.Mesh(geometry, material);

            sphere.position.set(node.x * 2, y, node.y * 2);
            sphere.userData = { id: node.id, node: node };

            scene.add(sphere);
            nodeMeshesById.set(node.id, sphere);
            nodes.push(sphere);
        });

        // Render Layer Edges
        if (data.layer_edges && data.layer_edges[i]) {
            const edges = data.layer_edges[i];
            const material = new THREE.LineBasicMaterial({ color: 0xcccccc, transparent: true, opacity: 0.3 });
            const points = [];

            edges.forEach(edge => {
                const sourceGlobal = edge[0];
                const targetGlobal = edge[1];
                const sourceNode = levelData.nodes.find(n => n.global_idx === sourceGlobal);
                const targetNode = levelData.nodes.find(n => n.global_idx === targetGlobal);

                if (sourceNode && targetNode) {
                    const p1 = new THREE.Vector3(sourceNode.x * 2, y, sourceNode.y * 2);
                    const p2 = new THREE.Vector3(targetNode.x * 2, y, targetNode.y * 2);
                    points.push(p1, p2);
                }
            });

            if (points.length > 0) {
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const line = new THREE.LineSegments(geometry, material);
                scene.add(line);
                layerEdgeLines.push(line);
            }
        }
    });

    // Render Rep Survival Lines
    const repMaterial = new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.1 });
    const repPoints = [];

    for (let i = 0; i < numLevels - 1; i++) {
        const currentLevel = data.levels[i];
        const nextLevel = data.levels[i + 1];

        currentLevel.nodes.forEach(node => {
            const nextNode = nextLevel.nodes.find(n => n.global_idx === node.global_idx);
            if (nextNode) {
                const p1 = new THREE.Vector3(node.x * 2, i * LAYER_SPACING, node.y * 2);
                const p2 = new THREE.Vector3(nextNode.x * 2, (i + 1) * LAYER_SPACING, nextNode.y * 2);
                repPoints.push(p1, p2);
            }
        });
    }

    if (repPoints.length > 0) {
        const geometry = new THREE.BufferGeometry().setFromPoints(repPoints);
        const line = new THREE.LineSegments(geometry, repMaterial);
        scene.add(line);
        repLines.push(line);
    }
}

function onMouseClick(event) {
    if (!renderer) return;
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    console.log("Mouse click at:", mouse.x, mouse.y);
    console.log("Checking intersection with", nodes.length, "nodes");

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(nodes);

    console.log("Intersects found:", intersects.length);

    if (intersects.length > 0) {
        const object = intersects[0].object;
        console.log("Selected node:", object.userData.node.id);
        try {
            selectNode(object.userData.node);
        } catch (e) {
            console.error("Critical error in selectNode:", e);
        }
    }
}

function selectNode(node) {
    selectedNode = node;
    clearOverlays();
    highlightPath(node);
    showSubsequence(node);
    showProjectionsAndNN(node);
    highlightTimeSeries(node);
    updateHistogram(node);
    renderAnomalyScore(data, node.global_idx);
}

function clearOverlays() {
    pathLines.forEach(l => scene.remove(l)); pathLines = [];
    ghostPoints.forEach(p => scene.remove(p)); ghostPoints = [];
    ghostLines.forEach(l => scene.remove(l)); ghostLines = [];
    nnLines.forEach(l => scene.remove(l)); nnLines = [];
}

function highlightPath(node) {
    let current = node;
    while (current.parent_id) {
        const parentMesh = nodeMeshesById.get(current.parent_id);
        if (parentMesh) {
            const parent = parentMesh.userData.node;
            const currentMesh = nodeMeshesById.get(current.id);

            const points = [currentMesh.position, parentMesh.position];
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const material = new THREE.LineBasicMaterial({ color: 0xff0000, linewidth: 2 });
            const line = new THREE.Line(geometry, material);
            scene.add(line);
            pathLines.push(line);

            current = parent;
        } else {
            break;
        }
    }
}

function showProjectionsAndNN(node) {
    const global_idx = node.global_idx;
    const verticalPoints = [];

    data.levels.forEach((levelData, i) => {
        const y = i * LAYER_SPACING;
        const x = node.x * 2;
        const z = node.y * 2;
        const ghostPos = new THREE.Vector3(x, y, z);
        verticalPoints.push(ghostPos);

        if (levelData.level !== node.level) {
            const geometry = new THREE.SphereGeometry(0.04, 16, 16);
            const material = new THREE.MeshBasicMaterial({ color: 0x00ff00, transparent: true, opacity: 0.5 });
            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.copy(ghostPos);
            scene.add(sphere);
            ghostPoints.push(sphere);
        }

        if (data.nn_indices && data.nn_indices[i]) {
            const nnGlobalIdx = data.nn_indices[i][global_idx];
            const nnNode = levelData.nodes.find(n => n.global_idx === nnGlobalIdx);

            if (nnNode) {
                const nnPos = new THREE.Vector3(nnNode.x * 2, y, nnNode.y * 2);
                const points = [ghostPos, nnPos];
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const material = new THREE.LineBasicMaterial({ color: 0x0000ff, linewidth: 2 });
                const line = new THREE.Line(geometry, material);
                scene.add(line);
                nnLines.push(line);
            }
        }
    });

    const geometry = new THREE.BufferGeometry().setFromPoints(verticalPoints);
    const material = new THREE.LineDashedMaterial({ color: 0x00ff00, dashSize: 0.1, gapSize: 0.05 });
    const line = new THREE.Line(geometry, material);
    line.computeLineDistances();
    scene.add(line);
    ghostLines.push(line);
}

// --- Histogram Logic ---

function initHistogram(data) {
    // Initial empty state or average
    updateHistogram(null);
}

function onWindowResize() {
    const container = document.getElementById('hierarchy-chart');
    // Use offsetWidth/Height to avoid subpixel issues or growing loops
    let width = container.offsetWidth;
    let height = container.offsetHeight;

    if (width === 0 || height === 0) {
        width = window.innerWidth * 0.7;
        height = window.innerHeight * 0.6;
    }

    renderer.setSize(width, height);

    const numLevels = data.levels.length;
    const dataWidth = 4.0; // Grid is 4x4
    const dataHeight = (numLevels - 1) * LAYER_SPACING + 2.0;

    const aspect = width / height;
    const centerY = (numLevels - 1) * LAYER_SPACING / 2;

    // Fit to View Logic
    // We want frustumHeight >= dataHeight AND frustumWidth >= dataWidth
    // frustumWidth = frustumHeight * aspect
    // So frustumHeight * aspect >= dataWidth => frustumHeight >= dataWidth / aspect

    let frustumHeight = Math.max(dataHeight, dataWidth / aspect);
    let frustumWidth = frustumHeight * aspect;

    camera.left = -frustumWidth / 2;
    camera.right = frustumWidth / 2;
    camera.top = frustumHeight / 2;
    camera.bottom = -frustumHeight / 2;

    // Position for Isometric view
    camera.position.set(20, centerY + 10, 20);
    camera.lookAt(0, centerY, 0);

    camera.updateProjectionMatrix();

    // Re-render other components
    renderTimeSeries(data, selectedNode ? selectedNode.global_idx : null);
    renderAnomalyScore(data, selectedNode ? selectedNode.global_idx : null);
    if (selectedNode) {
        updateHistogram(selectedNode);
        showSubsequence(selectedNode);
    }
}

function showSubsequence(node) {
    const subsequence = data.subsequences[node.global_idx];
    const container = document.getElementById('subsequence-chart'); // Use the chart div
    const width = container.offsetWidth - 20;
    const height = container.offsetHeight - 20;
    const margin = { top: 10, right: 30, bottom: 30, left: 30 };

    if (width <= 0 || height <= 0) return;

    d3.select("#subsequence-chart").html("");

    const svg = d3.select("#subsequence-chart")
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [0, 0, width, height]);

    const x = d3.scaleLinear()
        .domain([0, subsequence.length])
        .range([margin.left, width - margin.right]);

    const y = d3.scaleLinear()
        .domain([d3.min(subsequence), d3.max(subsequence)])
        .range([height - margin.bottom, margin.top]);

    const line = d3.line()
        .x((d, i) => x(i))
        .y(d => y(d));

    svg.append("path")
        .datum(subsequence)
        .attr("fill", "none")
        .attr("stroke", "red")
        .attr("stroke-width", 1.5)
        .attr("d", line);

    svg.append("g")
        .attr("transform", `translate(0,${height - margin.bottom})`)
        .call(d3.axisBottom(x).ticks(5));

    svg.append("g")
        .attr("transform", `translate(${margin.left},0)`)
        .call(d3.axisLeft(y).ticks(3)); // Fewer ticks for cleaner look
}

function highlightTimeSeries(node) {
    renderTimeSeries(data, node.global_idx);
}

function updateHistogram(node) {
    const container = document.getElementById('histogram-container');
    const width = container.offsetWidth;
    const height = container.offsetHeight - 40;

    if (width <= 0 || height <= 0) return;

    const margin = { top: 20, right: 20, bottom: 40, left: 60 };

    d3.select("#histogram-chart").html("");

    const svg = d3.select("#histogram-chart")
        .append("svg")
        .attr("width", width)
        .attr("height", height);

    // Map 3D Y to Screen Y
    const layerScreenPositions = [];
    data.levels.forEach((levelData, i) => {
        const y = i * LAYER_SPACING;
        const pos = new THREE.Vector3(0, y, 0);
        pos.project(camera);
        const screenY = margin.top + ((-pos.y + 1) / 2) * (height - margin.top - margin.bottom);
        layerScreenPositions.push(screenY);
    });

    // Data to plot
    let plotData = [];
    if (node) {
        const global_idx = node.global_idx;
        data.win_scores.forEach((scores, i) => {
            const score = scores[global_idx];
            // Match node color logic
            const maxScore = data.global_max_nn || 2.0;
            const t = Math.min(score / maxScore, 1.0);
            const color = d3.hsl(252 - t * 252, 1.0, 0.5).formatHex(); // HSL 0.7-0.0 maps to 252-0 degrees

            plotData.push({
                level: i,
                value: score,
                y: layerScreenPositions[i],
                color: color
            });
        });
    } else {
        return;
    }

    const x = d3.scaleLinear()
        .domain([0, data.global_max_nn || 1])
        .range([margin.left, width - margin.right]);

    const barHeight = 15;

    // Bars
    svg.selectAll("rect")
        .data(plotData)
        .join("rect")
        .attr("x", margin.left)
        .attr("y", d => d.y - barHeight / 2)
        .attr("width", d => Math.max(0, x(d.value) - margin.left))
        .attr("height", barHeight)
        .attr("fill", d => d.color);

    // Level Labels
    svg.selectAll("text.level-label")
        .data(plotData)
        .join("text")
        .attr("class", "level-label")
        .attr("x", margin.left - 5)
        .attr("y", d => d.y + 5)
        .attr("text-anchor", "end")
        .text(d => `L${d.level}`)
        .attr("font-size", "12px")
        .attr("font-weight", "bold");

    // Value Labels
    svg.selectAll("text.value-label")
        .data(plotData)
        .join("text")
        .attr("class", "value-label")
        .attr("x", d => x(d.value) + 5)
        .attr("y", d => d.y + 5)
        .text(d => d.value.toFixed(2))
        .attr("font-size", "10px");

    svg.append("g")
        .attr("transform", `translate(0,${height - margin.bottom})`)
        .call(d3.axisBottom(x).ticks(5));
}

function renderTimeSeries(data, highlightIdx = null) {
    const container = document.getElementById('ts-chart');
    if (!container) return;
    const width = container.offsetWidth - 20;
    const height = container.offsetHeight - 20;

    if (width <= 0 || height <= 0) return;

    const margin = { top: 20, right: 30, bottom: 30, left: 40 };

    d3.select("#ts-chart").html("");

    const svg = d3.select("#ts-chart")
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [0, 0, width, height]);

    // Define clip path
    svg.append("defs").append("clipPath")
        .attr("id", "clip-ts")
        .append("rect")
        .attr("width", width - margin.left - margin.right)
        .attr("height", height - margin.top - margin.bottom)
        .attr("x", margin.left)
        .attr("y", margin.top);

    const x = d3.scaleLinear()
        .domain(currentZoomRange || [0, data.time_series.length])
        .range([margin.left, width - margin.right]);

    const y = d3.scaleLinear()
        .domain([d3.min(data.time_series), d3.max(data.time_series)])
        .range([height - margin.bottom, margin.top]);

    // Brush for zooming
    const brush = d3.brushX()
        .extent([[margin.left, margin.top], [width - margin.right, height - margin.bottom]])
        .on("end", brushed);

    function brushed(event) {
        const selection = event.selection;
        if (selection) {
            const [x0, x1] = selection.map(x.invert);
            currentZoomRange = [x0, x1];
            document.getElementById('reset-zoom-btn').style.display = 'block';

            // Re-render both charts
            renderTimeSeries(data, highlightIdx);
            renderAnomalyScore(data, highlightIdx);

            // Remove brush overlay after selection
            svg.select(".brush").call(brush.move, null);
        }
    }

    const line = d3.line()
        .x((d, i) => x(i))
        .y(d => y(d));

    svg.append("g")
        .attr("class", "brush")
        .call(brush);

    svg.append("path")
        .datum(data.time_series)
        .attr("class", "ts-line")
        .attr("clip-path", "url(#clip-ts)")
        .attr("fill", "none")
        .attr("stroke", "steelblue")
        .attr("stroke-width", 1.5)
        .attr("d", line);

    if (highlightIdx !== null) {
        const winSize = data.win_size;
        svg.append("rect")
            .attr("clip-path", "url(#clip-ts)")
            .attr("x", x(highlightIdx))
            .attr("y", margin.top)
            .attr("width", x(highlightIdx + winSize) - x(highlightIdx))
            .attr("height", height - margin.bottom - margin.top)
            .attr("fill", "red")
            .attr("opacity", 0.3);
    }

    svg.append("g")
        .attr("transform", `translate(0,${height - margin.bottom})`)
        .call(d3.axisBottom(x));

    svg.append("g")
        .attr("transform", `translate(${margin.left},0)`)
        .call(d3.axisLeft(y));

    svg.append("text")
        .attr("x", margin.left + 5)
        .attr("y", margin.top + 5)
        .attr("font-size", "11px")
        .attr("font-weight", "bold")
        .attr("fill", "steelblue")
        .text("Input Time Series");
}

function animate() {
    requestAnimationFrame(animate);
    if (controls) controls.update();
    if (renderer && scene && camera) renderer.render(scene, camera);
}

function renderAnomalyScore(data, highlightIdx = null) {
    try {
        const container = document.getElementById('score-chart');
        if (!container) {
            console.error("Score chart container not found");
            return;
        }

        const width = container.offsetWidth - 20;
        const height = container.offsetHeight - 20;

        console.log("Rendering Anomaly Score:", {
            width,
            height,
            hasData: !!data,
            hasScores: data && data.ts_scores && data.ts_scores.length > 0
        });

        if (width <= 0 || height <= 0) return;

        const margin = { top: 10, right: 30, bottom: 30, left: 40 };

        d3.select("#score-chart").html("");

        const svg = d3.select("#score-chart")
            .append("svg")
            .attr("width", width)
            .attr("height", height)
            .attr("viewBox", [0, 0, width, height]);

        // Define clip path
        svg.append("defs").append("clipPath")
            .attr("id", "clip-score")
            .append("rect")
            .attr("width", width - margin.left - margin.right)
            .attr("height", height - margin.top - margin.bottom)
            .attr("x", margin.left)
            .attr("y", margin.top);
        // Select score series: ensemble or a specific layer
        let scores = null;
        let scoreTitle = "";

        if (scoreLayerMode === "ens") {
            if (!data.ts_score_ens || data.ts_score_ens.length === 0) {
                console.warn("No ts_score_ens found in data");
                return;
            }
            scores = data.ts_score_ens;
            scoreTitle = `HYDRA Aggregated Anomaly Score (${data_name})`;
        } else {
            const layerIdx = Number(scoreLayerMode);
            if (!data.ts_scores || !Array.isArray(data.ts_scores) || !data.ts_scores[layerIdx] || data.ts_scores[layerIdx].length === 0) {
                console.warn("No ts_scores for layer", layerIdx, "falling back to ensemble.");
                scores = data.ts_score_ens || [];
                scoreTitle = `HYDRA Aggregated Anomaly Score (${data_name})`;
            } else {
                scores = data.ts_scores[layerIdx];
                scoreTitle = `HYDRA Layer L${layerIdx} Score (${data_name})`;
            }
        }

        if (!scores || scores.length === 0) return;

        const x = d3.scaleLinear()
            .domain(currentZoomRange || [0, scores.length])
            .range([margin.left, width - margin.right]);

        const maxScore = d3.max(scores);
        const y = d3.scaleLinear()
            .domain([0, maxScore > 0 ? maxScore * 1.1 : 1])
            .range([height - margin.bottom, margin.top]);

        const area = d3.area()
            .x((d, i) => x(i))
            .y0(height - margin.bottom)
            .y1(d => y(d));

        const line = d3.line()
            .x((d, i) => x(i))
            .y(d => y(d));

        svg.append("path")
            .datum(scores)
            .attr("clip-path", "url(#clip-score)")
            .attr("fill", "#ccffcc")
            .attr("opacity", 0.5)
            .attr("d", area);

        svg.append("path")
            .datum(scores)
            .attr("clip-path", "url(#clip-score)")
            .attr("fill", "none")
            .attr("stroke", "green")
            .attr("stroke-width", 1.5)
            .attr("d", line);

        if (highlightIdx !== null) {
            // Highlight bar
            svg.append("line")
                .attr("clip-path", "url(#clip-score)")
                .attr("x1", x(highlightIdx))
                .attr("x2", x(highlightIdx))
                .attr("y1", margin.top)
                .attr("y2", height - margin.bottom)
                .attr("stroke", "black")
                .attr("stroke-dasharray", "4");

            const winSize = data.win_size || 0;
            svg.append("rect")
                .attr("clip-path", "url(#clip-score)")
                .attr("x", x(highlightIdx))
                .attr("y", margin.top)
                .attr("width", x(highlightIdx + winSize) - x(highlightIdx))
                .attr("height", height - margin.bottom - margin.top)
                .attr("fill", "green")
                .attr("opacity", 0.1);
        }

        svg.append("g")
            .attr("transform", `translate(0,${height - margin.bottom})`)
            .call(d3.axisBottom(x).ticks(10));

        svg.append("g")
            .attr("transform", `translate(${margin.left},0)`)
            .call(d3.axisLeft(y).ticks(5));

        svg.append("text")
            .attr("x", margin.left + 5)
            .attr("y", margin.top + 5)
            .attr("font-size", "11px")
            .attr("font-weight", "bold")
            .attr("fill", "green")
            .text(scoreTitle);
    } catch (err) {
        console.error("Error in renderAnomalyScore:", err);
    }
}

// Ensure global access for the start signal and zoom reset
window.hydraStartApp = hydraStartApp;
window.resetZoom = resetZoom;
window.hydraModuleReady = true;

console.log("HYDRA-AD module fully loaded and exposed to window");
