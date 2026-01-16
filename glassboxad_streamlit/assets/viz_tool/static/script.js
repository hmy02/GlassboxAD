async function init() {
    const response = await fetch('/api/data');
    const data = await response.json();
    
    renderTimeSeries(data);
    renderHierarchy(data);
}

function renderTimeSeries(data) {
    const width = 1000;
    const height = 200;
    const margin = {top: 20, right: 30, bottom: 30, left: 40};
    
    const svg = d3.select("#ts-chart")
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [0, 0, width, height]);
        
    const x = d3.scaleLinear()
        .domain([0, data.time_series.length])
        .range([margin.left, width - margin.right]);
        
    const y = d3.scaleLinear()
        .domain([d3.min(data.time_series), d3.max(data.time_series)])
        .range([height - margin.bottom, margin.top]);
        
    const line = d3.line()
        .x((d, i) => x(i))
        .y(d => y(d));
        
    svg.append("path")
        .datum(data.time_series)
        .attr("fill", "none")
        .attr("stroke", "steelblue")
        .attr("stroke-width", 1.5)
        .attr("d", line);
        
    svg.append("g")
        .attr("transform", `translate(0,${height - margin.bottom})`)
        .call(d3.axisBottom(x));
        
    svg.append("g")
        .attr("transform", `translate(${margin.left},0)`)
        .call(d3.axisLeft(y));
}

function renderHierarchy(data) {
    const width = 1000;
    const height = 600;
    const margin = {top: 20, right: 30, bottom: 30, left: 40};
    
    const svg = d3.select("#hierarchy-chart")
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [0, 0, width, height]);
        
    const numLevels = data.levels.length;
    const tsLength = data.time_series.length;
    
    const x = d3.scaleLinear()
        .domain([0, tsLength])
        .range([margin.left, width - margin.right]);
        
    const y = d3.scaleLinear()
        .domain([0, numLevels - 1])
        .range([height - margin.bottom, margin.top]);
        
    // Create a map for quick node lookup
    const nodeMap = new Map();
    data.levels.forEach(level => {
        level.nodes.forEach(node => {
            nodeMap.set(node.id, node);
        });
    });
    
    // Draw links
    svg.append("g")
        .attr("stroke", "#999")
        .attr("stroke-opacity", 0.3)
        .selectAll("line")
        .data(data.links)
        .join("line")
        .attr("x1", d => {
            const source = nodeMap.get(d.source);
            return x(source.global_idx);
        })
        .attr("y1", d => {
            const source = nodeMap.get(d.source);
            return y(source.level);
        })
        .attr("x2", d => {
            const target = nodeMap.get(d.target);
            return x(target.global_idx);
        })
        .attr("y2", d => {
            const target = nodeMap.get(d.target);
            return y(target.level);
        });
        
    // Color scale for anomaly score
    const maxScore = d3.max(data.levels[0].nodes, d => d.score);
    const color = d3.scaleSequential(d3.interpolateReds)
        .domain([0, maxScore]);
        
    // Draw nodes
    data.levels.forEach(level => {
        svg.append("g")
            .selectAll("circle")
            .data(level.nodes)
            .join("circle")
            .attr("cx", d => x(d.global_idx))
            .attr("cy", d => y(d.level))
            .attr("r", 3)
            .attr("fill", d => color(d.score))
            .attr("stroke", "black")
            .attr("stroke-width", 0.5)
            .on("mouseover", (event, d) => {
                d3.select("#tooltip")
                    .style("opacity", 1)
                    .html(`Level: ${d.level}<br>Idx: ${d.global_idx}<br>Score: ${d.score.toFixed(4)}`)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
                    
                // Highlight connected paths (optional, could be complex)
            })
            .on("mouseout", () => {
                d3.select("#tooltip").style("opacity", 0);
            });
    });
    
    // Add Y axis (Levels)
    svg.append("g")
        .attr("transform", `translate(${margin.left},0)`)
        .call(d3.axisLeft(y).ticks(numLevels));
        
    // Add X axis
    svg.append("g")
        .attr("transform", `translate(0,${height - margin.bottom})`)
        .call(d3.axisBottom(x));
}

init();
