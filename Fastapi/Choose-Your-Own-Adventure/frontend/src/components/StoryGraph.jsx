import { useEffect, useRef } from 'react';
import { Network } from 'vis-network';

function StoryGraph({ story, currentNodeId, onNodeClick, visitedNodes = new Set() }) {
    const graphRef = useRef(null);
    const networkRef = useRef(null);

    useEffect(() => {
        if (!story || !story.all_nodes) return;

        const nodes = [];
        const edges = [];
        const processedNodes = new Set();

        // Only build graph for visited nodes and their immediate children
        const buildVisitedGraph = (nodeId, level = 0) => {
            if (processedNodes.has(nodeId)) return;
            processedNodes.add(nodeId);

            const node = story.all_nodes[nodeId];
            if (!node) return;

            // Only show this node if it's been visited
            if (!visitedNodes.has(nodeId)) return;

            // Determine node color based on its type and state
            let color = '#2c3e50'; // Default dark
            let borderColor = '#34495e';
            let fontColor = '#ffffff';

            if (nodeId === currentNodeId) {
                color = '#e74c3c'; // Current node - red
                borderColor = '#c0392b';
            } else if (node.is_ending) {
                if (node.is_winning_ending) {
                    color = '#27ae60'; // Winning ending - green
                    borderColor = '#229954';
                } else {
                    color = '#8e44ad'; // Losing ending - purple
                    borderColor = '#7d3c98';
                }
            } else if (visitedNodes.size === 1) {
                color = '#3498db'; // Root node - blue
                borderColor = '#2980b9';
            }

            nodes.push({
                id: nodeId,
                label: node.is_ending 
                    ? (node.is_winning_ending ? 'WIN' : 'END')
                    : `Node ${nodes.length + 1}`,
                color: {
                    background: color,
                    border: borderColor,
                    highlight: {
                        background: '#f39c12',
                        border: '#e67e22'
                    }
                },
                font: {
                    color: fontColor,
                    size: 11,
                    face: 'Arial, sans-serif',
                    bold: true
                },
                shape: node.is_ending ? 'diamond' : 'circle',
                size: currentNodeId === nodeId ? 25 : 20,
                level: level
            });

            // Add edges for options, but only show edges to visited nodes or immediate children of current node
            if (node.options && node.options.length > 0) {
                node.options.forEach((option, index) => {
                    const targetNodeId = option.node_id;
                    
                    // Show edge if target is visited OR if this is the current node (to show available options)
                    const shouldShowEdge = visitedNodes.has(targetNodeId) || nodeId === currentNodeId;
                    
                    if (shouldShowEdge) {
                        // If target node is not visited but we're showing the edge, add the target node as "preview"
                        if (!visitedNodes.has(targetNodeId) && nodeId === currentNodeId) {
                            const targetNode = story.all_nodes[targetNodeId];
                            if (targetNode && !processedNodes.has(targetNodeId)) {
                                // Add preview node (slightly transparent)
                                nodes.push({
                                    id: targetNodeId,
                                    label: '?',
                                    color: {
                                        background: '#555555',
                                        border: '#777777',
                                        highlight: {
                                            background: '#f39c12',
                                            border: '#e67e22'
                                        }
                                    },
                                    font: {
                                        color: '#ffffff',
                                        size: 11,
                                        face: 'Arial, sans-serif',
                                        bold: true
                                    },
                                    shape: 'circle',
                                    size: 15,
                                    level: level + 1,
                                    opacity: 0.7
                                });
                                processedNodes.add(targetNodeId);
                            }
                        }
                        
                        edges.push({
                            from: nodeId,
                            to: targetNodeId,
                            label: option.text.substring(0, 12) + (option.text.length > 12 ? '...' : ''),
                            color: {
                                color: visitedNodes.has(targetNodeId) ? '#7f8c8d' : '#555555',
                                highlight: '#e67e22',
                                hover: '#f39c12'
                            },
                            font: {
                                color: '#ffffff',
                                size: 9,
                                strokeWidth: 1,
                                strokeColor: '#000000'
                            },
                            arrows: {
                                to: {
                                    enabled: true,
                                    scaleFactor: 0.6
                                }
                            },
                            smooth: {
                                type: 'curvedCW',
                                roundness: 0.1 + (index * 0.1)
                            },
                            dashes: !visitedNodes.has(targetNodeId) // Dashed lines for unvisited nodes
                        });
                    }

                    // Recursively build graph for visited child nodes
                    if (visitedNodes.has(targetNodeId)) {
                        buildVisitedGraph(targetNodeId, level + 1);
                    }
                });
            }
        };

        // Start building from root node (always visited)
        if (story.root_node) {
            buildVisitedGraph(story.root_node.id);
        }

        // Create network
        const data = { nodes, edges };
        const options = {
            layout: {
                hierarchical: {
                    enabled: true,
                    direction: 'UD',
                    sortMethod: 'directed',
                    shakeTowards: 'roots',
                    levelSeparation: 80,
                    nodeSpacing: 100,
                    treeSpacing: 120
                }
            },
            physics: {
                enabled: false
            },
            nodes: {
                borderWidth: 2,
                borderWidthSelected: 3,
                chosen: true,
                shadow: false
            },
            edges: {
                width: 1,
                selectionWidth: 3,
                shadow: false,
                labelHighlightBold: false
            },
            interaction: {
                hover: true,
                selectConnectedEdges: false,
                dragNodes: false,
                dragView: true,
                zoomView: true,
                tooltipDelay: 200
            },
            configure: {
                enabled: false
            }
        };

        if (networkRef.current) {
            networkRef.current.destroy();
        }

        networkRef.current = new Network(graphRef.current, data, options);

        // Add click event listener
        if (onNodeClick) {
            networkRef.current.on('click', (params) => {
                if (params.nodes.length > 0) {
                    const clickedNodeId = params.nodes[0];
                    onNodeClick(clickedNodeId);
                }
            });
        }

        return () => {
            if (networkRef.current) {
                networkRef.current.destroy();
                networkRef.current = null;
            }
        };
    }, [story, currentNodeId, onNodeClick, visitedNodes]);

    if (!story || !story.all_nodes) {
        return (
            <div className="story-graph-container">
                <div className="graph-placeholder">
                    <p>Graph will appear when story loads</p>
                </div>
            </div>
        );
    }

    return (
        <div className="story-graph-container">
            <div className="graph-header">
                <h3>Story Map</h3>
                <div className="graph-legend">
                    <div className="legend-item">
                        <div className="legend-dot current"></div>
                        <span>Current</span>
                    </div>
                    <div className="legend-item">
                        <div className="legend-dot root"></div>
                        <span>Start</span>
                    </div>
                    <div className="legend-item">
                        <div className="legend-dot win"></div>
                        <span>Win</span>
                    </div>
                    <div className="legend-item">
                        <div className="legend-dot lose"></div>
                        <span>Lose</span>
                    </div>
                </div>
            </div>
            <div 
                ref={graphRef} 
                className="story-graph"
            />
            <div className="graph-instructions">
                Click nodes to navigate | Drag to pan | Scroll to zoom
            </div>
        </div>
    );
}

export default StoryGraph;