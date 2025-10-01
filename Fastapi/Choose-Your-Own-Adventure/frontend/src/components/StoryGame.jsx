import { useState, useEffect } from 'react';
import StoryGraph from './StoryGraph';

function StoryGame({ story, onNewStory }) {
    const [currentNodeId, setCurrentNodeId] = useState(null);
    const [currentNode, setCurrentNode] = useState(null);
    const [options, setOptions] = useState([]);
    const [isEnding, setIsEnding] = useState(false);
    const [isWinningEnding, setIsWinningEnding] = useState(false);
    const [visitedNodes, setVisitedNodes] = useState(new Set());

    useEffect(() => {
        if (story && story.root_node) {
            const rootNodeId = story.root_node.id;
            setCurrentNodeId(rootNodeId);
            setVisitedNodes(new Set([rootNodeId]));
        }
    }, [story]);

    useEffect(() => {
        if (currentNodeId && story && story.all_nodes) {
            const node = story.all_nodes[currentNodeId];

            setCurrentNode(node);
            setIsEnding(node.is_ending);
            setIsWinningEnding(node.is_winning_ending);

            if (!node.is_ending && node.options && node.options.length > 0) {
                setOptions(node.options);
            } else {
                setOptions([]);
            }
        }
    }, [currentNodeId, story]);

    const chooseOption = (optionId) => {
        setCurrentNodeId(optionId);
        setVisitedNodes(prev => new Set([...prev, optionId]));
    };

    const handleGraphNodeClick = (nodeId) => {
        // Allow navigation to any visited node or adjacent nodes
        if (visitedNodes.has(nodeId) || isAdjacentNode(nodeId)) {
            setCurrentNodeId(nodeId);
            setVisitedNodes(prev => new Set([...prev, nodeId]));
        }
    };

    const isAdjacentNode = (nodeId) => {
        // Check if the node is directly accessible from current node
        if (currentNode && currentNode.options) {
            return currentNode.options.some(option => option.node_id === nodeId);
        }
        return false;
    };

    const restartStory = () => {
        if (story && story.root_node) {
            const rootNodeId = story.root_node.id;
            setCurrentNodeId(rootNodeId);
            setVisitedNodes(new Set([rootNodeId]));
        }
    };

    return (
        <div className="story-game">
            <div className="story-main-content">
                <header className="story-header">
                    <h2>{story.title}</h2>
                </header>

                <div className="story-content">
                    {currentNode && (
                        <div className="story-node">
                            <p>{currentNode.content}</p>

                            {isEnding ? (
                                <div className={`story-ending ${isWinningEnding ? 'winning' : 'losing'}`}>
                                    <h3>
                                        {isWinningEnding ? "Victory!" : "The End"}
                                    </h3>
                                    <div className="ending-message">
                                        {isWinningEnding 
                                            ? "Congratulations! You've achieved a winning ending!" 
                                            : "Your adventure has come to an end. Better luck next time!"
                                        }
                                    </div>
                                </div>
                            ) : (
                                <div className="story-options">
                                    <h3>Choose Your Path</h3>
                                    <div className="options-list">
                                        {options.map((option, index) => (
                                            <button
                                                key={index}
                                                onClick={() => chooseOption(option.node_id)}
                                                className="option-btn"
                                            >
                                                <span className="option-number">{index + 1}.</span>
                                                {option.text}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    <div className="story-controls">
                        <button onClick={restartStory} className="reset-btn">
                            Restart
                        </button>
                        {onNewStory && (
                            <button onClick={onNewStory} className="new-story-btn">
                                New Story
                            </button>
                        )}
                    </div>
                </div>
            </div>

            <StoryGraph 
                story={story} 
                currentNodeId={currentNodeId}
                onNodeClick={handleGraphNodeClick}
                visitedNodes={visitedNodes}
            />
        </div>
    );
}

export default StoryGame;