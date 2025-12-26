#pragma once

#include "telemetry_client.h"
#include <imgui.h>

namespace azl {

/**
 * Neural Network Activation Visualization
 * Shows how the network "thinks" - policy/value heads and MCTS results
 * Inspired by snake.cpp's dynamic network visualization
 */
class NetworkRenderer {
public:
    NetworkRenderer();
    
    void render(const FrameMessage& frame);
    
private:
    void draw_policy_head(const FrameMessage& frame);
    void draw_value_head(const FrameMessage& frame);
    void draw_mcts_thinking(const FrameMessage& frame);
    
    // Visual configuration
    static constexpr float BAR_WIDTH = 50.0f;
    static constexpr float BAR_MAX_HEIGHT = 100.0f;
    static constexpr int NUM_ACTIONS = 7;  // Connect Four columns
};

}  // namespace azl
