#pragma once

#include "telemetry_client.h"
#include <imgui.h>
#include <vector>

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
    void render_architecture(const FrameMessage& frame);  // NEW: Snake.cpp-style architecture viz
    
private:
    void draw_policy_head(const FrameMessage& frame);
    void draw_value_head(const FrameMessage& frame);
    void draw_mcts_thinking(const FrameMessage& frame);
    
    // NEW: Architecture visualization helpers
    void draw_resnet_architecture(const FrameMessage& frame);
    void draw_layer_node(ImDrawList* draw_list, ImVec2 pos, float radius, float activation, const char* label);
    void draw_connection(ImDrawList* draw_list, ImVec2 from, ImVec2 to, float weight, float activation);
    
    // Visual configuration
    static constexpr float BAR_WIDTH = 50.0f;
    static constexpr float BAR_MAX_HEIGHT = 100.0f;
    static constexpr int NUM_ACTIONS = 7;  // Connect Four columns
    
    // Architecture visualization config
    static constexpr float NODE_RADIUS = 8.0f;
    static constexpr float LAYER_SPACING = 150.0f;
    static constexpr float NODE_SPACING = 25.0f;
    
    // Animation state for dynamic effects
    float animation_time_ = 0.0f;
};

}  // namespace azl
