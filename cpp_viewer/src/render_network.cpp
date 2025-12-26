#include "render_network.h"
#include <imgui.h>
#include <algorithm>
#include <cmath>

namespace azl {

NetworkRenderer::NetworkRenderer() {}

void NetworkRenderer::render(const FrameMessage& frame) {
    ImGui::Begin("Neural Network Activity", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    
    ImGui::Text("Iteration: %d | Game: %d | Move: %d", 
                frame.iteration, frame.game_idx, frame.move_idx);
    ImGui::Separator();
    ImGui::Spacing();
    
    // Three panels showing different aspects of the network
    draw_policy_head(frame);
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    draw_value_head(frame);
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    draw_mcts_thinking(frame);
    
    ImGui::End();
}

void NetworkRenderer::draw_policy_head(const FrameMessage& frame) {
    ImGui::Text("Raw Policy Head (Neural Network Output)");
    ImGui::Text("Shows what the network thinks before MCTS search");
    ImGui::Spacing();
    
    if (frame.policy_head.empty()) {
        ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "No policy data");
        return;
    }
    
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    
    // Draw bars for each action
    float max_val = *std::max_element(frame.policy_head.begin(), frame.policy_head.end());
    if (max_val < 0.001f) max_val = 1.0f;  // Avoid division by zero
    
    for (int i = 0; i < NUM_ACTIONS && i < static_cast<int>(frame.policy_head.size()); i++) {
        float value = frame.policy_head[i];
        float normalized = value / max_val;
        float height = normalized * BAR_MAX_HEIGHT;
        
        // Position for this bar
        float x = canvas_pos.x + i * (BAR_WIDTH + 10);
        float y = canvas_pos.y + BAR_MAX_HEIGHT;
        
        // Check if this action is valid
        bool is_valid = i < static_cast<int>(frame.valid_moves.size()) && frame.valid_moves[i] > 0;
        
        // Color based on activation strength and validity
        ImU32 color;
        if (!is_valid) {
            color = IM_COL32(50, 50, 50, 255);  // Gray for invalid moves
        } else {
            // Blue intensity based on activation
            int intensity = static_cast<int>(normalized * 255);
            color = IM_COL32(0, intensity / 2, intensity, 255);
        }
        
        // Draw bar
        ImVec2 p_min(x, y - height);
        ImVec2 p_max(x + BAR_WIDTH, y);
        draw_list->AddRectFilled(p_min, p_max, color);
        
        // Outline
        draw_list->AddRect(p_min, p_max, IM_COL32(100, 100, 100, 255), 0.0f, 0, 1.5f);
        
        // Column label
        char label[8];
        snprintf(label, sizeof(label), "Col %d", i);
        ImVec2 text_pos(x + 5, y + 5);
        draw_list->AddText(text_pos, IM_COL32(200, 200, 200, 255), label);
        
        // Value label
        char val_label[16];
        snprintf(val_label, sizeof(val_label), "%.3f", value);
        ImVec2 val_pos(x + 5, y - height - 15);
        draw_list->AddText(val_pos, IM_COL32(255, 255, 255, 255), val_label);
    }
    
    // Reserve space
    ImGui::Dummy(ImVec2(NUM_ACTIONS * (BAR_WIDTH + 10), BAR_MAX_HEIGHT + 30));
}

void NetworkRenderer::draw_value_head(const FrameMessage& frame) {
    ImGui::Text("Value Head (Position Evaluation)");
    ImGui::Text("Network's evaluation: -1 (loss) to +1 (win)");
    ImGui::Spacing();
    
    float value = frame.value_head;
    
    // Create a horizontal bar from -1 to +1
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    
    float bar_width = 400.0f;
    float bar_height = 30.0f;
    
    // Background bar
    ImVec2 bg_min = canvas_pos;
    ImVec2 bg_max(canvas_pos.x + bar_width, canvas_pos.y + bar_height);
    draw_list->AddRectFilled(bg_min, bg_max, IM_COL32(40, 40, 40, 255));
    
    // Center line
    float center_x = canvas_pos.x + bar_width / 2;
    draw_list->AddLine(
        ImVec2(center_x, canvas_pos.y),
        ImVec2(center_x, canvas_pos.y + bar_height),
        IM_COL32(150, 150, 150, 255), 2.0f
    );
    
    // Value indicator
    float normalized_value = (value + 1.0f) / 2.0f;  // Map -1,1 to 0,1
    float indicator_x = canvas_pos.x + normalized_value * bar_width;
    
    // Color based on value (red for negative, green for positive)
    ImU32 indicator_color;
    if (value > 0) {
        int intensity = static_cast<int>(value * 200) + 55;
        indicator_color = IM_COL32(0, intensity, 0, 255);
    } else {
        int intensity = static_cast<int>(-value * 200) + 55;
        indicator_color = IM_COL32(intensity, 0, 0, 255);
    }
    
    // Draw indicator bar
    float indicator_width = 10.0f;
    ImVec2 ind_min(indicator_x - indicator_width / 2, canvas_pos.y);
    ImVec2 ind_max(indicator_x + indicator_width / 2, canvas_pos.y + bar_height);
    draw_list->AddRectFilled(ind_min, ind_max, indicator_color);
    
    // Labels
    char val_text[32];
    snprintf(val_text, sizeof(val_text), "%.3f", value);
    ImVec2 text_pos(indicator_x - 15, canvas_pos.y + bar_height + 5);
    draw_list->AddText(text_pos, IM_COL32(255, 255, 255, 255), val_text);
    
    // Reserve space
    ImGui::Dummy(ImVec2(bar_width, bar_height + 25));
}

void NetworkRenderer::draw_mcts_thinking(const FrameMessage& frame) {
    ImGui::Text("MCTS Search Results (After Tree Search)");
    ImGui::Text("Final policy after exploration");
    ImGui::Spacing();
    
    if (frame.mcts_policy.empty()) {
        ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "No MCTS data");
        return;
    }
    
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    
    // Find max for normalization
    float max_val = *std::max_element(frame.mcts_policy.begin(), frame.mcts_policy.end());
    if (max_val < 0.001f) max_val = 1.0f;
    
    for (int i = 0; i < NUM_ACTIONS && i < static_cast<int>(frame.mcts_policy.size()); i++) {
        float value = frame.mcts_policy[i];
        float normalized = value / max_val;
        float height = normalized * BAR_MAX_HEIGHT;
        
        float x = canvas_pos.x + i * (BAR_WIDTH + 10);
        float y = canvas_pos.y + BAR_MAX_HEIGHT;
        
        // Check if this is the chosen action
        bool is_chosen = (i == frame.chosen_action);
        
        // Color: green for high probability, with highlight for chosen
        int intensity = static_cast<int>(normalized * 200) + 55;
        ImU32 color;
        if (is_chosen) {
            color = IM_COL32(intensity, 255, 0, 255);  // Bright yellow/green for chosen
        } else {
            color = IM_COL32(0, intensity, intensity / 2, 255);  // Cyan/teal
        }
        
        // Draw bar
        ImVec2 p_min(x, y - height);
        ImVec2 p_max(x + BAR_WIDTH, y);
        draw_list->AddRectFilled(p_min, p_max, color);
        
        // Thicker outline for chosen action
        float outline_thickness = is_chosen ? 3.0f : 1.5f;
        ImU32 outline_color = is_chosen ? IM_COL32(255, 255, 0, 255) : IM_COL32(100, 100, 100, 255);
        draw_list->AddRect(p_min, p_max, outline_color, 0.0f, 0, outline_thickness);
        
        // Column label
        char label[8];
        snprintf(label, sizeof(label), "Col %d", i);
        ImVec2 text_pos(x + 5, y + 5);
        draw_list->AddText(text_pos, IM_COL32(200, 200, 200, 255), label);
        
        // Value and visit count
        char val_label[32];
        int visits = i < static_cast<int>(frame.root_visits.size()) ? frame.root_visits[i] : 0;
        snprintf(val_label, sizeof(val_label), "%.2f\n(%d)", value, visits);
        ImVec2 val_pos(x + 2, y - height - 30);
        draw_list->AddText(val_pos, IM_COL32(255, 255, 255, 255), val_label);
    }
    
    // Reserve space
    ImGui::Dummy(ImVec2(NUM_ACTIONS * (BAR_WIDTH + 10), BAR_MAX_HEIGHT + 40));
}

void NetworkRenderer::render_architecture(const FrameMessage& frame) {
    ImGui::Begin("ResNet Architecture - Live Activations", nullptr, 
                 ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoScrollbar);
    
    ImGui::Text("Iteration: %d | Game: %d | Move: %d", 
                frame.iteration, frame.game_idx, frame.move_idx);
    ImGui::Text("Dynamic neural network visualization inspired by Snake.cpp");
    ImGui::Separator();
    ImGui::Spacing();
    
    // Increment animation time for pulsing effects
    animation_time_ += 0.05f;
    
    draw_resnet_architecture(frame);
    
    ImGui::End();
}

void NetworkRenderer::draw_resnet_architecture(const FrameMessage& frame) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    ImVec2 canvas_size(900, 600);
    
    // Background
    draw_list->AddRectFilled(canvas_pos, 
                            ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
                            IM_COL32(15, 15, 20, 255));
    
    // Title
    ImVec2 title_pos(canvas_pos.x + 10, canvas_pos.y + 10);
    draw_list->AddText(title_pos, IM_COL32(255, 255, 255, 255), 
                      "AlphaZero ResNet: Input -> Conv -> ResBlocks -> Policy/Value Heads");
    
    float y_offset = canvas_pos.y + 80;
    float x_start = canvas_pos.x + 50;
    
    // Simulate different layer activations based on board state and network outputs
    // In a real implementation, you'd get these from the Python telemetry
    
    // Calculate pseudo-activations based on available data
    float board_complexity = 0.0f;
    for (const auto& row : frame.board) {
        for (int val : row) {
            if (val != 0) board_complexity += 0.1f;
        }
    }
    board_complexity = std::min(1.0f, board_complexity);
    
    float policy_entropy = 0.0f;
    for (float p : frame.policy_head) {
        if (p > 0.001f) policy_entropy += -p * std::log(p);
    }
    policy_entropy = std::min(1.0f, policy_entropy / 2.0f);
    
    float value_magnitude = std::abs(frame.value_head);
    
    // Layer 1: Input (Board State - 3 channels: player, opponent, empty)
    float input_x = x_start;
    draw_layer_node(draw_list, ImVec2(input_x, y_offset), NODE_RADIUS * 1.5f, 
                   board_complexity, "Input\n3×6×7");
    
    // Layer 2: Initial Conv Block
    float conv_x = input_x + LAYER_SPACING;
    float conv_activation = 0.5f + 0.3f * std::sin(animation_time_ + board_complexity);
    draw_layer_node(draw_list, ImVec2(conv_x, y_offset - 40), NODE_RADIUS, 
                   conv_activation, "Conv");
    draw_layer_node(draw_list, ImVec2(conv_x, y_offset), NODE_RADIUS, 
                   conv_activation * 0.9f, "128");
    draw_layer_node(draw_list, ImVec2(conv_x, y_offset + 40), NODE_RADIUS, 
                   conv_activation * 0.85f, "ch");
    
    // Draw connections from input to conv
    draw_connection(draw_list, ImVec2(input_x, y_offset), 
                   ImVec2(conv_x, y_offset - 40), 0.7f, conv_activation);
    draw_connection(draw_list, ImVec2(input_x, y_offset), 
                   ImVec2(conv_x, y_offset), 0.8f, conv_activation);
    draw_connection(draw_list, ImVec2(input_x, y_offset), 
                   ImVec2(conv_x, y_offset + 40), 0.6f, conv_activation);
    
    // Layer 3-12: Residual Blocks (show simplified)
    float res_x = conv_x + LAYER_SPACING;
    float res_activation = 0.6f + 0.2f * std::sin(animation_time_ * 1.5f + policy_entropy);
    
    for (int i = 0; i < 3; i++) {
        float y = y_offset - 60 + i * 60;
        draw_layer_node(draw_list, ImVec2(res_x, y), NODE_RADIUS * 0.8f, 
                       res_activation * (1.0f - i * 0.1f), 
                       i == 0 ? "Res" : (i == 1 ? "10x" : "Blk"));
        
        // Connections from conv to res blocks
        draw_connection(draw_list, ImVec2(conv_x, y_offset), 
                       ImVec2(res_x, y), 0.7f + i * 0.1f, res_activation);
    }
    
    // Residual skip connections (visual flourish)
    draw_list->AddBezierCubic(
        ImVec2(res_x - 20, y_offset - 60),
        ImVec2(res_x + 30, y_offset - 80),
        ImVec2(res_x + 30, y_offset + 80),
        ImVec2(res_x - 20, y_offset + 60),
        IM_COL32(100, 150, 255, 100), 2.0f);
    
    // Layer 13: Policy Head Branch
    float policy_x = res_x + LAYER_SPACING * 1.2f;
    float policy_y = y_offset - 100;
    
    draw_layer_node(draw_list, ImVec2(policy_x, policy_y - 30), NODE_RADIUS, 
                   policy_entropy, "Policy");
    draw_layer_node(draw_list, ImVec2(policy_x, policy_y), NODE_RADIUS * 0.7f, 
                   policy_entropy * 0.9f, "Conv");
    draw_layer_node(draw_list, ImVec2(policy_x, policy_y + 30), NODE_RADIUS * 0.7f, 
                   policy_entropy * 0.8f, "32ch");
    
    // Connections to policy head
    for (int i = 0; i < 3; i++) {
        float from_y = y_offset - 60 + i * 60;
        draw_connection(draw_list, ImVec2(res_x, from_y), 
                       ImVec2(policy_x, policy_y), 0.6f + i * 0.1f, policy_entropy);
    }
    
    // Policy output nodes (7 actions)
    float policy_out_x = policy_x + LAYER_SPACING * 0.8f;
    for (int i = 0; i < NUM_ACTIONS; i++) {
        float action_y = policy_y - 90 + i * 30;
        float action_activation = i < frame.policy_head.size() ? frame.policy_head[i] : 0.0f;
        bool is_chosen = (i == frame.chosen_action);
        
        char label[4];
        snprintf(label, sizeof(label), "%d", i);
        draw_layer_node(draw_list, ImVec2(policy_out_x, action_y), 
                       is_chosen ? NODE_RADIUS * 1.2f : NODE_RADIUS * 0.6f,
                       action_activation, label);
        
        // Connection with strength based on policy probability
        draw_connection(draw_list, ImVec2(policy_x, policy_y), 
                       ImVec2(policy_out_x, action_y), 
                       action_activation, action_activation);
        
        // Highlight chosen action with a ring
        if (is_chosen) {
            float pulse = 1.0f + 0.3f * std::sin(animation_time_ * 3.0f);
            draw_list->AddCircle(ImVec2(policy_out_x, action_y), 
                                NODE_RADIUS * 1.5f * pulse,
                                IM_COL32(255, 255, 0, 200), 16, 2.5f);
        }
    }
    
    // Layer 14: Value Head Branch
    float value_x = res_x + LAYER_SPACING * 1.2f;
    float value_y = y_offset + 100;
    
    draw_layer_node(draw_list, ImVec2(value_x, value_y - 30), NODE_RADIUS, 
                   value_magnitude, "Value");
    draw_layer_node(draw_list, ImVec2(value_x, value_y), NODE_RADIUS * 0.7f, 
                   value_magnitude * 0.9f, "Conv");
    draw_layer_node(draw_list, ImVec2(value_x, value_y + 30), NODE_RADIUS * 0.7f, 
                   value_magnitude * 0.8f, "3ch");
    
    // Connections to value head
    for (int i = 0; i < 3; i++) {
        float from_y = y_offset - 60 + i * 60;
        draw_connection(draw_list, ImVec2(res_x, from_y), 
                       ImVec2(value_x, value_y), 0.6f + i * 0.1f, value_magnitude);
    }
    
    // Value output node
    float value_out_x = value_x + LAYER_SPACING * 0.8f;
    draw_layer_node(draw_list, ImVec2(value_out_x, value_y), NODE_RADIUS * 1.5f, 
                   value_magnitude, "Out");
    draw_connection(draw_list, ImVec2(value_x, value_y), 
                   ImVec2(value_out_x, value_y), 1.0f, value_magnitude);
    
    // Value indicator with color
    ImU32 value_color;
    if (frame.value_head > 0) {
        int intensity = static_cast<int>(frame.value_head * 200) + 55;
        value_color = IM_COL32(0, intensity, 0, 255);
    } else {
        int intensity = static_cast<int>(-frame.value_head * 200) + 55;
        value_color = IM_COL32(intensity, 0, 0, 255);
    }
    
    char value_text[32];
    snprintf(value_text, sizeof(value_text), "%.3f", frame.value_head);
    ImVec2 value_text_pos(value_out_x + 20, value_y - 5);
    draw_list->AddText(value_text_pos, value_color, value_text);
    
    // Legend
    ImVec2 legend_pos(canvas_pos.x + canvas_size.x - 180, canvas_pos.y + canvas_size.y - 100);
    draw_list->AddText(legend_pos, IM_COL32(200, 200, 200, 255), "Legend:");
    draw_list->AddCircleFilled(ImVec2(legend_pos.x + 10, legend_pos.y + 20), 5, 
                              IM_COL32(255, 100, 100, 255));
    draw_list->AddText(ImVec2(legend_pos.x + 20, legend_pos.y + 15), 
                      IM_COL32(200, 200, 200, 255), "High activation");
    draw_list->AddCircleFilled(ImVec2(legend_pos.x + 10, legend_pos.y + 40), 5, 
                              IM_COL32(50, 50, 100, 255));
    draw_list->AddText(ImVec2(legend_pos.x + 20, legend_pos.y + 35), 
                      IM_COL32(200, 200, 200, 255), "Low activation");
    draw_list->AddLine(ImVec2(legend_pos.x, legend_pos.y + 55), 
                      ImVec2(legend_pos.x + 30, legend_pos.y + 55),
                      IM_COL32(150, 200, 255, 255), 3.0f);
    draw_list->AddText(ImVec2(legend_pos.x + 35, legend_pos.y + 50), 
                      IM_COL32(200, 200, 200, 255), "Strong weight");
    
    ImGui::Dummy(canvas_size);
}

void NetworkRenderer::draw_layer_node(ImDrawList* draw_list, ImVec2 pos, float radius, 
                                      float activation, const char* label) {
    // Clamp activation to [0, 1]
    activation = std::max(0.0f, std::min(1.0f, activation));
    
    // Color based on activation level
    int base_intensity = static_cast<int>(activation * 200) + 55;
    ImU32 node_color = IM_COL32(base_intensity / 2, base_intensity / 2, base_intensity, 255);
    
    // Outer glow for high activations
    if (activation > 0.7f) {
        float glow_radius = radius * (1.5f + 0.3f * std::sin(animation_time_ * 2.0f));
        draw_list->AddCircle(pos, glow_radius, 
                            IM_COL32(150, 150, 255, 100), 16, 1.5f);
    }
    
    // Main node circle
    draw_list->AddCircleFilled(pos, radius, node_color);
    draw_list->AddCircle(pos, radius, IM_COL32(255, 255, 255, 180), 16, 1.5f);
    
    // Label
    if (label && strlen(label) > 0) {
        ImVec2 text_size = ImGui::CalcTextSize(label);
        ImVec2 text_pos(pos.x - text_size.x / 2, pos.y + radius + 3);
        draw_list->AddText(text_pos, IM_COL32(200, 200, 200, 255), label);
    }
}

void NetworkRenderer::draw_connection(ImDrawList* draw_list, ImVec2 from, ImVec2 to, 
                                      float weight, float activation) {
    // Clamp values
    weight = std::max(0.0f, std::min(1.0f, weight));
    activation = std::max(0.0f, std::min(1.0f, activation));
    
    // Line thickness based on weight magnitude
    float thickness = 0.5f + weight * 2.5f;
    
    // Color based on activation and weight
    // Positive weights: blue/cyan, strong activations brighten
    int blue = static_cast<int>(weight * 150) + 50;
    int intensity = static_cast<int>(activation * 150);
    
    // Pulsing effect for high activations
    float pulse = 1.0f;
    if (activation > 0.6f) {
        pulse = 1.0f + 0.2f * std::sin(animation_time_ * 2.5f + weight * 3.14159f);
        thickness *= pulse;
    }
    
    ImU32 line_color = IM_COL32(intensity, intensity, blue, 150 + static_cast<int>(activation * 105));
    
    draw_list->AddLine(from, to, line_color, thickness);
}

}  // namespace azl
