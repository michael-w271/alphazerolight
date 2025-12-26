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

}  // namespace azl
