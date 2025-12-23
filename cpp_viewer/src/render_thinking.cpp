#include "render_thinking.h"
#include <imgui.h>
#include <implot.h>

namespace azl {

ThinkingRenderer::ThinkingRenderer() {}

void ThinkingRenderer::render(const FrameMessage& frame) {
    ImGui::Begin("MCTS Thinking", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    
    ImGui::Text("MCTS Searches: %d total visits", 
                frame.root_visits.empty() ? 0 : 
                std::accumulate(frame.root_visits.begin(), frame.root_visits.end(), 0));
    ImGui::Text("Value Head: %.3f", frame.value_head);
    ImGui::Spacing();
    
    render_visit_bars(frame);
    ImGui::Spacing();
    render_policy_comparison(frame);
    ImGui::Spacing();
    render_q_values(frame);
    
    ImGui::End();
}

void ThinkingRenderer::render_visit_bars(const FrameMessage& frame) {
    if (frame.root_visits.empty()) {
        ImGui::TextColored(ImVec4(1,0.5f,0,1), "No MCTS data");
        return;
    }
    
    ImGui::Text("MCTS Visit Counts per Column:");
    
    if (ImPlot::BeginPlot("##visits", ImVec2(400, 200))) {
        std::vector<double> x_vals, y_vals;
        for (size_t i = 0; i < frame.root_visits.size(); i++) {
            x_vals.push_back(static_cast<double>(i));
            y_vals.push_back(static_cast<double>(frame.root_visits[i]));
        }
        
        ImPlot::SetupAxes("Column", "Visits");
        ImPlot::PlotBars("Visits", x_vals.data(), y_vals.data(), x_vals.size(), 0.67);
        
        ImPlot::EndPlot();
    }
}

void ThinkingRenderer::render_policy_comparison(const FrameMessage& frame) {
    if (frame.mcts_policy.empty() || frame.policy_head.empty()) {
        return;
    }
    
    ImGui::Text("Policy Comparison (Network vs MCTS):");
    
    if (ImPlot::BeginPlot("##policy", ImVec2(400, 200))) {
        std::vector<double> x_vals;
        std::vector<double> mcts_vals, nn_vals;
        
        for (size_t i = 0; i < std::min(frame.mcts_policy.size(), frame.policy_head.size()); i++) {
            x_vals.push_back(static_cast<double>(i));
            mcts_vals.push_back(static_cast<double>(frame.mcts_policy[i]));
            nn_vals.push_back(static_cast<double>(frame.policy_head[i]));
        }
        
        ImPlot::SetupAxes("Column", "Probability");
        ImPlot::SetupLegend(ImPlotLocation_NorthEast);  // MUST be before Plot* calls
        ImPlot::PlotBars("MCTS Policy", x_vals.data(), mcts_vals.data(), x_vals.size(), 0.4, -0.2);
        ImPlot::PlotBars("NN Policy", x_vals.data(), nn_vals.data(), x_vals.size(), 0.4, 0.2);
        
        ImPlot::EndPlot();
    }
}

void ThinkingRenderer::render_q_values(const FrameMessage& frame) {
    if (frame.root_q.empty()) {
        return;
    }
    
    ImGui::Text("Root Q-Values (from root's perspective):");
    
    if (ImPlot::BeginPlot("##qvalues", ImVec2(400, 200))) {
        std::vector<double> x_vals, q_vals;
        
        for (size_t i = 0; i < frame.root_q.size(); i++) {
            x_vals.push_back(static_cast<double>(i));
            q_vals.push_back(static_cast<double>(frame.root_q[i]));
        }
        
        ImPlot::SetupAxes("Column", "Q-value");
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.1, 1.1);
        ImPlot::PlotLine("Q", x_vals.data(), q_vals.data(), x_vals.size());
        ImPlot::PlotScatter("Q", x_vals.data(), q_vals.data(), x_vals.size());
        
        ImPlot::EndPlot();
    }
}

}  // namespace azl
