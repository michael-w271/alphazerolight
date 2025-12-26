#include "render_metrics.h"
#include <imgui.h>
#include <implot.h>
#include <numeric>

namespace azl {

MetricsRenderer::MetricsRenderer() {}

void MetricsRenderer::add_metrics(const MetricsMessage& metrics) {
    history_.add(metrics);
}

void MetricsRenderer::render() {
    ImGui::Begin("Training Metrics", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    
    if (history_.iterations.empty()) {
        ImGui::TextColored(ImVec4(1,0.5f,0,1), "No metrics data yet");
        ImGui::End();
        return;
    }
    
    // Display latest metrics
    int latest_iter = history_.iterations.back();
    float latest_total = history_.total_loss.back();
    float latest_policy = history_.policy_loss.back();
    float latest_value = history_.value_loss.back();
    float latest_entropy = history_.policy_entropy.back();
    float latest_sharpness = history_.policy_sharpness.back();
    float latest_mcts_imp = history_.mcts_improvement_avg.back();
    
    ImGui::Text("Latest (Iteration %d):", latest_iter);
    ImGui::Indent();
    ImGui::Text("Total Loss: %.4f", latest_total);
    ImGui::Text("Policy Loss: %.4f", latest_policy);
    ImGui::Text("Value Loss: %.4f", latest_value);
    ImGui::Separator();
    ImGui::Text("Policy Entropy: %.4f", latest_entropy);
    ImGui::Text("Policy Sharpness: %.4f", latest_sharpness);
    ImGui::Text("MCTS Improvement: %.4f", latest_mcts_imp);
    ImGui::Unindent();
    ImGui::Spacing();
    
    // Convert deques to vectors for ImPlot (reused across all plots)
    std::vector<double> x_data(history_.iterations.begin(), history_.iterations.end());
    
    // Loss Curves
    if (ImPlot::BeginPlot("Loss Curves", ImVec2(-1, 250))) {
        std::vector<double> total_data(history_.total_loss.begin(), history_.total_loss.end());
        std::vector<double> policy_data(history_.policy_loss.begin(), history_.policy_loss.end());
        std::vector<double> value_data(history_.value_loss.begin(), history_.value_loss.end());
        
        ImPlot::SetupAxes("Iteration", "Loss");
        ImPlot::PlotLine("Total", x_data.data(), total_data.data(), x_data.size());
        ImPlot::PlotLine("Policy", x_data.data(), policy_data.data(), x_data.size());
        ImPlot::PlotLine("Value", x_data.data(), value_data.data(), x_data.size());
        ImPlot::SetupLegend(ImPlotLocation_NorthEast);
        
        ImPlot::EndPlot();
    }
    
    ImGui::Spacing();
    
    // Policy Confidence Metrics
    if (ImPlot::BeginPlot("Policy Confidence", ImVec2(-1, 250))) {
        std::vector<double> entropy_data(history_.policy_entropy.begin(), history_.policy_entropy.end());
        std::vector<double> sharpness_data(history_.policy_sharpness.begin(), history_.policy_sharpness.end());
        
        ImPlot::SetupAxes("Iteration", "Value");
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 5, ImPlotCond_Once);
        ImPlot::PlotLine("Entropy (uncertainty)", x_data.data(), entropy_data.data(), x_data.size());
        ImPlot::PlotLine("Sharpness (confidence)", x_data.data(), sharpness_data.data(), x_data.size());
        ImPlot::SetupLegend(ImPlotLocation_NorthEast);
        
        ImPlot::EndPlot();
    }
    
    ImGui::Spacing();
    
    // MCTS Improvement (learning effectiveness)
    if (ImPlot::BeginPlot("MCTS Improvement Factor", ImVec2(-1, 200))) {
        std::vector<double> mcts_data(history_.mcts_improvement_avg.begin(), history_.mcts_improvement_avg.end());
        
        ImPlot::SetupAxes("Iteration", "Improvement");
        ImPlot::PlotLine("Avg Improvement", x_data.data(), mcts_data.data(), x_data.size());
        ImPlot::SetupLegend(ImPlotLocation_NorthEast);
        
        ImPlot::EndPlot();
    }
    
    ImGui::Spacing();
    
    // Gradient Norms by Layer (shows which layers are learning)
    if (!history_.gradient_norms_history.empty()) {
        if (ImPlot::BeginPlot("Gradient Norms (Layer Learning)", ImVec2(-1, 250))) {
            ImPlot::SetupAxes("Iteration", "Gradient Norm");
            ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
            
            for (const auto& [layer_name, norms] : history_.gradient_norms_history) {
                if (!norms.empty()) {
                    std::vector<double> norm_data(norms.begin(), norms.end());
                    ImPlot::PlotLine(layer_name.c_str(), x_data.data(), norm_data.data(), x_data.size());
                }
            }
            ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_Outside);
            
            ImPlot::EndPlot();
        }
    }
    
    ImGui::Spacing();
    
    // Win Rate and Game Length (if available)
    if (!history_.eval_winrate.empty()) {
        if (ImPlot::BeginPlot("Evaluation Metrics", ImVec2(-1, 200))) {
            std::vector<double> winrate_data(history_.eval_winrate.begin(), history_.eval_winrate.end());
            std::vector<double> game_len_data(history_.avg_game_length.begin(), history_.avg_game_length.end());
            
            ImPlot::SetupAxes("Iteration", "Win Rate / Game Length");
            ImPlot::PlotLine("Win Rate", x_data.data(), winrate_data.data(), x_data.size());
            ImPlot::SetupLegend(ImPlotLocation_NorthEast);
            
            ImPlot::EndPlot();
        }
    }
    
    ImGui::End();
}

}  // namespace azl
