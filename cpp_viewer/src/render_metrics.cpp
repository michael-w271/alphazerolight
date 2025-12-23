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
    ImGui::Begin("Training Metrics", nullptr);
    
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
    
    ImGui::Text("Latest (Iteration %d):", latest_iter);
    ImGui::Indent();
    ImGui::Text("Total Loss: %.4f", latest_total);
    ImGui::Text("Policy Loss: %.4f", latest_policy);
    ImGui::Text("Value Loss: %.4f", latest_value);
    ImGui::Unindent();
    ImGui::Spacing();
    
    // Plot loss curves
    if (ImPlot::BeginPlot("Loss Curves", ImVec2(-1, 300))) {
        // Convert deques to vectors for ImPlot
        std::vector<double> x_data(history_.iterations.begin(), history_.iterations.end());
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
    
    ImGui::End();
}

}  // namespace azl
