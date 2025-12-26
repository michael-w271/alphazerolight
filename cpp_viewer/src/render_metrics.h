#pragma once

#include "telemetry_client.h"
#include <vector>
#include <deque>

namespace azl {

/**
 * Render training metrics (loss curves, win rate, etc.)
 */
class MetricsRenderer {
public:
    MetricsRenderer();
    
    void add_metrics(const MetricsMessage& metrics);
    void render();
    
private:
    struct MetricsHistory {
        std::deque<int> iterations;
        std::deque<float> total_loss;
        std::deque<float> policy_loss;
        std::deque<float> value_loss;
        std::deque<float> policy_entropy;
        std::deque<float> policy_sharpness;
        std::deque<float> mcts_improvement_avg;
        std::deque<float> eval_winrate;
        std::deque<float> avg_game_length;
        
        // Store gradient norms per layer over time
        std::map<std::string, std::deque<float>> gradient_norms_history;
        
        static constexpr size_t MAX_HISTORY = 1000;
        
        void add(const MetricsMessage& m) {
            iterations.push_back(m.iteration);
            total_loss.push_back(m.total_loss);
            policy_loss.push_back(m.policy_loss);
            value_loss.push_back(m.value_loss);
            policy_entropy.push_back(m.policy_entropy);
            policy_sharpness.push_back(m.policy_sharpness);
            mcts_improvement_avg.push_back(m.mcts_improvement_avg);
            eval_winrate.push_back(m.eval_winrate);
            avg_game_length.push_back(m.avg_game_length);
            
            // Store gradient norms for each layer
            for (const auto& [layer_name, norm] : m.gradient_norms) {
                gradient_norms_history[layer_name].push_back(norm);
                if (gradient_norms_history[layer_name].size() > MAX_HISTORY) {
                    gradient_norms_history[layer_name].pop_front();
                }
            }
            
            if (iterations.size() > MAX_HISTORY) {
                iterations.pop_front();
                total_loss.pop_front();
                policy_loss.pop_front();
                value_loss.pop_front();
                policy_entropy.pop_front();
                policy_sharpness.pop_front();
                mcts_improvement_avg.pop_front();
                eval_winrate.pop_front();
                avg_game_length.pop_front();
            }
        }
    };
    
    MetricsHistory history_;
};

}  // namespace azl
