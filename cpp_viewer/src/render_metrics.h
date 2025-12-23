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
        
        static constexpr size_t MAX_HISTORY = 1000;
        
        void add(const MetricsMessage& m) {
            iterations.push_back(m.iteration);
            total_loss.push_back(m.total_loss);
            policy_loss.push_back(m.policy_loss);
            value_loss.push_back(m.value_loss);
            
            if (iterations.size() > MAX_HISTORY) {
                iterations.pop_front();
                total_loss.pop_front();
                policy_loss.pop_front();
                value_loss.pop_front();
            }
        }
    };
    
    MetricsHistory history_;
};

}  // namespace azl
