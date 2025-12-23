#pragma once

#include "telemetry_client.h"

namespace azl {

/**
 * Render MCTS thinking visualization
 * Snake.cpp equivalent: network weight window + dynamic activity window
 */
class ThinkingRenderer {
public:
    ThinkingRenderer();
    
    void render(const FrameMessage& frame);
    
private:
    void render_visit_bars(const FrameMessage& frame);
    void render_policy_comparison(const FrameMessage& frame);
    void render_q_values(const FrameMessage& frame);
};

}  // namespace azl
