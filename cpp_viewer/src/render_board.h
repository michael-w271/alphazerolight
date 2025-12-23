#pragma once

#include "telemetry_client.h"
#include <imgui.h>

namespace azl {

/**
 * Render Connect Four board (6x7 grid with discs)
 * Snake.cpp equivalent: game window
 */
class BoardRenderer {
public:
    BoardRenderer();
    
    void render(const FrameMessage& frame);
    
private:
    void draw_grid();
    void draw_disc(int row, int col, int player, bool highlight = false);
    void draw_info(const FrameMessage& frame);
    
    static constexpr float CELL_SIZE = 60.0f;
    static constexpr float DISC_RADIUS = 25.0f;
    static constexpr int ROWS = 6;
    static constexpr int COLS = 7;
};

}  // namespace azl
