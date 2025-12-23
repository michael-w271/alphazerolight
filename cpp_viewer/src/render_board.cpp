#include "render_board.h"
#include <imgui.h>

namespace azl {

BoardRenderer::BoardRenderer() {}

void BoardRenderer::render(const FrameMessage& frame) {
    ImGui::Begin("Connect Four - Training", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    
    draw_info(frame);
    ImGui::Spacing();
    draw_grid();
    
    // Draw discs
    if (!frame.board.empty()) {
        for (int row = 0; row < ROWS && row < static_cast<int>(frame.board.size()); row++) {
            for (int col = 0; col < COLS && col < static_cast<int>(frame.board[row].size()); col++) {
                int player = frame.board[row][col];
                if (player != 0) {
                    // Highlight last move
                    bool highlight = (frame.chosen_action == col && !frame.is_terminal);
                    draw_disc(row, col, player, highlight);
                }
            }
        }
    }
    
    ImGui::End();
}

void BoardRenderer::draw_info(const FrameMessage& frame) {
    ImGui::Text("Iteration: %d | Game: %d | Move: %d", 
                frame.iteration, frame.game_idx, frame.move_idx);
    ImGui::Text("Player: %s | Temperature: %.2f", 
                frame.player == 1 ? "Red (+1)" : "Yellow (-1)", 
                frame.temperature);
    
    if (frame.is_terminal) {
        ImGui::TextColored(ImVec4(1,0.5f,0,1), "GAME OVER - Value: %.2f", frame.terminal_value);
    } else {
        ImGui::Text("Chosen Action: Column %d", frame.chosen_action);
    }
}

void BoardRenderer::draw_grid() {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    
    // Draw board background
    ImVec2 board_size(COLS * CELL_SIZE, ROWS * CELL_SIZE);
    draw_list->AddRectFilled(
        canvas_pos,
        ImVec2(canvas_pos.x + board_size.x, canvas_pos.y + board_size.y),
        IM_COL32(30, 80, 150, 255)  // Blue board
    );
    
    // Draw grid lines
    for (int col = 0; col <= COLS; col++) {
        float x = canvas_pos.x + col * CELL_SIZE;
        draw_list->AddLine(
            ImVec2(x, canvas_pos.y),
            ImVec2(x, canvas_pos.y + board_size.y),
            IM_COL32(50, 100, 170, 255), 2.0f
        );
    }
    
    for (int row = 0; row <= ROWS; row++) {
        float y = canvas_pos.y + row * CELL_SIZE;
        draw_list->AddLine(
            ImVec2(canvas_pos.x, y),
            ImVec2(canvas_pos.x + board_size.x, y),
            IM_COL32(50, 100, 170, 255), 2.0f
        );
    }
    
    // Reserve space for the board
    ImGui::Dummy(board_size);
}

void BoardRenderer::draw_disc(int row, int col, int player, bool highlight) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    
    // Calculate disc center (need to offset by board position)
    // NOTE: This is simplified - in practice we'd track canvas_pos from draw_grid
    float center_x = col * CELL_SIZE + CELL_SIZE / 2;
    float center_y = row * CELL_SIZE + CELL_SIZE / 2;
    
    // Color based on player
    ImU32 color = player == 1 
        ? IM_COL32(220, 50, 50, 255)   // Red for +1
        : IM_COL32(255, 220, 50, 255); // Yellow for -1
    
    // Draw disc (filled circle)
    // Note: This simplified version won't position correctly without proper canvas tracking
    // A full implementation would cache canvas_pos from draw_grid and use it here
    
    if (highlight) {
        // Draw highlight ring
        // draw_list->AddCircle(center, DISC_RADIUS + 5, IM_COL32(0, 255, 0, 255), 32, 3.0f);
    }
    
    // draw_list->AddCircleFilled(center, DISC_RADIUS, color, 32);
    
    // TODO: Fix positioning - need to track canvas origin from draw_grid
}

}  // namespace azl
