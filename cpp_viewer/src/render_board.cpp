#include "render_board.h"
#include <imgui.h>

namespace azl {

BoardRenderer::BoardRenderer() {}

void BoardRenderer::render(const FrameMessage& frame) {
    ImGui::Begin("Connect Four - Training", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    
    draw_info(frame);
    ImGui::Spacing();
    
    // Get canvas position before drawing grid
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    draw_grid();
    
    // Draw discs on top of the grid
    if (!frame.board.empty()) {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        
        for (int row = 0; row < ROWS && row < static_cast<int>(frame.board.size()); row++) {
            for (int col = 0; col < COLS && col < static_cast<int>(frame.board[row].size()); col++) {
                int player = frame.board[row][col];
                if (player != 0) {
                    // Calculate disc center
                    float center_x = canvas_pos.x + col * CELL_SIZE + CELL_SIZE / 2;
                    float center_y = canvas_pos.y + row * CELL_SIZE + CELL_SIZE / 2;
                    ImVec2 center(center_x, center_y);
                    
                    // Color based on player
                    ImU32 color = player == 1 
                        ? IM_COL32(220, 50, 50, 255)   // Red for +1
                        : IM_COL32(255, 220, 50, 255); // Yellow for -1
                    
                    // Highlight last move with a green ring
                    bool is_last_move = (frame.chosen_action == col && !frame.is_terminal);
                    if (is_last_move) {
                        draw_list->AddCircle(center, DISC_RADIUS + 5, IM_COL32(0, 255, 0, 255), 32, 3.0f);
                    }
                    
                    // Draw disc (filled circle)
                    draw_list->AddCircleFilled(center, DISC_RADIUS, color, 32);
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

}  // namespace azl
