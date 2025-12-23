#include <iostream>
#include <string>
#include <memory>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>

#include "telemetry_client.h"
#include "render_board.h"
#include "render_thinking.h"
#include "render_metrics.h"

using namespace azl;

// Parse command line arguments
struct Args {
    std::string telemetry_endpoint = "tcp://127.0.0.1:5556";
    bool help = false;
};

Args parse_args(int argc, char** argv) {
    Args args;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--telemetry" && i + 1 < argc) {
            args.telemetry_endpoint = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            args.help = true;
        }
    }
    
    return args;
}

void print_usage(const char* prog_name) {
    std::cout << "AlphaZero-Light C++ Viewer\n";
    std::cout << "Usage: " << prog_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --telemetry <endpoint>  ZeroMQ endpoint (default: tcp://127.0.0.1:5556)\n";
    std::cout << "  --help, -h              Show this help message\n\n";
    std::cout << "Controls:\n";
    std::cout << "  P         Pause/Resume rendering\n";
    std::cout << "  Space     Force render next frame\n";
    std::cout << "  ESC       Quit\n";
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    
    if (args.help) {
        print_usage(argv[0]);
        return 0;
    }
    
    std::cout << "\n";
    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << "  AlphaZero-Light C++ Viewer  \n";
    std::cout << "  Snake.cpp-style Live Training Visualization\n";
    std::cout << "═══════════════════════════════════════════════════\n\n";
    std::cout << "📡 Endpoint: " << args.telemetry_endpoint << "\n";
    std::cout << "🎮 Controls: P=Pause, Space=Force Render, ESC=Quit\n\n";
    
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
        return 1;
    }
    
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
    
    // Create window with graphics context
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
    
    SDL_Window* window = SDL_CreateWindow(
        "AlphaZero-Light Viewer",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        1600, 900,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI
    );
    
    if (!window) {
        std::cerr << "SDL_CreateWindow Error: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }
    
    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    SDL_GL_MakeCurrent(window, gl_context);
    SDL_GL_SetSwapInterval(1); // Enable vsync
    
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    
    // Setup Platform/Renderer backends
    ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL3_Init(glsl_version);
    
    // Initialize telemetry client
    TelemetryClient telemetry(args.telemetry_endpoint);
    telemetry.start();
    
    // Initialize renderers
    BoardRenderer board_renderer;
    ThinkingRenderer thinking_renderer;
    MetricsRenderer metrics_renderer;
    
    // Main state
    bool running = true;
    bool paused = false;
    FrameMessage current_frame{};
    bool has_frame = false;
    
    std::cout << "🚀 Viewer started - waiting for telemetry data...\n\n";
    
    // Main loop
    while (running) {
        // Poll and handle events
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            
            if (event.type == SDL_QUIT) {
                running = false;
            }
            
            if (event.type == SDL_KEYDOWN) {
                switch (event.key.keysym.sym) {
                    case SDLK_ESCAPE:
                        running = false;
                        break;
                    case SDLK_p:
                        paused = !paused;
                        std::cout << (paused ? "⏸  Paused" : "▶  Resumed") << std::endl;
                        break;
                    case SDLK_SPACE:
                        std::cout << "⏭  Force render" << std::endl;
                        break;
                }
            }
        }
        
        // Update: Read latest telemetry
        if (!paused) {
            FrameMessage new_frame;
            if (telemetry.get_latest_frame(new_frame)) {
                current_frame = new_frame;
                has_frame = true;
            }
            
            MetricsMessage metrics;
            if (telemetry.get_latest_metrics(metrics)) {
                metrics_renderer.add_metrics(metrics);
            }
        }
        
        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        
        // Status window
        ImGui::Begin("Status", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text("Telemetry: %s", telemetry.is_connected() ? "🟢 Connected" : "🔴 Disconnected");
        ImGui::Text("Frames received: %d", telemetry.get_frames_received());
        ImGui::Text("Metrics received: %d", telemetry.get_metrics_received());
        ImGui::Spacing();
        ImGui::Text("Rendering: %s", paused ? "PAUSED" : "LIVE");
        ImGui::Text("FPS: %.1f", io.Framerate);
        ImGui::End();
        
        // Render content windows
        if (has_frame) {
            board_renderer.render(current_frame);
            thinking_renderer.render(current_frame);
        }
        
        metrics_renderer.render();
        
        // Rendering
        ImGui::Render();
        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(window);
    }
    
    // Cleanup
    std::cout << "\n🛑 Shutting down...\n";
    telemetry.stop();
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    
    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    std::cout << "✅ Shutdown complete\n";
    return 0;
}
