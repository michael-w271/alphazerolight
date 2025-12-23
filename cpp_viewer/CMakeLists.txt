# AlphaZero-Light C++ Viewer
# Snake.cpp-style live training visualization

cmake_minimum_required(VERSION 3.15)
project(azl_viewer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find required packages
find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

# Dear ImGui (expected in deps/imgui)
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/deps/imgui)
if(NOT EXISTS ${IMGUI_DIR})
    message(STATUS "ImGui not found, will need to be cloned:")
    message(STATUS "  git clone https://github.com/ocornut/imgui.git deps/imgui")
endif()

# ImPlot (expected in deps/implot)
set(IMPLOT_DIR ${CMAKE_SOURCE_DIR}/deps/implot)
if(NOT EXISTS ${IMPLOT_DIR})
    message(STATUS "ImPlot not found, will need to be cloned:")
    message(STATUS "  git clone https://github.com/epezent/implot.git deps/implot")
endif()

# JSON library (header-only, expected in deps/json)
set(JSON_DIR ${CMAKE_SOURCE_DIR}/deps/json)
if(NOT EXISTS ${JSON_DIR})
    message(STATUS "nlohmann/json not found, will need to be cloned:")
    message(STATUS "  git clone https://github.com/nlohmann/json.git deps/json")
endif()

# ImGui sources
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl2.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# ImPlot sources
set(IMPLOT_SOURCES
    ${IMPLOT_DIR}/implot.cpp
    ${IMPLOT_DIR}/implot_items.cpp
)

# Viewer sources
set(VIEWER_SOURCES
    src/main.cpp
    src/telemetry_client.cpp
    src/render_board.cpp
    src/render_thinking.cpp
    src/render_metrics.cpp
)

# Executable
add_executable(azl_viewer 
    ${VIEWER_SOURCES}
    ${IMGUI_SOURCES}
    ${IMPLOT_SOURCES}
)

# Include directories
target_include_directories(azl_viewer PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${IMPLOT_DIR}
    ${JSON_DIR}/include
    ${SDL2_INCLUDE_DIRS}
    ${ZMQ_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(azl_viewer PRIVATE
    ${SDL2_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${ZMQ_LIBRARIES}
    dl
    pthread
)

# Compiler flags
target_compile_options(azl_viewer PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -O2
)

# Print configuration
message(STATUS "=== AlphaZero-Light Viewer Configuration ===")
message(STATUS "SDL2: ${SDL2_LIBRARIES}")
message(STATUS "ZMQ: ${ZMQ_LIBRARIES}")
message(STATUS "ImGui: ${IMGUI_DIR}")
message(STATUS "ImPlot: ${IMPLOT_DIR}")
message(STATUS "JSON: ${JSON_DIR}")
