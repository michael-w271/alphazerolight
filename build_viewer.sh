#!/bin/bash
# Build script for AlphaZero-Light C++ Viewer

set -e

echo "================================================"
echo "  AlphaZero-Light C++ Viewer - Build Script"
echo "================================================"
echo""

# Check for dependencies
echo "📦 Checking dependencies..."

# Check for SDL2
if ! pkg-config --exists sdl2; then
    echo "❌ SDL2 not found. Please install:"
    echo "   Ubuntu/Debian: sudo apt install libsdl2-dev"
    echo "   Fedora: sudo dnf install SDL2-devel"
    exit 1
fi

# Check for ZeroMQ
if ! pkg-config --exists libzmq; then
    echo "❌ ZeroMQ not found. Please install:"
    echo "   Ubuntu/Debian: sudo apt install libzmq3-dev"
    echo "   Fedora: sudo dnf install zeromq-devel"
    exit 1
fi

echo "✅ System dependencies found"
echo ""

# Clone header-only libraries if needed
echo "📚 Checking header-only libraries..."

cd cpp_viewer

if [ ! -d "deps/imgui" ]; then
    echo "Cloning Dear ImGui..."
    git clone https://github.com/ocornut/imgui.git deps/imgui
fi

if [ ! -d "deps/implot" ]; then
    echo "Cloning ImPlot..."
    git clone https://github.com/epezent/implot.git deps/implot
fi

if [ ! -d "deps/json" ]; then
    echo "Cloning nlohmann/json..."
    git clone https://github.com/nlohmann/json.git deps/json
fi

echo "✅ Header-only libraries ready"
echo ""

# Build
echo "🔨 Building..."
mkdir -p build
cd build
cmake ..
make -j$(nproc)

echo ""
echo "✅ Build complete!"
echo ""
echo "To run the viewer:"
echo "  ./cpp_viewer/build/azl_viewer --telemetry tcp://127.0.0.1:5556"
echo ""
echo "To start demo training:"
echo "  python -m alpha_zero_light.training.trainer_demo \\"
echo "    --telemetry tcp://127.0.0.1:5556 \\"
echo "    --device cuda \\"
echo "    --demo"
